"""
Kavach/LedgerPilot — Flask API

Core endpoints:
  POST /api/analyze-text  — manual entry: description + amount + flags
  POST /api/analyze-image — upload a receipt photo, OCR does the reading
  POST /api/chat          — plain-language Q&A grounded in real backend data

analyze-text and analyze-image both call the SAME pipeline.process_transaction()
underneath — the only difference is where the description/amount text comes
from. This is deliberate: one pipeline function, two entry points, so
behavior can never silently diverge between "typed in" and "photographed"
transactions.
"""

import json
import os
import tempfile
import uuid
from datetime import date

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

from pipeline import process_transaction, extract_amount_from_text
from preprocessing import preprocess_pipeline
from ocr_extract import extract_text
from categories import CATEGORIES
from rule_kb import get_connection
from rule_engine import get_gst_rate, get_itc_determination
from audit_log import get_transaction_history
from chat_context import build_chat_context

load_dotenv()

GEMINI_MODEL = "gemini-flash-lite-latest"
CHAT_SYSTEM_INSTRUCTION = (
    "You are LedgerPilot's compliance assistant. Only answer using the JSON "
    "context provided below. If the answer is not present in it, say so "
    "honestly instead of guessing. Never state a GST rate, ITC rule, or "
    "number that is not present in the context.\n\nCONTEXT:\n{context_json}"
)

app = Flask(__name__)
CORS(app)  # frontend runs on a different port during development — needed for that to work at all


@app.route("/api/health", methods=["GET"])
def health():
    """Quick check that the server (and all the loaded models) came up
    correctly — call this first when debugging 'nothing works'."""
    return jsonify({"status": "ok"})


@app.route("/api/rules", methods=["GET"])
def get_rules():
    """Read-only: the 14 categories' GST rate + ITC eligibility as
    currently ACTIVE in the versioned rule KB (rule_kb.py), not
    categories.py's static defaults — so this reflects any admin-issued
    rate change (rule_engine.add_rate_change()). Optional ?as_of=YYYY-MM-DD
    to see what was active on a past date; defaults to today."""
    as_of = request.args.get("as_of") or date.today().isoformat()
    conn = get_connection()
    try:
        rules = []
        for cat in CATEGORIES:
            gst_rate = get_gst_rate(cat.key, as_of, conn=conn)
            itc_eligible, itc_reason = get_itc_determination(cat.key, as_of, conn=conn)
            rules.append({
                "key": cat.key,
                "label": cat.label,
                "gst_rate": gst_rate,
                "itc_eligible": itc_eligible,
                "itc_reason": itc_reason,
            })
    finally:
        conn.close()
    return jsonify({"as_of": as_of, "rules": rules})


@app.route("/api/history", methods=["GET"])
def get_history():
    """Read-only: the real transaction history, regrouped from Phase 4's
    append-only audit_log.jsonl (one line per decision event) into one
    record per transaction — most recent first. Transactions logged
    before description/amount/gst_rate/gst_split were added to the audit
    log will have those fields as null; that's a real gap in old data,
    not something this endpoint should paper over."""
    records = get_transaction_history()
    records.reverse()
    return jsonify({"transactions": records})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Expects JSON: {message: str}. Grounds Gemini strictly in real backend
    data (last 20 audit-log transactions, the current stats summary, and the
    full categories.py rule book) — stateless, single-turn, no conversation
    memory carried between requests."""
    data = request.get_json()
    if not data or "message" not in data or not str(data["message"]).strip():
        return jsonify({"error": "Request must include a non-empty 'message'"}), 400

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "Chat is not configured: GEMINI_API_KEY is not set"}), 503

    context = build_chat_context()
    system_instruction = CHAT_SYSTEM_INSTRUCTION.format(
        context_json=json.dumps(context, ensure_ascii=False)
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=data["message"],
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )
        reply = response.text
    except Exception:
        return jsonify({"error": "The chat assistant is temporarily unavailable. Please try again."}), 502

    return jsonify({"reply": reply})


@app.route("/api/analyze-text", methods=["POST"])
def analyze_text():
    """Expects JSON: {description, amount, transaction_date?, is_intra_state?,
    claimed_itc?, late_filing_days?, has_valid_gstin?}"""
    data = request.get_json()
    if not data or "description" not in data or "amount" not in data:
        return jsonify({"error": "Request must include 'description' and 'amount'"}), 400

    transaction_id = f"TXN-{uuid.uuid4().hex[:8]}"
    result = process_transaction(
        transaction_id=transaction_id,
        description_text=data["description"],
        amount=float(data["amount"]),
        transaction_date=data.get("transaction_date"),
        is_intra_state=data.get("is_intra_state", True),
        claimed_itc=data.get("claimed_itc", True),
        late_filing_days=data.get("late_filing_days", 0),
        has_valid_gstin=data.get("has_valid_gstin", True),
    )
    return jsonify(result)


@app.route("/api/analyze-image", methods=["POST"])
def analyze_image():
    """Expects a multipart form upload with field name 'document'.
    Runs the FULL Phase 2 pipeline (OpenCV preprocessing + Tesseract OCR)
    before handing off to the same process_transaction() as the text endpoint."""
    if "document" not in request.files:
        return jsonify({"error": "No file uploaded under field name 'document'"}), 400

    file = request.files["document"]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        processed_image, meta = preprocess_pipeline(tmp_path)
        ocr_text, ocr_confidence = extract_text(processed_image)

        amount = extract_amount_from_text(ocr_text)
        if amount is None:
            return jsonify({
                "error": "Could not extract an amount from the document text",
                "ocr_text": ocr_text,
                "ocr_confidence": ocr_confidence,
            }), 422

        transaction_id = f"TXN-{uuid.uuid4().hex[:8]}"
        result = process_transaction(
            transaction_id=transaction_id,
            description_text=ocr_text,
            amount=amount,
        )
        result["ocr_confidence"] = ocr_confidence
        result["ocr_text"] = ocr_text
        result["preprocessing_meta"] = meta
        return jsonify(result)
    finally:
        os.remove(tmp_path)  # never leave uploaded documents sitting on disk unencrypted


if __name__ == "__main__":
    app.run(debug=True, port=5000)
