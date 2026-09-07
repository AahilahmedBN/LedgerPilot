"""
Kavach/LedgerPilot — Flask API

Two endpoints:
  POST /api/analyze-text  — manual entry: description + amount + flags
  POST /api/analyze-image — upload a receipt photo, OCR does the reading

Both call the SAME pipeline.process_transaction() underneath — the only
difference is where the description/amount text comes from. This is
deliberate: one pipeline function, two entry points, so behavior can never
silently diverge between "typed in" and "photographed" transactions.
"""

import os
import tempfile
import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS

from pipeline import process_transaction, extract_amount_from_text
from preprocessing import preprocess_pipeline
from ocr_extract import extract_text

app = Flask(__name__)
CORS(app)  # frontend runs on a different port during development — needed for that to work at all


@app.route("/api/health", methods=["GET"])
def health():
    """Quick check that the server (and all the loaded models) came up
    correctly — call this first when debugging 'nothing works'."""
    return jsonify({"status": "ok"})


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
