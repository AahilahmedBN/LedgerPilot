"""
Kavach/LedgerPilot — End-to-End Pipeline Orchestrator

This is the file that makes 6 separate phases into ONE product. Given a
transaction (either raw OCR text or a manually-entered description +
amount), this runs, in order:

  1. Categorization (Phase 3) — what category is this transaction?
  2. Rule engine (Phase 5) — given that category and the transaction date,
     what's the GST split and ITC eligibility, per the versioned rule KB?
  3. Risk classifier (Phase 6) — given the ITC eligibility from step 2 and
     other signals (late filing, GSTIN validity), what's the penalty
     risk level?
  4. Audit log (Phase 4) — record the categorization decision AND the ITC
     determination, so every step is traceable after the fact.

KEY DESIGN POINT: the risk classifier's 'itc_claimed_but_blocked' feature
comes from the RULE ENGINE's determination, not a separate guess — this
is what makes steps 2 and 3 genuinely connected rather than coincidentally
adjacent. If the categorization model is wrong, that error propagates
through the whole chain — which is realistic, and worth saying honestly
in viva: this pipeline's accuracy is bounded by its weakest stage.
"""

import re
from datetime import date

import joblib
import pandas as pd

from rule_kb import get_connection, create_schema, seed_initial_rules
from rule_engine import evaluate_transaction, RuleNotFoundError
from audit_log import log_categorization_decision, log_itc_determination, log_risk_flag

CATEGORIZATION_CONFIDENCE_THRESHOLD = 0.55

# Loaded once at import time — these are small classical ML models,
# loading them per-request would be wasteful for no benefit.
_cat_model = joblib.load("categorization_model.joblib")
_cat_vectorizer = joblib.load("tfidf_vectorizer.joblib")
_risk_model = joblib.load("risk_classifier.joblib")
_risk_label_encoder = joblib.load("risk_label_encoder.joblib")
_risk_feature_columns = joblib.load("risk_feature_columns.joblib")

# Ensure the rule KB database exists and is seeded — safe to call every
# startup, seed_initial_rules() skips if already populated.
_rule_conn = get_connection()
create_schema(_rule_conn)
seed_initial_rules(_rule_conn)


def extract_amount_from_text(text):
    """Simple regex-based amount extraction from OCR text — looks for
    'Total: Rs. X' or similar patterns. THIS IS A SIMPLIFICATION, say so
    honestly: real invoice parsing (distinguishing subtotal/tax/total,
    handling varied formats) is a meaningfully harder problem than this
    one regex. For the demo, this is good enough to extract SOMETHING
    real from OCR text rather than requiring manual entry every time."""
    matches = re.findall(r"Total[:\s]*Rs\.?\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
    if matches:
        return float(matches[-1].replace(",", ""))
    # Fallback: any Rs. amount in the text, take the largest (often the total)
    all_amounts = re.findall(r"Rs\.?\s*([\d,]+\.?\d*)", text)
    if all_amounts:
        return max(float(a.replace(",", "")) for a in all_amounts)
    return None


def categorize(description_text):
    """Phase 3 step: predict category from text, with the confidence
    threshold safety net."""
    vec = _cat_vectorizer.transform([description_text])
    category_key = _cat_model.predict(vec)[0]
    confidence = float(_cat_model.predict_proba(vec).max())
    needs_review = confidence < CATEGORIZATION_CONFIDENCE_THRESHOLD
    return category_key, confidence, needs_review


def assess_risk(category_key, amount, tax_amount, itc_eligible, claimed_itc, late_filing_days, has_valid_gstin, is_round_number):
    """Phase 6 step: predict risk level. Builds the exact feature row the
    trained model expects, including one-hot category columns — must
    match train_risk_classifier.py's feature construction exactly, or
    predictions will be silently wrong (this is the #1 real-world bug
    source when wiring a saved model into an application). tax_amount
    MUST come from the rule engine's actual GST computation, not a guess —
    an earlier version of this function got that wrong; fixed before this
    ever reached the repo."""
    row = {col: 0 for col in _risk_feature_columns}
    row["amount"] = amount
    row["tax_amount"] = tax_amount
    row["late_filing_days"] = late_filing_days
    row["claimed_itc"] = int(claimed_itc)
    row["itc_claimed_but_blocked"] = int(claimed_itc and itc_eligible == "Blocked")
    row["has_valid_gstin"] = int(has_valid_gstin)
    row["is_round_number_amount"] = int(is_round_number)
    cat_col = f"cat_{category_key}"
    if cat_col in row:
        row[cat_col] = 1

    X = pd.DataFrame([row])[_risk_feature_columns]
    pred_encoded = _risk_model.predict(X)[0]
    risk_level = _risk_label_encoder.inverse_transform([pred_encoded])[0]
    confidence = float(_risk_model.predict_proba(X).max())
    return risk_level, confidence


def process_transaction(
    transaction_id, description_text, amount, transaction_date=None,
    is_intra_state=True, claimed_itc=True, late_filing_days=0, has_valid_gstin=True,
):
    """THE main entry point — one transaction in, full compliance result out.
    This is what both the Flask API and a future WhatsApp bot would call."""
    transaction_date = transaction_date or date.today().isoformat()
    is_round_number = (amount == round(amount / 1000) * 1000)

    # Step 1: Categorization
    category_key, cat_confidence, needs_review = categorize(description_text)
    log_categorization_decision(transaction_id, category_key, cat_confidence, needs_review)

    # Step 2: Rule engine (GST + ITC), date-aware
    try:
        rule_result = evaluate_transaction(category_key, amount, transaction_date, is_intra_state, conn=_rule_conn)
    except RuleNotFoundError as e:
        return {"error": str(e), "category_key": category_key}

    log_itc_determination(
        transaction_id, category_key, rule_result["itc_eligible"], rule_result["itc_reason"],
    )

    # Step 3: Risk classifier — uses the RULE ENGINE's itc_eligible AND its
    # actual computed tax amount, not a guess
    total_tax = rule_result["gst_split"]["total_tax"]
    risk_level, risk_confidence = assess_risk(
        category_key, amount, total_tax, rule_result["itc_eligible"], claimed_itc,
        late_filing_days, has_valid_gstin, is_round_number,
    )
    log_risk_flag(
        transaction_id, risk_level,
        basis=f"category={category_key}, itc_eligible={rule_result['itc_eligible']}, "
              f"claimed_itc={claimed_itc}, late_filing_days={late_filing_days}",
        confidence=risk_confidence,
    )

    return {
        "transaction_id": transaction_id,
        "description": description_text,
        "amount": amount,
        "transaction_date": transaction_date,
        "category": category_key,
        "categorization_confidence": round(cat_confidence, 3),
        "categorization_needs_review": needs_review,
        "gst_rate": rule_result["gst_rate"],
        "gst_split": rule_result["gst_split"],
        "itc_eligible": rule_result["itc_eligible"],
        "itc_reason": rule_result["itc_reason"],
        "risk_level": risk_level,
        "risk_confidence": round(risk_confidence, 3),
    }
