"""
Kavach — Phase 4: Audit / Decision Log (hardening requirement #11)

WHAT: every categorization decision and risk flag gets a timestamped,
structured log entry recording what was decided, on what basis, at what
confidence. This is what lets you (and a viva panel) answer "why did the
system flag this transaction?" for any specific record, months later.

FORMAT: JSON Lines (.jsonl) — one JSON object per line, append-only. Chosen
over a single JSON array because appending to a JSON array file safely
requires rewriting the whole file; JSONL lets you just append a line, which
matters once this is a live web app with concurrent uploads, not just a
test script.

WHAT THIS IS NOT: this is not a tamper-proof/cryptographically-signed audit
trail (that would need hash-chaining, which is out of scope here) — say
that plainly if asked in viva. It's a structured, honest record of
decisions, not a blockchain-grade guarantee.
"""

import json
import os
from datetime import datetime, timezone

LOG_FILENAME = "audit_log.jsonl"


def _write_entry(entry, log_path=LOG_FILENAME):
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def log_categorization_decision(
    transaction_id, predicted_category, confidence, needs_review,
    model_version="tfidf_logreg_v1", log_path=LOG_FILENAME,
):
    """Call this every time the categorization model makes a prediction —
    not just the ones that get flagged. A complete log needs the trusted
    decisions too, or you can't compute honest statistics on how often the
    threshold actually fires."""
    entry = {
        "event_type": "categorization_decision",
        "transaction_id": transaction_id,
        "predicted_category": predicted_category,
        "confidence": round(float(confidence), 4),
        "needs_review": bool(needs_review),
        "model_version": model_version,
    }
    return _write_entry(entry, log_path)


def log_risk_flag(
    transaction_id, risk_level, basis, confidence,
    model_version="xgboost_risk_v1", log_path=LOG_FILENAME,
):
    """For Phase 6 (penalty/ITC-risk classifier) — logs what was flagged,
    why (basis), and how confident the model was. 'basis' should be a
    short human-readable reason, not just a raw feature vector — this is
    what a compliance officer would actually read."""
    entry = {
        "event_type": "risk_flag",
        "transaction_id": transaction_id,
        "risk_level": risk_level,
        "basis": basis,
        "confidence": round(float(confidence), 4),
        "model_version": model_version,
    }
    return _write_entry(entry, log_path)


def log_itc_determination(
    transaction_id, category, itc_eligible, reason, log_path=LOG_FILENAME,
):
    """For the deterministic rule engine (Phase 5) — this one has no
    confidence score, because rule-engine output is deterministic, not
    probabilistic. Keeping this as a separate function (not reusing
    log_categorization_decision) makes that distinction explicit in the
    log itself — don't blur ML decisions and rule-engine decisions
    together, they need different kinds of scrutiny."""
    entry = {
        "event_type": "itc_determination",
        "transaction_id": transaction_id,
        "category": category,
        "itc_eligible": itc_eligible,
        "reason": reason,
    }
    return _write_entry(entry, log_path)


def read_log(log_path=LOG_FILENAME):
    """Reads the full log back as a list of dicts — for building the
    admin/maintainer review dashboard later, or for spot-checking in a
    demo."""
    if not os.path.exists(log_path):
        return []
    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def summarize_log(log_path=LOG_FILENAME):
    """Quick stats for a demo: how many decisions logged, how many flagged
    for review, breakdown by event type."""
    entries = read_log(log_path)
    if not entries:
        return {"total_entries": 0}

    by_type = {}
    needs_review_count = 0
    for e in entries:
        et = e.get("event_type", "unknown")
        by_type[et] = by_type.get(et, 0) + 1
        if e.get("needs_review"):
            needs_review_count += 1

    return {
        "total_entries": len(entries),
        "by_event_type": by_type,
        "flagged_needs_review": needs_review_count,
    }
