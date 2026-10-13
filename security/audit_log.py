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
    description=None, amount=None,
    model_version="tfidf_logreg_v1", log_path=LOG_FILENAME,
):
    """Call this every time the categorization model makes a prediction —
    not just the ones that get flagged. A complete log needs the trusted
    decisions too, or you can't compute honest statistics on how often the
    threshold actually fires.

    description/amount are optional (default None) so older callers (e.g.
    security/test_hardening.py) keep working unmodified — they exist so
    the audit log can double as the source of truth for a transaction
    history view, not just a decision trail. Entries logged before these
    params existed will simply have description/amount as null."""
    entry = {
        "event_type": "categorization_decision",
        "transaction_id": transaction_id,
        "predicted_category": predicted_category,
        "confidence": round(float(confidence), 4),
        "needs_review": bool(needs_review),
        "description": description,
        "amount": amount,
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
    transaction_id, category, itc_eligible, reason,
    gst_rate=None, gst_split=None, log_path=LOG_FILENAME,
):
    """For the deterministic rule engine (Phase 5) — this one has no
    confidence score, because rule-engine output is deterministic, not
    probabilistic. Keeping this as a separate function (not reusing
    log_categorization_decision) makes that distinction explicit in the
    log itself — don't blur ML decisions and rule-engine decisions
    together, they need different kinds of scrutiny.

    gst_rate/gst_split are optional (default None) for the same backward-
    compatibility reason as log_categorization_decision's description/
    amount — see that docstring."""
    entry = {
        "event_type": "itc_determination",
        "transaction_id": transaction_id,
        "category": category,
        "itc_eligible": itc_eligible,
        "reason": reason,
        "gst_rate": gst_rate,
        "gst_split": gst_split,
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


def get_transaction_history(log_path=LOG_FILENAME):
    """Regroups the flat, append-only event log back into one record per
    transaction_id — the shape a transaction history view actually wants,
    vs. the 3-events-per-transaction stream this file stores it as.
    Returned in the order transactions first appeared (oldest first);
    callers wanting most-recent-first should reverse the result.

    Entries logged before description/amount/gst_rate/gst_split existed
    (see log_categorization_decision/log_itc_determination) will have
    those keys as None — real gaps in old data, not fabricated values."""
    by_txn = {}
    order = []
    for e in read_log(log_path):
        txn_id = e.get("transaction_id")
        if txn_id is None:
            continue
        if txn_id not in by_txn:
            by_txn[txn_id] = {"transaction_id": txn_id}
            order.append(txn_id)
        record = by_txn[txn_id]
        event_type = e.get("event_type")
        if event_type == "categorization_decision":
            record["description"] = e.get("description")
            record["amount"] = e.get("amount")
            record["category"] = e.get("predicted_category")
            record["categorization_confidence"] = e.get("confidence")
            record["categorization_needs_review"] = e.get("needs_review")
            record["analyzed_at"] = e.get("timestamp")
        elif event_type == "itc_determination":
            record["itc_eligible"] = e.get("itc_eligible")
            record["itc_reason"] = e.get("reason")
            record["gst_rate"] = e.get("gst_rate")
            record["gst_split"] = e.get("gst_split")
        elif event_type == "risk_flag":
            record["risk_level"] = e.get("risk_level")
            record["risk_confidence"] = e.get("confidence")
    return [by_txn[t] for t in order]


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
