"""
Kavach/LedgerPilot — Phase 5: Deterministic GST/TDS Rule Engine

DETERMINISTIC, NOT ML: given the same transaction (category, amount, date,
intra/inter-state), this ALWAYS returns the same answer. There is no model,
no confidence score, no "probably." This is intentional — tax law isn't a
probability, and this file should never contain a model.predict() call. If
anyone asks "why isn't this ML too," that's the answer.

DATE-AWARE LOOKUP: every computation takes a transaction_date and looks up
whichever rule was active on that date — see rule_kb.py's docstring for why.
"""

import sqlite3
from datetime import date

from rule_kb import get_connection


class RuleNotFoundError(Exception):
    """Raised when no rule covers the given category + date — this should
    never happen for a seeded category, but fails loudly rather than
    silently guessing if it does (e.g. a typo'd category_key, or a date
    before GST even existed)."""
    pass


def _find_active_rule(conn, table, category_key, as_of_date):
    """Core date-range lookup, shared by GST rate and ITC rule queries.
    'Active' means: effective_from <= as_of_date AND
    (effective_to IS NULL OR effective_to >= as_of_date)."""
    row = conn.execute(
        f"""
        SELECT * FROM {table}
        WHERE category_key = ?
          AND effective_from <= ?
          AND (effective_to IS NULL OR effective_to >= ?)
        ORDER BY effective_from DESC
        LIMIT 1
        """,
        (category_key, as_of_date, as_of_date),
    ).fetchone()

    if row is None:
        raise RuleNotFoundError(
            f"No {table} entry covers category '{category_key}' as of {as_of_date}. "
            f"Check the category_key is valid and a rule was seeded for this date range."
        )
    return row


def get_gst_rate(category_key, as_of_date, conn=None):
    """Returns the GST rate (as a string, e.g. '18' or 'exempt') that was
    active for this category on this date."""
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        row = _find_active_rule(conn, "gst_rate_rules", category_key, as_of_date)
        return row["gst_rate"]
    finally:
        if own_conn:
            conn.close()


def get_itc_determination(category_key, as_of_date, conn=None):
    """Returns (itc_eligible, itc_reason) active for this category on this date."""
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        row = _find_active_rule(conn, "itc_rules", category_key, as_of_date)
        return row["itc_eligible"], row["itc_reason"]
    finally:
        if own_conn:
            conn.close()


def compute_gst_split(amount, gst_rate, is_intra_state):
    """Splits total GST into CGST+SGST (intra-state) or IGST (inter-state).
    Deterministic arithmetic, no lookups here — kept separate from rate
    lookup so it's independently testable."""
    if gst_rate == "exempt":
        return {"cgst": 0.0, "sgst": 0.0, "igst": 0.0, "total_tax": 0.0}

    rate = float(gst_rate) / 100.0
    total_tax = round(amount * rate, 2)

    if is_intra_state:
        half = round(total_tax / 2, 2)
        return {"cgst": half, "sgst": half, "igst": 0.0, "total_tax": half * 2}
    else:
        return {"cgst": 0.0, "sgst": 0.0, "igst": total_tax, "total_tax": total_tax}


def evaluate_transaction(category_key, amount, transaction_date, is_intra_state, conn=None):
    """The main entry point — given a transaction, returns the FULL
    deterministic compliance result: GST split AND ITC determination,
    both looked up as of the transaction's actual date, not today's date.

    transaction_date and as_of_date are deliberately the SAME value here —
    we always evaluate a transaction using the rules that were active when
    it happened, never today's rules for a past transaction. This is the
    behavior that makes old filings stay correct even after rules change."""
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        gst_rate = get_gst_rate(category_key, transaction_date, conn)
        itc_eligible, itc_reason = get_itc_determination(category_key, transaction_date, conn)
        gst_split = compute_gst_split(amount, gst_rate, is_intra_state)

        return {
            "category_key": category_key,
            "transaction_date": transaction_date,
            "amount": amount,
            "gst_rate": gst_rate,
            "is_intra_state": is_intra_state,
            "gst_split": gst_split,
            "itc_eligible": itc_eligible,
            "itc_reason": itc_reason,
        }
    finally:
        if own_conn:
            conn.close()


def add_rate_change(category_key, new_rate, effective_from, changed_by, reason, conn=None):
    """ADMIN/MAINTAINER ACTION: introduces a new GST rate for a category,
    effective from a given date. Does NOT delete the old rule — closes its
    effective_to the day before the new one starts, so historic
    transactions still resolve to the OLD rate. This is the function that
    makes 'versioned' real rather than just a schema decoration."""
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        current = _find_active_rule(conn, "gst_rate_rules", category_key, effective_from)

        # Close the currently-active rule the day before the new one starts
        from datetime import timedelta
        close_date = (date.fromisoformat(effective_from) - timedelta(days=1)).isoformat()

        conn.execute(
            "UPDATE gst_rate_rules SET effective_to = ? WHERE id = ?",
            (close_date, current["id"]),
        )
        conn.execute(
            """INSERT INTO gst_rate_rules (category_key, gst_rate, effective_from, effective_to)
               VALUES (?, ?, ?, NULL)""",
            (category_key, new_rate, effective_from),
        )
        conn.execute(
            """INSERT INTO rule_change_log (changed_at, changed_by, rule_table, category_key, change_description)
               VALUES (?, ?, 'gst_rate_rules', ?, ?)""",
            (date.today().isoformat(), changed_by, category_key,
             f"Rate changed from {current['gst_rate']}% to {new_rate}% effective {effective_from}. Reason: {reason}"),
        )
        conn.commit()
        print(f"Rate change recorded: {category_key} {current['gst_rate']}% -> {new_rate}% "
              f"effective {effective_from} (old rule closed {close_date})")
    finally:
        if own_conn:
            conn.close()
