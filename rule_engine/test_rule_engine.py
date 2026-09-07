"""
Kavach/LedgerPilot — Phase 5: Rule Engine Test Suite

Every test here is a manually-verified scenario, not a random check —
each docstring states WHY the expected answer is correct, citing real GST
rules, so this file doubles as documentation you can walk a viva panel
through line by line.

Run with: pytest test_rule_engine.py -v
"""

import os
import pytest

from rule_kb import get_connection, create_schema, seed_initial_rules
from rule_engine import (
    evaluate_transaction, get_gst_rate, get_itc_determination,
    compute_gst_split, add_rate_change, RuleNotFoundError,
)

TEST_DB_PATH = "test_ledgerpilot_rules.db"


@pytest.fixture
def conn():
    """Fresh, isolated test database for every test — never touches the
    real ledgerpilot_rules.db, and never leaves stale state between tests."""
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    connection = get_connection(TEST_DB_PATH)
    create_schema(connection)
    seed_initial_rules(connection, changed_by="test_fixture")
    yield connection
    connection.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


# ---------------------------------------------------------------------------
# GST rate lookup
# ---------------------------------------------------------------------------

def test_rent_rate_is_18_percent(conn):
    """Rent on commercial premises is a standard input service, taxed at
    18% GST — the typical rate for services under GST law."""
    rate = get_gst_rate("rent", "2026-08-01", conn)
    assert rate == "18"


def test_exempt_non_gst_has_exempt_rate(conn):
    """Salaries, government fees etc. are entirely outside GST's scope —
    the rate should be 'exempt', not '0' (0% is a real GST slab; exempt
    means GST doesn't apply at all — these are legally different things)."""
    rate = get_gst_rate("exempt_non_gst", "2026-08-01", conn)
    assert rate == "exempt"


def test_unknown_category_raises_not_silently_guesses(conn):
    """A typo'd or invalid category key must fail loudly — silently
    returning a default rate would be a compliance bug, not a convenience."""
    with pytest.raises(RuleNotFoundError):
        get_gst_rate("not_a_real_category", "2026-08-01", conn)


# ---------------------------------------------------------------------------
# ITC eligibility
# ---------------------------------------------------------------------------

def test_food_catering_is_itc_blocked(conn):
    """Sec 17(5)(b)(i) of the CGST Act explicitly blocks ITC on food and
    outdoor catering — this is one of the most commonly tested ITC rules."""
    eligible, reason = get_itc_determination("food_catering", "2026-08-01", conn)
    assert eligible == "Blocked"
    assert "17(5)" in reason


def test_purchase_goods_is_itc_eligible(conn):
    """Ordinary input goods used for business have no Sec 17(5) restriction
    — standard ITC eligibility applies."""
    eligible, reason = get_itc_determination("purchase_goods", "2026-08-01", conn)
    assert eligible == "Eligible"


def test_sales_revenue_is_not_applicable(conn):
    """A sale is outward supply (revenue) — ITC is a question about
    purchases, so 'Not Applicable' is the only correct answer here, not
    'Eligible' or 'Blocked'."""
    eligible, _ = get_itc_determination("sales_revenue", "2026-08-01", conn)
    assert eligible == "Not Applicable"


# ---------------------------------------------------------------------------
# GST split arithmetic
# ---------------------------------------------------------------------------

def test_intra_state_splits_evenly_between_cgst_and_sgst():
    """Intra-state supply splits GST equally into CGST + SGST — this is
    exactly how Indian GST law structures intra-state tax, not an
    arbitrary 50/50 choice."""
    result = compute_gst_split(10000, "18", is_intra_state=True)
    assert result["cgst"] == 900.0
    assert result["sgst"] == 900.0
    assert result["igst"] == 0.0
    assert result["total_tax"] == 1800.0


def test_inter_state_uses_igst_only():
    """Inter-state supply uses IGST instead of CGST+SGST — a different tax
    head entirely under Indian GST law, not just a relabeled split."""
    result = compute_gst_split(10000, "18", is_intra_state=False)
    assert result["igst"] == 1800.0
    assert result["cgst"] == 0.0
    assert result["sgst"] == 0.0


def test_exempt_rate_produces_zero_tax():
    result = compute_gst_split(50000, "exempt", is_intra_state=True)
    assert result["total_tax"] == 0.0


# ---------------------------------------------------------------------------
# Full end-to-end transaction evaluation
# ---------------------------------------------------------------------------

def test_full_evaluation_rent_transaction(conn):
    """A realistic ₹18,500 intra-state rent payment — checks GST split AND
    ITC determination are both correct together, not just in isolation."""
    result = evaluate_transaction(
        category_key="rent", amount=18500, transaction_date="2026-08-03",
        is_intra_state=True, conn=conn,
    )
    assert result["gst_rate"] == "18"
    assert result["gst_split"]["cgst"] == 1665.0
    assert result["gst_split"]["sgst"] == 1665.0
    assert result["itc_eligible"] == "Eligible"


# ---------------------------------------------------------------------------
# THE KEY TEST: date-effective rule versioning actually works
# ---------------------------------------------------------------------------

def test_rate_change_does_not_alter_historic_transactions(conn):
    """THIS is the test that proves the whole point of Phase 5. We simulate
    a real-world rate change: capital_goods GST rate rises from 18% to 12%
    effective 2026-09-01 (a hypothetical government notification).

    A transaction BEFORE the change date must still use the OLD rate.
    A transaction ON/AFTER the change date must use the NEW rate.

    If this test passes, it proves old filings stay correct even after
    the rule changes — which is the entire justification for building a
    versioned rule KB instead of hardcoding rates in Python."""
    add_rate_change(
        category_key="capital_goods", new_rate="12", effective_from="2026-09-01",
        changed_by="test_admin", reason="Simulated government rate revision", conn=conn,
    )

    rate_before_change = get_gst_rate("capital_goods", "2026-08-15", conn)
    rate_after_change = get_gst_rate("capital_goods", "2026-09-15", conn)
    rate_on_change_day = get_gst_rate("capital_goods", "2026-09-01", conn)

    assert rate_before_change == "18", "Transaction before the change must use the OLD rate"
    assert rate_after_change == "12", "Transaction after the change must use the NEW rate"
    assert rate_on_change_day == "12", "The change date itself uses the NEW rate (effective_from is inclusive)"


def test_rate_change_is_logged_in_audit_trail(conn):
    """The admin action itself must be traceable — who changed what, when,
    and why. This is what an examiner means by 'versioned with an audit
    trail', not just a database column that happens to hold two rows."""
    add_rate_change(
        category_key="rent", new_rate="12", effective_from="2027-01-01",
        changed_by="admin_purva", reason="Simulated test change", conn=conn,
    )
    log_entry = conn.execute(
        "SELECT * FROM rule_change_log WHERE category_key = 'rent' AND changed_by = 'admin_purva'"
    ).fetchone()
    assert log_entry is not None
    assert "12" in log_entry["change_description"]
