"""
Kavach/LedgerPilot — Phase 5: Versioned Rule Knowledge Base (schema + seeding)

WHY VERSIONED, DATE-EFFECTIVE RULES (this is the whole point of this phase,
explain it exactly like this in viva):

GST rates and ITC rules DO change over time — the government revises rates,
adds exceptions, closes loopholes. A naive system that just hardcodes "rent
= 18%" breaks the moment that changes, AND — more importantly — it makes
old filings wrong retroactively. If you recompute a March 2026 transaction
using a rate that only took effect in September 2026, that's a compliance
bug, not a cosmetic one.

The fix: every rule has an effective_from date, and optionally an
effective_to date. When computing GST/ITC for a transaction, we look up
whichever rule was ACTIVE on that transaction's date — not just "whatever
the current rule is." This means:
  - Historic transactions always recompute correctly, forever.
  - Updating a rate is an ADMIN ACTION (insert a new rule row with a future
    effective_from), not a code deployment.
  - You get a full audit trail of what the rule was at any point in time.

TABLES:
  categories       — static reference (the 14 categories, rarely change)
  gst_rate_rules   — versioned: category + rate + effective_from/to
  itc_rules        — versioned: category + eligibility + reason + effective_from/to
  rule_change_log  — who changed what rule, when, and why (the "admin/
                      maintainer" audit trail — separate from the main
                      audit_log.jsonl from Phase 4, because this logs RULE
                      changes, not TRANSACTION decisions — different thing)
"""

import sqlite3
from datetime import date

from rule_engine.categories import CATEGORIES

DB_PATH = "ledgerpilot_rules.db"

# GST rollout date in India — the earliest sensible effective_from for any
# base rule, since nothing GST-related existed before this.
GST_ROLLOUT_DATE = "2017-07-01"


def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # lets us access columns by name, not just index
    return conn


def create_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS categories (
            key TEXT PRIMARY KEY,
            label TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS gst_rate_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_key TEXT NOT NULL,
            gst_rate TEXT NOT NULL,
            effective_from TEXT NOT NULL,
            effective_to TEXT,
            FOREIGN KEY (category_key) REFERENCES categories(key)
        );

        CREATE TABLE IF NOT EXISTS itc_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_key TEXT NOT NULL,
            itc_eligible TEXT NOT NULL,
            itc_reason TEXT NOT NULL,
            effective_from TEXT NOT NULL,
            effective_to TEXT,
            FOREIGN KEY (category_key) REFERENCES categories(key)
        );

        CREATE TABLE IF NOT EXISTS rule_change_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            changed_at TEXT NOT NULL,
            changed_by TEXT NOT NULL,
            rule_table TEXT NOT NULL,
            category_key TEXT NOT NULL,
            change_description TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_gst_category_dates
            ON gst_rate_rules (category_key, effective_from, effective_to);
        CREATE INDEX IF NOT EXISTS idx_itc_category_dates
            ON itc_rules (category_key, effective_from, effective_to);
    """)
    conn.commit()


def seed_initial_rules(conn, changed_by="system_seed"):
    """Seeds the KB from categories.py's definitions, all effective from
    GST rollout date with no end date (still active). Run this ONCE on a
    fresh database — it will skip seeding if categories already exist, so
    it's safe to call on every startup without duplicating data."""
    existing = conn.execute("SELECT COUNT(*) as c FROM categories").fetchone()["c"]
    if existing > 0:
        print(f"Categories already seeded ({existing} found) — skipping seed.")
        return

    now = date.today().isoformat()

    for cat in CATEGORIES:
        conn.execute(
            "INSERT INTO categories (key, label) VALUES (?, ?)",
            (cat.key, cat.label),
        )
        conn.execute(
            """INSERT INTO gst_rate_rules (category_key, gst_rate, effective_from, effective_to)
               VALUES (?, ?, ?, NULL)""",
            (cat.key, cat.gst_rate, GST_ROLLOUT_DATE),
        )
        conn.execute(
            """INSERT INTO itc_rules (category_key, itc_eligible, itc_reason, effective_from, effective_to)
               VALUES (?, ?, ?, ?, NULL)""",
            (cat.key, cat.itc_eligible, cat.itc_reason, GST_ROLLOUT_DATE),
        )
        conn.execute(
            """INSERT INTO rule_change_log (changed_at, changed_by, rule_table, category_key, change_description)
               VALUES (?, ?, 'initial_seed', ?, ?)""",
            (now, changed_by, cat.key, f"Initial seed from categories.py: rate={cat.gst_rate}, itc={cat.itc_eligible}"),
        )

    conn.commit()
    print(f"Seeded {len(CATEGORIES)} categories with initial rate + ITC rules, effective from {GST_ROLLOUT_DATE}.")


if __name__ == "__main__":
    conn = get_connection()
    create_schema(conn)
    seed_initial_rules(conn)
    conn.close()
