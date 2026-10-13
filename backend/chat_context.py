"""
Kavach/LedgerPilot — Chat grounding context.

Assembles the real, backend-sourced JSON context that the /api/chat endpoint
hands to Gemini alongside its system instruction. Nothing here is invented
for the chat feature — it's the same audit log and category taxonomy the
rest of the app already uses, just reshaped for an LLM prompt.
"""

from audit_log import get_transaction_history, summarize_log
from categories import CATEGORIES

RECENT_TRANSACTIONS_LIMIT = 20


def build_chat_context():
    """Returns a JSON-serializable dict: the last 20 transactions (most
    recent first), the current audit-log stats summary, and the full
    static rule book from categories.py. This is the entire universe of
    facts Gemini is allowed to answer from."""
    history = get_transaction_history()
    recent_transactions = list(reversed(history[-RECENT_TRANSACTIONS_LIMIT:]))

    rule_book = [
        {
            "key": cat.key,
            "label": cat.label,
            "gst_rate": cat.gst_rate,
            "itc_eligible": cat.itc_eligible,
            "itc_reason": cat.itc_reason,
        }
        for cat in CATEGORIES
    ]

    return {
        "recent_transactions": recent_transactions,
        "stats_summary": summarize_log(),
        "rule_book": rule_book,
    }
