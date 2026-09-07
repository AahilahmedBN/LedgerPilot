"""
Kavach/LedgerPilot — Phase 6: Synthetic Penalty/ITC-Risk Dataset Generator

CALIBRATION SOURCE (cite this in your report/viva — these are REAL,
published GST penalty rules, not invented numbers):
  - Late filing fee: Rs 50/day per return (Rs 20/day for NIL returns) under
    GST law.
  - Incorrect/ineligible ITC claim penalty: 10% of the tax involved, or a
    minimum of Rs 10,000, whichever is higher (for non-fraudulent
    offences under the 21 offences listed in the CGST Act).
  - Interest on unpaid tax: 18% per annum, applied pro-rata for the days
    outstanding.
  - "Minor breach" legal threshold: tax amount involved under Rs 5,000 —
    officers cannot impose heavy penalties for these under GST law.

WHY THIS MATTERS: instead of inventing an arbitrary 'risk score', we
compute each synthetic transaction's REAL estimated penalty exposure using
these actual formulas, then bucket that into Low/Medium/High risk labels.
The ML model is trained to predict the bucket from transaction features —
it never sees the penalty formula itself, only the underlying signals
(same as how a real classifier would only have transaction-level data,
not a ground-truth penalty calculation).

KNOWN SIMPLIFICATION (say this honestly): real GST penalty determinations
involve officer discretion, case history, and fraud-intent findings that
this synthetic model does not capture. This estimates penalty EXPOSURE
risk from structural transaction signals, not a legal determination.
"""

import csv
import random

from categories import CATEGORIES, CATEGORY_BY_KEY

RANDOM_STATE = 42

# Real, published GST penalty constants — see module docstring for sources.
LATE_FEE_PER_DAY = 50
MIN_INCORRECT_ITC_PENALTY = 10000
INCORRECT_ITC_PENALTY_RATE = 0.10
ANNUAL_INTEREST_RATE = 0.18
MINOR_BREACH_THRESHOLD = 5000

# Risk bucket thresholds (INR estimated penalty exposure) — reasoned
# cutoffs, not published law; say so if asked.
LOW_RISK_MAX = 2000
MEDIUM_RISK_MAX = 15000


def compute_gst_amount(amount, gst_rate):
    if gst_rate == "exempt":
        return 0.0
    return round(amount * float(gst_rate) / 100.0, 2)


def compute_penalty_exposure(amount, gst_rate, itc_eligible, claimed_itc, late_filing_days, has_valid_gstin):
    """Computes estimated penalty exposure using REAL published GST
    penalty formulas — see module docstring for citations."""
    tax_amount = compute_gst_amount(amount, gst_rate)

    late_fee = late_filing_days * LATE_FEE_PER_DAY

    incorrect_itc_penalty = 0.0
    if claimed_itc and itc_eligible == "Blocked":
        # Claiming ITC on a legally blocked category — real penalty exposure
        incorrect_itc_penalty = max(tax_amount * INCORRECT_ITC_PENALTY_RATE, MIN_INCORRECT_ITC_PENALTY)

    interest = 0.0
    if late_filing_days > 0 and tax_amount > 0:
        interest = round(tax_amount * ANNUAL_INTEREST_RATE * (late_filing_days / 365), 2)

    # Missing GSTIN on a real invoice is a documentation deficiency —
    # modeled as a smaller fixed exposure, not one of the cited formulas
    # above (this part IS a reasoned estimate, not published law).
    documentation_risk = 0.0 if has_valid_gstin else 500.0

    total = late_fee + incorrect_itc_penalty + interest + documentation_risk
    return round(total, 2), {
        "tax_amount": tax_amount,
        "late_fee": late_fee,
        "incorrect_itc_penalty": incorrect_itc_penalty,
        "interest": interest,
        "documentation_risk": documentation_risk,
    }


def bucket_risk(penalty_exposure):
    if penalty_exposure < LOW_RISK_MAX:
        return "Low"
    elif penalty_exposure < MEDIUM_RISK_MAX:
        return "Medium"
    else:
        return "High"


def generate_row(txn_id, rng):
    cat = rng.choice(CATEGORIES)
    amount = round(rng.uniform(500, 200000), 2)

    # Whether the business actually claimed ITC on this transaction —
    # independent of whether they're legally allowed to (that's the whole
    # point: claiming when blocked is the risk signal).
    claimed_itc = rng.random() < 0.7  # most businesses attempt to claim ITC where plausible

    # Late filing: most filings are on time (a majority at 0 days), but a
    # genuine minority run meaningfully late — using an actual mixture
    # instead of a tight gaussian, since the gaussian version produced an
    # unrealistically thin tail (verified by checking the output
    # distribution — see DECISIONS.md for this correction).
    if rng.random() < 0.55:
        late_filing_days = 0
    else:
        late_filing_days = rng.randint(1, 120)

    has_valid_gstin = rng.random() < 0.88  # most invoices have valid GSTIN; some don't
    is_round_number = amount == round(amount / 1000) * 1000  # naturally rare, also inject some
    if rng.random() < 0.08:
        amount = round(amount / 1000) * 1000
        is_round_number = True

    penalty_exposure, breakdown = compute_penalty_exposure(
        amount, cat.gst_rate, cat.itc_eligible, claimed_itc, late_filing_days, has_valid_gstin,
    )
    risk_level = bucket_risk(penalty_exposure)

    return {
        "transaction_id": f"RISK{txn_id:06d}",
        "category_key": cat.key,
        "amount": amount,
        "gst_rate": cat.gst_rate,
        "itc_eligible": cat.itc_eligible,
        "claimed_itc": claimed_itc,
        "itc_claimed_but_blocked": claimed_itc and cat.itc_eligible == "Blocked",
        "late_filing_days": late_filing_days,
        "has_valid_gstin": has_valid_gstin,
        "is_round_number_amount": is_round_number,
        "tax_amount": breakdown["tax_amount"],
        "estimated_penalty_exposure": penalty_exposure,
        "risk_level": risk_level,
    }


def main(n_rows=3000, seed=RANDOM_STATE, out_path="risk_dataset.csv"):
    rng = random.Random(seed)
    rows = [generate_row(i + 1, rng) for i in range(n_rows)]

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["risk_level"]] = counts.get(r["risk_level"], 0) + 1

    print(f"Generated {n_rows} rows -> {out_path}")
    print("\nRisk level distribution:")
    for level in ["Low", "Medium", "High"]:
        print(f"  {level:8s} {counts.get(level, 0):5d} ({counts.get(level, 0) / n_rows * 100:.1f}%)")

    return rows


if __name__ == "__main__":
    main()
