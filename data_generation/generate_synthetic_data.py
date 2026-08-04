"""
Kavach — Synthetic Transaction Dataset Generator (Phase 1 deliverable)

WHY SYNTHETIC (put this in DECISIONS.md verbatim, it's your justification):
Real MSME bank statements/invoices are sensitive and scarce for a student team
to legally collect at scale. Since ground truth here comes from applying
known GST/ITC rules (not subjective human judgement), synthetic generation
from rule-grounded templates gives correct-by-construction labels, avoids
any privacy exposure, and lets us control class balance precisely — all of
which a purely manually-labeled real dataset could not guarantee on this
timeline. This is a deliberate methodological choice, not a shortcut; report
it as such in the synopsis.

WHAT THIS SCRIPT DOES NOT DO:
It does not simulate scanned-document image noise (blur, skew, poor
lighting) — that is a separate concern for the OCR pipeline (Phase 2), which
needs real or realistically-rendered document images, not CSV rows. Keep the
two pipelines conceptually separate: this script tests/trains the
CATEGORIZATION model on clean-ish text; OCR robustness is tested elsewhere.

USAGE:
    python3 generate_synthetic_data.py --rows 3000 --seed 42 --out transactions.csv
"""

import argparse
import csv
import random
import string
from datetime import date, timedelta

from faker import Faker

from categories import CATEGORIES, CATEGORY_BY_KEY

fake = Faker("en_IN")

# ---------------------------------------------------------------------------
# Realistic amount ranges (INR) per category — tune these against real
# published MSME transaction-size data if you find a benchmark; for now
# these are reasonable small-business orders of magnitude, documented here
# so you can defend the numbers in viva rather than claim they're empirical.
# ---------------------------------------------------------------------------
AMOUNT_RANGES = {
    "purchase_goods": (500, 150000),
    "capital_goods": (20000, 800000),
    "input_services": (1000, 60000),
    "rent": (8000, 100000),
    "utilities": (500, 15000),
    "bank_charges": (50, 5000),
    "motor_vehicle": (500, 50000),
    "food_catering": (200, 20000),
    "club_membership": (2000, 50000),
    "employee_travel": (2000, 80000),
    "employee_insurance": (3000, 100000),
    "works_contract": (10000, 1000000),
    "sales_revenue": (1000, 500000),
    "exempt_non_gst": (1000, 60000),
}

INDIAN_STATE_CODES = [
    "27", "29", "33", "07", "19", "24", "09", "23", "36", "32",
]  # MH, KA, TN, DL, GJ, MP, UP, WB, TG, KL — enough variety for demo

NOISE_ABBREVIATIONS = {
    "payment": ["pymt", "pmt", "paymnt"],
    "purchase": ["purch", "purchse"],
    "invoice": ["inv", "invc"],
    "services": ["svcs", "srvcs"],
    "charges": ["chrgs", "chgs"],
}


def fake_gstin(state_code):
    """Generates a format-plausible (NOT validated/real) GSTIN for synthetic data.
    Format: 2-digit state + 10-char PAN-like + 1 entity + Z + 1 checksum char.
    Clearly fake — do not use for anything beyond training data realism."""
    pan_like = "".join(random.choices(string.ascii_uppercase, k=5))
    pan_like += "".join(random.choices(string.digits, k=4))
    pan_like += random.choice(string.ascii_uppercase)
    entity_code = random.choice(string.digits[1:])
    checksum = random.choice(string.ascii_uppercase + string.digits)
    return f"{state_code}{pan_like}{entity_code}Z{checksum}"


def add_text_noise(text, noise_prob=0.35):
    """Applies light, realistic noise: case variation, abbreviation swaps,
    occasional extra whitespace — mimics real-world OCR'd/typed descriptions
    without corrupting the underlying meaning."""
    words = text.split()
    out_words = []
    for w in words:
        lw = w.lower().strip(",.-")
        if lw in NOISE_ABBREVIATIONS and random.random() < noise_prob:
            w = random.choice(NOISE_ABBREVIATIONS[lw])
        out_words.append(w)
    result = " ".join(out_words)
    if random.random() < 0.15:
        result = result.upper()
    elif random.random() < 0.15:
        result = result.lower()
    if random.random() < 0.1:
        result = result.replace(" ", "  ", 1)  # stray double space
    return result


def random_date(start_year=2024, end_year=2026):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def gst_split(gst_rate, amount):
    """Randomly decide intra-state (CGST+SGST) vs inter-state (IGST) and
    compute the split. gst_rate may be '0','5','12','18','28','exempt'."""
    if gst_rate == "exempt":
        return 0.0, 0.0, 0.0
    rate = float(gst_rate) / 100.0
    tax = round(amount * rate, 2)
    if random.random() < 0.7:  # intra-state more common for small local MSMEs
        half = round(tax / 2, 2)
        return half, half, 0.0
    return 0.0, 0.0, tax


def generate_row(txn_id, category):
    cat = CATEGORY_BY_KEY[category.key]
    template = random.choice(cat.templates)
    item = random.choice(cat.items) if cat.items else ""
    vendor = fake.company()

    description_clean = template.format(vendor=vendor, item=item).strip()
    description_raw = add_text_noise(description_clean)

    low, high = AMOUNT_RANGES[cat.key]
    amount = round(random.uniform(low, high), 2)

    cgst, sgst, igst = gst_split(cat.gst_rate, amount)

    document_type = random.choice(cat.document_types)
    state_code = random.choice(INDIAN_STATE_CODES)
    vendor_gstin = fake_gstin(state_code) if document_type in ("invoice", "receipt") else ""

    return {
        "transaction_id": f"TXN{txn_id:06d}",
        "date": random_date().isoformat(),
        "description_raw": description_raw,
        "description_clean": description_clean,
        "amount": amount,
        "gst_rate": cat.gst_rate,
        "cgst_amount": cgst,
        "sgst_amount": sgst,
        "igst_amount": igst,
        "vendor_name": vendor,
        "vendor_gstin": vendor_gstin,
        "document_type": document_type,
        "category": cat.label,
        "category_key": cat.key,
        "itc_eligible": cat.itc_eligible,
        "itc_blocked_reason": cat.itc_reason if cat.itc_eligible == "Blocked" else "",
        "source": "synthetic",
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Kavach transaction dataset")
    parser.add_argument("--rows", type=int, default=2800, help="Total rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="transactions.csv", help="Output CSV path")
    args = parser.parse_args()

    random.seed(args.seed)
    Faker.seed(args.seed)

    rows_per_category = args.rows // len(CATEGORIES)
    remainder = args.rows - rows_per_category * len(CATEGORIES)

    all_rows = []
    txn_id = 1
    for i, cat in enumerate(CATEGORIES):
        n = rows_per_category + (1 if i < remainder else 0)
        for _ in range(n):
            all_rows.append(generate_row(txn_id, cat))
            txn_id += 1

    random.shuffle(all_rows)

    fieldnames = list(all_rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Quick sanity report — run this every time you regenerate the dataset
    print(f"Generated {len(all_rows)} rows -> {args.out}")
    counts = {}
    for r in all_rows:
        counts[r["category_key"]] = counts.get(r["category_key"], 0) + 1
    print("\nClass balance:")
    for k, v in sorted(counts.items()):
        print(f"  {k:22s} {v:5d}")


if __name__ == "__main__":
    main()
