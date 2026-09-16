# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LedgerPilot (internal/academic codename "Kavach") is an AI compliance co-pilot for Indian MSMEs: it takes a
transaction (typed in, or photographed as a receipt) and returns GST categorization, GST/ITC (Input Tax
Credit) eligibility, and a penalty-risk level. It was built as a sequence of numbered "Phases," each with its
own standalone folder, then wired together into one Flask app in `backend/`.

## Repo layout — two kinds of folders

**`backend/`** is the only folder that runs as a live application. It is a **flat, self-contained copy** of
the modules and trained model artifacts needed to serve requests — not a package that imports from the phase
folders below. It has no subpackages and uses plain relative imports (`from pipeline import ...`) and relative
file paths (`joblib.load("categorization_model.joblib")`, `DB_PATH = "ledgerpilot_rules.db"`), so **it must be
run with `backend/` as the working directory**, and any edit to shared logic (categories, rule engine, audit
log, encryption, preprocessing, OCR) needs to be made in `backend/`'s copy to actually affect the running app.

**Everything else is a "Phase" workspace** — where a piece was originally developed, trained, or tested in
isolation, each with its own copy of `categories.py` and (where relevant) its own `model_artifacts/`:

| Folder | Phase | Purpose |
|---|---|---|
| `data_generation/` | — | Generates the synthetic `transactions.csv` training set (`generate_synthetic_data.py`), using Faker; `DATA_DICTIONARY.md` documents its columns. |
| `ocr_pipeline/` | 2 | OpenCV preprocessing (`preprocessing.py`: grayscale → denoise → auto-crop/perspective-correct → deskew → adaptive threshold) + Tesseract OCR (`ocr_extract.py`). `generate_test_documents.py` / `run_pipeline_test.py` synthesize and test sample receipts. |
| `categorization_model/` | 3 | Trains the transaction categorizer: TF-IDF + Logistic Regression on `description_raw → category_key` (`train_categorization_model.py`), fit on the **raw/noisy** description text on purpose (so accuracy reflects real OCR output, not the clean version). Outputs to `model_artifacts/`. |
| `security/` | 4 | Encryption-at-rest (`encryption.py`, Fernet) and the structured decision/audit log (`audit_log.py`, JSONL, append-only). `test_hardening.py` is a standalone script (not pytest) that round-trips a real sample receipt through encrypt/decrypt and logs real model predictions. |
| `rule_engine/` | 5 | Deterministic, **non-ML** GST rate + ITC rule engine (`rule_engine.py`) backed by a **versioned, date-effective** SQLite rule KB (`rule_kb.py`). `test_rule_engine.py` is real pytest and is the best single file to read to understand the rule engine's contract. |
| `risk_classifier/` | 6 | Trains the penalty-risk classifier: XGBoost on transaction-level signals (`train_risk_classifier.py`), using a synthetic dataset (`generate_risk_dataset.py`) whose labels are computed from real, cited GST penalty formulas (late-filing fee, ITC-misuse penalty, interest, minor-breach threshold). |

Each phase folder's `categories.py` defines the same 14 fixed transaction categories (key, GST rate, ITC
eligibility + legal reason, description templates) — treat it as the canonical taxonomy; if it needs to
change, it needs to change everywhere it's duplicated (see below).

**`frontend/`** is a single static `index.html` (Tailwind via CDN, vanilla JS, no build step) that posts to
the Flask API at `http://127.0.0.1:5000`. Open it directly in a browser; no bundler/dev server involved.

## Known duplication (by design, not an oversight)

`categories.py`, `rule_engine.py`, `rule_kb.py`, `audit_log.py`, `encryption.py`, `preprocessing.py`, and
`ocr_extract.py` each exist as independent copies in multiple folders (their origin phase folder, plus
`backend/`). They are not symlinked or imported cross-folder. When asked to change shared behavior, check
which copy the user actually means (usually `backend/`, since that's what serves the live app) and consider
whether the same change is expected in the originating phase folder too.

## The end-to-end pipeline (`backend/pipeline.py`)

`process_transaction()` is the single entry point both Flask endpoints call, and runs, in order:

1. **Categorize** (`categorize()`) — TF-IDF + LogisticRegression predicts `category_key`; predictions below
   `CATEGORIZATION_CONFIDENCE_THRESHOLD` (0.55) are flagged `needs_review`.
2. **Rule engine** (`evaluate_transaction()`) — looks up the GST rate and ITC eligibility that were **active
   on the transaction's own date** (not today's date) from the versioned SQLite KB, then computes the
   CGST/SGST/IGST split.
3. **Risk classifier** (`assess_risk()`) — XGBoost predicts Low/Medium/High risk. Its
   `itc_claimed_but_blocked` feature is derived from step 2's actual rule-engine determination (not a
   separately-guessed value) — this is the deliberate coupling that makes steps 2 and 3 genuinely connected;
   if you touch either step, preserve this data flow.
4. **Audit log** — every categorization decision, ITC determination, and risk flag is appended to
   `audit_log.jsonl` (JSON Lines, append-only, not tamper-proof — just a structured decision trail).

Errors propagate honestly: if the categorizer mis-predicts a category, that wrong category flows into the
rule engine and risk classifier too — there's no correction step.

Rule versioning (`rule_kb.py` / `rule_engine.py`): every `gst_rate_rules` / `itc_rules` row has
`effective_from` / `effective_to`. Changing a rate is `add_rate_change()` (an admin action that closes the old
row and inserts a new one, logged to `rule_change_log`), never an edit to existing rows — this is what keeps
historic transactions resolving to the rate that was actually in force when they happened.

## Commands

Run everything from inside the relevant folder — none of these scripts add the repo root to `sys.path`.

```bash
# Run the API (loads all model artifacts + the rule DB at import time)
cd backend && python app.py                      # Flask dev server, port 5000, debug=True

# Frontend: no build — just open frontend/index.html in a browser
# (it calls http://127.0.0.1:5000, so the backend must be running)

# Rule engine tests (real pytest)
cd rule_engine && pytest test_rule_engine.py -v
cd rule_engine && pytest test_rule_engine.py -v -k test_rate_change_does_not_alter_historic_transactions

# Security/hardening check (standalone script, not pytest — run in order)
cd security && python test_hardening.py

# OCR pipeline smoke test (generates sample receipts, runs preprocessing + OCR)
cd ocr_pipeline && python run_pipeline_test.py

# Retrain the categorization model (reads transactions.csv, writes model_artifacts/)
cd categorization_model && python train_categorization_model.py

# Regenerate the synthetic risk dataset, then retrain the risk classifier
cd risk_classifier && python generate_risk_dataset.py
cd risk_classifier && python train_risk_classifier.py

# Regenerate the synthetic transactions dataset used for categorization training
cd data_generation && python generate_synthetic_data.py
```

Dependencies are in root `requirements.txt` (opencv-python, pytesseract, pillow, numpy, faker, pandas,
scikit-learn, joblib, flask, flask-cors, xgboost, pytest). Note: `security/encryption.py` imports
`cryptography`, which is **not** listed in `requirements.txt`. OCR also requires the Tesseract binary
installed on the system (pytesseract is just a wrapper around it). A `venv/` is committed at the repo root.

## Two API endpoints (`backend/app.py`)

- `POST /api/analyze-text` — JSON `{description, amount, transaction_date?, is_intra_state?, claimed_itc?,
  late_filing_days?, has_valid_gstin?}`, calls `process_transaction()` directly.
- `POST /api/analyze-image` — multipart upload, field name `document`; runs OpenCV preprocessing + Tesseract
  OCR, extracts an amount via regex (`extract_amount_from_text()` — a deliberate simplification, not full
  invoice parsing), then calls the same `process_transaction()`. The temp upload file is always deleted in a
  `finally` block.

Both endpoints converge on the same pipeline function so manual-entry and photographed transactions can never
silently diverge in behavior.
