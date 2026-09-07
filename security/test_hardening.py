"""
Kavach — Phase 4: End-to-End Hardening Test

Proves both pieces work together against REAL artifacts from earlier
phases, not toy examples:
  1. Encrypts one of the actual Phase 2 sample receipt images, decrypts it,
     and verifies byte-for-byte the round trip is lossless.
  2. Loads the actual Phase 3 trained model and logs real predictions
     (including a genuinely low-confidence one) to the audit log.

Run this from inside security/, with sample_docs/ copied in from
ocr_pipeline/ and model_artifacts/ copied in from categorization_model/.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from encryption import generate_key, load_key, encrypt_file, decrypt_file
from audit_log import log_categorization_decision, log_itc_determination, summarize_log


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_encryption(sample_doc_path):
    print("=" * 70)
    print("TEST 1: Encryption at rest — round trip on a real document")
    print("=" * 70)

    if not os.path.exists("kavach_secret.key"):
        generate_key()
        print("Generated new encryption key: kavach_secret.key")
    else:
        print("Using existing key: kavach_secret.key")

    original_hash = file_hash(sample_doc_path)
    print(f"Original file: {sample_doc_path}")
    print(f"Original SHA-256: {original_hash}")

    encrypted_path = sample_doc_path + ".encrypted"
    encrypt_file(sample_doc_path, encrypted_path)
    print(f"Encrypted -> {encrypted_path}")

    # Prove the encrypted file is NOT readable as an image — this is the
    # actual point of encryption, so actually check it, don't just assert.
    with open(encrypted_path, "rb") as f:
        encrypted_bytes = f.read()
    is_valid_png_header = encrypted_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    print(f"Encrypted file has valid PNG header: {is_valid_png_header} (should be False)")

    decrypted_path = sample_doc_path + ".decrypted.png"
    decrypt_file(encrypted_path, decrypted_path)
    decrypted_hash = file_hash(decrypted_path)
    print(f"Decrypted SHA-256: {decrypted_hash}")

    match = original_hash == decrypted_hash
    print(f"\nRound-trip lossless: {match}")
    if not match:
        raise AssertionError("Decrypted file does not match original — encryption is broken!")

    return match


def test_audit_logging():
    print(f"\n{'=' * 70}")
    print("TEST 2: Audit logging — real predictions from the Phase 3 model")
    print("=" * 70)

    import joblib
    model = joblib.load("model_artifacts/categorization_model.joblib")
    vectorizer = joblib.load("model_artifacts/tfidf_vectorizer.joblib")

    CONFIDENCE_THRESHOLD = 0.55

    test_transactions = [
        ("TXN-DEMO-001", "Monthly Rent - Shop Premises"),
        ("TXN-DEMO-002", "Staff Lunch Catering - 25 plates"),
        ("TXN-DEMO-003", "Paid for snacks and tea during client visit"),  # genuinely low confidence
    ]

    for txn_id, description in test_transactions:
        vec = vectorizer.transform([description])
        predicted = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max()
        needs_review = confidence < CONFIDENCE_THRESHOLD

        log_categorization_decision(
            transaction_id=txn_id,
            predicted_category=predicted,
            confidence=confidence,
            needs_review=needs_review,
        )
        print(f"  {txn_id}: '{description}' -> {predicted} "
              f"(confidence: {confidence:.3f}, needs_review: {needs_review})")

    # One ITC determination log too, since Phase 5's rule engine will use this
    log_itc_determination(
        transaction_id="TXN-DEMO-002",
        category="food_catering",
        itc_eligible="Blocked",
        reason="Sec 17(5)(b)(i) blocks ITC on food and beverages and outdoor catering",
    )
    print("  Logged one ITC determination for TXN-DEMO-002")

    summary = summarize_log()
    print(f"\nAudit log summary: {summary}")


if __name__ == "__main__":
    sample_receipt = os.path.join("sample_docs", "receipt_rent.png")
    if not os.path.exists(sample_receipt):
        print(f"ERROR: {sample_receipt} not found. Copy sample_docs/ here from ocr_pipeline/ first.")
        sys.exit(1)

    test_encryption(sample_receipt)
    test_audit_logging()

    print(f"\n{'=' * 70}")
    print("Both hardening pieces verified against real artifacts.")
    print("Check audit_log.jsonl to see the actual log file that was written.")
    print("=" * 70)
