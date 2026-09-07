"""
LedgerPilot - Phase 3: Transaction Categorization Classifier

WHAT: TF-IDF (text -> numeric features) + Logistic Regression, trained on
description_raw -> category_key from transactions.csv.

"""

import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from categories import CATEGORY_BY_KEY

CONFIDENCE_THRESHOLD = 0.55  # tune this once you see real precision/recall — start conservative
RANDOM_STATE = 42


def load_data(csv_path="transactions.csv"):
    df = pd.read_csv(csv_path)
    # Train on description_raw (the noisy version) — that's the realistic
    # case. Training on description_clean would inflate accuracy
    # artificially and wouldn't reflect real OCR output.
    return df["description_raw"], df["category_key"]


def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),  # unigrams + bigrams — "outdoor catering" matters more than "outdoor" alone
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",  # protects against any residual class imbalance
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)

    # Confidence-threshold behavior: how many test predictions would get
    # flagged 'needs_review' at the current threshold, and — critically —
    # what fraction of those flagged ones were WRONG vs how many unflagged
    # ones were wrong. This is what tells you if the threshold is doing its job.
    max_proba = y_proba.max(axis=1)
    needs_review = max_proba < CONFIDENCE_THRESHOLD
    correct = (y_pred == y_test.values)

    flagged_wrong = int((needs_review & ~correct).sum())
    flagged_total = int(needs_review.sum())
    unflagged_wrong = int((~needs_review & ~correct).sum())
    unflagged_total = int((~needs_review).sum())

    print("=" * 70)
    print(f"OVERALL ACCURACY: {accuracy:.4f}")
    print("=" * 70)
    print(report_text)

    print("=" * 70)
    print(f"CONFIDENCE THRESHOLD BEHAVIOR (threshold = {CONFIDENCE_THRESHOLD})")
    print("=" * 70)
    print(f"Flagged 'needs_review': {flagged_total}/{len(y_test)} predictions")
    print(f"  Of those flagged, {flagged_wrong} were actually wrong "
          f"({(flagged_wrong / flagged_total * 100) if flagged_total else 0:.1f}% of flagged)")
    print(f"Of the {unflagged_total} predictions trusted automatically, "
          f"{unflagged_wrong} were wrong "
          f"({(unflagged_wrong / unflagged_total * 100) if unflagged_total else 0:.1f}% of unflagged)")
    print("\nInterpretation for your report: the threshold is doing its job if")
    print("the 'wrong %' among flagged predictions is much higher than among")
    print("unflagged ones — that means low-confidence really does correlate")
    print("with actual errors, not just noise.")

    # Save everything needed to reproduce and defend these numbers
    os.makedirs("model_artifacts", exist_ok=True)
    joblib.dump(model, "model_artifacts/categorization_model.joblib")
    joblib.dump(vectorizer, "model_artifacts/tfidf_vectorizer.joblib")

    metrics_out = {
        "accuracy": accuracy,
        "per_class_report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "class_labels": model.classes_.tolist(),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "threshold_behavior": {
            "flagged_total": flagged_total,
            "flagged_wrong": flagged_wrong,
            "unflagged_total": unflagged_total,
            "unflagged_wrong": unflagged_wrong,
        },
        "test_set_size": len(y_test),
        "train_set_size": len(y_train),
    }
    with open("model_artifacts/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nSaved: model_artifacts/categorization_model.joblib")
    print(f"Saved: model_artifacts/tfidf_vectorizer.joblib")
    print(f"Saved: model_artifacts/metrics.json  <- put these numbers in your report, don't retype them by hand")

    return model, vectorizer, metrics_out


if __name__ == "__main__":
    train()
