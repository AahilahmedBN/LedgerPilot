"""
Kavach/LedgerPilot — Phase 6: Penalty/ITC-Risk Classifier (XGBoost)

WHAT: predicts Low/Medium/High penalty-risk from transaction-level
signals (category, ITC-claim behavior, filing lateness, GSTIN validity,
round-number amounts) — trained on the synthetic dataset from
generate_risk_dataset.py, whose labels were computed from REAL published
GST penalty formulas (see that file's docstring for citations).

WHY XGBOOST HERE (vs Logistic Regression for categorization): these
features interact non-linearly — e.g. "claimed ITC on a blocked category"
matters enormously more when ALSO combined with high tax amount, but
barely matters alone if the tax amount is tiny (falls under the Rs 5,000
minor-breach threshold). Tree-based models capture these interactions
naturally; a linear model would need manually engineered interaction
terms to do the same job.

CLASS IMBALANCE: High-risk transactions are correctly rare (~4% here) —
that imbalance reflects reality, not a data quality problem, so we do NOT
artificially rebalance the dataset. Instead we use class weights during
training and report per-class recall/precision honestly, especially for
the High class, since that's the one that actually matters for the
product's purpose (missing a real High-risk transaction is a worse
failure than misclassifying a Low one).
"""

import json
import os

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "amount", "tax_amount", "late_filing_days",
    "claimed_itc", "itc_claimed_but_blocked",
    "has_valid_gstin", "is_round_number_amount",
]
CATEGORICAL_COLUMN = "category_key"
TARGET_COLUMN = "risk_level"


def load_data(csv_path="risk_dataset.csv"):
    df = pd.read_csv(csv_path)

    # Booleans arrive from CSV as strings 'True'/'False' — convert properly,
    # don't rely on truthy-string coercion which would make even 'False' truthy.
    for col in ["claimed_itc", "itc_claimed_but_blocked", "has_valid_gstin", "is_round_number_amount"]:
        df[col] = df[col].astype(str).map({"True": 1, "False": 0})

    # One-hot encode category — 14 categories, small enough that this is
    # simpler and more interpretable than target/label encoding.
    category_dummies = pd.get_dummies(df[CATEGORICAL_COLUMN], prefix="cat")
    X = pd.concat([df[FEATURE_COLUMNS], category_dummies], axis=1)
    y = df[TARGET_COLUMN]

    return X, y


def train():
    X, y = load_data()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # XGBoost needs numeric labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded,
    )

    # Class weights to counter the (correct, realistic) imbalance —
    # computed from training data, not hardcoded, so this stays correct
    # if the dataset composition changes later.
    class_counts = pd.Series(y_train).value_counts()
    total = len(y_train)
    sample_weight = pd.Series(y_train).map(lambda c: total / (len(class_counts) * class_counts[c])).values

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    target_names = label_encoder.classes_.tolist()
    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0,
    )
    report_text = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("=" * 70)
    print(f"OVERALL ACCURACY: {accuracy:.4f}")
    print("=" * 70)
    print(report_text)

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print(f"Rows = actual, Columns = predicted. Order: {target_names}")
    print(conf_matrix)

    # The number that actually matters most for this product: how many
    # TRUE High-risk transactions did we catch (recall), not just overall
    # accuracy — missing a real High-risk case is the expensive failure.
    high_idx = target_names.index("High") if "High" in target_names else None
    if high_idx is not None:
        high_recall = report[target_names[high_idx]]["recall"]
        print(f"\nHigh-risk recall: {high_recall:.3f} "
              f"({'GOOD — catching most real High-risk cases' if high_recall > 0.7 else 'NEEDS IMPROVEMENT — missing too many real High-risk cases'})")

    # Feature importance — genuinely useful for viva: "which signals drove
    # the model's decisions" is a natural follow-up question.
    importances = sorted(
        zip(X.columns, model.feature_importances_), key=lambda x: -x[1]
    )[:8]
    print(f"\nTop 8 most important features:")
    for name, importance in importances:
        print(f"  {name:30s} {importance:.4f}")

    os.makedirs("model_artifacts", exist_ok=True)
    joblib.dump(model, "model_artifacts/risk_classifier.joblib")
    joblib.dump(label_encoder, "model_artifacts/risk_label_encoder.joblib")
    joblib.dump(list(X.columns), "model_artifacts/risk_feature_columns.joblib")

    metrics_out = {
        "accuracy": accuracy,
        "per_class_report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "class_labels": target_names,
        "top_features": [{"name": n, "importance": float(i)} for n, i in importances],
        "test_set_size": len(y_test),
        "train_set_size": len(y_train),
    }
    with open("model_artifacts/risk_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nSaved: model_artifacts/risk_classifier.joblib")
    print(f"Saved: model_artifacts/risk_metrics.json")

    return model, label_encoder, metrics_out


if __name__ == "__main__":
    train()
