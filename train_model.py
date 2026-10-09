"""Retrain the loan approval model with a leak-free evaluation.

The original notebook applied SMOTE before splitting the data, so synthetic
rows built from future test rows could leak into training. This script splits
first, learns every fill value from the training split only, handles class
imbalance with class_weight="balanced" (no synthetic rows), and evaluates on
the held-out test set once.

Usage:
    python train_model.py --data data/loan_sanction_train.csv

It overwrites new_trained_model.sav and metrics.json, which the app reads.
Run it in the environment from requirements.txt so the saved model matches the
scikit-learn version used for deployment.
"""
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from loan_model import FEATURES, METRICS_PATH, MODEL_PATH, RAW_ENCODINGS

CATEGORICAL_FILL = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
NUMERIC_FILL = ["LoanAmount", "Loan_Amount_Term"]


def learn_fill_values(train_raw):
    """Modes for categorical columns and means for numeric ones (train only)."""
    fill = {col: train_raw[col].mode()[0] for col in CATEGORICAL_FILL}
    fill.update({col: float(train_raw[col].mean()) for col in NUMERIC_FILL})
    return fill


def prepare(raw, fill):
    """Fill missing values and encode the raw dataset columns as numbers."""
    df = raw[FEATURES].copy().fillna(fill)
    df["Dependents"] = df["Dependents"].astype(str)
    for column, mapping in RAW_ENCODINGS.items():
        df[column] = df[column].map(mapping)
    df["Credit_History"] = df["Credit_History"].astype(int)
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Unexpected values found in columns: {bad}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/loan_sanction_train.csv")
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data = data.dropna(subset=["Loan_Status"])
    target = (data["Loan_Status"] == "Y").astype(int)
    raw = data[FEATURES]

    raw_train, raw_test, y_train, y_test = train_test_split(
        raw, target, test_size=0.2, random_state=args.seed, stratify=target
    )
    fill = learn_fill_values(raw_train)
    x_train, x_test = prepare(raw_train, fill), prepare(raw_test, fill)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=5, class_weight="balanced", random_state=1
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    scores = {}
    for name, estimator in candidates.items():
        scores[name] = cross_val_score(
            estimator, x_train, y_train, cv=cv, scoring="balanced_accuracy"
        ).mean()
        print(f"{name}: 5-fold CV balanced accuracy = {scores[name]:.3f}")

    best = max(scores, key=scores.get)
    model = candidates[best].fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(f"\nSelected: {best}\nHeld-out test set ({len(y_test)} rows):")
    print(classification_report(y_test, predictions, target_names=["Not approved", "Approved"]))

    metrics = {
        "model": best,
        "evaluation": (
            f"Held-out test set ({len(y_test)} rows, stratified 80/20 split). "
            "Fill values learned on the training split only; no oversampling."
        ),
        "test_accuracy": round(float(accuracy_score(y_test, predictions)), 3),
        "recall_not_approved": round(float(recall_score(y_test, predictions, pos_label=0)), 3),
        "recall_approved": round(float(recall_score(y_test, predictions, pos_label=1)), 3),
    }
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {MODEL_PATH.name} and {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
