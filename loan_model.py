"""Shared model logic for the Loan Approval Prediction project.

The encodings below are exactly the ones produced by the training notebook
(sklearn LabelEncoder sorts labels alphabetically; '3+' dependents became 3).
Both the Streamlit app and train_model.py import from here so the training and
prediction code can never drift apart.
"""
import json
import math
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "new_trained_model.sav"
METRICS_PATH = ROOT / "metrics.json"

# Column order must match the training data exactly.
FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

GENDER = {"Female": 0, "Male": 1}
MARRIED = {"No": 0, "Yes": 1}
DEPENDENTS = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION = {"Graduate": 0, "Not Graduate": 1}
SELF_EMPLOYED = {"No": 0, "Yes": 1}
CREDIT_HISTORY = {"No / poor (0)": 0, "Yes / good (1)": 1}
PROPERTY_AREA = {"Rural": 0, "Semiurban": 1, "Urban": 2}

# Raw dataset labels -> codes, used when preparing the CSV for training.
RAW_ENCODINGS = {
    "Gender": GENDER,
    "Married": MARRIED,
    "Dependents": DEPENDENTS,
    "Education": EDUCATION,
    "Self_Employed": SELF_EMPLOYED,
    "Property_Area": PROPERTY_AREA,
}


# Input bounds used by the app when metrics.json has no "feature_ranges" yet.
# train_model.py writes the exact min/max seen in your training data.
FALLBACK_RANGES = {
    "ApplicantIncome": (0, 81000),
    "LoanAmount": (9, 700),
    "Loan_Amount_Term": (12, 480),
}


def load_model(path=MODEL_PATH):
    """Load the trained model. Only load pickle files you created yourself."""
    with open(path, "rb") as model_file:
        return pickle.load(model_file)


def load_metrics(path=METRICS_PATH):
    """Return the saved evaluation summary, or None if it is missing."""
    try:
        with open(path, encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _code(mapping, value, field):
    try:
        return mapping[value]
    except KeyError:
        raise ValueError(
            f"Invalid {field}: {value!r}. Expected one of {list(mapping)}."
        ) from None


def build_features(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area,
):
    """Turn form values into the one-row DataFrame the model expects."""
    if applicant_income < 0 or loan_amount <= 0 or loan_term <= 0:
        raise ValueError("Income must be >= 0; loan amount and term must be > 0.")
    row = [
        _code(GENDER, gender, "gender"),
        _code(MARRIED, married, "married"),
        _code(DEPENDENTS, dependents, "dependents"),
        _code(EDUCATION, education, "education"),
        _code(SELF_EMPLOYED, self_employed, "self-employed"),
        float(applicant_income),
        float(loan_amount),
        float(loan_term),
        _code(CREDIT_HISTORY, credit_history, "credit history"),
        _code(PROPERTY_AREA, property_area, "property area"),
    ]
    return pd.DataFrame([row], columns=FEATURES)


def predict(model, **form_values):
    """Return (label, score): label 1 = predicted approved, 0 = not approved.

    score is the model's predicted probability of class 1. It comes from a small
    random forest and is not a calibrated probability.
    """
    features = build_features(**form_values)
    label = int(model.predict(features)[0])
    score = float(model.predict_proba(features)[0][list(model.classes_).index(1)])
    return label, score


def load_ranges(metrics=None):
    """Min/max the model saw in training for the numeric inputs.

    Tree models cannot extrapolate: every value beyond the training maximum is
    treated exactly like the maximum. The app therefore limits the inputs to
    these ranges instead of letting users type absurd values.
    """
    ranges = dict(FALLBACK_RANGES)
    saved = (metrics or {}).get("feature_ranges", {})
    for name in ranges:
        if name in saved:
            low, high = saved[name]
            ranges[name] = (int(math.floor(low)), int(math.ceil(high)))
    return ranges


def repayment_warning(applicant_income, loan_amount, loan_term):
    """Return a caution if the rough repayment exceeds income, else None.

    LoanAmount is in thousands in this dataset. Assumes income is monthly and
    ignores interest, so this is only a sanity check. The model itself never
    sees a repayment-to-income ratio.
    """
    monthly = loan_amount * 1000 / loan_term
    if monthly > applicant_income:
        return (
            f"Rough check: repaying {loan_amount:,} thousand over {loan_term} months "
            f"is about {monthly:,.0f} per month before interest, more than the "
            f"income entered (assuming income is monthly). This model does not "
            f"check affordability, so treat the prediction with extra caution."
        )
    return None
