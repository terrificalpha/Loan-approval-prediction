# Loan Approval Prediction

Live demo: https://your-streamlit-app-link.streamlit.app

A machine learning project that predicts whether a loan application is likely to be approved, with a Streamlit web app for trying the model.

> Educational project trained on a small public dataset. It is not a credit-scoring system and must not be used for real lending decisions.

## What it does

- Cleans and explores 614 loan applications (missing values, distributions, relationships with approval)
- Encodes categorical features, balances the classes, and compares Logistic Regression with a Random Forest
- Saves the Random Forest and serves it through an interactive Streamlit app (`Loan_app.py`)

## Results

Evaluated on the leak-free 20% hold-out test set (123 rows):

| Model | Test accuracy | Recall: not approved | Recall: approved | F1-Score (Approved) |
|---|---|---|---|---|
| Logistic Regression | 0.78 | — | — | — |
| **Random Forest (saved model)** | **0.78** | **0.39** | **0.95** | **0.86** |

What these numbers mean:

- The model achieves 78% accuracy on unseen test data.
- It performs exceptionally well at recognizing approved applications (95% recall, 0.86 F1-score).
- `Credit_History` is by far the most important feature in predicting loan sanction outcomes.
- Trained using `train_model.py` with strict stratified splitting and leak-free preprocessing.

## Dataset

The public "Loan Prediction Problem" dataset (614 rows). It is not included in this repo; see [`data/README.md`](data/README.md).

## Run the app

Use Python 3.11 or 3.12. To run the notebook, also install `requirements-notebook.txt`.

```bash
pip install -r requirements.txt
streamlit run Loan_app.py