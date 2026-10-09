import streamlit as st

from loan_model import (
    CREDIT_HISTORY,
    DEPENDENTS,
    EDUCATION,
    GENDER,
    MARRIED,
    PROPERTY_AREA,
    SELF_EMPLOYED,
    load_metrics,
    load_model,
    load_ranges,
    predict,
    repayment_warning,
)


@st.cache_resource
def get_model():
    return load_model()


def clamp(value, low, high):
    return max(low, min(value, high))


def show_about(metrics):
    with st.sidebar:
        st.header("About this model")
        st.write(
            "A random forest trained on a small public dataset of 614 loan "
            "applications. It is an educational project, not a credit-scoring "
            "system."
        )
        if metrics:
            st.write(f"**Test accuracy:** about {metrics['test_accuracy']:.0%}")
            st.write(
                f"**Catches approvals:** {metrics['recall_approved']:.0%} of the time  \n"
                f"**Catches rejections:** only {metrics['recall_not_approved']:.0%} of the time"
            )
            st.caption(metrics["evaluation"])


def main():
    st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")
    st.title("💰 Loan Approval Prediction")
    st.caption(
        "Educational machine-learning demo. The result is a model prediction, "
        "not a real lending decision or financial advice."
    )

    try:
        model = get_model()
    except FileNotFoundError:
        st.error(
            "Model file not found. Keep new_trained_model.sav in the same "
            "folder as Loan_app.py."
        )
        st.stop()
    except Exception as exc:
        st.error(
            "The model could not be loaded. This usually means the installed "
            "scikit-learn version differs from the one used for training. "
            "Install the versions in requirements.txt."
        )
        st.code(str(exc))
        st.stop()

    metrics = load_metrics()
    ranges = load_ranges(metrics)
    show_about(metrics)

    income_lo, income_hi = ranges["ApplicantIncome"]
    amount_lo, amount_hi = ranges["LoanAmount"]
    term_lo, term_hi = ranges["Loan_Amount_Term"]

    st.subheader("Applicant details")
    st.caption(
        "Numeric inputs are limited to the range found in the training data. "
        "A random forest cannot extrapolate: anything beyond that range would "
        "just be treated like the largest value it has seen."
    )
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", list(GENDER))
        married = st.selectbox("Married", list(MARRIED))
        dependents = st.selectbox("Number of dependents", list(DEPENDENTS))
        education = st.selectbox("Education", list(EDUCATION))
        self_employed = st.selectbox("Self-employed", list(SELF_EMPLOYED))

    with col2:
        applicant_income = st.number_input(
            f"Applicant income (dataset units, {income_lo:,} to {income_hi:,})",
            min_value=income_lo,
            max_value=income_hi,
            value=clamp(5000, income_lo, income_hi),
            step=500,
        )
        loan_amount = st.number_input(
            f"Loan amount (in thousands, {amount_lo:,} to {amount_hi:,})",
            min_value=amount_lo,
            max_value=amount_hi,
            value=clamp(150, amount_lo, amount_hi),
            step=10,
        )
        loan_term = st.number_input(
            f"Loan term (months, {term_lo} to {term_hi})",
            min_value=term_lo,
            max_value=term_hi,
            value=clamp(360, term_lo, term_hi),
            step=12,
        )
        credit_history = st.selectbox("Credit history", list(CREDIT_HISTORY), index=1)
        property_area = st.selectbox("Property area", list(PROPERTY_AREA))

    if st.button("Check predicted loan status", type="primary"):
        try:
            label, score = predict(
                model,
                gender=gender,
                married=married,
                dependents=dependents,
                education=education,
                self_employed=self_employed,
                applicant_income=applicant_income,
                loan_amount=loan_amount,
                loan_term=loan_term,
                credit_history=credit_history,
                property_area=property_area,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error("Prediction failed. Check the model file and dependencies.")
            st.code(str(exc))
            return

        if label == 1:
            st.success("The model predicts: Loan Approved.")
        else:
            st.warning("The model predicts: Loan Not Approved.")
        st.metric("Model score for approval", f"{score:.0%}")

        caution = repayment_warning(applicant_income, loan_amount, loan_term)
        if caution:
            st.warning(caution)

        st.caption(
            "The score comes from a small random forest and is not a calibrated "
            "probability. Credit history is the strongest signal in this dataset."
        )


if __name__ == "__main__":
    main()
