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
    predict,
)


@st.cache_resource
def get_model():
    return load_model()


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

    show_about(load_metrics())

    st.subheader("Applicant details")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", list(GENDER))
        married = st.selectbox("Married", list(MARRIED))
        dependents = st.selectbox("Number of dependents", list(DEPENDENTS))
        education = st.selectbox("Education", list(EDUCATION))
        self_employed = st.selectbox("Self-employed", list(SELF_EMPLOYED))

    with col2:
        applicant_income = st.number_input(
            "Applicant income (dataset units)", min_value=0, value=5000, step=500
        )
        loan_amount = st.number_input(
            "Loan amount (in thousands)", min_value=1, value=150, step=10
        )
        loan_term = st.number_input(
            "Loan term (months)", min_value=1, max_value=480, value=360, step=12
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
        st.caption(
            "The score comes from a small random forest and is not a calibrated "
            "probability. Credit history is the strongest signal in this dataset."
        )


if __name__ == "__main__":
    main()
