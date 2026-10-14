
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


st.set_page_config(
    page_title="Bank Loan Credit Risk",
    page_icon="🏦",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "random_forest_credit_risk.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🏦 Bank Loan & Credit Risk Prediction")
st.write("Enter loan applicant details to estimate default risk.")
st.divider()

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        customer_age = st.number_input(
            "Customer Age", 18, 100, 30
        )
        annual_income = st.number_input(
            "Annual Income", 0, 100000000, 60000, step=1000
        )
        home_ownership = st.selectbox(
            "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )
        employment_length = st.number_input(
            "Employment Length (years)", 0.0, 60.0, 5.0
        )
        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "EDUCATION", "MEDICAL", "VENTURE",
                "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"
            ]
        )
        loan_grade = st.selectbox(
            "Loan Grade", ["A", "B", "C", "D", "E", "F", "G"]
        )

    with col2:
        loan_amount = st.number_input(
            "Loan Amount", 500, 35000, 10000, step=500
        )
        interest_rate = st.number_input(
            "Interest Rate (%)", 0.0, 30.0, 11.0, step=0.1
        )
        loan_percent_income = st.number_input(
            "Loan as Proportion of Income",
            0.0, 1.0, 0.20, step=0.01
        )
        previous_default_history = st.selectbox(
            "Previous Default History", ["N", "Y"]
        )
        credit_history_length = st.number_input(
            "Credit History Length (years)", 0, 60, 5
        )

    submitted = st.form_submit_button("Predict Default Risk")

if submitted:
    input_data = pd.DataFrame([{
        "customer_age": float(customer_age),
        "annual_income": int(annual_income),
        "home_ownership": home_ownership,
        "employment_length": float(employment_length),
        "loan_purpose": loan_purpose,
        "loan_grade": loan_grade,
        "loan_amount": float(loan_amount),
        "interest_rate": float(interest_rate),
        "loan_percent_income": float(loan_percent_income),
        "previous_default_history": previous_default_history,
        "credit_history_length": int(credit_history_length)
    }])

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    st.subheader("Prediction Result")
    st.metric("Estimated Default Probability", f"{probability:.1%}")

    if prediction == 1:
        st.error("Model Prediction: Higher Default Risk")
    else:
        st.success("Model Prediction: Lower Default Risk")

    st.caption(
        "Educational model output only; this is not an actual bank decision."
    )





# ==========================================
# EDA GRAPHS
# ==========================================

st.divider()
st.header("📊 Exploratory Data Analysis")

app_folder = Path(__file__).resolve().parent
eda_folder = app_folder / "EDA_Graphs"


eda_graphs = [
    ("Loan Status Distribution", "01_loan_status.png"),
    ("Customer Age Analysis", "02_customer_age.png"),
    ("Annual Income Analysis", "03_annual_income.png"),
    ("Home Ownership Analysis", "04_home_ownership.png"),
    ("Employment Length Analysis", "05_employment_length.png"),
    ("Loan Purpose Analysis", "06_loan_purpose.png"),
    ("Loan Grade Analysis", "07_loan_grade.png"),
    ("Loan Amount Analysis", "08_loan_amount.png"),
    ("Interest Rate Analysis", "09_interest_rate.png"),
    ("Loan-to-Income Ratio", "10_loan_income_ratio.png"),
    ("Previous Default History", "11_previous_default_history.png"),
]

for i in range(0, len(eda_graphs), 2):
    cols = st.columns(2)

    for col, (title, filename) in zip(cols, eda_graphs[i:i + 2]):
        image_path = eda_folder / filename

        with col:
            st.subheader(title)

            if image_path.exists():
                st.image(str(image_path), width="stretch")
            else:
                st.error(f"Graph not found: {image_path}")

# ==========================================
# MODEL EVALUATION GRAPHS
# ==========================================

st.divider()
st.header("🤖 Machine Learning Model Evaluation")

evaluation_graphs = [
    ("Confusion Matrix", "random_forest_confusion_matrix.png"),
    ("ROC Curve", "random_forest_roc_curve.png"),
]

cols = st.columns(2)

for col, (title, filename) in zip(cols, evaluation_graphs):
    image_path = app_folder / filename

    with col:
        st.subheader(title)
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(f"Graph not found: {filename}")
