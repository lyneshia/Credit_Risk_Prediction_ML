"""
Credit Risk Prediction Model - Streamlit UI
--------------------------------------------
Run with:  streamlit run credit_risk_app.py

This app collects borrower details in a tabular (grid) layout, auto-calculates
derived ratios, and predicts:
  - Probability of Default
  - Credit Rating (Poor / Average / Good / Excellent)
  - Credit Score

NOTE ON THE MODEL:
This file ships with a transparent, weighted-scorecard fallback function
(`predict_credit_risk`) so the app is fully runnable out of the box without
any external model file. If you have a trained ML model (e.g. a scikit-learn
pipeline saved with joblib as `model.pkl`), drop it in the same folder and
the app will automatically load and use it instead — just make sure the
feature order/names match `FEATURE_COLUMNS` below.
"""

import streamlit as st
from prediction_helper import predict_credit_risk

FEATURE_COLUMNS = [
    "age", "income_lpa", "loan_amount_lakhs", "loan_tenure_months",
    "bank_balance", "sanction_amount", "credit_utilization_ratio",
    "open_loan_accounts", "residence_type", "loan_purpose", "loan_type",
    "loan_to_income_ratio", "delinquency_ratio", "avg_dpd_per_delinquency",
]

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

st.title("📊 Credit Risk Prediction Model")
st.caption("Fill in the borrower details below to estimate default risk, credit rating, and credit score.")

st.divider()
st.subheader("Borrower & Loan Details")

# ---------------------------------------------------------------------------
# TABULAR INPUT GRID
# Each "row" is a set of columns holding a label + input widget, mimicking a
# table layout since Streamlit has no native editable grid for mixed widgets.
# ---------------------------------------------------------------------------

def labeled_number_input(label, key, min_value=0.0, max_value=None, value=0.0, step=1.0, help_text=None):
    st.markdown(f"**{label}**")
    return st.number_input(
        label, min_value=min_value, max_value=max_value, value=value,
        step=step, key=key, label_visibility="collapsed", help=help_text,
    )

def labeled_selectbox(label, key, options, help_text=None):
    st.markdown(f"**{label}**")
    return st.selectbox(
        label, options=options, key=key,
        label_visibility="collapsed", help=help_text,
    )

# --- Row 1 ---
r1c1, r1c2, r1c3, r1c4 = st.columns(4)
with r1c1:
    age = labeled_number_input("Age", "age", min_value=18.0, max_value=100.0, value=30.0, step=1.0)
with r1c2:
    income_lpa = labeled_number_input("Income (LPA)", "income_lpa", min_value=0.0, max_value=1000.0, value=6.0, step=0.5)
with r1c3:
    loan_amount_lakhs = labeled_number_input("Loan Amount (Lakhs)", "loan_amount_lakhs", min_value=0.0, max_value=1000.0, value=10.0, step=0.5)
with r1c4:
    loan_tenure_months = labeled_number_input("Loan Tenure (Months)", "loan_tenure_months", min_value=1.0, max_value=480.0, value=36.0, step=1.0)

# --- Row 2 ---
r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    bank_balance = labeled_number_input("Bank Balance", "bank_balance", min_value=0.0, max_value=10000.0, value=1.5, step=0.1)
with r2c2:
    credit_utilization_ratio = labeled_number_input("Credit Utilization Ratio (%)", "credit_utilization_ratio", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
with r2c3:
    open_loan_accounts = labeled_number_input("Open Loan Accounts", "open_loan_accounts", min_value=0.0, max_value=50.0, value=2.0, step=1.0)

# --- Row 3 ---
r3c1, r3c2, r3c3 = st.columns(3)
with r3c1:
    residence_type = labeled_selectbox("Residence Type", "residence_type", ["Owned", "Mortgage", "Rented"])
with r3c2:
    loan_purpose = labeled_selectbox("Loan Purpose", "loan_purpose", ["Home", "Personal", "Education", "Auto"])
with r3c3:
    loan_type = labeled_selectbox("Loan Type", "loan_type", ["Secured", "Unsecured"])


# --- Row 4 ---
r4c1, r4c2,r4c3 = st.columns(3)
with r4c1:
    avg_dpd_per_delinquency = labeled_number_input("Avg DPD per Delinquency", "avg_dpd_per_delinquency", min_value=0.0, max_value=365.0, value=5.0, step=1.0)

with r4c2:
    delinquency_ratio = labeled_number_input("Delinquency Ratio (%)", "delinquency_ratio", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
# ---------------------------------------------------------------------------
# CALCULATED FIELD (background calculation -> static/read-only display)
# ---------------------------------------------------------------------------
loan_to_income_ratio = round(loan_amount_lakhs / income_lpa, 2) if income_lpa > 0 else 0.0

with r4c3:
    st.markdown("**Loan to Income Ratio** *(auto-calculated)*")
    st.text_input(
        "Loan to Income Ratio", value=f"{loan_to_income_ratio}",
        disabled=True, label_visibility="collapsed",
        key="loan_to_income_ratio_display",
    )

st.divider()

# ---------------------------------------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------------------------------------

col_btn = st.columns([1, 1, 1])[1]
with col_btn:
    predict_clicked = st.button("🔍 Predict Credit Risk", use_container_width=True)

if predict_clicked:
    features = {
        "age": age,
        "income_lpa": income_lpa,
        "loan_amount_lakhs": loan_amount_lakhs,
        "loan_tenure_months": loan_tenure_months,
        "bank_balance": bank_balance,
        "credit_utilization_ratio": credit_utilization_ratio,
        "open_loan_accounts": open_loan_accounts,
        "residence_type": residence_type,
        "loan_purpose": loan_purpose,
        "loan_type": loan_type,
        "loan_to_income_ratio": loan_to_income_ratio,
        "delinquency_ratio": delinquency_ratio,
        "avg_dpd_per_delinquency": avg_dpd_per_delinquency,
    }

    credit_score, probability_of_default, rating = predict_credit_risk(features)

    st.divider()
    st.subheader("Prediction Results")

    res1, res2, res3 = st.columns(3)
    with res1:
        st.metric("Probability of Default", f"{probability_of_default * 100:.2f}%")
    with res2:
        rating_color = {
            "Poor": "🔴", "Average": "🟠", "Good": "🟢", "Excellent": "🟢",
        }
        st.metric("Rating", f"{rating_color.get(rating, '')} {rating}")
    with res3:
        st.metric("Credit Score", f"{credit_score}")

    st.progress(min(max(probability_of_default, 0.0), 1.0))