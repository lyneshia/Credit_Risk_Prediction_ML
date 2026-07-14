import joblib
import numpy as np
import pandas as pd

model_path = "Artifacts/model_data.joblib"

model_data = joblib.load(model_path)

model = model_data["model"]
feature_names = model_data["features"]
scaler = model_data["scaler"]
cols_to_scale = model_data["cols_to_scale"]


def scaled_data(input_dict):
    df = pd.DataFrame([input_dict])

    # Scale only the required columns
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


def prepare_df(features):

    input_dict = {

        # ---------------- Numerical Features ---------------- #
        "age": features["age"],
        "sanction_amount": 0,
        "processing_fee": 0,                      # Not available
        "loan_tenure_months": features["loan_tenure_months"],
        "bank_balance_at_application": features["bank_balance"],
        "number_of_open_accounts": features["open_loan_accounts"],
        "credit_utilization_ratio": features["credit_utilization_ratio"],
        "loan_to_income_ratio": features["loan_to_income_ratio"],
        "delinquency_ratio": features["delinquency_ratio"],
        "avg_dpd_per_delinquency": features["avg_dpd_per_delinquency"],

        # ------------ Dummy values required for scaling ------------ #
        "number_of_dependants": 0,
        "years_at_current_address": 0,
        "number_of_closed_accounts": 0,
        "enquiry_count": 0,
        "gst":0,
        "net_disbursement": 0,
        "principal_outstanding": 0,

        # ---------------- Residence Type ---------------- #
        "residence_type_Owned": 1 if features["residence_type"] == "Owned" else 0,
        "residence_type_Rented": 1 if features["residence_type"] == "Rented" else 0,

        # ---------------- Loan Purpose ---------------- #
        "loan_purpose_Education": 1 if features["loan_purpose"] == "Education" else 0,
        "loan_purpose_Home": 1 if features["loan_purpose"] == "Home" else 0,
        "loan_purpose_Personal": 1 if features["loan_purpose"] == "Personal" else 0,

        # ---------------- Loan Type ---------------- #
        "loan_type_Unsecured": 1 if features["loan_type"] == "Unsecured" else 0,
    }

    # Scale
    df = scaled_data(input_dict)

    # Remove columns added only for scaling
    df = df.drop(
        columns=[
            "number_of_dependants",
            "years_at_current_address",
            "number_of_closed_accounts",
            "enquiry_count",
        ]
    )

    # Ensure exact same order as training
    df = df[feature_names]

    return df


def calculate_credit_score(df, base_score=300, scale_length=600):

    x = np.dot(df.values, model.coef_.T) + model.intercept_

    default_prob = 1 / (1 + np.exp(-x))
    non_default_prob = 1 - default_prob

    credit_score = base_score + non_default_prob.flatten() * scale_length

    def credit_rating(score):
        if score < 500:
            return "Poor"
        elif score < 650:
            return "Average"
        elif score < 750:
            return "Good"
        else:
            return "Excellent"

    rating = credit_rating(credit_score[0])

    return (
        int(credit_score[0]),
        float(default_prob.flatten()[0]),
        rating,
    )


def predict_credit_risk(features):
    df = prepare_df(features)

    credit_score, probability, rating = calculate_credit_score(df)

    return credit_score, probability, rating