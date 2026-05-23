import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction System")
st.write("Enter customer details to predict churn risk")

# =========================
# INPUT FIELDS
# =========================

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
MonthlyCharges = st.number_input("Monthly Charges", 0.0)
TotalCharges = st.number_input("Total Charges", 0.0)

# =========================
# ENCODING (same as train.py)
# =========================

def encode(val):
    return 1 if val == "Yes" else 0

input_data = [
    1 if gender == "Male" else 0,
    SeniorCitizen,
    encode(Partner),
    encode(Dependents),
    tenure,
    encode(PhoneService),
    0 if MultipleLines == "No" else (1 if MultipleLines == "Yes" else 2),
    {"DSL": 0, "Fiber optic": 1, "No": 2}[InternetService],
    0 if OnlineSecurity == "No" else (1 if OnlineSecurity == "Yes" else 2),
    0 if OnlineBackup == "No" else (1 if OnlineBackup == "Yes" else 2),
    0 if DeviceProtection == "No" else (1 if DeviceProtection == "Yes" else 2),
    0 if TechSupport == "No" else (1 if TechSupport == "Yes" else 2),
    0 if StreamingTV == "No" else (1 if StreamingTV == "Yes" else 2),
    0 if StreamingMovies == "No" else (1 if StreamingMovies == "Yes" else 2),
    {"Month-to-month": 0, "One year": 1, "Two year": 2}[Contract],
    encode(PaperlessBilling),
    {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }[PaymentMethod],
    MonthlyCharges,
    TotalCharges
]

# =========================
# PREDICTION
# =========================

if st.button("Predict Churn"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    
    if probability < 0.3:
        st.info(f"🟢 Low Risk ({probability:.2f})")
    elif probability < 0.7:
        st.warning(f"🟡 Medium Risk ({probability:.2f})")
    else:
        st.error(f"🔴 High Risk ({probability:.2f})")