import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('random_forest_model.pkl')

st.title("Loan Repayment Prediction")

# Input form
st.header("Applicant Info")
# Replace these with your real features
feature1 = st.number_input("Feature 1 (e.g., Age)", min_value=0.0)
feature2 = st.number_input("Feature 2 (e.g., Income)", min_value=0.0)
feature3 = st.number_input("Feature 3 (e.g., Loan Amount)", min_value=0.0)
feature4 = st.selectbox("Feature 4 (e.g., Term)", options=[12, 24, 36, 48, 60])
feature5 = st.number_input("Feature 5 (e.g., Credit Score)", min_value=0.0)

# You must match the order & name of columns used during training
input_data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5]],
                          columns=['age', 'income', 'loan_amount', 'term', 'credit_score'])  # ← match this to your model

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ The applicant is likely to repay the loan.")
    else:
        st.error("❌ The applicant is likely to default on the loan.")
