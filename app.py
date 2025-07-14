import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('random_forest_model.pkl')

st.title("Loan Repayment Prediction")

st.header("Applicant Info")

age = st.number_input("Age", min_value=18, max_value=100)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
income = st.number_input("Annual Income ($)", min_value=0.0)
loan_amount = st.number_input("Loan Amount ($)", min_value=0.0)
term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

# Add missing features
credit_policy = st.selectbox("Credit Policy (1 = Meets Policy, 0 = Doesn't)", [0, 1])
days_with_cr_line = st.number_input("Days with Credit Line", min_value=0.0)
delinq_2yrs = st.number_input("Number of Delinquencies in 2 Years", min_value=0)
fico = st.number_input("FICO Score", min_value=300, max_value=850)

# Prepare input DataFrame
input_data = pd.DataFrame([[
    age, credit_score, income, loan_amount, term,
    credit_policy, days_with_cr_line, delinq_2yrs, fico
]], columns=[
    'age', 'credit_score', 'income', 'loan_amount', 'term',
    'credit.policy', 'days.with.cr.line', 'delinq.2yrs', 'fico'
])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ The applicant is likely to repay the loan.")
    else:
        st.error("❌ The applicant is likely to default on the loan.")
