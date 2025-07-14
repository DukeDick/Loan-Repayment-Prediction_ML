import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

st.title("Loan Repayment Prediction")
st.header("Applicant Info")

# Input fields (matching training features exactly)
credit_policy = st.selectbox("Credit Policy (1 = Meets Policy, 0 = Doesn't)", [0, 1])
purpose = st.selectbox("Purpose of Loan", [
    'credit_card', 'debt_consolidation', 'educational', 'major_purchase',
    'small_business', 'all_other'
])
int_rate = st.number_input("Interest Rate (%)", min_value=0.0)
installment = st.number_input("Installment ($)", min_value=0.0)
log_annual_inc = st.number_input("Log of Annual Income", min_value=0.0)
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0)
fico = st.number_input("FICO Score", min_value=300, max_value=850)
days_with_cr_line = st.number_input("Days with Credit Line", min_value=0.0)
revol_bal = st.number_input("Revolving Balance", min_value=0.0)
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0)
inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0)
delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", min_value=0)
pub_rec = st.number_input("Number of Public Records", min_value=0)

# Create input DataFrame with correct column names
input_data = pd.DataFrame([[
    credit_policy, purpose, int_rate, installment, log_annual_inc, dti, fico,
    days_with_cr_line, revol_bal, revol_util, inq_last_6mths,
    delinq_2yrs, pub_rec
]], columns=[
    'credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti',
    'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
    'inq.last.6mths', 'delinq.2yrs', 'pub.rec'
])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ The applicant is **likely to repay** the loan. (Confidence: {proba:.2%})")
    else:
        st.error(f"❌ The applicant is **likely to default** on the loan. (Confidence: {proba:.2%})")
