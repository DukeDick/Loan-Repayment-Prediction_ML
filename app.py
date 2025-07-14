import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

st.title("Loan Repayment Prediction")
st.header("Applicant Info")

# Input fields matching the model
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

# One-hot encoding for 'purpose'
purpose_dummies = pd.get_dummies(pd.Series(purpose), prefix="purpose")
purpose_df = pd.DataFrame(columns=[
    'purpose_all_other', 'purpose_credit_card', 'purpose_debt_consolidation',
    'purpose_educational', 'purpose_major_purchase', 'purpose_small_business'
])
for col in purpose_df.columns:
    purpose_df[col] = [1 if col == f'purpose_{purpose}' else 0]

# Final input row
input_data = pd.DataFrame([[
    credit_policy, int_rate, installment, log_annual_inc, dti, fico,
    days_with_cr_line, revol_bal, revol_util, inq_last_6mths,
    delinq_2yrs, pub_rec
]], columns=[
    'credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti',
    'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
    'inq.last.6mths', 'delinq.2yrs', 'pub.rec'
])

# Concatenate one-hot encoded purpose
input_final = pd.concat([input_data, purpose_df], axis=1)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_final)[0]
    if prediction == 1:
        st.success("✅ The applicant is likely to repay the loan.")
    else:
        st.error("❌ The applicant is likely to default on the loan.")
