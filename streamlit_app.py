import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
try:
    model = pickle.load(open('attrition_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Model or Scaler file not found! Please run `python train_model.py` first.")
    st.stop()

st.set_page_config(page_title="HR Attrition Analyzer", layout="centered")
st.title("📊 Employee Attrition Risk Analyzer")
st.write("Modify parameters below to compute churn risk.")

st.markdown("---")
st.subheader("👥 Employee Metrics Input")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 35)
    income = st.number_input("Monthly Income ($)", 1000, 25000, 5000, step=500)
    years = st.slider("Years at Company", 0, 40, 5)

with col2:
    satisfaction = st.selectbox("Job Satisfaction Rating", [1, 2, 3, 4], index=2)
    wlb = st.selectbox("Work-Life Balance Rating", [1, 2, 3, 4], index=2)
    overtime = st.radio("Works Overtime?", ["Yes", "No"], index=1)

overtime_val = 1 if overtime == "Yes" else 0

if st.button("Calculate Risk Profile", use_container_width=True):
    raw_inputs = np.array([[age, income, satisfaction, wlb, years, overtime_val]])
    scaled_inputs = scaler.transform(raw_inputs)
    
    prediction = model.predict(scaled_inputs)[0]
    probability = model.predict_proba(scaled_inputs)[0][1] * 100
    
    st.markdown("---")
    st.subheader("🎯 Model Prediction Output")
    
    if prediction == 1:
        st.error(f"⚠️ **High Attrition Risk:** {probability:.1f}% probability of leaving.")
    else:
        st.success(f"✅ **Low Attrition Risk:** {probability:.1f}% likelihood trend to leave.")