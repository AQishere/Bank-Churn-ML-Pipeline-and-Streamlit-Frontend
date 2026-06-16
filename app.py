import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the optimized model and fresh scaler
model = joblib.load('lightgbm_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Bank Customer Churn Predictor", layout="centered")

st.title("🏦 Bank Customer Churn Prediction App")
st.write("Enter the customer's details below to predict their risk of leaving the bank.")

st.header("📊 Customer Demographics & Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=85, value=35)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])

with col2:
    credit_score = st.slider("Credit Score", min_value=350, max_value=850, value=650)
    tenure = st.slider("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)
    is_active_member = st.selectbox("Is Active Member?", options=["Yes", "No"])

st.header("💰 Financial Profile")
col3, col4 = st.columns(2)

with col3:
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
    num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)

with col4:
    point_earned = st.number_input("Points Earned", min_value=0, value=500, step=10)

if st.button("🔮 Predict Churn Risk", use_container_width=True):
    
    # Encodings
    gender_encoded = 1 if gender == "Male" else 0
    is_active_encoded = 1 if is_active_member == "Yes" else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0
    
    # Feature Engineering (Recreated exactly as your training data expected)
    balance_per_product = balance / num_of_products
    is_zero_balance = 1 if balance == 0 else 0
    
    if age < 35:
        age_group = 2  # Young mapping from LabelEncoder
    elif age <= 60:
        age_group = 0  # Middle mapping from LabelEncoder
    else:
        age_group = 1  # Senior mapping from LabelEncoder
        
    germany_female = 1 if (geography == "Germany" and gender == "Female") else 0
    products_active = 1 if (num_of_products > 1 and is_active_encoded == 1) else 0

    # Scale ONLY the remaining numerical columns used during training
    numerical_features = np.array([[credit_score, point_earned]])
    scaled_numerical = scaler.transform(numerical_features)[0]
    
    # Reconstruct the precise feature layout for LightGBM (No raw Age or Balance)
    input_data = pd.DataFrame([{
        'CreditScore': scaled_numerical[0],
        'Gender': gender_encoded,
        'Tenure': tenure,
        'NumOfProducts': num_of_products,
        'IsActiveMember': is_active_encoded,
        'Point Earned': scaled_numerical[1],
        'Geography_Germany': geo_germany,
        'Geography_Spain': geo_spain,
        'Balance_Per_Product': balance_per_product,
        'Age_Group': age_group,
        'Is_Zero_Balance': is_zero_balance,
        'Germany_Female': germany_female,
        'Products_Active': products_active
    }])
    
    # Run Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 **High Risk Alert!** This customer is highly likely to churn. (Churn Probability: {probability:.2%})")
    else:
        st.success(f"✅ **Safe!** This customer is likely to stay with the bank. (Churn Probability: {probability:.2%})")