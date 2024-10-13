# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:21:23 2024

@author: gaoli
"""

   
    
    
    
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the trained logistic regression model, scaler, and feature names
@st.cache(allow_output_mutation=True)
def load_model():
    saved = joblib.load('LOG.pkl')
    model = saved['model']
    scaler = saved['scaler']
    feature_names = saved['features']
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# Define the feature names based on the final selected features
# Ensure this matches the order used during model training
# feature_names = ['HbA1c', 'Weight', 'Tyg', 'LDL', 'age', 'RBC', 'sex', 'GLU', 'WBC', 'PLR', 'CRP']

# Streamlit user interface
st.title("Type 2 Diabetes Mellitus (T2DM) Predictor")

st.header("Input Your Health Metrics")

# Collect user inputs for each feature

# HbA1c (%)
HbA1c = st.number_input("HbA1c (%):", min_value=4.0, max_value=15.0, value=5.5, format="%.1f")

# Weight (kg)
Weight = st.number_input("Weight (kg):", min_value=30.0, max_value=200.0, value=70.0, format="%.1f")

# Tyg
Tyg = st.number_input("Triglyceride and Glucose Index (Tyg):", min_value=-10.0, max_value=10.0, value=0.0, format="%.2f")

# LDL Cholesterol (mg/dL)
LDL = st.number_input("Low-Density Lipoprotein Cholesterol (LDL) (mg/dL):", min_value=10.0, max_value=300.0, value=100.0, format="%.1f")

# Age (years)
age = st.number_input("Age (years):", min_value=1, max_value=120, value=50)

# RBC (×10¹²/L)
RBC = st.number_input("Red Blood Cell Count (RBC) (×10¹²/L):", min_value=2.0, max_value=10.0, value=4.5, format="%.2f")

# Sex
sex_options = {'Female': 0, 'Male': 1}
sex_input = st.selectbox("Sex:", options=list(sex_options.keys()))
sex = sex_options[sex_input]

# GLU (Blood Glucose Level) (mg/dL)
GLU = st.number_input("Blood Glucose Level (GLU) (mg/dL):", min_value=50.0, max_value=500.0, value=100.0, format="%.1f")

# WBC (×10⁹/L)
WBC = st.number_input("White Blood Cell Count (WBC) (×10⁹/L):", min_value=1.0, max_value=20.0, value=6.0, format="%.1f")

# PLR (Platelet-to-Lymphocyte Ratio)
PLR = st.number_input("Platelet-to-Lymphocyte Ratio (PLR):", min_value=0.0, max_value=500.0, value=100.0, format="%.2f")

# CRP (mg/L)
CRP = st.number_input("C-Reactive Protein (CRP) (mg/L):", min_value=0.0, max_value=200.0, value=5.0, format="%.1f")

# Collect the inputs into a feature array
feature_values = [HbA1c, Weight, Tyg, LDL, age, RBC, sex, GLU, WBC, PLR, CRP]
features = np.array([feature_values])

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Preprocess input using the loaded scaler
    X_scaled = scaler.transform(features)
    
    # Make prediction
    predicted_class = model.predict(X_scaled)[0]
    predicted_proba = model.predict_proba(X_scaled)[0]
    
    # Display prediction results
    st.subheader("Prediction Results")
    if predicted_class == 1:
        st.write(f"**Prediction:** High risk of Type 2 Diabetes Mellitus (T2DM)")
    else:
        st.write(f"**Prediction:** Low risk of Type 2 Diabetes Mellitus (T2DM)")
    st.write(f"**Probability of T2DM:** {predicted_proba[1]*100:.1f}%")
    st.write(f"**Probability of Non-T2DM:** {predicted_proba[0]*100:.1f}%")
    
    # Generate advice based on prediction results
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a **high risk** of Type 2 Diabetes Mellitus (T2DM). "
            f"The model predicts that your probability of having T2DM is **{predicted_proba[1]*100:.1f}%**. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "We recommend that you consult a healthcare professional for further evaluation."
        )
    else:
        advice = (
            f"According to our model, you have a **low risk** of Type 2 Diabetes Mellitus (T2DM). "
            f"The model predicts that your probability of not having T2DM is **{predicted_proba[0]*100:.1f}%**. "
            "Maintaining a healthy lifestyle is still very important. "
            "Regular check-ups are recommended to monitor your health."
        )
    
    st.write(advice)
    
    # Calculate SHAP values and display force plot
    st.subheader("Feature Contribution to Prediction (SHAP Values)")
    
    # Initialize SHAP explainer for logistic regression model
    explainer = shap.LinearExplainer(model, X_scaled, feature_perturbation="interventional")
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_scaled)
    
    # Create a DataFrame for the input features
    feature_df = pd.DataFrame(features, columns=feature_names)
    
    # Plot SHAP force plot
    shap.initjs()
    fig, ax = plt.subplots(figsize=(10, 2))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        feature_df.iloc[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
    
    # Optionally, display a SHAP summary plot
    st.subheader("Overall Feature Importance")
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    st.pyplot(plt)
