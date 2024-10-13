# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:21:23 2024

@author: gaoli
"""
import os
os.system('pip install joblib')
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the trained logistic regression model
model = joblib.load('LOG.pkl')

# Define the feature names based on the final selected features
feature_names = ['HbA1c', 'Tyg', 'LDL', 'age', 'RBC', 'sex', 'GLU', 'WBC', 'PLR', 'CRP']

# Streamlit user interface
st.title("Type 2 Diabetes Mellitus Predictor")

# Collect user inputs for each feature

# HbA1c (%)
HbA1c = st.number_input("HbA1c (%):", min_value=4.0, max_value=15.0, value=5.5, format="%.1f")

# Triglyceride and Glucose to compute Tyg
triglyceride = st.number_input("Triglyceride (mg/dL):", min_value=10.0, max_value=1000.0, value=150.0, format="%.1f")
glucose = st.number_input("Glucose (mg/dL):", min_value=50.0, max_value=500.0, value=100.0, format="%.1f")

# Compute Tyg using the formula: ln [triglyceride (mg/dL) × glucose (mg/dL)/2]
Tyg = np.log((triglyceride * glucose) / 2)

# LDL Cholesterol
LDL = st.number_input("Low-Density Lipoprotein Cholesterol (LDL) (mg/dL):", min_value=10.0, max_value=300.0, value=100.0, format="%.1f")

# Age
age = st.number_input("Age:", min_value=1, max_value=120, value=50)

# RBC
RBC = st.number_input("Red Blood Cell Count (RBC) (×10¹²/L):", min_value=2.0, max_value=10.0, value=4.5, format="%.2f")

# Sex
sex_options = {'Female': 0, 'Male': 1}
sex_input = st.selectbox("Sex:", options=list(sex_options.keys()))
sex = sex_options[sex_input]

# GLU (Assuming it's fasting blood glucose level)
GLU = st.number_input("Blood Glucose Level (GLU) (mg/dL):", min_value=50.0, max_value=500.0, value=100.0, format="%.1f")

# WBC
WBC = st.number_input("White Blood Cell Count (WBC) (×10⁹/L):", min_value=1.0, max_value=20.0, value=6.0, format="%.1f")

# Platelet count and Lymphocyte count to compute PLR
platelet_count = st.number_input("Platelet Count (×10⁹/L):", min_value=10.0, max_value=1000.0, value=250.0, format="%.1f")
lymphocyte_count = st.number_input("Lymphocyte Count (×10⁹/L):", min_value=0.1, max_value=10.0, value=2.0, format="%.2f")

# Compute PLR using the formula: Platelet count / Lymphocyte count
PLR = platelet_count / lymphocyte_count

# CRP
CRP = st.number_input("C-Reactive Protein (CRP) (mg/L):", min_value=0.0, max_value=200.0, value=5.0, format="%.1f")

# Collect the inputs into a feature array
feature_values = [HbA1c, Tyg, LDL, age, RBC, sex, GLU, WBC, PLR, CRP]
features = np.array([feature_values])

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    if predicted_class == 1:
        st.write(f"**Prediction:** High risk of Type 2 Diabetes Mellitus (T2DM)")
    else:
        st.write(f"**Prediction:** Low risk of Type 2 Diabetes Mellitus (T2DM)")
    st.write(f"**Probability of T2DM:** {predicted_proba[1]*100:.1f}%")
    st.write(f"**Probability of Non-T2DM:** {predicted_proba[0]*100:.1f}%")

    # Generate advice based on prediction results
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of Type 2 Diabetes Mellitus (T2DM). "
            f"The model predicts that your probability of having T2DM is {predicted_proba[1]*100:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "We recommend that you consult a healthcare professional for further evaluation."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of Type 2 Diabetes Mellitus (T2DM). "
            f"The model predicts that your probability of not having T2DM is {predicted_proba[0]*100:.1f}%. "
            "Maintaining a healthy lifestyle is still very important. "
            "Regular check-ups are recommended to monitor your health."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    # Note: For the SHAP explainer, we need background data. We'll use a small sample for demonstration.

    # Create background data (mean values)
    background_data = np.mean(features, axis=0).reshape(1, -1)

    # Create SHAP explainer for logistic regression model
    explainer = shap.LinearExplainer(model, background_data)

    # Calculate SHAP values
    shap_values = explainer.shap_values(features)

    # Plot SHAP force plot
    shap.initjs()
    st.subheader("Feature Contribution to Prediction (SHAP Values)")
    # Convert feature values to a DataFrame for better display in SHAP plots
    feature_df = pd.DataFrame(features, columns=feature_names)
    shap.force_plot(
        explainer.expected_value, shap_values[0], feature_df.iloc[0], feature_names=feature_names, matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_force_plot.png")

    # Optionally, display SHAP summary plot
    st.subheader("Overall Feature Importance")
    shap.summary_plot(shap_values, feature_df, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_summary_plot.png")
