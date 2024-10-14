# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:21:23 2024

@author: gaoli
"""

import joblib
    print("Joblib imported successfully.")
except ImportError as e:
    print("Error importing joblib:", e)

import sys
print(sys.executable)
import sys
print(sys.path)
 
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('LOG.pkl')  # Load the best Logistic Regression model trained with selected features

# Define feature names
feature_names = [
    "HbA1c", "Weight", "Tyg", "LDL", "Age", "RBC", "Sex", "GLU", "WBC", "PLR", "CRP"
]

# Streamlit user interface
st.title("Type 2 Diabetes Predictor")

# Input features
age = st.number_input("Age:", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')
weight = st.number_input("Weight (kg):", min_value=20.0, max_value=200.0, value=70.0)
hba1c = st.number_input("HbA1c (%):", min_value=2.0, max_value=15.0, value=5.5)
ldl = st.number_input("LDL (mg/dL):", min_value=50.0, max_value=300.0, value=100.0)
rbc = st.number_input("RBC (10^6/uL):", min_value=2.0, max_value=10.0, value=5.0)
glu = st.number_input("GLU (mg/dL):", min_value=50.0, max_value=400.0, value=120.0)
wbc = st.number_input("WBC (10^3/uL):", min_value=1.0, max_value=20.0, value=7.0)
plr = st.number_input("PLR (Platelet-Lymphocyte Ratio):", min_value=0.0, max_value=500.0, value=150.0)
crp = st.number_input("CRP (mg/L):", min_value=0.0, max_value=100.0, value=5.0)
tyg = st.number_input("Tyg (Triacylglycerol):", min_value=0.0, max_value=10.0, value=1.5)

# Process inputs and make predictions
feature_values = [hba1c, weight, tyg, ldl, age, rbc, sex, glu, wbc, plr, crp]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of type 2 diabetes. "
            f"The model predicts that your probability of having type 2 diabetes is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a healthcare provider as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of type 2 diabetes. "
            f"The model predicts that your probability of not having type 2 diabetes is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )
    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.LinearExplainer(model, features, feature_perturbation="interventional")
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
