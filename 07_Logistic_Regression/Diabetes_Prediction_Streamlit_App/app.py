import streamlit as st
import joblib
import numpy as np
import os
# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# ==========================================
# Load Model and Scaler
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "logistic_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
# ==========================================
# Title
# ==========================================

st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the patient's health details below to predict whether the patient has diabetes."
)

st.markdown("---")

# ==========================================
# User Inputs
# ==========================================

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=0
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=300,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=79
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.000,
    max_value=3.000,
    value=0.471,
    format="%.3f"
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=33
)

# ==========================================
# Prediction
# ==========================================

if st.button("Predict"):

    input_data = np.array([
        [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]
    ])

    # Scale the user input
    scaled_data = scaler.transform(input_data)

    # Predict the result
    prediction = model.predict(scaled_data)

    # Predict the probability
    probability = model.predict_proba(scaled_data)

    st.markdown("---")

    if prediction[0] == 1:

        st.error("⚠️ The patient is likely to have Diabetes.")

    else:

        st.success("✅ The patient is not likely to have Diabetes.")

    st.write(f"**Probability of Diabetes:** {probability[0][1]*100:.2f}%")

    st.write(f"**Probability of No Diabetes:** {probability[0][0]*100:.2f}%")
