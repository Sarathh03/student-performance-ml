# app.py

import streamlit as st
import joblib
import numpy as np

# Load trained artifacts
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/encoders.pkl")

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction")
st.write("Predict whether a student will **Pass or Fail** based on academic and behavioral features.")

# ---- User Inputs ----
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

math_score = st.slider("Math Score", 0, 100, 50)
reading_score = st.slider("Reading Score", 0, 100, 50)
writing_score = st.slider("Writing Score", 0, 100, 50)

# ---- Encode Inputs ----
input_data = {
    "gender": gender,
    "race/ethnicity": race,
    "parental level of education": parent_edu,
    "lunch": lunch,
    "test preparation course": test_prep,
    "math score": math_score,
    "reading score": reading_score,
    "writing score": writing_score
}

# Convert categorical to numeric using saved encoders
encoded_features = []
for col, value in input_data.items():
    if col in encoders:
        encoded_features.append(encoders[col].transform([value])[0])
    else:
        encoded_features.append(value)

# Scale features
scaled_features = scaler.transform([encoded_features])

# ---- Prediction ----
if st.button("Predict"):
    prediction = model.predict(scaled_features)[0]

    if prediction == 1:
        st.success("✅ Prediction: PASS")
    else:
        st.error("❌ Prediction: FAIL")
