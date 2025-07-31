import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🧠 Personality Prediction App")
st.write("Enter the input data below to predict personality.")

# Example features — change these based on your actual model input
feature_names = ['Openness', 'Neuroticism', 'Conscientiousness', 'Agreeableness', 'Extraversion', 'Impulsiveness', 'SensationSeeking', 'Age']

input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", step=0.1)

if st.button("Predict Personality"):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    st.success(f"Predicted Personality: **{prediction}**")
