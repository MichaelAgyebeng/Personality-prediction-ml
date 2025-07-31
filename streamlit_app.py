import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🧠 Personality Prediction App")
st.write("Enter the input data below to predict personality.")

Time_spent_Alone = st.slider("How much time do you spend alone (hours per day)?", 0, 24, 1)
Stage_fear = st.selectbox("Do you have stage fear?", ['Yes', 'No'])
Social_event_attendance = st.slider("How many social events do you attend monthly?", 0, 30, 5)
Going_outside = st.slider("How often do you go outside (times per week)?", 0, 7, 3)
Drained_after_socializing = st.selectbox("Do you feel drained after socializing?", ['Yes', 'No'])
Friends_circle_size = st.number_input("How many close friends do you have?", min_value=0, max_value=100, value=5)
Post_frequency = st.slider("How often do you post on social media (times per week)?", 0, 20, 3)

# Prepare the input data
input_data = pd.DataFrame({
    'Time_spent_Alone': [Time_spent_Alone],
    'Stage_fear': [Stage_fear],
    'Social_event_attendance': [Social_event_attendance],
    'Going_outside': [Going_outside],
    'Drained_after_socializing': [Drained_after_socializing],
    'Friends_circle_size': [Friends_circle_size],
    'Post_frequency': [Post_frequency],
})

# Encode categorical variables if model was trained with encoded data
input_data['Stage_fear'] = input_data['Stage_fear'].map({'Yes': 1, 'No': 0})
input_data['Drained_after_socializing'] = input_data['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Handle missing values if needed
input_data.fillna(0, inplace=True)


# Example features — change these based on your actual model input
# feature_names = ['Openness', 'Neuroticism', 'Conscientiousness', 'Agreeableness', 'Extraversion', 'Impulsiveness', 'SensationSeeking', 'Age']

# input_data = {}
# for feature in feature_names:
#     input_data[feature] = st.number_input(f"{feature}", step=0.1)

if st.button("Predict Personality"):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    st.success(f"Predicted Personality: **{prediction}**")
