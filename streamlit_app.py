import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("model.pkl")


# Streamlit app UI
st.title("Personality Prediction App")

# Collect input
time_spent_alone = st.number_input("Time spent alone (hours)", min_value=0, max_value=24, value=1)
stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
social_event_attendance = st.number_input("Number of social events attended last month", min_value=0, max_value=30, value=4)
going_outside = st.number_input("How many times do you go outside in a week?", min_value=0, max_value=30, value=3)
drained_after_socializing = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
friends_circle_size = st.number_input("How many friends do you have?", min_value=0, max_value=100, value=10)
post_frequency = st.number_input("How many times do you post on social media weekly?", min_value=0, max_value=100, value=5)

# Predict
if st.button("Predict Personality"):
    try:
        input_data = {
            'Time_spent_Alone': time_spent_alone,
            'Stage_fear': stage_fear,
            'Social_event_attendance': social_event_attendance,
            'Going_outside': going_outside,
            'Drained_after_socializing': drained_after_socializing,
            'Friends_circle_size': friends_circle_size,
            'Post_frequency': post_frequency
        }

        df = pd.DataFrame([input_data])  # Ensures 2D input
        df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
        df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})


        prediction = model.predict(df)[0]
        st.success(f"Predicted Personality: {prediction}")

    except Exception as e:
        st.error(f"Prediction failed. Error: {str(e)}")
