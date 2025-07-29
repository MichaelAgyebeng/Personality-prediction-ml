from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("C:\\Users\\T PLUG\\Downloads\\playground-series-s5e7\\model.pkl")

@app.route('/')
def home():
    return "ML API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])  # data should be a dict with correct columns
    prediction = model.predict(df)
    return jsonify({'prediction': prediction[0]})
