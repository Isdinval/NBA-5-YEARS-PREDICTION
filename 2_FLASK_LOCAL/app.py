# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:45:01 2024

@author: Olivi
"""


from flask import Flask, request, jsonify
from joblib import load
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the saved model
model = load('nba_career_prediction_model.joblib')

# Load the feature names and scaler
feature_names = load('feature_names.joblib')
scaler = load('scaler.joblib')

# Load the decision threshold from the .txt file
with open('decision_threshold.txt', 'r') as f:
    decision_threshold = float(f.read().strip())
    
@app.route('/')
def home():
    return "NBA Career Prediction API is running. Use POST /predict for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Ensure all required features are present
    required_features = set(feature_names)
    provided_features = set(data.keys())
    
    if not required_features.issubset(provided_features):
        missing_features = required_features - provided_features
        return jsonify({'error': f'Missing features: {", ".join(missing_features)}'}), 400
    
    # Extract features in the correct order
    feature_vector = [data[feature] for feature in feature_names]
    
    # Preprocess the input data
    feature_vector = np.array(feature_vector).reshape(1, -1)
    feature_vector = scaler.transform(feature_vector)
    
    # Make prediction using model loaded from disk
    probability = model.predict_proba(feature_vector)[0][1]  # Probability of lasting 5+ years
    
    # Determine the class based on the decision threshold
    prediction = 1 if probability >= decision_threshold else 0
    
    # Prepare the response
    response = {
        'prediction': prediction,  # 0 for <5 years, 1 for 5+ years
        'probability': float(probability),
        'interpretation': 'Likely to last 5+ years in NBA' if prediction == 1 else 'Likely to last <5 years in NBA',
        'threshold_used': decision_threshold
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)