import os
from flask import Flask, json, render_template, request, jsonify
import joblib

from collections import Counter
import numpy as np
import os
import pandas as pd
import sys
import csv
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import argparse
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Placeholder function to simulate the machine learning model prediction
def make_prediction(input_data):
    print(input_data)
    return 1

@app.route('/')
def index():
    return render_template('viscus.html')

@app.route('/viscus_logo.png')
def viscus_logo():
    return render_template('VISCUS_BIGGER.png')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        feature_names = ['sex',
                         'age',
                         'hypertension',
                         'heart_disease',
                         'ever_married',
                         'work_type',
                         'Residence_type',
                         'avg_glucose_level',
                         'bmi',
                         'smoking_status']

        input_data = np.array([data['input_data']], dtype=np.float64) 
        input_data = pd.DataFrame(input_data, columns=feature_names)

        model = joblib.load('./output/Save_Model/logistic_regression.sav')
        test_data = np.array(input_data)

        predicted_values = model.decision_function(test_data)
        predicted_labels = model.predict(test_data)
        
        print(predicted_values, predicted_labels)

        response = jsonify({'predictions': predicted_values.tolist()})
        return response
        #return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
