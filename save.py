import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Placeholder function to simulate the machine learning model prediction
def make_prediction(input_data):
    print(input_data)
    return 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/viscus')
def viscus():
    return render_template('viscus.html')

@app.route('/viscus_logo.png')
def viscus_logo():
    return render_template('VISCUS_BIGGER.png')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = data['input_data']
        prediction = make_prediction(input_data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
