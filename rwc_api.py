from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

# Load the trained model with noise
with open('roadworthy_model_with_noise.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define a route for the home page (optional)
@app.route('/')
def home():
    return "Welcome to the Roadworthy Prediction API with Noise Level 0.1"

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Expected feature names (same as the ones used in training)
    expected_features = ['vehicle_age', 'kilometer', 'brake_condition_bad',
                         'tire_condition_bad', 'suspension_condition_bad', 'emission_compliance_fail']

    # Get the input data from the POST request
    vehicle_data = request.json
    
    # Create a feature array in the correct order
    features = []
    for feature in expected_features:
        features.append(vehicle_data.get(feature, 0))
    
    # Convert the feature list into a numpy array for the model
    features_array = np.array([features])

    # Make a prediction using the loaded model
    prediction = model.predict(features_array)

    # Convert the prediction result to string (or int) to make it JSON serializable
    prediction_str = str(prediction[0])  # Convert to string

    # Return the prediction in JSON format
    return jsonify({'roadworthy_prediction': prediction_str})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
