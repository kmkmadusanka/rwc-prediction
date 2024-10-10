from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
from textblob import TextBlob  # Import TextBlob for sentiment analysis
import nltk

# Download necessary NLTK corpora for TextBlob
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the trained model with noise for roadworthy prediction
with open('roadworthy_model_with_noise.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define a route for the home page (optional)
@app.route('/')
def home():
    return "Welcome to the Roadworthy Prediction API with Sentiment Analysis and Noise Level 0.1"

# Define a route for roadworthy predictions
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

# Define a route for feedback sentiment analysis
@app.route('/analyze-feedback', methods=['POST'])
def analyze_feedback():
     # Get the input feedback data from the POST request
    feedback_data = request.json
    
    # Check if feedback was provided
    if not feedback_data:
        return jsonify({'error': 'No feedback provided'}), 400
    
    # Create a dictionary to store the sentiment results
    sentiment_result = {}

    # Loop through each key-value pair in the feedback data
    for component, status in feedback_data.items():
        # Perform sentiment analysis on each value (status like "checked", "pending")
        analysis = TextBlob(status)
        if analysis.sentiment.polarity > 0:
            sentiment_result[component] = 1  # Positive
        else:
            sentiment_result[component] = 0  # Negative or neutral

    # Return the sentiment analysis result in JSON format
    return jsonify(sentiment_result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
