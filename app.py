from flask import Flask, request, jsonify
import pandas as pd
import pickle


with open('fraud_detection_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Inference function
def make_inference(new_data):
    new_data_df = pd.DataFrame(new_data)
    predictions = pipeline.predict(new_data_df)
    return predictions.tolist()

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from POST request
    print('data:---', data)
    try:
        # Ensure new data is in the correct format
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid data format. Data should be a JSON object.'}), 400

        # Make predictions
        predictions = make_inference(data)
        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run( debug=False)