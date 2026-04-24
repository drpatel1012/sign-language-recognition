from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
# Enable CORS for the frontend origin (or allow all)
CORS(app)

MODEL_FILE = 'model.pkl'
model = None

# Load the model on startup
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
else:
    print(f"Warning: Model file {MODEL_FILE} not found. Please train the model first.")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained or loaded.'}), 500

    try:
        data = request.json
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'Invalid request. "landmarks" key is required.'}), 400

        landmarks = data['landmarks']
        # The model expects a 2D array: (1, 63)
        features = np.array(landmarks).reshape(1, -1)
        
        # Ensure we have exactly 63 features
        if features.shape[1] != 63:
             return jsonify({'error': f'Invalid landmark data size. Expected 63, got {features.shape[1]}'}), 400

        prediction = model.predict(features)
        predicted_class = prediction[0]

        return jsonify({'prediction': str(predicted_class)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
