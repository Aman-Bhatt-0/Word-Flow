from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import traceback
from tensorflow.keras.models import load_model
from predictor import get_top_k_predictions

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Load vocabulary
try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded successfully.")
except Exception as e:
    print("Failed to load tokenizer:")
    traceback.print_exc()

# Load the Keras model
try:
    model = load_model("next_word_model.h5")   # or .keras
    print("Keras model loaded successfully.")
except Exception as e:
    print("Error loading model:")
    traceback.print_exc()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        print("GET request received at /predict.")
        return jsonify({'message': 'Send a POST request with text data'})

    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        print(f"Received text: '{text}'")

        if not text:
            print("Empty input received.")
            return jsonify({'predictions': []})

        top_words = get_top_k_predictions(model, tokenizer, text, top_k=5)
        print(f"Top predictions: {top_words}")
        return jsonify({'predictions': top_words})

    except Exception as e:
        print("Prediction error:")
        traceback.print_exc()
        return jsonify({'error': 'Prediction failed.'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)