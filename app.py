from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import pickle
import nltk
import traceback
from predictor import get_top_k_predictions

# Download NLTK tokenizer
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=150, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

# Load vocabulary
try:
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    print(f" Vocab loaded successfully. Size: {len(vocab)}")
except Exception as e:
    print("Failed to load vocab:")
    traceback.print_exc()

# Initialize and load model
vocab_size = len(vocab)
model = LSTMModel(vocab_size, embedding_dim=100, hidden_dim=150)

try:
    state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print(" Model loaded successfully.")
except Exception as e:
    print(" Error loading model:")
    traceback.print_exc()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        print(" GET request received at /predict.")
        return jsonify({'message': 'Send a POST request with text data'})

    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        print(f" Received text: '{text}'")

        if not text:
            print(" Empty input received.")
            return jsonify({'predictions': []})

        top_words = get_top_k_predictions(model, vocab, text, top_k=5)
        print(f" Top predictions: {top_words}")
        return jsonify({'predictions': top_words})

    except Exception as e:
        print(" Prediction error:")
        traceback.print_exc()
        return jsonify({'error': 'Prediction failed.'}), 500

# Run the app
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f" Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)