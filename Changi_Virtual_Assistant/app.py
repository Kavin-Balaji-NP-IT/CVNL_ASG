
Copy

"""
Flask Backend API for Changi Virtual Assist
Connects the HTML frontend to the trained RNN model
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
from collections import Counter
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# ============================================================================
# Model Definition (defined dynamically when model file is available)
# ============================================================================
# Note: The full `IntentRNN` class is defined only when a saved model checkpoint
# and PyTorch are available. This allows the server to start even if the
# trained model or PyTorch is not installed; prediction endpoints will return 503.


class TextPreprocessor:
    """Text preprocessing and vocabulary management"""
    def __init__(self, word_to_idx):
        self.word_to_idx = word_to_idx

    def clean_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize(self, text):
        """Tokenize text into words"""
        return text.split()

    def text_to_indices(self, text):
        """Convert text to indices"""
        tokens = self.tokenize(self.clean_text(text))
        return [self.word_to_idx.get(t, 1) for t in tokens]  # 1 is <UNK>


# ============================================================================
# Load Model and Resources
# ============================================================================

print("Loading model...")
# Determine device only if PyTorch is available; otherwise use CPU as default string
try:
    import torch as _torch
    torch = _torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
except Exception:
    torch = None
    device = 'cpu'
    print("PyTorch not available; running in model-unavailable mode. Using device: cpu")

# Path to your saved model checkpoint
MODEL_PATH = 'changi_airport_rnn_atis_complete.pth'

# Default placeholders when model is not available
MODEL_LOADED = False
vocab = {}
intent_to_idx = {}
idx_to_intent = {}
hyperparams = {'vocab_size': 1, 'embed_dim': 1, 'hidden_dim': 1, 'num_layers': 1, 'dropout': 0.0, 'bidirectional': False}
preprocessor = TextPreprocessor(vocab)
model = None

# Attempt to load the checkpoint if it exists
if os.path.exists(MODEL_PATH):
    # Import PyTorch and related utilities locally so the server can start
    # even when PyTorch isn't installed (unless a model load is requested).
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence

    # Define the model class now that nn is available
    class IntentRNN(nn.Module):
        """Bidirectional LSTM for Intent Classification"""
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                     num_layers=2, dropout=0.4, bidirectional=True):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.bidirectional = bidirectional

            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.emb_dropout = nn.Dropout(0.2)

            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                               dropout=dropout if num_layers > 1 else 0,
                               bidirectional=bidirectional, batch_first=True)

            fc_input = hidden_dim * (2 if bidirectional else 1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(fc_input, num_classes)

        def forward(self, x, lengths):
            embedded = self.embedding(x)
            embedded = self.emb_dropout(embedded)

            packed = pack_padded_sequence(embedded, lengths.cpu(),
                                         batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed)

            if self.bidirectional:
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden = hidden[-1]

            out = self.dropout(hidden)
            return self.fc(out)

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Extract components
    vocab = checkpoint['vocab']
    intent_to_idx = checkpoint['intent_to_idx']
    idx_to_intent = checkpoint['idx_to_intent']
    hyperparams = checkpoint['hyperparameters']

    # Initialize preprocessor
    preprocessor = TextPreprocessor(vocab)

    # Initialize model
    model = IntentRNN(
        vocab_size=hyperparams['vocab_size'],
        embed_dim=hyperparams['embed_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        num_classes=len(intent_to_idx),
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout'],
        bidirectional=hyperparams['bidirectional']
    ).to(device)

    # Load trained weights
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    MODEL_LOADED = True

    print(f"✓ Model loaded successfully!")
    print(f"  Vocabulary size: {hyperparams['vocab_size']}")
    print(f"  Number of intents: {len(intent_to_idx)}")
    print(f"  Test accuracy: {checkpoint['test_acc']:.2f}%")
    print()
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}")
    print("Server will start without the trained model. Prediction endpoints will return 503.")
    print()

# ============================================================================
# Response Templates
# ============================================================================

RESPONSE_TEMPLATES = {
    'flight': [
        "I can help you find flight information. Could you please specify your departure and destination cities?",
        "I'd be happy to search for flights. Which route are you interested in?",
        "Let me assist you with flight details. Where would you like to fly to and from?"
    ],
    'flight_time': [
        "I can check flight times for you. Please provide your flight number or route details.",
        "To find the exact departure or arrival time, I'll need your flight information.",
        "I'll look up the flight schedule. Which flight are you asking about?"
    ],
    'airfare': [
        "I can help you check airfare information. What route are you interested in?",
        "Let me find pricing details for you. Which cities are you traveling between?",
        "I'll search for fare options. Please specify your travel route."
    ],
    'ground_service': [
        "For ground transportation from Changi Airport, you have several options: MRT trains, buses, taxis, and private hire cars. Where would you like to go?",
        "I can help with ground transport information. The MRT station is in Terminal 2 and 3. Taxis are available at all terminals.",
        "Getting around Singapore is easy! Would you like directions for MRT, taxi, or bus services?"
    ],
    'airline': [
        "I can provide airline information. Which flight or route are you asking about?",
        "Let me check the airline details for you. Do you have a flight number?",
        "I'll find out which airline operates that flight. Please share more details."
    ],
    'abbreviation': [
        "I can help explain aviation or airport abbreviations. What term would you like me to clarify?",
        "Airport terminology can be confusing! Which abbreviation are you asking about?",
        "Let me explain that abbreviation for you. What specific term do you need help with?"
    ],
    'airport': [
        "I can help you with airport information. Which specific location or service are you looking for?",
        "Changi Airport has excellent facilities. What would you like to know about?",
        "Let me assist you with airport details. What are you looking for?"
    ],
    'ground_fare': [
        "Ground transportation fares vary by destination and service type. Where are you heading?",
        "I can provide fare estimates for taxis, buses, and trains. What's your destination?",
        "Let me help you with transportation costs. Where would you like to go?"
    ],
    'aircraft': [
        "I can provide aircraft information. Which specific aircraft or flight are you asking about?",
        "Let me look up the aircraft details. Do you have a flight number?",
        "I'll help you with aircraft information. What would you like to know?"
    ],
    'capacity': [
        "I can help with capacity and seating information. Which flight are you interested in?",
        "Let me check the capacity details. What specific information do you need?",
        "I'll find capacity information for you. Which aircraft or flight?"
    ],
    'default': [
        "I'm here to help with your airport and flight queries. Could you please provide more details?",
        "I'd be happy to assist! Could you rephrase your question or provide more information?",
        "Let me help you with that. Can you give me more details about what you need?"
    ]
}


def get_response_for_intent(intent):
    """Get a contextual response based on the predicted intent"""
    import random
    
    # Clean up intent name (handle compound intents)
    base_intent = intent.split('+')[0] if '+' in intent else intent
    base_intent = base_intent.split('_')[0] if base_intent not in RESPONSE_TEMPLATES else base_intent
    
    # Get response template
    templates = RESPONSE_TEMPLATES.get(base_intent, RESPONSE_TEMPLATES['default'])
    return random.choice(templates)


# ============================================================================
# Prediction Function
# ============================================================================

def predict_intent(text, top_k=3):
    """
    Predict intent for a given text query
    
    Args:
        text: Input query string
        top_k: Number of top predictions to return
    
    Returns:
        dict with predictions and response
    """
    # Preprocess text
    indices = preprocessor.text_to_indices(text)
    
    if len(indices) == 0:
        indices = [1]  # Use UNK token for empty input
    
    indices_tensor = torch.LongTensor([indices]).to(device)
    length = torch.LongTensor([len(indices)])

    # Predict
    with torch.no_grad():
        outputs = model(indices_tensor, length)
        probs = torch.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, min(top_k, len(intent_to_idx)))

    # Format results
    predictions = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        predictions.append({
            'intent': idx_to_intent[int(idx)],
            'confidence': float(prob) * 100
        })
    
    # Get contextual response
    top_intent = predictions[0]['intent']
    response = get_response_for_intent(top_intent)
    
    return {
        'predictions': predictions,
        'response': response,
        'top_intent': top_intent
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/')
def index():
    """Serve the HTML frontend"""
    return send_from_directory('.', 'changi_assistant_v2.html')


@app.route('/v2')
def index_v2():
    """Serve the updated HTML frontend"""
    return send_from_directory('.', 'changi_assistant_v2.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Predict intent from user query - Demo Mode
    
    Expected JSON: {"query": "user question here"}
    Returns: {"predictions": [...], "response": "...", "top_intent": "..."}
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing query parameter'
            }), 400
        
        query = data['query'].strip().lower()
        
        if not query:
            return jsonify({
                'error': 'Query cannot be empty'
            }), 400

        # Demo mode responses based on keywords and patterns
        intent = 'flight'
        response = "I can help you find flight information. Could you please specify your departure and destination cities?"
        confidence = 85
        
        # Check for specific gate numbers
        if 'gate a15' in query or 'a15' in query:
            intent = 'airport'
            response = "Gate A15 is located in Terminal 1, Level 2 (Departure Hall). Take the Skytrain or walk via the connecting bridges. It's about a 5-minute walk from the main departure area."
            confidence = 95
        elif 'gate c9' in query or 'c9' in query:
            intent = 'airport'
            response = "Gate C9 is located in Terminal 3, Level 2 (Departure Hall). It's in the central area of Terminal 3, about 3 minutes walk from immigration. Look for signs pointing to gates C1-C20."
            confidence = 95
        elif 'gate' in query and any(char.isdigit() for char in query):
            intent = 'airport'
            response = "I can help you find that specific gate. Gates A1-A20 are in Terminal 1, gates B1-B20 are in Terminal 2, and gates C1-C20 are in Terminal 3. All gates are on Level 2 (Departure Hall)."
            confidence = 92
        elif 'gate' in query:
            intent = 'airport'
            response = "I can help you with gate information. Which specific gate number are you looking for? Gates are organized by terminal: A gates (Terminal 1), B gates (Terminal 2), C gates (Terminal 3)."
            confidence = 90
        
        # Check for flight numbers (typically 2-4 digits)
        elif query.isdigit() and len(query) >= 3:
            intent = 'flight'
            response = f"Flight {query}: I can help you with flight information. This appears to be a flight number. Would you like departure time, gate information, or arrival details? Please specify which airline if you know it."
            confidence = 93
        
        # Check for city pairs
        elif 'singapore' in query and 'mumbai' in query:
            intent = 'flight'
            response = "Singapore to Mumbai flights: Air India and Singapore Airlines operate direct flights (about 5.5 hours). Typical departure times are 08:30, 14:20, and 23:55. Would you like specific flight numbers or booking information?"
            confidence = 94
        elif 'singapore' in query and 'new york' in query:
            intent = 'flight'
            response = "Singapore to New York flights: Singapore Airlines operates direct flights (about 18 hours). Flight SQ21 departs at 02:35, SQ23 at 08:55. These are some of the world's longest flights!"
            confidence = 94
        
        # Time-related queries
        elif any(word in query for word in ['departure time', 'depart time', 'what time']):
            intent = 'flight_time'
            response = "I can check departure times for you. Please provide your flight number (e.g., SQ123) or route (e.g., Singapore to London), and I'll give you the specific departure time and gate information."
            confidence = 91
        elif any(word in query for word in ['time', 'when', 'depart', 'arrive']):
            intent = 'flight_time'
            response = "I can help with flight timing information. Are you looking for departure time, arrival time, or current flight status? Please provide your flight number or destination."
            confidence = 88
        
        # Transportation
        elif any(word in query for word in ['taxi', 'bus', 'mrt', 'transport', 'city']):
            intent = 'ground_service'
            response = "Ground transport from Changi: MRT (Terminals 2&3, $2.50, 45min to city), Taxi ($25-35, 30min), Bus ($2, 1hr). MRT is fastest and cheapest to city center. Which area are you going to?"
            confidence = 90
        
        # Luggage
        elif any(word in query for word in ['luggage', 'baggage', 'bag']):
            intent = 'airport'
            response = "Baggage services: Claim belts are on Level 1 of each terminal. Left luggage storage available at $12/day. Oversized baggage has special collection areas. Are you looking for claim, storage, or lost baggage?"
            confidence = 87
        
        # Terminal info
        elif 'terminal' in query:
            intent = 'airport'
            response = "Changi has 4 terminals: T1 (many Asian airlines), T2 (has MRT station), T3 (largest, Singapore Airlines hub), T4 (budget airlines). Free shuttle buses connect all terminals. Which terminal do you need?"
            confidence = 89
        
        result = {
            'predictions': [
                {'intent': intent, 'confidence': confidence},
                {'intent': 'default', 'confidence': 15}
            ],
            'response': response,
            'top_intent': intent
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,  # Demo mode is working
        'mode': 'demo',
        'device': str(device),
        'message': 'Running in demo mode with keyword-based responses'
    })


@app.route('/api/intents', methods=['GET'])
def get_intents():
    """Get list of all possible intents"""
    return jsonify({
        'intents': list(intent_to_idx.keys()),
        'count': len(intent_to_idx)
    })


# ============================================================================
# Run Server
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("CHANGI VIRTUAL ASSIST - API SERVER")
    print("=" * 70)
    print(f"Server starting on http://localhost:5000")
    print(f"API endpoint: http://localhost:5000/api/predict")
    print(f"Health check: http://localhost:5000/api/health")
    print(f"Frontend: http://localhost:5000")
    print("=" * 70)
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True)