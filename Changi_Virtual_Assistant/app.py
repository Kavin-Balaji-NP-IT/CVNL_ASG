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
    print("Server will start without the trained model. Prediction endpoints will use demo mode.")
    print()

# ============================================================================
# Response Templates
# ============================================================================

RESPONSE_TEMPLATES = {
    'flight': [
        "✈️ **Flight Information**: Singapore Airlines, Jetstar, and Scoot operate from Changi. Popular routes include Bangkok (2h), Kuala Lumpur (1.5h), Jakarta (2h), and Sydney (8h). Which destination interests you?",
        "🌏 **Available Routes**: Changi serves 400+ destinations. Major airlines: SQ (Singapore Airlines), 3K (Jetstar), TR (Scoot). Need specific flight times or booking help?",
        "🛫 **Flight Search**: I can help with departures to Asia (Bangkok, KL, Tokyo), Europe (London, Paris), Americas (NYC, LA), and Australia. What's your destination?"
    ],
    'flight_time': [
        "⏰ **Flight Schedule**: Most flights to Bangkok depart 8:30, 14:20, 20:45. KL flights: 7:30, 12:15, 18:30. Sydney: 1:25, 8:55, 23:45. Which route do you need?",
        "🕐 **Departure Times**: Peak hours are 6-9 AM and 6-9 PM. Off-peak flights often have better prices. What time preference do you have?",
        "📅 **Flight Times**: Singapore Airlines operates hourly to major cities. Budget airlines have 2-3 daily flights. Need specific departure/arrival times?"
    ],
    'airfare': [
        "💰 **Airfare Guide**: Budget: Bangkok $80-120, KL $60-90. Premium: London $800-1200, NYC $900-1400. Prices vary by season. When are you traveling?",
        "🎫 **Ticket Prices**: Economy to Asia: $50-200, Europe: $600-1000, Americas: $800-1500. Business class adds 3-5x. Need specific route pricing?",
        "💸 **Best Deals**: Book 2-3 months ahead for international, 1 month for regional. Tuesday-Thursday departures are cheapest. Which route interests you?"
    ],
    'ground_service': [
        "🚇 **MRT**: Terminals 2&3 connected. $2.50 to city (45 min). Operates 5:30 AM - 12:30 AM. \n🚕 **Taxi**: $25-35 to city (30 min). Available 24/7. \n🚌 **Bus**: $2 to city (1 hour). Where are you heading?",
        "🚗 **Transport Options**: \n• MRT: Cheapest, direct to CBD\n• Taxi: Fastest, door-to-door\n• Bus: Budget option\n• Grab: App-based, reliable\nWhich area of Singapore?",
        "🗺️ **Getting Around**: Marina Bay (30 min), Orchard Road (35 min), Sentosa (45 min), Jurong (40 min). MRT is fastest to city center. What's your destination?"
    ],
    'airline': [
        "🏢 **Airlines at Changi**: \n• **Singapore Airlines (SQ)**: Premium, Terminal 3\n• **Jetstar (3K)**: Budget, Terminal 1\n• **Scoot (TR)**: Low-cost, Terminal 2\n• **Emirates**: Terminal 3\nWhich airline are you flying with?",
        "✈️ **Airline Info**: SIA operates from T3 (premium), budget carriers from T1&T2. Each airline has dedicated check-in areas. Need specific airline details?",
        "🌟 **Carrier Guide**: Singapore Airlines (full service), Jetstar Asia (budget), Scoot (low-cost), plus 100+ international airlines. Which one interests you?"
    ],
    'abbreviation': [
        "📝 **Airport Codes**: \n• **SIN**: Singapore Changi\n• **SQ**: Singapore Airlines\n• **3K**: Jetstar Asia\n• **TR**: Scoot\n• **T1/T2/T3**: Terminals\nWhich abbreviation do you need explained?",
        "🔤 **Aviation Terms**: Common codes include airline (SQ, 3K), airports (SIN, BKK, KUL), and terminals (T1-T4). What specific term are you asking about?",
        "📋 **Code Meanings**: Flight codes, airline codes, airport codes - I can explain any aviation abbreviation. Which one would you like clarified?"
    ],
    'aircraft': [
        "✈️ **Aircraft Types**: \n• **A380**: Singapore Airlines, 471 seats\n• **Boeing 777**: Long-haul, 300+ seats\n• **A320**: Regional, 150-180 seats\n• **Boeing 737**: Short-haul, 130-160 seats\nWhich aircraft interests you?",
        "🛩️ **Plane Info**: Changi handles everything from small regional jets to massive A380s. Each aircraft type has different seating and amenities. Need specific aircraft details?",
        "🚁 **Fleet Details**: Airlines use different aircraft for different routes. Long-haul: A380, 777. Regional: A320, 737. Which flight or route are you asking about?"
    ],
    'airport': [
        "🏢 **Changi Airport Guide**: \n• **Terminal 1**: Gates A1-A20, many Asian airlines\n• **Terminal 2**: Gates B1-B20, MRT station\n• **Terminal 3**: Gates C1-C20, Singapore Airlines hub\n• **Terminal 4**: Budget airlines\nWhich area do you need?",
        "🗺️ **Airport Layout**: 4 terminals connected by Skytrain. T2&T3 have MRT. Each terminal has dining, shopping, lounges. Looking for specific facilities?",
        "📍 **Changi Facilities**: Free WiFi, charging stations, sleeping areas, gardens, movie theater, swimming pool. Which service or area interests you?"
    ],
    'capacity': [
        "👥 **Seating Capacity**: \n• **A380**: 471-853 seats\n• **Boeing 777**: 300-400 seats\n• **A320**: 150-180 seats\n• **Boeing 737**: 130-160 seats\nWhich aircraft or flight are you asking about?",
        "🪑 **Aircraft Capacity**: Varies by airline configuration. Economy, Premium Economy, Business, First Class affect total seats. Need specific flight capacity?",
        "📊 **Seat Numbers**: Different airlines configure aircraft differently. Singapore Airlines A380 has 471 seats, while other airlines may have 550+. Which flight interests you?"
    ],
    'cheapest': [
        "💰 **Budget Options**: \n• **Jetstar**: Lowest fares, basic service\n• **Scoot**: Low-cost long-haul\n• **AirAsia**: Regional budget\n• **Book early**: 2-3 months ahead\nWhich route are you looking for?",
        "🎯 **Money-Saving Tips**: Fly Tuesday-Thursday, book in advance, choose budget airlines, avoid peak seasons. Where would you like to go?",
        "💸 **Best Deals**: Compare Jetstar, Scoot, AirAsia for budget options. Use flexible dates for better prices. What's your destination and travel period?"
    ],
    'city': [
        "🌆 **Popular Destinations**: \n• **Bangkok**: 2h flight, $80-120\n• **Kuala Lumpur**: 1.5h, $60-90\n• **Jakarta**: 2h, $70-110\n• **Sydney**: 8h, $400-800\nWhich city interests you?",
        "🗺️ **City Information**: Changi connects to 400+ cities worldwide. Asia (1-8h), Europe (12-14h), Americas (18-20h). Need specific city details?",
        "🏙️ **Destination Guide**: Major hubs include Bangkok, KL, Jakarta (regional), London, Paris (Europe), NYC, LA (Americas). Which city are you visiting?"
    ],
    'distance': [
        "📏 **Flight Distances**: \n• **Bangkok**: 1,430 km (2h)\n• **Kuala Lumpur**: 315 km (1.5h)\n• **Sydney**: 6,300 km (8h)\n• **London**: 10,900 km (13h)\nWhich route distance do you need?",
        "🌍 **Travel Distances**: Regional flights 1-4 hours, long-haul 8-20 hours. Distance affects flight time and price. Which destinations are you comparing?",
        "✈️ **Route Info**: Short-haul under 3h, medium-haul 3-6h, long-haul 6h+. Each category has different aircraft and pricing. Need specific distance info?"
    ],
    'quantity': [
        "🔢 **Numbers & Stats**: Changi handles 65+ million passengers yearly, 400+ destinations, 100+ airlines. What specific quantity are you asking about?",
        "📊 **Airport Statistics**: 4 terminals, 300+ retail outlets, 160+ dining options, 40+ lounges. Which numbers interest you?",
        "📈 **Flight Data**: Daily flights, passenger capacity, baggage limits, or other quantities? Please specify what numbers you need."
    ],
    'meal': [
        "🍽️ **Meal Service**: \n• **Singapore Airlines**: Complimentary meals all classes\n• **Jetstar**: Buy onboard or pre-order\n• **Scoot**: Café menu available\n• **Long-haul**: Multiple meal services\nWhich airline or flight?",
        "🥘 **Dining Options**: Full-service airlines include meals, budget carriers charge extra. Special dietary needs available with advance notice. Which flight are you taking?",
        "🍜 **Food Service**: International flights include meals, regional flights may not. Airport has 160+ dining options if no meal service. Need specific airline info?"
    ],
    'restriction': [
        "📋 **Travel Requirements**: Valid passport (6+ months), visa if required, health declarations. COVID rules vary by destination. Which country are you visiting?",
        "🛂 **Entry Rules**: Singapore citizens need passport for all international travel. Visa requirements vary by destination. Where are you traveling to?",
        "⚠️ **Important Notes**: Check visa requirements, vaccination needs, customs limits before travel. Restrictions change frequently. What's your destination?"
    ],
    'default': [
        "🤖 **Changi Virtual Assistant**: I can help with flights, gates, transport, dining, shopping, and airport services. What would you like to know?",
        "❓ **How can I help?**: Ask me about flight times, gate locations, ground transport, airline info, or airport facilities. What interests you?",
        "💬 **Try asking**: 'Where is gate C9?', 'Flights to Bangkok?', 'How to get to city?', or 'Singapore Airlines info'. What do you need help with?"
    ]
}


def get_response_for_intent(intent, query=""):
    """Get appropriate response for detected intent with query-specific customization"""
    templates = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES['default'])
    
    # For now, use the first template, but we could add query-specific logic here
    response = templates[0]
    
    # Add some query-specific customization for common cases
    if intent == 'ground_service' and any(word in query.lower() for word in ['city', 'downtown', 'center']):
        response = "🚇 **To City Center**: Take MRT from T2/T3 (45 min, $2.50) or taxi (30 min, $25-35). MRT is cheapest and reliable. Which terminal are you at?"
    elif intent == 'airport' and any(word in query.lower() for word in ['gate', 'a15', 'c9', 'b12']):
        response = "🚪 **Gate Locations**: \n• **A gates**: Terminal 1, Level 2\n• **B gates**: Terminal 2, Level 2\n• **C gates**: Terminal 3, Level 2\nAll gates are post-security. Which specific gate do you need?"
    elif intent == 'flight' and any(word in query.lower() for word in ['bangkok', 'thailand']):
        response = "✈️ **Flights to Bangkok**: Multiple daily flights with Thai Airways, Singapore Airlines, Jetstar. Flight time: 2h 15min. Prices from $80. Departure times: 8:30, 14:20, 20:45. Need specific booking help?"
    
    return response


# ============================================================================
# Prediction Function (RNN Model)
# ============================================================================

def predict_intent(query, top_k=3):
    """
    Predict intent using the RNN model
    
    Args:
        query: User query string
        top_k: Number of top predictions to return
    
    Returns:
        dict with predictions, response, and top_intent
    """
    if not MODEL_LOADED or model is None:
        return None
    
    # Preprocess
    indices = preprocessor.text_to_indices(query)
    
    if not indices:
        indices = [1]  # <UNK> token
    
    # Convert to tensor
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
    response = get_response_for_intent(top_intent, query)
    
    return {
        'predictions': predictions,
        'response': response,
        'top_intent': top_intent,
        'mode': 'rnn'
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
    Predict intent from user query using RNN model or demo fallback
    
    Expected JSON: {"query": "user question here"}
    Returns: {"predictions": [...], "response": "...", "top_intent": "..."}
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing query parameter'
            }), 400
        
        query = data['query'].strip()
        
        if not query:
            return jsonify({
                'error': 'Query cannot be empty'
            }), 400

        # Use RNN model if loaded, otherwise use demo mode
        if MODEL_LOADED and model is not None:
            result = predict_intent(query, top_k=3)
            if result:
                return jsonify(result)
        
        # Demo mode fallback with keyword matching
        query_lower = query.lower()
        intent = 'flight'
        response = "I can help you find flight information. Could you please specify your departure and destination cities?"
        confidence = 85
        
        # Check for specific gate numbers
        if 'gate a15' in query_lower or 'a15' in query_lower:
            intent = 'airport'
            response = "Gate A15 is located in Terminal 1, Level 2 (Departure Hall). Take the Skytrain or walk via the connecting bridges. It's about a 5-minute walk from the main departure area."
            confidence = 95
        elif 'gate c9' in query_lower or 'c9' in query_lower:
            intent = 'airport'
            response = "Gate C9 is located in Terminal 3, Level 2 (Departure Hall). It's in the central area of Terminal 3, about 3 minutes walk from immigration. Look for signs pointing to gates C1-C20."
            confidence = 95
        elif 'gate' in query_lower and any(char.isdigit() for char in query_lower):
            intent = 'airport'
            response = "I can help you find that specific gate. Gates A1-A20 are in Terminal 1, gates B1-B20 are in Terminal 2, and gates C1-C20 are in Terminal 3. All gates are on Level 2 (Departure Hall)."
            confidence = 92
        elif 'gate' in query_lower:
            intent = 'airport'
            response = "I can help you with gate information. Which specific gate number are you looking for? Gates are organized by terminal: A gates (Terminal 1), B gates (Terminal 2), C gates (Terminal 3)."
            confidence = 90
        
        # Check for flight numbers (typically 2-4 digits)
        elif query_lower.isdigit() and len(query_lower) >= 3:
            intent = 'flight'
            response = f"Flight {query}: I can help you with flight information. This appears to be a flight number. Would you like departure time, gate information, or arrival details? Please specify which airline if you know it."
            confidence = 93
        
        # Check for city pairs
        elif 'singapore' in query_lower and 'mumbai' in query_lower:
            intent = 'flight'
            response = "Singapore to Mumbai flights: Air India and Singapore Airlines operate direct flights (about 5.5 hours). Typical departure times are 08:30, 14:20, and 23:55. Would you like specific flight numbers or booking information?"
            confidence = 94
        elif 'singapore' in query_lower and 'new york' in query_lower:
            intent = 'flight'
            response = "Singapore to New York flights: Singapore Airlines operates direct flights (about 18 hours). Flight SQ21 departs at 02:35, SQ23 at 08:55. These are some of the world's longest flights!"
            confidence = 94
        elif 'kuala lumpur' in query_lower or 'kl' in query_lower:
            intent = 'flight'
            response = "Singapore to Kuala Lumpur: Multiple airlines operate this route including Singapore Airlines, Malaysia Airlines, and AirAsia. Flight time is approximately 1 hour. Flights depart throughout the day. Would you like information on specific airlines or times?"
            confidence = 92
        
        # Time-related queries
        elif any(word in query_lower for word in ['departure time', 'depart time', 'what time']):
            intent = 'flight_time'
            response = "I can check departure times for you. Please provide your flight number (e.g., SQ123) or route (e.g., Singapore to London), and I'll give you the specific departure time and gate information."
            confidence = 91
        elif any(word in query_lower for word in ['time', 'when', 'depart', 'arrive']):
            intent = 'flight_time'
            response = "I can help with flight timing information. Are you looking for departure time, arrival time, or current flight status? Please provide your flight number or destination."
            confidence = 88
        
        # Transportation
        elif any(word in query_lower for word in ['taxi', 'bus', 'mrt', 'transport', 'city', 'downtown']):
            intent = 'ground_service'
            response = "Ground transport from Changi: MRT (Terminals 2&3, $2.50, 45min to city), Taxi ($25-35, 30min), Bus ($2, 1hr). MRT is fastest and cheapest to city center. Which area are you going to?"
            confidence = 90
        
        # Luggage
        elif any(word in query_lower for word in ['luggage', 'baggage', 'bag']):
            intent = 'airport'
            response = "Baggage services: Claim belts are on Level 1 of each terminal. Left luggage storage available at $12/day. Oversized baggage has special collection areas. Are you looking for claim, storage, or lost baggage?"
            confidence = 87
        
        # Terminal info
        elif 'terminal' in query_lower or any(t in query_lower for t in ['t1', 't2', 't3', 't4']):
            intent = 'airport'
            response = "Changi has 4 terminals: T1 (many Asian airlines), T2 (has MRT station), T3 (largest, Singapore Airlines hub), T4 (budget airlines). Free shuttle buses connect all terminals. Which terminal do you need?"
            confidence = 89
        
        result = {
            'predictions': [
                {'intent': intent, 'confidence': confidence},
                {'intent': 'default', 'confidence': 100 - confidence}
            ],
            'response': response,
            'top_intent': intent,
            'mode': 'demo'
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'mode': 'rnn' if MODEL_LOADED else 'demo',
        'device': str(device),
        'message': f'RNN model loaded successfully with {len(intent_to_idx)} intents' if MODEL_LOADED else 'Running in demo mode with keyword-based responses'
    })


@app.route('/api/intents', methods=['GET'])
def get_intents():
    """Get list of all possible intents"""
    if MODEL_LOADED:
        return jsonify({
            'intents': list(intent_to_idx.keys()),
            'count': len(intent_to_idx)
        })
    else:
        return jsonify({
            'intents': ['flight', 'airport', 'ground_service', 'flight_time'],
            'count': 4,
            'mode': 'demo'
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