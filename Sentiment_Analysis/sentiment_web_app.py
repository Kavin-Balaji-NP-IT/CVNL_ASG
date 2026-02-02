"""
Flask Web Application for Changi Airport Sentiment Analysis
Using RNN Model for Sentiment Classification
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import pickle
import json
import re

app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ComprehensivePreprocessor:
    """Comprehensive preprocessor for airline sentiment"""
    
    def __init__(self, max_vocab_size=3000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_size = 2
        
    def clean_text(self, text):
        """Clean text while preserving sentiment"""
        if not isinstance(text, str):
            text = str(text)
        
        text = text.lower().strip()
        
        # Handle contractions
        text = text.replace("n't", " not")
        text = text.replace("'re", " are")
        text = text.replace("'ve", " have")
        text = text.replace("'ll", " will")
        text = text.replace("'d", " would")
        text = text.replace("'m", " am")
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s!?.]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def text_to_sequence(self, text):
        """Convert text to sequence"""
        words = self.clean_text(text).split()
        return [self.word2idx.get(word, 1) for word in words]


class ComprehensiveRNN(nn.Module):
    """Comprehensive RNN with attention for sentiment analysis"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=5):
        super(ComprehensiveRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)
        
        # Bidirectional LSTM with 2 layers
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, 2,
            batch_first=True, dropout=0.2, bidirectional=True
        )
        
        # Multi-head attention (simplified)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification layers
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # Bidirectional LSTM
        rnn_out, _ = self.rnn(embedded)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(rnn_out), dim=1)
        attended_output = torch.sum(attention_weights * rnn_out, dim=1)
        
        # Classification with batch norm
        output = self.dropout(attended_output)
        
        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.fc3(output)
        
        return output


class SentimentAnalyzer:
    """RNN-based sentiment analyzer"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_info = None
        self.load_model()
    
    def load_model(self):
        """Load the trained RNN model and preprocessor"""
        try:
            # Load model info
            with open('comprehensive_sentiment_model_info.json', 'r') as f:
                self.model_info = json.load(f)
            
            # Load preprocessor
            with open('comprehensive_sentiment_preprocessor.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            # Initialize model
            self.model = ComprehensiveRNN(
                vocab_size=self.model_info['vocab_size'],
                embedding_dim=self.model_info['embedding_dim'],
                hidden_dim=self.model_info['hidden_dim'],
                num_classes=self.model_info['num_classes']
            ).to(device)
            
            # Load trained weights
            checkpoint = torch.load('comprehensive_sentiment_model.pth', map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print("Comprehensive RNN sentiment model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, text):
        """Predict sentiment for given text using RNN"""
        if not self.model:
            return None
        
        try:
            # Preprocess text
            sequence = self.preprocessor.text_to_sequence(text)
            max_length = self.model_info['max_length']
            
            # Pad or truncate
            if len(sequence) < max_length:
                sequence = sequence + [0] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
            
            # Convert to tensor
            sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get results
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            all_probabilities = probabilities[0].cpu().numpy()
            
            result = {
                'predicted_sentiment': self.model_info['label_names'][predicted_class],
                'confidence': float(confidence_score),
                'all_probabilities': {
                    label: float(prob) for label, prob in 
                    zip(self.model_info['label_names'], all_probabilities)
                },
                'processed_text': self.preprocessor.clean_text(text)
            }
            
            return result
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None


# Initialize the sentiment analyzer
analyzer = SentimentAnalyzer()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not analyzer.model:
            return jsonify({'error': 'RNN model not loaded'}), 500
        
        result = analyzer.predict(text)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_type': 'RNN',
        'model_loaded': analyzer.model is not None,
        'device': str(device)
    })


if __name__ == '__main__':
    print("="*60)
    print("CHANGI AIRPORT SENTIMENT ANALYSIS WEB APP")
    print("="*60)
    print("Using RNN Model for Sentiment Classification")
    print(f"Device: {device}")
    print(f"Model loaded: {analyzer.model is not None}")
    print("Starting Flask server...")
    print("Access the app at: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)