"""
Changi Virtual Assistant - Prediction Module
Standalone module for loading and using the trained RNN model
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import re
from collections import Counter
import os


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
        # Embedding layer
        embedded = self.embedding(x)
        embedded = self.emb_dropout(embedded)

        # Pack padded sequence for efficient LSTM processing
        packed = pack_padded_sequence(embedded, lengths.cpu(),
                                     batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)

        # Concatenate forward and backward hidden states
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # Classification layer
        out = self.dropout(hidden)
        return self.fc(out)


class ChangiPredictor:
    """Main predictor class for Changi Virtual Assistant"""
    
    def __init__(self, model_path='changi_airport_rnn_atis_complete.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = None
        self.intent_to_idx = {}
        self.idx_to_intent = {}
        self.model_loaded = False
        
        if os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model and all components"""
        try:
            print(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract components
            vocab = checkpoint['vocab']
            self.intent_to_idx = checkpoint['intent_to_idx']
            self.idx_to_intent = checkpoint['idx_to_intent']
            hyperparams = checkpoint['hyperparameters']
            
            # Initialize preprocessor
            self.preprocessor = TextPreprocessor(vocab)
            
            # Initialize model
            self.model = IntentRNN(
                vocab_size=hyperparams['vocab_size'],
                embed_dim=hyperparams['embed_dim'],
                hidden_dim=hyperparams['hidden_dim'],
                num_classes=len(self.intent_to_idx),
                num_layers=hyperparams['num_layers'],
                dropout=hyperparams['dropout'],
                bidirectional=hyperparams['bidirectional']
            ).to(self.device)
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            
            self.model_loaded = True
            
            print(f"✓ Model loaded successfully!")
            print(f"  Vocabulary size: {hyperparams['vocab_size']}")
            print(f"  Number of intents: {len(self.intent_to_idx)}")
            print(f"  Test accuracy: {checkpoint['test_acc']:.2f}%")
            print(f"  Device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model_loaded = False
    
    def predict(self, query, top_k=3):
        """
        Predict intent from user query
        
        Args:
            query: User query string
            top_k: Number of top predictions to return
        
        Returns:
            dict with predictions, confidence scores, and top intent
        """
        if not self.model_loaded:
            return None
        
        # Preprocess
        indices = self.preprocessor.text_to_indices(query)
        
        if not indices:
            indices = [1]  # <UNK> token
        
        # Convert to tensor
        indices_tensor = torch.LongTensor([indices]).to(self.device)
        length = torch.LongTensor([len(indices)])

        # Predict
        with torch.no_grad():
            outputs = self.model(indices_tensor, length)
            probs = torch.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, min(top_k, len(self.intent_to_idx)))

        # Format results
        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            predictions.append({
                'intent': self.idx_to_intent[int(idx)],
                'confidence': float(prob) * 100
            })
        
        return {
            'predictions': predictions,
            'top_intent': predictions[0]['intent'],
            'top_confidence': predictions[0]['confidence']
        }
    
    def get_intents(self):
        """Get list of all possible intents"""
        if self.model_loaded:
            return list(self.intent_to_idx.keys())
        return []
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model_loaded


def main():
    """Test the predictor"""
    predictor = ChangiPredictor()
    
    if not predictor.is_loaded():
        print("Model not loaded. Please train the model first using changi_rnn_model.py")
        return
    
    # Test queries
    test_queries = [
        "What flights are available to Bangkok?",
        "Where is gate C9?",
        "How to get to city center?",
        "What does SQ mean?",
        "Flight times to Mumbai",
        "Terminal 3 information"
    ]
    
    print("\n" + "="*50)
    print("TESTING PREDICTOR")
    print("="*50)
    
    for query in test_queries:
        result = predictor.predict(query, top_k=3)
        if result:
            print(f"\nQuery: '{query}'")
            print(f"Top Intent: {result['top_intent']} ({result['top_confidence']:.1f}%)")
            print("All predictions:")
            for pred in result['predictions']:
                print(f"  - {pred['intent']}: {pred['confidence']:.1f}%")
        else:
            print(f"\nQuery: '{query}' - Prediction failed")


if __name__ == "__main__":
    main()