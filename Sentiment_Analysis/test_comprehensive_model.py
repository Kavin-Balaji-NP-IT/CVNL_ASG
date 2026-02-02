"""
Test the comprehensive sentiment model
"""

import torch
import torch.nn as nn
import pickle
import json
import re

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


def load_and_test_model():
    """Load and test the comprehensive model"""
    
    print("="*60)
    print("TESTING COMPREHENSIVE SENTIMENT MODEL")
    print("="*60)
    
    try:
        # Load model info
        with open('comprehensive_sentiment_model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # Load preprocessor
        with open('comprehensive_sentiment_preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Initialize model
        model = ComprehensiveRNN(
            vocab_size=model_info['vocab_size'],
            embedding_dim=model_info['embedding_dim'],
            hidden_dim=model_info['hidden_dim'],
            num_classes=model_info['num_classes']
        ).to(device)
        
        # Load trained weights
        checkpoint = torch.load('comprehensive_sentiment_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("Model loaded successfully!")
        print(f"Test accuracy: {model_info['test_accuracy']:.4f}")
        print(f"Vocabulary size: {model_info['vocab_size']}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test examples
        test_examples = [
            "Very bad service",
            "Nice service but food was lacking taste",
            "Excellent service and friendly staff",
            "Terrible experience, worst airline ever",
            "Amazing flight, highly recommend",
            "Average experience, nothing special",
            "Good staff but poor facilities",
            "Outstanding customer service",
            "Flight was okay",
            "Horrible experience, never again"
        ]
        
        print("\n" + "="*60)
        print("SENTIMENT PREDICTIONS")
        print("="*60)
        
        for example in test_examples:
            # Preprocess
            sequence = preprocessor.text_to_sequence(example)
            max_length = model_info['max_length']
            
            # Pad or truncate
            if len(sequence) < max_length:
                sequence = sequence + [0] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
            
            # Convert to tensor
            sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_label = model_info['label_names'][predicted.item()]
            confidence_score = confidence.item()
            
            print(f"'{example}'")
            print(f"  -> {predicted_label} ({confidence_score:.3f})")
            print()
        
        print("="*60)
        print("Model testing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    load_and_test_model()