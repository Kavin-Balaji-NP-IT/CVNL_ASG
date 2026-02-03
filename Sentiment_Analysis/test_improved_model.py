"""
Test the improved sentiment model with the problematic examples
"""

import torch
import torch.nn as nn
import pickle
import json
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImprovedPreprocessor:
    """Improved preprocessor with better sentiment word handling"""
    
    def __init__(self, max_vocab_size=2000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_size = 2
        
        # Critical sentiment words that MUST be in vocabulary
        self.critical_words = {
            # Very Negative
            'very', 'extremely', 'really', 'so', 'absolutely', 'completely',
            'worst', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'bad', 'poor', 'disappointing', 'unacceptable', 'disaster',
            
            # Very Positive  
            'best', 'excellent', 'amazing', 'fantastic', 'wonderful', 'brilliant',
            'outstanding', 'perfect', 'superb', 'incredible', 'love', 'great',
            'good', 'nice', 'pleasant', 'helpful', 'friendly', 'beautiful',
            
            # Neutral/Mixed
            'but', 'however', 'though', 'although', 'decent', 'okay', 'average',
            'standard', 'normal', 'regular', 'fine', 'acceptable'
        }
        
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


class ImprovedSentimentRNN(nn.Module):
    """Improved RNN with better architecture for clear sentiment detection"""
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_classes=5):
        super(ImprovedSentimentRNN, self).__init__()
        
        # Smaller, more focused architecture
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Single bidirectional LSTM layer
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, 1,
            batch_first=True, bidirectional=True
        )
        
        # Simple attention
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Simpler classification layers
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM
        rnn_out, _ = self.rnn(embedded)
        
        # Attention
        attention_weights = torch.softmax(self.attention(rnn_out), dim=1)
        attended_output = torch.sum(attention_weights * rnn_out, dim=1)
        
        # Classification
        output = self.dropout(attended_output)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output


def test_improved_model():
    """Test the improved model with problematic examples"""
    
    print("="*70)
    print("TESTING IMPROVED SENTIMENT MODEL V2")
    print("="*70)
    
    # Load model info
    with open('improved_sentiment_model_info_v2.json', 'r') as f:
        model_info = json.load(f)
    
    # Load preprocessor
    with open('improved_sentiment_preprocessor_v2.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Initialize model
    model = ImprovedSentimentRNN(
        vocab_size=model_info['vocab_size'],
        embedding_dim=model_info['embedding_dim'],
        hidden_dim=model_info['hidden_dim'],
        num_classes=model_info['num_classes']
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load('improved_sentiment_model_v2.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Test Accuracy: {model_info['test_accuracy']:.1%}")
    print(f"Vocabulary Size: {model_info['vocab_size']}")
    
    # Test the problematic examples from the context
    problematic_examples = [
        "Very bad service",  # Was: Neutral (64.9%) - should be Very Negative
        "Excellent service and friendly staff",  # Was: Very Positive (50.5%) - should be higher confidence
        "Terrible experience",  # Was: Very Negative (47.7%) - should be higher confidence
        "Worst service ever!",  # Should be Very Negative with high confidence
        "nice service but food was lacking taste",  # Should be Neutral (mixed sentiment)
        "Amazing flight experience",  # Should be Very Positive with high confidence
        "average experience, nothing special",  # Should be Neutral
        "outstanding customer service"  # Should be Very Positive with high confidence
    ]
    
    print("\n" + "="*70)
    print("TESTING PROBLEMATIC EXAMPLES")
    print("="*70)
    
    for example in problematic_examples:
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
        
        # Color coding for confidence
        if confidence_score >= 0.9:
            confidence_color = "🟢"  # High confidence
        elif confidence_score >= 0.7:
            confidence_color = "🟡"  # Medium confidence
        else:
            confidence_color = "🔴"  # Low confidence
        
        print(f"'{example}'")
        print(f"  → {predicted_label} ({confidence_score:.1%}) {confidence_color}")
        print()
    
    print("="*70)
    print("LEGEND:")
    print("🟢 High Confidence (≥90%)")
    print("🟡 Medium Confidence (70-89%)")
    print("🔴 Low Confidence (<70%)")
    print("="*70)


if __name__ == "__main__":
    test_improved_model()