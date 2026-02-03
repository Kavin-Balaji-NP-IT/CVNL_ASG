"""
Demo Script for Comprehensive Sentiment Analysis
Interactive command-line interface for testing the sentiment model
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


class SentimentAnalyzer:
    """Sentiment analyzer for interactive demo"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_info = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            # Load model info
            with open('improved_sentiment_model_info_v2.json', 'r') as f:
                self.model_info = json.load(f)
            
            # Load preprocessor
            with open('improved_sentiment_preprocessor_v2.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            # Initialize model
            self.model = ImprovedSentimentRNN(
                vocab_size=self.model_info['vocab_size'],
                embedding_dim=self.model_info['embedding_dim'],
                hidden_dim=self.model_info['hidden_dim'],
                num_classes=self.model_info['num_classes']
            ).to(device)
            
            # Load trained weights
            checkpoint = torch.load('improved_sentiment_model_v2.pth', map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print("✅ Improved RNN sentiment model V2 loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def predict(self, text):
        """Predict sentiment for given text"""
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
            print(f"❌ Error in prediction: {e}")
            return None
    
    def analyze_text(self, text):
        """Analyze text and display results"""
        result = self.predict(text)
        
        if not result:
            print("❌ Failed to analyze text")
            return
        
        print("\n" + "="*70)
        print("🎯 SENTIMENT ANALYSIS RESULT")
        print("="*70)
        print(f"📝 Input: {text}")
        print(f"🔍 Processed: {result['processed_text']}")
        print(f"\n🎭 Predicted Sentiment: {result['predicted_sentiment']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        
        # Get sentiment emoji
        sentiment_emojis = {
            'Very Negative': '😡',
            'Negative': '😞', 
            'Neutral': '😐',
            'Positive': '😊',
            'Very Positive': '😍'
        }
        
        emoji = sentiment_emojis.get(result['predicted_sentiment'], '🤔')
        print(f"😀 Emotion: {emoji}")
        
        print(f"\n📈 Probability Distribution:")
        # Sort by probability
        sorted_probs = sorted(result['all_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for sentiment, prob in sorted_probs:
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {sentiment:<15}: {prob:.1%} |{bar}|")
        
        print("="*70)


def main():
    """Interactive demo"""
    
    print("="*70)
    print("🛫 CHANGI AIRPORT SENTIMENT ANALYSIS DEMO")
    print("="*70)
    print("🤖 AI-Powered Passenger Feedback Analysis using Improved RNN V2")
    print("📊 Model: Bidirectional LSTM with Attention Mechanism")
    print("🎯 Classes: Very Negative, Negative, Neutral, Positive, Very Positive")
    print("="*70)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    if not analyzer.model:
        print("❌ Failed to load model. Exiting...")
        return
    
    print(f"📈 Model Info:")
    print(f"  • Test Accuracy: {analyzer.model_info['test_accuracy']:.1%}")
    print(f"  • Vocabulary Size: {analyzer.model_info['vocab_size']:,}")
    print(f"  • Model Parameters: 223K (Improved & Optimized)")
    
    # Example demonstrations
    print("\n" + "="*70)
    print("🧪 EXAMPLE DEMONSTRATIONS")
    print("="*70)
    
    examples = [
        "Very bad service",
        "Nice service but food was lacking taste", 
        "Excellent service and friendly staff",
        "Terrible experience, worst airline ever",
        "Amazing flight, highly recommend",
        "Average experience, nothing special"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        analyzer.analyze_text(example)
    
    # Interactive mode
    print("\n" + "="*70)
    print("🎮 INTERACTIVE MODE")
    print("="*70)
    print("💬 Enter passenger feedback to analyze sentiment")
    print("💡 Try examples like:")
    print("   • 'The staff were amazing but the food was terrible'")
    print("   • 'Outstanding customer service!'")
    print("   • 'Flight was delayed and uncomfortable'")
    print("🚪 Type 'quit' to exit")
    print("="*70)
    
    while True:
        try:
            user_input = input("\n🎤 Enter feedback: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using Changi Airport Sentiment Analysis!")
                break
            
            if not user_input:
                print("⚠️  Please enter some text to analyze.")
                continue
            
            analyzer.analyze_text(user_input)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()