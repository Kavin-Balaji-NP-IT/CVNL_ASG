"""
Deployment Script for RNN Intent Classification Model
Loads trained model and provides inference capabilities
"""

import torch
import torch.nn as nn
import pickle
import json
import re
from typing import List, Tuple, Dict


class TextPreprocessor:
    """Handles text preprocessing and vocabulary building"""
    
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def clean_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        words = text.split()
        sequence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        return sequence


class IntentLSTM(nn.Module):
    """LSTM-based Intent Classifier"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, num_classes=3, dropout=0.5):
        super(SentimentLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class IntentAnalyzer:
    """
    Deployment class for intent classification inference
    """
    
    def __init__(self, model_path='best_model.pth', 
                 preprocessor_path='preprocessor.pkl',
                 model_info_path='model_info.json'):
        """
        Initialize the intent analyzer
        
        Args:
            model_path: Path to saved model checkpoint
            preprocessor_path: Path to saved preprocessor
            model_info_path: Path to model configuration
        """
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load preprocessor
        print("Loading preprocessor...")
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Load model info
        print("Loading model configuration...")
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Initialize model
        print("Initializing model...")
        self.model = IntentLSTM(
            vocab_size=self.model_info['vocab_size'],
            embedding_dim=self.model_info['embedding_dim'],
            hidden_dim=self.model_info['hidden_dim'],
            num_layers=self.model_info['num_layers'],
            num_classes=self.model_info['num_classes'],
            dropout=self.model_info['dropout']
        ).to(self.device)
        
        # Load trained weights
        print("Loading model weights...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Get class names and mapping
        self.class_names = self.model_info['class_names']
        self.idx_to_label = self.model_info['idx_to_label']
        # Convert string keys to int keys for idx_to_label
        self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}
        self.max_length = self.model_info['max_length']
        
        print("Model loaded successfully!")
        print(f"Classes: {self.class_names}")
        
    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocess text for model input
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed tensor
        """
        # Clean text
        clean_text = self.preprocessor.clean_text(text)
        
        # Convert to sequence
        sequence = self.preprocessor.text_to_sequence(clean_text)
        
        # Pad or truncate
        if len(sequence) < self.max_length:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        # Convert to tensor
        tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
        
        return tensor
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict intent for a single text
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (predicted_intent, confidence, all_probabilities)
        """
        # Preprocess
        input_tensor = self.preprocess_text(text)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # Get results
        predicted_class = predicted.item()
        predicted_intent = self.idx_to_label[predicted_class]
        confidence_score = confidence.item()
        
        # Get all probabilities
        all_probs = {
            self.class_names[i]: probabilities[0][i].item() 
            for i in range(len(self.class_names))
        }
        
        return predicted_intent, confidence_score, all_probs
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Predict intent for a batch of texts
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of tuples (predicted_intent, confidence, all_probabilities)
        """
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
    
    def analyze_text(self, text: str, verbose: bool = True) -> Dict:
        """
        Analyze text and return detailed results
        
        Args:
            text: Input text string
            verbose: Whether to print results
            
        Returns:
            Dictionary with analysis results
        """
        intent, confidence, probabilities = self.predict(text)
        
        result = {
            'text': text,
            'predicted_intent': intent,
            'confidence': confidence,
            'probabilities': probabilities
        }
        
        if verbose:
            print("\n" + "="*80)
            print("INTENT CLASSIFICATION RESULT")
            print("="*80)
            print(f"Text: {text}")
            print(f"\nPredicted Intent: {intent.upper()}")
            print(f"Confidence: {confidence:.4f}")
            print(f"\nProbability Distribution:")
            # Sort by probability
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                bar = "█" * int(prob * 50)
                print(f"  {class_name:<30}: {prob:.4f} {bar}")
            print("="*80)
        
        return result


def main():
    """Demo usage of the intent analyzer"""
    
    print("="*80)
    print("RNN INTENT CLASSIFIER - DEPLOYMENT")
    print("="*80)
    
    # Initialize analyzer
    analyzer = IntentAnalyzer()
    
    # Example texts from SNIPS dataset
    example_texts = [
        "Book a restaurant for two people at 7pm",
        "What's the weather like in New York tomorrow?",
        "Play some jazz music",
        "Add milk to my shopping list",
        "Set an alarm for 6 AM",
        "Tell me a joke",
        "What time is it in London?",
        "Search for Italian restaurants nearby",
    ]
    
    print("\n" + "="*80)
    print("ANALYZING EXAMPLE QUERIES")
    print("="*80)
    
    # Analyze each example
    for i, text in enumerate(example_texts, 1):
        print(f"\n--- Example {i} ---")
        analyzer.analyze_text(text, verbose=True)
    
    # Batch prediction example
    print("\n" + "="*80)
    print("BATCH PREDICTION EXAMPLE")
    print("="*80)
    
    batch_results = analyzer.predict_batch(example_texts)
    
    print(f"\n{'Text':<50} {'Intent':<30} {'Confidence':<12}")
    print("-" * 92)
    for text, (intent, confidence, _) in zip(example_texts, batch_results):
        text_short = text[:47] + "..." if len(text) > 50 else text
        print(f"{text_short:<50} {intent:<30} {confidence:<12.4f}")
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter queries for intent classification.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("Enter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            analyzer.analyze_text(user_input, verbose=True)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
