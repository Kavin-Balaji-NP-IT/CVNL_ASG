"""
RNN Model Development for Changi Airport Sentiment Analysis
Complete implementation following academic requirements:
- Dataset: Kaggle crowdflower/twitter-airline-sentiment
- Model: LSTM-based RNN with justification
- Evaluation: Comprehensive metrics and analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import pickle
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import time
from datetime import datetime

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TextPreprocessor:
    """
    Text preprocessor for sentiment analysis
    Handles tokenization, vocabulary building, and sequence creation
    """
    
    def __init__(self, max_vocab_size=5000, max_length=50):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def clean_text(self, text):
        """
        Clean and normalize text
        - Convert to lowercase
        - Handle contractions
        - Remove special characters
        - Preserve sentiment-relevant punctuation
        """
        if not isinstance(text, str):
            text = str(text)
        
        text = text.lower().strip()
        
        # Handle contractions
        contractions = {
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "can't": "cannot", "won't": "will not",
            "shouldn't": "should not", "wouldn't": "would not", "couldn't": "could not"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove URLs and mentions
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        
        # Keep letters, numbers, and important punctuation
        text = re.sub(r'[^\w\s!?.]', ' ', text)
        
        # Handle repeated characters (sooo -> so)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from training texts
        Prioritizes most frequent words up to max_vocab_size
        """
        word_counts = Counter()
        
        # Count all words
        for text in texts:
            words = self.clean_text(text).split()
            word_counts.update(words)
        
        # Add most common words to vocabulary
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        for word, count in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
        
        print(f"Built vocabulary with {self.vocab_size} words")
        print(f"Most common words: {list(word_counts.most_common(10))}")
    
    def text_to_sequence(self, text):
        """Convert text to sequence of token indices"""
        words = self.clean_text(text).split()
        return [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>
    
    def pad_sequences(self, sequences):
        """
        Pad or truncate sequences to ensure uniform length
        - Pad shorter sequences with 0 (<PAD>)
        - Truncate longer sequences
        """
        padded_sequences = []
        for seq in sequences:
            if len(seq) < self.max_length:
                # Pad with zeros
                padded_seq = seq + [0] * (self.max_length - len(seq))
            else:
                # Truncate
                padded_seq = seq[:self.max_length]
            padded_sequences.append(padded_seq)
        return padded_sequences


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis"""
    
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        
        # Convert texts to sequences
        sequences = [preprocessor.text_to_sequence(text) for text in texts]
        self.sequences = preprocessor.pad_sequences(sequences)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label


class SentimentLSTM(nn.Module):
    """
    LSTM-based RNN for sentiment classification
    
    Architecture Choice Justification:
    - LSTM chosen over vanilla RNN: Better at handling long sequences and vanishing gradient problem
    - LSTM chosen over GRU: More parameters allow for better performance on complex sentiment patterns
    - Bidirectional: Captures context from both directions
    - Dropout: Prevents overfitting
    - Multiple layers: Increases model capacity for complex patterns
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, 
                 num_classes=3, dropout=0.3, bidirectional=True):
        super(SentimentLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state (for bidirectional, concatenate both directions)
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Classification
        output = self.dropout(hidden)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.relu(self.fc2(output))
        output = self.dropout(output)
        output = self.fc3(output)
        
        return output


def load_kaggle_dataset():
    """
    Load and preprocess the Kaggle airline sentiment dataset
    Returns texts and labels for training
    """
    print("="*60)
    print("LOADING KAGGLE AIRLINE SENTIMENT DATASET")
    print("="*60)
    
    try:
        # Download dataset using kagglehub
        path = kagglehub.dataset_download("crowdflower/twitter-airline-sentiment")
        print(f"Path to dataset files: {path}")
        
        # Load the CSV file
        data_file = f"{path}/Tweets.csv"
        df = pd.read_csv(data_file)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Extract texts and sentiments
        texts = df['text'].astype(str).tolist()
        sentiments = df['airline_sentiment'].tolist()
        
        # Map sentiments to numerical labels
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        labels = [sentiment_map[s] for s in sentiments]
        
        # Display dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(texts)}")
        for sentiment, label in sentiment_map.items():
            count = labels.count(label)
            percentage = (count / len(labels)) * 100
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Show sample texts
        print(f"\nSample texts:")
        for i in range(3):
            print(f"  {i+1}. [{sentiments[i].upper()}] {texts[i][:100]}...")
        
        return texts, labels, sentiment_map
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """
    Train the RNN model with hyperparameter tuning and performance monitoring
    """
    print("="*60)
    print("TRAINING RNN MODEL")
    print("="*60)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_epochs} epochs...")
    print()
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
        print(f'  Val Acc: {val_acc:.2f}%')
        print(f'  Time: {time.time() - start_time:.1f}s')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch + 1,
                'model_config': {
                    'vocab_size': model.embedding.num_embeddings,
                    'embedding_dim': model.embedding.embedding_dim,
                    'hidden_dim': model.hidden_dim,
                    'num_layers': model.num_layers,
                    'num_classes': model.fc3.out_features,
                    'bidirectional': model.bidirectional
                }
            }, 'sentiment_analysis_model.pth')  # Changed filename to match web app
            print(f'  ✅ New best model saved! Val Acc: {val_acc:.2f}%')
        
        print("-" * 60)
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"✅ Best validation accuracy: {best_val_acc:.2f}%")
    
    return train_losses, train_accuracies, val_accuracies, best_val_acc


def evaluate_model(model, test_loader, class_names):
    """
    Comprehensive model evaluation with multiple metrics
    - Accuracy, Precision, Recall, F1-score
    - Confusion Matrix
    - Misclassification Analysis
    """
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, labels=[0, 1, 2]
    )
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted'
    )
    
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # Per-class metrics
    print("Per-class Metrics:")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall:    {recall[i]:.4f}")
        print(f"  F1-score:  {f1[i]:.4f}")
        print(f"  Support:   {support[i]}")
        print()
    
    # Average metrics
    print("Average Metrics:")
    print("-" * 50)
    print(f"Macro Average:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall:    {recall_macro:.4f}")
    print(f"  F1-score:  {f1_macro:.4f}")
    print()
    print(f"Weighted Average:")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall:    {recall_weighted:.4f}")
    print(f"  F1-score:  {f1_weighted:.4f}")
    print()
    
    # Detailed classification report
    print("Detailed Classification Report:")
    print("-" * 50)
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - RNN Sentiment Analysis')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('rnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Misclassification Analysis
    print("Misclassification Analysis:")
    print("-" * 50)
    misclassified = []
    for i, (true_label, pred_label, prob) in enumerate(zip(all_targets, all_predictions, all_probabilities)):
        if true_label != pred_label:
            confidence = prob[pred_label]
            misclassified.append({
                'index': i,
                'true_label': class_names[true_label],
                'predicted_label': class_names[pred_label],
                'confidence': confidence
            })
    
    print(f"Total misclassifications: {len(misclassified)}")
    print(f"Misclassification rate: {len(misclassified)/len(all_targets)*100:.2f}%")
    
    # Show most confident misclassifications
    misclassified.sort(key=lambda x: x['confidence'], reverse=True)
    print("\nMost confident misclassifications:")
    for i, misc in enumerate(misclassified[:5]):
        print(f"{i+1}. True: {misc['true_label']}, Predicted: {misc['predicted_label']}, "
              f"Confidence: {misc['confidence']:.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'misclassified_count': len(misclassified)
    }


def plot_training_history(train_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Training and validation accuracy
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('rnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training and evaluation pipeline"""
    print("="*60)
    print("RNN MODEL DEVELOPMENT FOR SENTIMENT ANALYSIS")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print("="*60)
    
    # Load dataset
    texts, labels, sentiment_map = load_kaggle_dataset()
    if texts is None:
        return
    
    class_names = ['Negative', 'Neutral', 'Positive']
    
    # Create preprocessor and build vocabulary
    preprocessor = TextPreprocessor(max_vocab_size=5000, max_length=50)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Build vocabulary on training data
    preprocessor.build_vocabulary(X_train)
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, preprocessor)
    val_dataset = SentimentDataset(X_val, y_val, preprocessor)
    test_dataset = SentimentDataset(X_test, y_test, preprocessor)
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SentimentLSTM(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        bidirectional=True
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(f"- Type: Bidirectional LSTM")
    print(f"- Vocabulary size: {preprocessor.vocab_size:,}")
    print(f"- Embedding dimension: 128")
    print(f"- Hidden dimension: 256")
    print(f"- Number of layers: 2")
    print(f"- Dropout: 0.3")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, train_accuracies, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader, num_epochs=5, learning_rate=0.001  # Reduced epochs
    )
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies, val_accuracies)
    
    # Load best model for evaluation
    checkpoint = torch.load('sentiment_analysis_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    evaluation_results = evaluate_model(model, test_loader, class_names)
    
    # Save preprocessor and model info
    with open('sentiment_analysis_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    model_info = {
        'vocab_size': preprocessor.vocab_size,
        'max_length': preprocessor.max_length,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'num_classes': 3,
        'label_names': class_names,  # Changed from class_names to label_names to match web app
        'test_accuracy': evaluation_results['accuracy'],
        'best_val_accuracy': best_val_acc / 100,
        'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('sentiment_analysis_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("="*60)
    print("MODEL DEVELOPMENT COMPLETED")
    print("="*60)
    print(f"✅ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✅ Test accuracy: {evaluation_results['accuracy']*100:.2f}%")
    print(f"✅ Model saved as: best_sentiment_rnn_model.pth")
    print(f"✅ Preprocessor saved as: rnn_preprocessor.pkl")
    print(f"✅ Model info saved as: rnn_model_info.json")
    print(f"✅ Visualizations saved as: rnn_confusion_matrix.png, rnn_training_history.png")
    print("="*60)


if __name__ == "__main__":
    main()