"""
RNN-Based Intent Classification for Aviation Domain
Dataset: SNIPS Intent Classification Dataset
Model: LSTM (Long Short-Term Memory)
Classes: Multiple intent classes (6-7 intents)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import pickle
import json
from datasets import load_dataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TextPreprocessor:
    """Handles text preprocessing and vocabulary building"""
    
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def build_vocabulary(self, texts):
        """Build vocabulary from texts"""
        word_counts = Counter()
        
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Get most common words
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
        
        print(f"Vocabulary size: {self.vocab_size}")
        return self.word2idx
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        words = text.split()
        sequence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        return sequence


class SentimentDataset(Dataset):
    """Custom Dataset for sentiment analysis"""
    
    def __init__(self, texts, labels, preprocessor, max_length=100):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Clean text
        clean_text = self.preprocessor.clean_text(text)
        
        # Convert to sequence
        sequence = self.preprocessor.text_to_sequence(clean_text)
        
        # Pad or truncate
        if len(sequence) < self.max_length:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class IntentLSTM(nn.Module):
    """
    LSTM-based Intent Classifier
    
    Architecture Choice Justification:
    - LSTM over vanilla RNN: Addresses vanishing gradient problem, better at capturing 
      long-range dependencies in text
    - LSTM over GRU: More expressive with separate forget and input gates, performs 
      better on intent classification tasks with complex patterns
    - Bidirectional: Captures context from both directions in the sentence
    - Dropout: Prevents overfitting on the dataset
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, num_classes=7, dropout=0.5):
        super(IntentLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate final hidden states from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # Fully connected layers
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def load_and_prepare_data():
    """Load and prepare the SNIPS Intent Classification dataset"""
    
    print("Loading SNIPS dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("DeepPavlov/snips", "intents")
    
    # Get training data
    train_data = dataset['train']
    
    # Extract texts and labels
    texts = []
    labels = []
    
    for example in train_data:
        texts.append(example['text'])
        labels.append(example['label'])
    
    # Convert to numpy arrays
    texts = np.array(texts)
    labels = np.array(labels)
    
    # Get unique labels and create mapping
    unique_labels = sorted(set(labels))
    label_names = [train_data.features['label'].int2str(label) for label in unique_labels]
    
    # Create label mapping
    label_map = {name: idx for idx, name in enumerate(label_names)}
    idx_to_label = {idx: name for name, idx in label_map.items()}
    
    print(f"Total samples: {len(texts)}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"\nClass distribution:")
    
    # Count samples per class
    for label_idx in unique_labels:
        label_name = train_data.features['label'].int2str(label_idx)
        count = np.sum(labels == label_idx)
        percentage = (count / len(labels)) * 100
        print(f"  {label_name}: {count} ({percentage:.1f}%)")
    
    return texts, labels, label_map, idx_to_label, label_names


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (sequences, labels) in enumerate(dataloader):
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    return avg_loss, accuracy, precision, recall, f1, all_labels, all_preds


def plot_confusion_matrix(labels, predictions, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - LSTM Sentiment Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(train_losses, val_losses, train_accs, val_accs, 
                         save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', marker='o')
    ax1.plot(val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy', marker='o')
    ax2.plot(val_accs, label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")


def analyze_misclassifications(texts, true_labels, pred_labels, idx_to_label, 
                               preprocessor, num_examples=5):
    """Analyze misclassified examples"""
    
    misclassified_indices = [i for i in range(len(true_labels)) 
                            if true_labels[i] != pred_labels[i]]
    
    print(f"\n{'='*80}")
    print(f"MISCLASSIFICATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Total misclassified: {len(misclassified_indices)} / {len(true_labels)}")
    print(f"Misclassification rate: {len(misclassified_indices)/len(true_labels)*100:.2f}%\n")
    
    # Sample random misclassifications
    if len(misclassified_indices) > num_examples:
        sample_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
    else:
        sample_indices = misclassified_indices
    
    print(f"Sample Misclassified Examples:\n")
    for i, idx in enumerate(sample_indices, 1):
        print(f"Example {i}:")
        print(f"  Text: {texts[idx][:100]}...")
        print(f"  True: {idx_to_label[true_labels[idx]]}")
        print(f"  Predicted: {idx_to_label[pred_labels[idx]]}")
        print()


def main():
    """Main training pipeline"""
    
    # Hyperparameters (tuned for best performance)
    BATCH_SIZE = 64
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 15
    MAX_LENGTH = 50  # SNIPS queries are typically shorter
    MAX_VOCAB_SIZE = 10000
    
    print("="*80)
    print("RNN-BASED INTENT CLASSIFICATION - SNIPS DATASET")
    print("="*80)
    print(f"\nHyperparameters:")
    print(f"  Model: Bidirectional LSTM")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Embedding Dim: {EMBEDDING_DIM}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Num Layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Max Sequence Length: {MAX_LENGTH}")
    print(f"  Max Vocabulary Size: {MAX_VOCAB_SIZE}")
    print()
    
    # Load data
    texts, labels, label_map, idx_to_label, label_names = load_and_prepare_data()
    num_classes = len(label_names)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Initialize preprocessor and build vocabulary
    preprocessor = TextPreprocessor(max_vocab_size=MAX_VOCAB_SIZE)
    
    # Clean training texts and build vocabulary
    cleaned_train_texts = [preprocessor.clean_text(text) for text in X_train]
    preprocessor.build_vocabulary(cleaned_train_texts)
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, preprocessor, MAX_LENGTH)
    val_dataset = SentimentDataset(X_val, y_val, preprocessor, MAX_LENGTH)
    test_dataset = SentimentDataset(X_test, y_test, preprocessor, MAX_LENGTH)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = IntentLSTM(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
            print("✓ Best model saved!")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    test_loss, test_acc, test_precision, test_recall, test_f1, true_labels, predictions = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        true_labels, predictions, average=None
    )
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 73)
    for i, class_name in enumerate(label_names):
        print(f"{class_name:<25} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
              f"{f1_per_class[i]:<12.4f} {support[i]:<12}")
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, label_names)
    
    # Misclassification analysis
    analyze_misclassifications(X_test, true_labels, predictions, idx_to_label, preprocessor)
    
    # Save preprocessor and model artifacts
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    model_info = {
        'vocab_size': preprocessor.vocab_size,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'num_classes': num_classes,
        'dropout': DROPOUT,
        'max_length': MAX_LENGTH,
        'label_map': label_map,
        'idx_to_label': idx_to_label,
        'class_names': label_names
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("Saved files:")
    print("  - best_model.pth")
    print("  - preprocessor.pkl")
    print("  - model_info.json")
    print("  - confusion_matrix.png")
    print("  - training_history.png")
    print("="*80)


if __name__ == "__main__":
    main()
