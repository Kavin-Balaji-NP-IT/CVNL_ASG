"""
Changi Virtual Assistant - RNN Intent Classification Model
Converted from Jupyter notebook to standalone Python module
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import numpy as np
from collections import Counter
import re
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TextPreprocessor:
    """Text preprocessing and vocabulary management"""
    def __init__(self):
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.word_to_idx = self.vocab.copy()

    def clean_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize(self, text):
        """Tokenize text into words"""
        return text.split()

    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        word_freq = Counter()
        for text in texts:
            word_freq.update(self.tokenize(self.clean_text(text)))

        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)

        print(f"\nVocabulary built!")
        print(f"  Vocab size: {len(self.word_to_idx)} (min_freq={min_freq})")
        return self.word_to_idx

    def text_to_indices(self, text):
        """Convert text to indices"""
        tokens = self.tokenize(self.clean_text(text))
        return [self.word_to_idx.get(t, 1) for t in tokens]  # 1 is <UNK>


class IntentDataset(Dataset):
    """PyTorch Dataset for intent classification"""
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = self.preprocessor.text_to_indices(self.texts[idx])
        return {
            'indices': torch.LongTensor(indices),
            'label': torch.LongTensor([self.labels[idx]]),
            'length': len(indices)
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    indices = pad_sequence([b['indices'] for b in batch], batch_first=True)
    labels = torch.cat([b['label'] for b in batch])
    lengths = torch.LongTensor([b['length'] for b in batch])
    return {'indices': indices, 'labels': labels, 'lengths': lengths}


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


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        indices = batch['indices'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths']

        optimizer.zero_grad()
        outputs = model(indices, lengths)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            indices = batch['indices'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']

            outputs = model(indices, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), 100 * correct / total, all_preds, all_labels


def load_and_prepare_data():
    """Load ATIS dataset and prepare for training"""
    print("Loading ATIS dataset from Hugging Face...")
    ds = load_dataset("tuetschek/atis")

    # Extract data from dataset and convert to lists
    train_texts = list(ds['train']['text'])
    train_labels_raw = list(ds['train']['intent'])

    test_texts = list(ds['test']['text'])
    test_labels_raw = list(ds['test']['intent'])

    # Create validation split from training data (10%) - SHUFFLE FIRST
    import random
    combined = list(zip(train_texts, train_labels_raw))
    random.shuffle(combined)
    train_texts, train_labels_raw = zip(*combined)
    train_texts, train_labels_raw = list(train_texts), list(train_labels_raw)

    # Create validation split from training data (10%)
    val_size = int(0.1 * len(train_texts))
    val_texts = train_texts[:val_size]
    val_labels_raw = train_labels_raw[:val_size]
    train_texts = train_texts[val_size:]
    train_labels_raw = train_labels_raw[val_size:]

    print(f"\nDataset loaded successfully!")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Val:   {len(val_texts)} samples")
    print(f"  Test:  {len(test_texts)} samples")

    # Add Singapore/Changi short follow-up examples (helps 1-word replies)
    extra = [
        ("t1", "abbreviation"),
        ("t2", "abbreviation"),
        ("t3", "abbreviation"),
        ("terminal 1", "abbreviation"),
        ("terminal 2", "abbreviation"),
        ("terminal 3", "abbreviation"),
        ("what is t1", "abbreviation"),
        ("what does sin mean", "abbreviation"),
        ("sin airport code", "abbreviation"),
        ("sq airline", "abbreviation"),
        ("tr airline", "abbreviation"),
        ("3k airline", "abbreviation"),
        ("mumbai", "city"),
        ("bangkok", "city"),
        ("kuala lumpur", "city"),
        ("jakarta", "city"),
        ("sydney", "city"),
    ]

    train_texts.extend([t for t, y in extra])
    train_labels_raw.extend([y for t, y in extra])

    # Get unique intents and create mapping
    all_intents = sorted(list(set(train_labels_raw + test_labels_raw + val_labels_raw)))
    intent_to_idx = {intent: idx for idx, intent in enumerate(all_intents)}
    idx_to_intent = {idx: intent for intent, idx in intent_to_idx.items()}

    NUM_CLASSES = len(intent_to_idx)

    # Convert intent strings to indices
    train_labels = [intent_to_idx[intent] for intent in train_labels_raw]
    val_labels = [intent_to_idx[intent] for intent in val_labels_raw]
    test_labels = [intent_to_idx[intent] for intent in test_labels_raw]

    print(f"\nNumber of unique intents: {NUM_CLASSES}")
    print(f"\nTop 15 most common intents:")
    intent_counts = Counter(train_labels_raw)
    for i, (intent, count) in enumerate(intent_counts.most_common(15), 1):
        print(f"  {i:2d}. {intent:<35} {count:>4} samples")

    return (train_texts, train_labels, val_texts, val_labels, 
            test_texts, test_labels, intent_to_idx, idx_to_intent, NUM_CLASSES)


def create_data_loaders(train_texts, train_labels, val_texts, val_labels, 
                       test_texts, test_labels, preprocessor, batch_size=32):
    """Create PyTorch DataLoaders"""
    train_dataset = IntentDataset(train_texts, train_labels, preprocessor)
    val_dataset = IntentDataset(val_texts, val_labels, preprocessor)
    test_dataset = IntentDataset(test_texts, test_labels, preprocessor)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size, collate_fn=collate_fn)

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, device, epochs=20, lr=0.001):
    """Train the RNN model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0

    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:2d}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:6.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_atis.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        print()

    print(f"{'='*70}")
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}\n")

    return history, best_val_acc


def save_complete_model(model, preprocessor, intent_to_idx, idx_to_intent, 
                       test_acc, hyperparams, filename='changi_airport_rnn_atis_complete.pth'):
    """Save complete model with all necessary components"""
    checkpoint = {
        'model_state': model.state_dict(),
        'vocab': preprocessor.word_to_idx,
        'intent_to_idx': intent_to_idx,
        'idx_to_intent': idx_to_intent,
        'hyperparameters': hyperparams,
        'test_acc': test_acc,
        'model_class': 'IntentRNN'
    }
    
    torch.save(checkpoint, filename)
    print(f"✓ Complete model saved as '{filename}'")
    print(f"  Model includes: state_dict, vocabulary, intent mappings, hyperparameters")
    print(f"  Test accuracy: {test_acc:.2f}%")


def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load and prepare data
    (train_texts, train_labels, val_texts, val_labels, 
     test_texts, test_labels, intent_to_idx, idx_to_intent, NUM_CLASSES) = load_and_prepare_data()

    # Build vocabulary
    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(train_texts, min_freq=2)

    # Analyze sequence lengths
    lengths = [len(preprocessor.text_to_indices(t)) for t in train_texts]
    print(f"\nSequence length statistics:")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.0f}")
    print(f"  Max: {np.max(lengths)}")
    print(f"  Min: {np.min(lengths)}")
    print(f"  95th percentile: {np.percentile(lengths, 95):.0f}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_texts, train_labels, val_texts, val_labels, 
        test_texts, test_labels, preprocessor, batch_size=32)

    # Initialize model
    VOCAB_SIZE = len(preprocessor.word_to_idx)
    hyperparams = {
        'vocab_size': VOCAB_SIZE,
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True
    }

    model = IntentRNN(
        vocab_size=hyperparams['vocab_size'],
        embed_dim=hyperparams['embed_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        num_classes=NUM_CLASSES,
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout'],
        bidirectional=hyperparams['bidirectional']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Train model
    history, best_val_acc = train_model(model, train_loader, val_loader, device, epochs=20)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_model_atis.pth'))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print(f"{'='*70}\n")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Save complete model
    save_complete_model(model, preprocessor, intent_to_idx, idx_to_intent, 
                       test_acc, hyperparams)

    # Detailed classification report
    unique_test_labels = sorted(list(set(y_true)))
    intent_names_in_test = [idx_to_intent[i] for i in unique_test_labels]

    print(f"\n{'='*70}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*70}\n")
    print(classification_report(y_true, y_pred, labels=unique_test_labels,
                               target_names=intent_names_in_test, zero_division=0))

    return model, preprocessor, intent_to_idx, idx_to_intent, test_acc


if __name__ == "__main__":
    main()