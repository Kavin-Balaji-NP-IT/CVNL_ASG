"""
Changi Virtual Assist Triage - RNN Text Classification
========================================================
A PyTorch-based RNN model for classifying passenger/staff queries at Changi Airport
into operational categories for efficient triage and response.

Intent Categories:
1. Flight Status / Gate Change
2. Baggage Issues  
3. Terminal Directions
4. Special Assistance
5. Transport / Ground Connectivity
6. Security / Customs Queries
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import numpy as np
import pandas as pd
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Model and training configuration"""
    # Data parameters
    RANDOM_SEED = 42
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Model hyperparameters
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True
    USE_ATTENTION = True
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    GRAD_CLIP = 5.0
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

# ============================================================================
# DATASET CREATION - CHANGI AIRPORT QUERIES
# ============================================================================

def create_changi_dataset():
    """
    Create a synthetic dataset of Changi Airport passenger queries
    with realistic examples for each intent category
    """
    
    # Intent categories
    intents = {
        'flight_status': [
            "What's the status of flight SQ123?",
            "Has my flight to London departed?",
            "Is flight CA456 delayed?",
            "When does the Singapore Airlines flight to New York leave?",
            "What gate is my flight departing from?",
            "Has the gate changed for flight EK405?",
            "Is my flight on time?",
            "Flight status for SQ702 please",
            "Where can I check flight departures?",
            "Has boarding started for my flight?",
            "What time is boarding for gate B7?",
            "Is the flight to Sydney still departing at 9pm?",
            "My flight number is TR345, what's the gate?",
            "Gate information for Emirates flight",
            "Check flight arrival time",
            "When will flight land?",
            "Incoming flights from Tokyo",
            "Departure board location",
            "Latest gate assignment for my connection",
            "Flight tracking information",
            "Real-time flight updates",
            "Gate closure time for flight",
            "Final call for which gates?",
            "Next flight to Bangkok",
            "Connecting flight gate change",
            "Where to find flight information screens?",
            "My boarding pass says gate C12, is that correct?",
            "Flight cancellation status",
            "Alternative flights available?",
            "Delayed flight compensation",
        ],
        
        'baggage': [
            "My luggage is missing",
            "Lost baggage claim",
            "Where do I report damaged luggage?",
            "Baggage carousel for flight SQ890",
            "My suitcase didn't arrive",
            "How to file a baggage claim?",
            "Delayed baggage tracking",
            "Where is the baggage service counter?",
            "My bag was damaged during transit",
            "Oversized baggage collection",
            "Left item in checked luggage",
            "Baggage weight limits",
            "Extra baggage fees",
            "Where to collect baggage for international flights?",
            "Baggage claim area Terminal 3",
            "Lost and found office location",
            "Track my delayed luggage",
            "Baggage wrap services",
            "Fragile item handling",
            "Sports equipment baggage",
            "Duty-free purchases collection",
            "Storing luggage temporarily",
            "Baggage trolley locations",
            "Maximum baggage allowance",
            "Declare baggage contents",
            "Prohibited items in luggage",
            "Baggage storage facilities",
            "Damaged suitcase compensation",
            "Missing items from luggage",
            "Baggage delivery to hotel",
        ],
        
        'directions': [
            "How do I get to Terminal 2?",
            "Where is immigration?",
            "Directions to Jewel Changi",
            "Where are the lounges?",
            "How to get to the transfer area?",
            "Where is the prayer room?",
            "Location of duty-free shops",
            "How do I reach gate E23?",
            "Where can I find a pharmacy?",
            "Smoking area location",
            "Nearest restroom please",
            "Where to exchange currency?",
            "ATM locations in the terminal",
            "Food court directions",
            "Transit hotel location",
            "Shower facilities in terminal",
            "Children's play area",
            "Where is the SkyTrain?",
            "Walking distance to gates",
            "Fastest route to Terminal 1",
            "Jewel Waterfall viewing area",
            "Shopping district location",
            "Medical clinic in airport",
            "Baby care room location",
            "How to access viewing gallery?",
            "Butterfly Garden directions",
            "Canopy Park entrance",
            "Left luggage facility",
            "Meeting point locations",
            "Quiet zones in terminal",
        ],
        
        'special_assistance': [
            "I need wheelchair assistance",
            "Request for elderly passenger help",
            "Special needs support",
            "Traveling with an infant, need help",
            "Disability assistance services",
            "Request escort through immigration",
            "Medical assistance required",
            "Unaccompanied minor service",
            "Need help with mobility",
            "Assistance for blind passenger",
            "Electric cart service",
            "Porter service availability",
            "Priority boarding assistance",
            "Deaf passenger support",
            "Oxygen support on board",
            "Stretcher booking",
            "Guide dog procedures",
            "Pregnancy travel assistance",
            "Reduced mobility help",
            "Special meal requests",
            "Accessibility features in terminal",
            "Assistance through security",
            "Baby stroller availability",
            "Family lane for immigration",
            "Accessible restrooms location",
            "Sign language interpreter",
            "Cognitive assistance services",
            "First aid station",
            "Ambulance service",
            "Urgent medical attention needed",
        ],
        
        'transport': [
            "How do I get to the city?",
            "Taxi queue location",
            "MRT station directions",
            "Bus services to Orchard Road",
            "Car rental counters",
            "Grab pickup point",
            "Shuttle bus to hotels",
            "Private hire car booking",
            "Public transport options",
            "Cheapest way to downtown",
            "Airport transfer services",
            "Limousine service booking",
            "Where to board the airport shuttle?",
            "Train schedule to city",
            "Bus timings",
            "How long to city center?",
            "Transportation costs",
            "Night bus services",
            "Express train tickets",
            "Car park locations",
            "Ride-sharing pickup zones",
            "Coach services",
            "Ferry terminal transport",
            "Cruise terminal shuttle",
            "Hotel courtesy bus",
            "Budget transport options",
            "Fastest way to Marina Bay",
            "Transport with large luggage",
            "Accessible transport options",
            "Multi-modal journey planning",
        ],
        
        'security_customs': [
            "What items can I bring in hand luggage?",
            "Liquid restrictions for carry-on",
            "Customs declaration procedures",
            "Duty-free allowance limits",
            "Do I need to declare this?",
            "Security screening questions",
            "Battery restrictions on flights",
            "Food items through customs",
            "Medication declaration",
            "What's not allowed in cabin baggage?",
            "Customs clearance process",
            "Import duty calculator",
            "Tobacco allowance",
            "Alcohol limits",
            "Tax-free shopping limits",
            "Declare purchased goods",
            "Sharp objects in luggage?",
            "Electronic devices security",
            "Perfume volume limits",
            "Currency declaration threshold",
            "Restricted items list",
            "Security check procedures",
            "Quarantine regulations",
            "Agricultural products customs",
            "Designer goods declaration",
            "Jewelry declaration limits",
            "How long is security screening?",
            "Fast track security",
            "Diplomatic customs lane",
            "Customs inspection process",
        ]
    }
    
    # Create dataset with labels
    data = []
    for intent, queries in intents.items():
        for query in queries:
            data.append({'text': query, 'intent': intent})
    
    return pd.DataFrame(data)

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    @staticmethod
    def clean_text(text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s\?\!\.\,]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def tokenize(text):
        """Simple whitespace tokenization"""
        return text.split()

# ============================================================================
# VOCABULARY BUILDER
# ============================================================================

class Vocabulary:
    """Build and manage vocabulary from corpus"""
    
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        
    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        # Count word frequencies
        for text in texts:
            tokens = TextPreprocessor.tokenize(TextPreprocessor.clean_text(text))
            self.word_freq.update(tokens)
        
        # Add words that meet minimum frequency
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)} words")
        print(f"Words with freq >= {self.min_freq}: {len(self.word2idx) - 2}")
        
    def encode(self, text):
        """Convert text to sequence of indices"""
        tokens = TextPreprocessor.tokenize(TextPreprocessor.clean_text(text))
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
    
    def __len__(self):
        return len(self.word2idx)

# ============================================================================
# PYTORCH DATASET
# ============================================================================

class ChangiQueryDataset(Dataset):
    """PyTorch Dataset for Changi queries"""
    
    def __init__(self, texts, labels, vocab, label_encoder):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label_encoder = label_encoder
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded = self.vocab.encode(text)
        
        # Encode label
        label_idx = self.label_encoder[label]
        
        return {
            'text': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(label_idx, dtype=torch.long),
            'length': len(encoded)
        }

def collate_batch(batch):
    """Custom collate function for variable length sequences"""
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch])
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    
    return {
        'text': texts_padded,
        'label': labels,
        'length': lengths
    }

# ============================================================================
# ATTENTION MECHANISM
# ============================================================================

class AttentionLayer(nn.Module):
    """Attention mechanism for RNN outputs"""
    
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, rnn_outputs, lengths):
        """
        Args:
            rnn_outputs: (batch_size, seq_len, hidden_dim)
            lengths: (batch_size,)
        Returns:
            context: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        # Compute attention scores
        scores = self.attention(rnn_outputs).squeeze(-1)  # (batch_size, seq_len)
        
        # Create mask for padding
        batch_size, seq_len = scores.size()
        mask = torch.arange(seq_len, device=scores.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        
        # Apply mask (set padding positions to very negative value)
        scores = scores.masked_fill(~mask, -1e9)
        
        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), rnn_outputs).squeeze(1)  # (batch_size, hidden_dim)
        
        return context, attention_weights

# ============================================================================
# RNN MODEL WITH ATTENTION
# ============================================================================

class BiLSTMAttentionClassifier(nn.Module):
    """
    Bidirectional LSTM with Attention mechanism for text classification
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, 
                 num_layers=2, dropout=0.3, bidirectional=True, use_attention=True):
        super(BiLSTMAttentionClassifier, self).__init__()
        
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if use_attention:
            self.attention = AttentionLayer(lstm_output_dim)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, text, lengths):
        """
        Args:
            text: (batch_size, seq_len)
            lengths: (batch_size,)
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (batch_size, seq_len) if use_attention else None
        """
        # Embedding
        embedded = self.embedding(text)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pack sequence for efficient LSTM processing
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack sequence
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)  # (batch_size, seq_len, hidden_dim*2)
        
        attention_weights = None
        
        if self.use_attention:
            # Use attention mechanism
            context, attention_weights = self.attention(lstm_output, lengths)
        else:
            # Use last hidden state (from both directions if bidirectional)
            if self.bidirectional:
                # Concatenate final states from forward and backward
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                hidden = hidden[-1]
            context = hidden
        
        # Fully connected layers
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits, attention_weights

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        lengths = batch['length'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(text, lengths)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length'].to(device)
            
            # Forward pass
            logits, _ = model(text, lengths)
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()

def visualize_attention(text, attention_weights, vocab):
    """Visualize attention weights for a given text"""
    tokens = TextPreprocessor.tokenize(TextPreprocessor.clean_text(text))
    weights = attention_weights[:len(tokens)]
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(tokens)), weights)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.ylabel('Attention Weight')
    plt.title('Attention Weights Visualization')
    plt.tight_layout()
    return plt.gcf()

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("=" * 80)
    print("CHANGI VIRTUAL ASSIST TRIAGE - RNN TEXT CLASSIFIER")
    print("=" * 80)
    print(f"\nDevice: {Config.DEVICE}")
    
    # ========================================================================
    # 1. CREATE DATASET
    # ========================================================================
    print("\n[1/8] Creating Changi Airport dataset...")
    df = create_changi_dataset()
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Number of intents: {df['intent'].nunique()}")
    print(f"\nIntent distribution:")
    print(df['intent'].value_counts())
    
    # ========================================================================
    # 2. SPLIT DATA
    # ========================================================================
    print("\n[2/8] Splitting dataset...")
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=Config.RANDOM_SEED).reset_index(drop=True)
    
    # Calculate split sizes
    total_size = len(df)
    train_size = int(Config.TRAIN_SPLIT * total_size)
    val_size = int(Config.VAL_SPLIT * total_size)
    
    # Split data
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    print(f"  Train: {len(train_df)} samples ({Config.TRAIN_SPLIT:.0%})")
    print(f"  Val:   {len(val_df)} samples ({Config.VAL_SPLIT:.0%})")
    print(f"  Test:  {len(test_df)} samples ({Config.TEST_SPLIT:.0%})")
    
    # ========================================================================
    # 3. BUILD VOCABULARY
    # ========================================================================
    print("\n[3/8] Building vocabulary...")
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(train_df['text'].tolist())
    
    # ========================================================================
    # 4. CREATE LABEL ENCODER
    # ========================================================================
    print("\n[4/8] Creating label encoder...")
    all_labels = sorted(df['intent'].unique())
    label_encoder = {label: idx for idx, label in enumerate(all_labels)}
    idx_to_label = {idx: label for label, idx in label_encoder.items()}
    
    print(f"  Intent classes: {len(label_encoder)}")
    for label, idx in label_encoder.items():
        print(f"    {idx}: {label}")
    
    # ========================================================================
    # 5. CREATE DATASETS AND DATALOADERS
    # ========================================================================
    print("\n[5/8] Creating PyTorch datasets...")
    
    train_dataset = ChangiQueryDataset(
        train_df['text'].tolist(),
        train_df['intent'].tolist(),
        vocab,
        label_encoder
    )
    
    val_dataset = ChangiQueryDataset(
        val_df['text'].tolist(),
        val_df['intent'].tolist(),
        vocab,
        label_encoder
    )
    
    test_dataset = ChangiQueryDataset(
        test_df['text'].tolist(),
        test_df['intent'].tolist(),
        vocab,
        label_encoder
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    # ========================================================================
    # 6. CREATE MODEL
    # ========================================================================
    print("\n[6/8] Creating model...")
    
    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=len(label_encoder),
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        bidirectional=Config.BIDIRECTIONAL,
        use_attention=Config.USE_ATTENTION
    ).to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # 7. TRAINING
    # ========================================================================
    print("\n[7/8] Training model...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"\nTraining for {Config.NUM_EPOCHS} epochs...")
    print("-" * 80)
    
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            Config.DEVICE, Config.GRAD_CLIP
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, Config.DEVICE)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        print(f"Epoch {epoch+1:02d}/{Config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    # ========================================================================
    # 8. EVALUATION
    # ========================================================================
    print("\n[8/8] Evaluating on test set...")
    
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, Config.DEVICE
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    
    # Classification report
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(
        test_labels, test_preds,
        target_names=all_labels,
        digits=4
    ))
    
    # ========================================================================
    # 9. SAVE ARTIFACTS
    # ========================================================================
    print("\nSaving model and artifacts...")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': len(vocab),
            'embedding_dim': Config.EMBEDDING_DIM,
            'hidden_dim': Config.HIDDEN_DIM,
            'num_classes': len(label_encoder),
            'num_layers': Config.NUM_LAYERS,
            'dropout': Config.DROPOUT,
            'bidirectional': Config.BIDIRECTIONAL,
            'use_attention': Config.USE_ATTENTION
        },
        'vocab': vocab.word2idx,
        'label_encoder': label_encoder,
        'idx_to_label': idx_to_label
    }, '/home/claude/changi_rnn_model.pth')
    
    # Save training history
    with open('/home/claude/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # ========================================================================
    # 10. VISUALIZATIONS
    # ========================================================================
    print("\nGenerating visualizations...")
    
    # Training history plot
    fig1 = plot_training_history(history)
    fig1.savefig('/home/claude/training_history.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Confusion matrix
    fig2 = plot_confusion_matrix(test_labels, test_preds, all_labels)
    fig2.savefig('/home/claude/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: /home/claude/changi_rnn_model.pth")
    print(f"Training history: /home/claude/training_history.json")
    print(f"Plots saved to: /home/claude/")
    
    return model, vocab, label_encoder, idx_to_label

# ============================================================================
# INFERENCE CLASS
# ============================================================================

class ChangiAssistantInference:
    """Production-ready inference class for Changi Virtual Assist"""
    
    def __init__(self, model_path, device=None):
        """Load model and artifacts"""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create vocabulary
        self.vocab = Vocabulary()
        self.vocab.word2idx = checkpoint['vocab']
        self.vocab.idx2word = {v: k for k, v in self.vocab.word2idx.items()}
        
        # Label encoder
        self.label_encoder = checkpoint['label_encoder']
        self.idx_to_label = checkpoint['idx_to_label']
        
        # Create model
        config = checkpoint['config']
        self.model = BiLSTMAttentionClassifier(**config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def predict(self, text, return_attention=False):
        """
        Predict intent for a given text query
        
        Args:
            text: Input text query
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with prediction results
        """
        # Encode text
        encoded = self.vocab.encode(text)
        text_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)
        length_tensor = torch.tensor([len(encoded)], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits, attention_weights = self.model(text_tensor, length_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        result = {
            'query': text,
            'intent': self.idx_to_label[str(pred_idx)],
            'confidence': confidence,
            'all_probabilities': {
                self.idx_to_label[str(i)]: probs[0, i].item()
                for i in range(len(self.idx_to_label))
            }
        }
        
        if return_attention and attention_weights is not None:
            result['attention_weights'] = attention_weights[0].cpu().numpy()
        
        return result
    
    def predict_batch(self, texts):
        """Predict intents for multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

# ============================================================================
# INTERACTIVE DEMO
# ============================================================================

def interactive_demo(model_path='/home/claude/changi_rnn_model.pth'):
    """Interactive demo for testing the model"""
    
    print("\n" + "=" * 80)
    print("CHANGI VIRTUAL ASSIST - INTERACTIVE DEMO")
    print("=" * 80)
    
    # Load model
    assistant = ChangiAssistantInference(model_path)
    
    print("\nIntent Categories:")
    for idx, label in sorted(assistant.idx_to_label.items(), key=lambda x: int(x[0])):
        print(f"  {int(idx) + 1}. {label.replace('_', ' ').title()}")
    
    print("\n" + "-" * 80)
    print("Enter your queries (type 'quit' to exit)")
    print("-" * 80)
    
    # Example queries
    example_queries = [
        "Where is my flight gate?",
        "My luggage is missing",
        "How do I get to Jewel?",
        "I need wheelchair assistance",
        "How do I get to the city?",
        "What can I bring in my hand luggage?"
    ]
    
    print("\nExample queries you can try:")
    for i, query in enumerate(example_queries, 1):
        print(f"  {i}. {query}")
    
    print("\n" + "-" * 80)
    
    while True:
        query = input("\nYour query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using Changi Virtual Assist! ✈️")
            break
        
        if not query:
            continue
        
        # Predict
        result = assistant.predict(query, return_attention=True)
        
        # Display results
        print(f"\n{'─' * 80}")
        print(f"Intent: {result['intent'].replace('_', ' ').upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"{'─' * 80}")
        
        # Show top 3 predictions
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        print("\nTop 3 predictions:")
        for i, (intent, prob) in enumerate(sorted_probs, 1):
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"  {i}. {intent.replace('_', ' ').title():<25} {bar} {prob:.1%}")

if __name__ == "__main__":
    # Train model
    model, vocab, label_encoder, idx_to_label = main()
    
    # Run interactive demo
    print("\n" + "=" * 80)
    response = input("\nWould you like to try the interactive demo? (y/n): ").strip().lower()
    if response == 'y':
        interactive_demo()
