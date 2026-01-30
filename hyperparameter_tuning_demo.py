#!/usr/bin/env python3
"""
Hyperparameter Tuning Demonstration
Shows systematic approach to optimizing RNN model performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from itertools import product
import json
from datetime import datetime

# Import our model components
from rnn_model_development_demo import (
    TextPreprocessor, IntentDataset, BiLSTMClassifier, 
    collate_fn, train_epoch, evaluate
)

def hyperparameter_search():
    """
    Systematic hyperparameter search for RNN optimization
    """
    print("🔍 Hyperparameter Tuning Demonstration")
    print("=" * 50)
    
    # Define hyperparameter search space
    hyperparams = {
        'embed_dim': [64, 128, 256],
        'hidden_dim': [128, 256, 512],
        'num_layers': [1, 2, 3],
        'dropout': [0.2, 0.3, 0.4, 0.5],
        'lea