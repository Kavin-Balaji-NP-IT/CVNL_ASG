"""
Hybrid Emotion Detection RNN using Both Kaggle and Synthetic Datasets
Enhanced accuracy with combined real-world and synthetic data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import json
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplo