"""
Fine-tune the fast model for better accuracy
Focus on clear positive/negative cases while maintaining mixed sentiment handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from collections import Counter
import re
import pickle
import json
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds
torch.manual_see