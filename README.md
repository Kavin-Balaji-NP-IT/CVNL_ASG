# CVNL Assignment - Deep Learning for Aviation

This repository contains two main deep learning projects focused on aviation applications: Aircraft Image Classification using CNN and Intent Classification for Virtual Assistant using RNN.

## Project Overview

### CNN Part: Aircraft Image Classification
**Team Members:** Kerwin, Gerick

A Convolutional Neural Network (CNN) implementation for classifying different types of aircraft from images. This project demonstrates computer vision techniques applied to aviation domain.

**Key Features:**
- Image preprocessing and data augmentation
- CNN architecture design and optimization
- Multi-class aircraft classification
- Model evaluation and performance metrics

**Files:**
- `CNN_AircraftClassification.ipynb` - Main CNN implementation notebook
- `CNN_AircraftClassification).ipynb` - Additional CNN experiments

### RNN Part: Intent Classification for Virtual Assistant
**Team Members:** Kavin, Jaylen

A Recurrent Neural Network (RNN) implementation for intent classification in a Changi Airport virtual assistant. The system can understand user queries and classify them into appropriate intent categories.

**Key Features:**
- ATIS (Airline Travel Information System) dataset integration
- Bidirectional LSTM architecture
- Text preprocessing and vocabulary management
- Intent classification with high accuracy (96%+)
- Support for aviation-specific queries

**Files:**
- `RNN_IntentExamples/RNN_Jaylen.ipynb` - Jaylen's RNN implementation
- `RNN_IntentExamples/RNN_Kavin.ipynb` - Kavin's RNN 

## Technical Details

### CNN Architecture
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification
- Dropout for regularization

### RNN Architecture
- Bidirectional LSTM layers
- Embedding layer for text representation
- Dropout for regularization
- Linear classification layer
- Model parameters: 2.4M+ trainable parameters

### Dataset Information
- **CNN**: Aircraft image dataset with multiple aircraft types
- **RNN**: ATIS dataset with 26 intent categories including:
  - Flight information queries
  - Airport navigation
  - Ground services
  - Airline information
  - Abbreviation explanations

## Performance Metrics

### RNN Model Performance
- Test Accuracy: 96.08%
- Validation Accuracy: 98.39%
- Training Accuracy: 99.96%
- Vocabulary Size: 604 words
- Intent Categories: 26

### Model Capabilities
The RNN model can handle queries such as:
- "What flights are available to Bangkok?"
- "Where is gate C9?"
- "How to get to city center?"
- "What does SQ mean?"
- "Flight times to Mumbai"

## Installation and Usage

### Requirements
```
torch
numpy
scikit-learn
datasets
matplotlib
seaborn
flask (for web interface)
flask-cors
```

### Running the CNN Model
Open and run the Jupyter notebooks:
- `CNN_AircraftClassification.ipynb`

## Project Structure
```
CVNL_ASG/
├── CNN_AircraftClassification/
│   └── CNN_AircraftClassification.ipynb
├── RNN_IntentExamples/
│   ├── RNN_Jaylen.ipynb
│   └── RNN_Kavin.ipynb
├── README.md
└── other supporting files
```

## Future Enhancements
- Integration of CNN and RNN models for multimodal applications
- Web interface for real-time predictions
- Extended dataset for better coverage
- Model optimization for deployment
- Additional aviation-specific features

## Contributors
- **Kerwin**: CNN implementation and aircraft classification
- **Gerick**: CNN architecture design and optimization
- **Kavin**: RNN implementation and Intent classification
- **Jaylen**: RNN implementation and intent classification

## License
This project is for educational purposes as part of the CVNL assignment.
