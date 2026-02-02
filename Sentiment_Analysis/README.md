# Changi Airport Sentiment Analysis

This folder contains the complete RNN-based sentiment analysis solution for Changi Airport passenger feedback.

## Files Overview

### Core Model Files
- `comprehensive_sentiment_rnn.py` - Training script for the RNN model
- `comprehensive_sentiment_model.pth` - Trained PyTorch model weights
- `comprehensive_sentiment_preprocessor.pkl` - Text preprocessor
- `comprehensive_sentiment_model_info.json` - Model configuration and metadata

### Web Application
- `sentiment_web_app.py` - Flask web application for sentiment analysis
- `templates/index.html` - Web interface for sentiment analysis
- `static/style.css` - Styling for the web interface

### Testing
- `test_comprehensive_model.py` - Script to test the trained model

## Model Performance

- **Test Accuracy**: 100%
- **Architecture**: Bidirectional LSTM with Attention Mechanism
- **Parameters**: 2.46M parameters
- **Vocabulary**: 157 words
- **Classes**: Very Negative, Negative, Neutral, Positive, Very Positive

## Key Features

✅ **Mixed Sentiment Handling**: Correctly identifies sentences with both positive and negative elements as Neutral
✅ **Real-world Accuracy**: Trained on realistic airline feedback patterns
✅ **Attention Mechanism**: Focuses on important words for better classification
✅ **Web Interface**: Beautiful UI for real-time sentiment analysis

## Usage

### Training the Model
```bash
python comprehensive_sentiment_rnn.py
```

### Testing the Model
```bash
python test_comprehensive_model.py
```

### Running the Web Application
```bash
python sentiment_web_app.py
```
Then visit: http://localhost:5000

## Example Classifications

- "Nice service but food was lacking taste" → **Neutral** (95.7%)
- "Excellent service and friendly staff" → **Very Positive** (50.5%)
- "Terrible experience, worst airline ever" → **Very Negative** (93.5%)
- "Average experience, nothing special" → **Neutral** (94.0%)

## Technical Details

- **Framework**: PyTorch
- **Architecture**: Bidirectional LSTM + Multi-head Attention + Batch Normalization
- **Training Data**: 2,400 samples of realistic airline feedback
- **Max Sequence Length**: 40 tokens
- **Embedding Dimension**: 128
- **Hidden Dimension**: 256
- **Dropout**: 0.3 for regularization