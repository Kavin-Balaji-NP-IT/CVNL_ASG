# User Guide: Changi Airport Emotion RNN System

## Quick Start Guide

### 1. System Requirements
- Python 3.8+
- PyTorch 1.9+
- Flask 2.0+
- 4GB RAM minimum
- Internet connection (for dataset download)

### 2. Installation

#### Step 1: Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install flask pandas numpy scikit-learn
pip install matplotlib seaborn kagglehub
```

#### Step 2: Download Project Files
```bash
# Ensure you have all required files:
# - sentiment_analysis_rnn.py
# - sentiment_web_app.py  
# - sentiment_analysis_model.pth
# - sentiment_analysis_preprocessor.pkl
# - sentiment_analysis_model_info.json
```

### 3. Running the Web Application

#### Start the Server
```bash
cd Sentiment_Analysis
python sentiment_web_app.py
```

#### Expected Output
```
============================================================
CHANGI AIRPORT SENTIMENT ANALYSIS WEB APP
============================================================
Using Sentiment Analysis RNN Model (Kaggle Dataset)
Model loaded: True
Model classes: ['Negative', 'Neutral', 'Positive']
Test accuracy: 77.80%
Vocabulary size: 3,000
Starting Flask server...
Access the app at: http://localhost:5000
============================================================
```

### 4. Using the Web Interface

#### Step 1: Open Browser
Navigate to: `http://localhost:5000`

#### Step 2: Enter Text
- Type or paste passenger feedback in the text area
- Examples that work well:
  - "Amazing service, absolutely love it!"
  - "Worst service I have ever experienced"
  - "Standard check-in process"

#### Step 3: Analyze
- Click "Analyze Sentiment" button
- Wait for results (typically <1 second)

#### Step 4: Interpret Results
- **Sentiment**: Negative/Neutral/Positive
- **Confidence**: Percentage confidence in prediction
- **Detailed Analysis**: Probability breakdown for all classes

### 5. API Usage

#### Basic API Call
```python
import requests
import json

# Analyze sentiment via API
response = requests.post(
    'http://localhost:5000/analyze',
    headers={'Content-Type': 'application/json'},
    data=json.dumps({'text': 'The airport is beautiful!'})
)

result = response.json()
print(f"Sentiment: {result['predicted_sentiment']}")
print(f"Confidence: {result['confidence']:.1%}")
```

#### Batch Processing
```python
texts = [
    "Great service!",
    "Terrible experience",
    "Average waiting time"
]

for text in texts:
    response = requests.post(
        'http://localhost:5000/analyze',
        json={'text': text}
    )
    result = response.json()
    print(f"'{text}' -> {result['predicted_sentiment']} ({result['confidence']:.1%})")
```

### 6. Model Performance Guidelines

#### What Works Well
- **Strong Emotions**: Clear positive/negative language
- **Direct Statements**: "Love this", "Hate that"
- **Service-Related**: Airport/airline specific feedback

#### What May Be Challenging
- **Sarcasm**: "Great, another delay!" 
- **Mixed Sentiment**: "Good food but bad service"
- **Subtle Emotions**: "The service was okay"

#### Accuracy Expectations
- **Overall Accuracy**: 77.8%
- **Negative Detection**: 87% precision
- **Positive Detection**: 68% precision  
- **Neutral Detection**: 58% precision

### 7. Troubleshooting

#### Common Issues

**Issue**: "Model not loaded" error
**Solution**: 
- Ensure all model files are in the same directory
- Check file permissions
- Verify PyTorch installation

**Issue**: Web app won't start
**Solution**:
- Check if port 5000 is available
- Try different port: `app.run(port=5001)`
- Verify Flask installation

**Issue**: Slow predictions
**Solution**:
- Model runs on CPU by default
- For faster inference, use GPU if available
- Consider reducing batch size

**Issue**: Unexpected predictions
**Solution**:
- Check input text preprocessing
- Review model limitations in technical report
- Consider retraining with domain-specific data

### 8. Advanced Usage

#### Custom Model Training
```bash
# Train new model with your data
python rnn_model_development.py
```

#### Model Evaluation
```python
# Load and test model
from sentiment_analysis_rnn import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.predict("Your text here")
print(result)
```

#### Integration with Other Systems
```python
# Example: Process CSV file
import pandas as pd

df = pd.read_csv('feedback.csv')
results = []

for text in df['feedback']:
    response = requests.post('http://localhost:5000/analyze', 
                           json={'text': text})
    results.append(response.json())

df['sentiment'] = [r['predicted_sentiment'] for r in results]
df['confidence'] = [r['confidence'] for r in results]
```

### 9. Support and Contact

For technical issues or questions:
- Review the technical report: `EMOTION_RNN_TECHNICAL_REPORT.md`
- Check model performance metrics
- Refer to code comments in source files

### 10. License and Attribution

This system uses the Kaggle Twitter Airline Sentiment dataset:
- **Source**: crowdflower/twitter-airline-sentiment
- **License**: Creative Commons
- **Citation**: Available on Kaggle dataset page

---

**Happy Analyzing! 🛫✈️🛬**