"""
Flask Web Application for Changi Airport Sentiment Analysis
Using RNN Model for Sentiment Classification
"""

from flask import Flask, render_template, request, jsonify
from sentiment_analysis_rnn import SentimentAnalyzer

app = Flask(__name__)

# Initialize the sentiment analyzer
print("Initializing Sentiment Analysis Web Application...")
analyzer = SentimentAnalyzer()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not analyzer.is_loaded():
            return jsonify({'error': 'RNN model not loaded'}), 500
        
        result = analyzer.predict(text)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_info = analyzer.get_model_info()
    return jsonify({
        'status': 'healthy',
        'model_type': 'RNN',
        'model_loaded': analyzer.is_loaded(),
        'test_accuracy': model_info.get('test_accuracy', 'N/A') if model_info else 'N/A',
        'vocab_size': model_info.get('vocab_size', 'N/A') if model_info else 'N/A'
    })


if __name__ == '__main__':
    print("="*60)
    print("CHANGI AIRPORT SENTIMENT ANALYSIS WEB APP")
    print("="*60)
    print("Using Sentiment Analysis RNN Model (Kaggle Dataset)")
    print(f"Model loaded: {analyzer.is_loaded()}")
    
    if analyzer.is_loaded():
        model_info = analyzer.get_model_info()
        print(f"Model classes: {model_info['label_names']}")
        print(f"Test accuracy: {model_info.get('test_accuracy', 'N/A'):.2%}")
        print(f"Vocabulary size: {model_info.get('vocab_size', 'N/A'):,}")
    
    print("Starting Flask server...")
    print("Access the app at: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)