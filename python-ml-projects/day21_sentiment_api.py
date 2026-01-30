"""
Day 21: Sentiment Analysis REST API
Production-ready API for sentiment analysis
Deploy to Render/Railway for live access
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

print("=" * 60)
print("SENTIMENT ANALYSIS API - DAY 21")
print("=" * 60)

# ============================================
# LOAD MODEL
# ============================================

print("\nLoading sentiment analysis model...")

try:
    # Load the model and vectorizer from Day 18
    MODEL_PATH = 'models/sentiment_analyzer.pkl'
    VECTORIZER_PATH = 'models/sentiment_vectorizer.pkl'
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Model loaded successfully!")
    else:
        print("❌ Model files not found!")
        print(f"   Looking for: {MODEL_PATH}")
        print(f"   Looking for: {VECTORIZER_PATH}")
        print("\n⚠️  Running without model (demo mode)")
        model = None
        vectorizer = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None

# ============================================
# HELPER FUNCTIONS
# ============================================

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    if model and vectorizer:
        # Use ML model
        text_vectorized = vectorizer.transform([text])
        sentiment = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = probabilities.max()
        
        # Get probability for each class
        classes = model.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
    else:
        # Fallback: Simple rule-based
        text_lower = text.lower()
        positive_words = ['good', 'great', 'excellent', 'love', 'perfect', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'poor']
        
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        if pos_count > neg_count:
            sentiment = 'positive'
            confidence = 0.7
        elif neg_count > pos_count:
            sentiment = 'negative'
            confidence = 0.7
        else:
            sentiment = 'neutral'
            confidence = 0.6
        
        prob_dict = {
            'positive': confidence if sentiment == 'positive' else (1 - confidence) / 2,
            'negative': confidence if sentiment == 'negative' else (1 - confidence) / 2,
            'neutral': confidence if sentiment == 'neutral' else (1 - confidence) / 2
        }
    
    return {
        'sentiment': sentiment,
        'confidence': float(confidence),
        'probabilities': prob_dict
    }

# ============================================
# API ROUTES
# ============================================

@app.route('/', methods=['GET'])
def home():
    """API home page with documentation"""
    return jsonify({
        'api_name': 'Sentiment Analysis API',
        'version': '1.0',
        'author': 'Emdad Hossain',
        'description': 'REST API for analyzing text sentiment (positive/negative/neutral)',
        'endpoints': {
            'GET /': 'This documentation',
            'GET /health': 'Health check',
            'POST /analyze': 'Analyze sentiment of text',
            'POST /batch': 'Analyze multiple texts'
        },
        'usage_example': {
            'endpoint': '/analyze',
            'method': 'POST',
            'body': {
                'text': 'This product is amazing!'
            },
            'response': {
                'sentiment': 'positive',
                'confidence': 0.95,
                'probabilities': {
                    'positive': 0.95,
                    'negative': 0.03,
                    'neutral': 0.02
                }
            }
        },
        'portfolio': 'https://emdadhossain25.github.io/emdad-portfolio/',
        'github': 'https://github.com/emdadhossain25'
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'version': '1.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze sentiment of single text
    
    Request body:
    {
        "text": "Your text here"
    }
    
    Response:
    {
        "text": "Your text here",
        "sentiment": "positive/negative/neutral",
        "confidence": 0.95,
        "probabilities": {...},
        "timestamp": "2026-01-28T10:30:00"
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text',
                'usage': {
                    'text': 'Your text to analyze'
                }
            }), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400
        
        # Analyze sentiment
        result = analyze_sentiment(text)
        
        # Return result
        return jsonify({
            'text': text,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/batch', methods=['POST'])
def batch_analyze():
    """
    Analyze sentiment of multiple texts
    
    Request body:
    {
        "texts": ["text 1", "text 2", "text 3"]
    }
    
    Response:
    {
        "results": [
            {"text": "text 1", "sentiment": "positive", ...},
            {"text": "text 2", "sentiment": "negative", ...}
        ],
        "count": 2
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing required field: texts (array)',
                'usage': {
                    'texts': ['text 1', 'text 2', 'text 3']
                }
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'error': 'texts must be an array'
            }), 400
        
        if len(texts) > 100:
            return jsonify({
                'error': 'Maximum 100 texts per batch request'
            }), 400
        
        # Analyze all texts
        results = []
        for text in texts:
            if text and text.strip():
                result = analyze_sentiment(text)
                results.append({
                    'text': text,
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
        
        return jsonify({
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def stats():
    """API statistics (could be expanded with database)"""
    return jsonify({
        'api_version': '1.0',
        'model_type': 'Logistic Regression / Random Forest',
        'training_accuracy': '96.7%',
        'features': 'TF-IDF Vectorization',
        'supported_sentiments': ['positive', 'negative', 'neutral'],
        'max_batch_size': 100,
        'author': 'Emdad Hossain',
        'portfolio': 'https://emdadhossain25.github.io/emdad-portfolio/'
    })

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/analyze', '/batch', '/health', '/stats']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# ============================================
# RUN SERVER
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "=" * 60)
    print("🚀 SENTIMENT ANALYSIS API STARTING")
    print("=" * 60)
    print(f"\n✅ Server running on: http://localhost:{port}")
    print(f"\nAvailable endpoints:")
    print(f"  • GET  /          - API documentation")
    print(f"  • GET  /health    - Health check")
    print(f"  • POST /analyze   - Analyze single text")
    print(f"  • POST /batch     - Analyze multiple texts")
    print(f"  • GET  /stats     - API statistics")
    print(f"\nTest with:")
    print(f"""  curl -X POST http://localhost:{port}/analyze \\
       -H "Content-Type: application/json" \\
       -d '{{"text": "This product is amazing!"}}'""")
    print("\n" + "=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=True)
