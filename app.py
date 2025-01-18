from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import string
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
import sys
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add file handler for logging
handler = RotatingFileHandler('spam_detector.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)

class SpamDetector:
    def __init__(self):
        self.ps = None
        self.stop_words = None
        self.tfidf = None
        self.model = None
        self.initialize_nltk()
        self.load_models()

    def initialize_nltk(self):
        """Initialize NLTK resources and components"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.ps = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK resources initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NLTK resources: {str(e)}")
            raise

    def load_models(self):
        """Load the pre-trained model and vectorizer"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
            vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                raise FileNotFoundError("Model or vectorizer file not found")
            
            with open(vectorizer_path, 'rb') as f:
                self.tfidf = pickle.load(f)
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise

    def transform_text(self, text):
        """Transform input text with preprocessing steps"""
        try:
            # Input validation
            if not isinstance(text, str):
                raise ValueError("Input must be a string")
            
            # Text preprocessing pipeline
            text = text.lower()
            text = nltk.word_tokenize(text)
            text = [word for word in text if word.isalnum()]
            text = [word for word in text if word not in self.stop_words and 
                   word not in string.punctuation]
            text = [self.ps.stem(word) for word in text]
            
            return " ".join(text)
        except Exception as e:
            logger.error(f"Text transformation error: {str(e)}")
            raise

    def predict_spam(self, message):
        """Predict if a message is spam"""
        try:
            transformed_sms = self.transform_text(message)
            vector_input = self.tfidf.transform([transformed_sms])
            result = self.model.predict(vector_input)[0]
            probability = self.model.predict_proba(vector_input)[0]
            
            return {
                'is_spam': bool(result),
                'confidence': float(max(probability)),
                'processed_text': transformed_sms
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Configure app
app.config.update(
    MAX_CONTENT_LENGTH=1024 * 1024,  # 1MB max-limit
    TEMPLATES_AUTO_RELOAD=True
)

# Initialize spam detector
try:
    spam_detector = SpamDetector()
except Exception as e:
    logger.critical(f"Failed to initialize SpamDetector: {str(e)}")
    sys.exit(1)

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return render_template('error.html', error="Internal Server Error"), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method != 'POST':
            return jsonify({'error': 'Method not allowed'}), 405

        # Get input message
        input_sms = request.form.get('message', '')
        
        if not input_sms:
            return jsonify({'error': 'No message provided'}), 400
        
        if len(input_sms) > 5000:  # Limit message length
            return jsonify({'error': 'Message too long'}), 400

        # Log prediction request
        logger.info(f"Processing prediction request - Length: {len(input_sms)}")
        
        # Get prediction
        result = spam_detector.predict_spam(input_sms)
        
        # Log prediction result
        logger.info(f"Prediction complete - Result: {'Spam' if result['is_spam'] else 'Not Spam'} "
                   f"(Confidence: {result['confidence']:.2f})")

        # Return result based on request type
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'prediction': 'Spam' if result['is_spam'] else 'Not Spam',
                'confidence': f"{result['confidence']:.2%}",
                'processed_text': result['processed_text']
            })
        else:
            return render_template('index.html', 
                                result='Spam' if result['is_spam'] else 'Not Spam',
                                confidence=f"{result['confidence']:.2%}",
                                processed_text=result['processed_text'])

    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal Server Error"), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Perform basic health checks
        test_prediction = spam_detector.predict_spam("Test message")
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': spam_detector.model is not None,
            'vectorizer_loaded': spam_detector.tfidf is not None
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)