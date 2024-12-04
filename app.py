from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import string

app = Flask(__name__)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer and stopwords list (one-time)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    # Text preprocessing pipeline
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text into words

    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stop_words and word not in string.punctuation]

    # Stem the words
    text = [ps.stem(word) for word in text]

    return " ".join(text)  # Return transformed text

def predict_spam(message):
    # Preprocess the input message and predict using the model
    transformed_sms = transform_text(message)
    vector_input = tfidf.transform([transformed_sms])  # Transform the message into a vector using TF-IDF
    result = model.predict(vector_input)[0]  # Predict the result using the trained model
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_sms = request.form['message']
        result = predict_spam(input_sms)
        
        # Display prediction result as "Spam" or "Not Spam"
        prediction_text = 'Spam' if result == 1 else 'Not Spam'
        
        return render_template('index.html', result=prediction_text)  # Pass result to the template

if __name__ == '__main__':
    # Load the pre-trained TF-IDF vectorizer and model from disk
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=4000, debug=True)
