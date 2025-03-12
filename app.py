from flask import Flask, render_template, request
import pickle
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os



print(os.path.exists("Myapp/spam_model.pkl"))  # Should return True if the file is present


# Download stopwords (ensure it's available)
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Load trained model & vectorizer
model = pickle.load(open('Myapp/spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    cleaned_msg = clean_text(message)
    message_vectorized = vectorizer.transform([cleaned_msg]).toarray()
    prediction = model.predict(message_vectorized)[0]

    result = "🚨 Spam Message Detected!" if prediction == 1 else "✅ Not a Spam Message."
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
