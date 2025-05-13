import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download NLTK data (first-time only)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load Model & Vectorizer (pretrained)
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    st.error("Model files not found! Train a model first.")
    st.stop()

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)  # Remove URLs, mentions, hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
    return ' '.join(tokens)

# Streamlit UI
st.title("Social Media Sentiment Analyzer 🚀")
st.write("Enter a tweet, and I'll predict if it's Positive or Negative!")

user_input = st.text_input("Type a tweet here:")

if user_input:
    # Clean & Predict
    cleaned_input = clean_text(user_input)
    input_vec = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vec)[0]
    
    # Display Result
    if prediction == 1:
        st.success("Positive Sentiment 😊")
    else:
        st.error("Negative Sentiment 😞")