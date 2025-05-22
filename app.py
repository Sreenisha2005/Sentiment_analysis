import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once at start (will skip if already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Cache stopwords and lemmatizer globally
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load Model & Vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"Model files not found or failed to load! {e}")
    st.stop()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit UI
st.title("Social Media Sentiment Analyzer")
st.write("Enter a tweet, and I'll predict if it's Positive or Negative!")

user_input = st.text_input("Type a tweet here:")

if user_input:
    cleaned_input = clean_text(user_input)
    input_vec = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vec)[0]

    if prediction == 1:
        st.success("Positive Sentiment 😊")
    else:
        st.error("Negative Sentiment 😞")

