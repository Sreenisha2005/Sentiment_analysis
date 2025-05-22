import os
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set nltk data path explicitly to local nltk_data folder in your project
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.mkdir(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'punkt']
    for res in resources:
        try:
            if res == 'punkt':
                nltk.data.find('tokenizers/punkt')
            else:
                nltk.data.find(f'corpora/{res}')
        except LookupError:
            nltk.download(res, download_dir=nltk_data_dir, quiet=True)

download_nltk_resources()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load your model & vectorizer as before
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
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        st.error("NLTK punkt tokenizer data not found. Please check your NLTK installation.")
        st.stop()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

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
