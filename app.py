import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Optional: add your nltk_data path here if needed, adjust accordingly
# nltk.data.path.append(r'D:\python\Naan mudhalvan\nltk_data')

def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'punkt']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if res == 'punkt' else f'corpora/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

# Download resources once at start (if missing)
download_nltk_resources()

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
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        st.error("NLTK punkt tokenizer data not found. Please check your NLTK installation.")
        st.stop()
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
