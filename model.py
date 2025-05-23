import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import kagglehub

path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")

print("Path to dataset files:", path)

df = pd.read_csv(f"{path}/twitter_training.csv", header=None, names=['index', 'borderlands', 'label', 'text'])
print(df.shape)
print(df.head)

# 2. Handle missing values
print(f"Missing values before: {df['text'].isnull().sum()}")
df = df.dropna(subset=['text'])  # or use fillna('')
print(f"Missing values after: {df['text'].isnull().sum()}")

# 3. Now vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text']) 

# Training a Simple Model
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# Saving Model & Vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved as model.pkl & vectorizer.pkl!")