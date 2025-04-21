import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load your trained model and other necessary objects
# Replace 'best_model.pkl' and 'label_encoder.pkl' with your actual file names
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Preprocessing function
def preprocess_reviews(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    ]
    cleaned_review = ' '.join(cleaned_tokens)
    return cleaned_review

# Sentiment prediction function
def predict_sentiment(text, model, label_encoder):
    prediction = model.predict([text])[0]
    return label_encoder.inverse_transform([prediction])[0]

# Streamlit UI
st.title("Movie Review Sentiment Analysis")

user_input = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    if user_input:
        cleaned_review = preprocess_reviews(user_input)
        prediction = predict_sentiment(cleaned_review, best_model, label_encoder)
        st.write(f"Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter a movie review.")