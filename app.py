import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load files
model = pickle.load(open("sentiment model.pkl", "rb"))

vectorizer = pickle.load(open("tfidf vectorizer.pkl", "rb"))



nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

st.title(" Flipkart Review Sentiment Analyzer")
st.write("Enter a product review and click Predict to see if it's Positive or Negative")

user_input = st.text_area(" Enter Review Here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        clean = clean_text(user_input)
        vector = vectorizer.transform([clean])
        result = model.predict(vector)[0]

        if result == "Positive":
            st.success(" Sentiment: POSITIVE")
        else:
            st.error(" Sentiment: NEGATIVE")
