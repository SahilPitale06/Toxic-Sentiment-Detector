import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_tfidf():
    tfidf = pickle.load(open("tf_idf.pkl", "rb"))
    return tfidf

def load_model():
    nb_model = pickle.load(open("Toxicity_Sentiment_model.pkl", "rb"))
    return nb_model

def toxicity_detection(text):
    tfidf = load_tfidf()
    #transform the input to Tfidf vectors
    text_tfidf = tfidf.transform([text]).toarray()
    model = load_model()
    #predict the class of the input text
    prediction = model.predict(text_tfidf)

    #map the predicted class to a string
    class_name = "Toxic" if prediction == 1 else "Non-toxic"

    return class_name

st.header("Toxic Sentiment Detection App")
st.subheader("Enter the text below:")
text_input = st.text_input("Enter your text")


if text_input != "":
    if st.button("Analyze"):
        result = toxicity_detection(text_input)
        st.subheader("Detected result:")
        st.info(f"The result is: {result}.")
else:
    st.warning("Please enter some text to analyze.")


