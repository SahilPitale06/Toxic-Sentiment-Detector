from fastapi import FastAPI
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI()

# Load the Tfidf and model
tfidf = pickle.load(open("tf_idf.pkl", "rb"))
nb_model = pickle.load(open("Toxicity_Sentiment_model.pkl", "rb"))

# Endpoint
@app.post("/predict")
async def predict(text: str):
    #transform the input to Tfidf vectors
    text_tfidf = tfidf.transform([text]).toarray()

    #predict the class of the input text
    prediction = nb_model.predict(text_tfidf)[0]

    #map the predicted class to a string
    class_name = "Toxic" if prediction == 1 else "Non-toxic"

    #return the prediction in a JSON response
    return{
        "text":text,
        "class":class_name
    }

# To run the application, use the following command in your terminal:
# uvicorn api:app --reload
# Go to http://127.0.0.1:8000/docs to access the Swagger UI.
# Click Try it out under /predict, enter text, and submit the request to get a prediction.