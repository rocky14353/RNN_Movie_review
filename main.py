import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import streamlit as st


# load the imdb data set 

words_index  = imdb.get_word_index()
reverse_index = {value:key for key,value in words_index.items()}

# load the model

model = load_model('simplernn_imdb.h5')


def decode_review(encoded):
    decoded_review = ' '.join([reverse_index.get(i-3, '?') for i in encoded])
    return decoded_review

def encode_review(text):
    words = text.lower().split()
    encoded_review = []
    for word in words:
        index =words_index.get(word,2)
        index = index+3
        if index > 10000:
            index = 2+3
        encoded_review.append(index)
    padded_review =sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review
                  

st.title("IMDB Movie Review Sentiment Analysis")

user_input = st.text_area("Enter your movie review here:")

if st.button("Predict Sentiment"):
    encoded_input = encode_review(user_input)
    prediction = model.predict(encoded_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    st.write(f"Predicted Sentiment: {sentiment} (Confidence: {prediction[0][0]:.2f})")
else:
    st.write("Please enter a movie review and click 'Predict Sentiment' to see the result.")