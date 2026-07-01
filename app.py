import streamlit as st
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 150
model = load_model("depression_bilstm.h5")
tokenizer = joblib.load("tokenizer.pkl")



st.title("Mental Health Detection")

text = st.text_area("Enter your text")

if st.button("Predict"):

    seq = tokenizer.texts_to_sequences([text])

    pad = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post"
    )

    prediction = model.predict(pad)

    if prediction[0][0] > 0.5:
        st.error("Depression Detected")
    else:
        st.success("No Depression Detected")