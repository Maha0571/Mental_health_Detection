import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 150

model = load_model(
    "models/depression_bilstm.h5"
)

tokenizer = joblib.load(
    "models/tokenizer.pkl"
)

text = input("Enter Text : ")

seq = tokenizer.texts_to_sequences(
    [text]
)

pad = pad_sequences(
    seq,
    maxlen=MAX_LEN,
    padding="post"
)

prediction = model.predict(pad)

if prediction[0][0] > 0.5:
    print("Depression Detected")
else:
    print("No Depression Detected")