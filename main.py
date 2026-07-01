# Mental Health Detection from Social Media Posts using Deep Learning

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

# =========================
# Load Dataset
# =========================

df = pd.read_csv("depression_dataset_reddit_cleaned.csv")

print(df.head())

# =========================
# Features & Target
# =========================

X_text = df["clean_text"].astype(str)

y = df["is_depression"]

# =========================
# Tokenization
# =========================

MAX_WORDS = 10000
MAX_LEN = 150

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_text)

sequences = tokenizer.texts_to_sequences(
    X_text
)

X = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =========================
# Bi-LSTM Model
# =========================

model = Sequential()

model.add(
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128,
        input_length=MAX_LEN
    )
)

model.add(
    Bidirectional(
        LSTM(64)
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

# =========================
# Compile
# =========================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# Train
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# =========================
# Evaluation
# =========================

pred_probs = model.predict(X_test)

predictions = (
    pred_probs > 0.5
).astype(int)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", accuracy)

print(
    classification_report(
        y_test,
        predictions
    )
)

# =========================
# Save Model
# =========================

model.save(
    "depression_bilstm.h5"
)

joblib.dump(
    tokenizer,
    "tokenizer.pkl"
)

print("Model Saved Successfully")