import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_lstm.h5')
import os

print("Current Directory:", os.getcwd())
print("Tokenizer Path:", os.path.abspath("tokenizer.pkl"))

with open('tokenizer.pkl','rb') as handle:
    tokenizer = pickle.load(handle)

print(type(tokenizer))


def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list],maxlen = max_sequence_len-1,padding = 'pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_idx = np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index == predicted_idx:
            return word
    return None

st.title("Next Word Prediction LSTM")
input_text = st.text_input("Enter the Sequence of words")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next Word prediction: {next_word}")

