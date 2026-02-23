from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)

# Load model and tokenizer
model = load_model('next_word_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def predict_words(seed_text, next_words):
    max_sequence_len = model.input_shape[1]
    output_text = seed_text
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted, axis=-1)[0]
        output_text += " " + tokenizer.index_word[predicted_index]
    
    return output_text

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        seed_text = request.form['seed_text']
        next_words = int(request.form['next_words'])
        prediction = predict_words(seed_text, next_words)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)