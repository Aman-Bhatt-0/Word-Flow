import numpy as np
import re

def simple_tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens

def text_to_indices(sentence, tokenizer, max_len=60):
    tokens = simple_tokenize(sentence)
    seq = tokenizer.texts_to_sequences([' '.join(tokens)])[0]

    seq = seq[-max_len:]
    padded = [0] * (max_len - len(seq)) + seq
    return np.array(padded)

def get_top_k_predictions(model, tokenizer, text, max_len=60, top_k=3):
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    input_indices = text_to_indices(text, tokenizer, max_len)
    input_indices = np.array([input_indices])

    preds = model.predict(input_indices, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]

    top_words = [index_to_word.get(i, "<unk>") for i in top_indices]
    return top_words
