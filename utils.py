"""Utilities for the model training and predictions"""

import re

import torch
from torch.nn.functional import pad

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


from NewsLSTM import NewsLSTM

nltk.download('punkt')
nltk.download('stopwords')

# TODO: the language should be in config
STOPWORDS = set(stopwords.words('english'))


def yield_tokens(text_list):
    for text in text_list:
        yield clean_text(text)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS]
    return tokens


def preprocess_text(text, vocab, max_length, pad_idx):
    tokens = clean_text(text)
    sequence = [vocab[token] if token in vocab else vocab["<unk>"]
                for token in tokens]
    tensor = torch.tensor(sequence, dtype=torch.long)
    padded_tensor = pad(tensor, (0, max_length - len(tensor)), value=pad_idx)
    return padded_tensor.unsqueeze(0)


def load_model(model_path, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
    print(model_path, vocab_size, embedding_dim,
          hidden_dim, output_dim, pad_idx)
    model = NewsLSTM(vocab_size, embedding_dim,
                     hidden_dim, output_dim, pad_idx)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def format_prediction(prediction: int, text: str, model: str):
    print("probabilty: ", prediction)

    response: str = f"""Article: {text[:50]}... → Prediction: {
        'FAKE NEWS' if prediction == 1 else 'REAL NEWS'}\n Model: {model}"""
    print(response)

    return response
