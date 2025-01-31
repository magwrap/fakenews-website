"""Prediction"""
import os
import toml
import pandas as pd

import torch
from torchtext.vocab import build_vocab_from_iterator

from utils import load_model, yield_tokens, preprocess_text

# get the config
with open('config.toml', 'r', encoding="utf8") as f:
    config = toml.load(f)

print("config", config)
config = config["config"]

cwd = os.getcwd()
data_dir = config["data_dir"]

print(os.listdir(data_dir))

fake_news_csv = data_dir + config["fake_news_csv"]
true_news_csv = data_dir + config["true_news_csv"]

fake_news_df = pd.read_csv(fake_news_csv)
true_news_df = pd.read_csv(true_news_csv)

train_dataset = pd.concat([fake_news_df, true_news_df])
cleaned_train_dataset = train_dataset.drop(columns=["subject", "date"])

print(cleaned_train_dataset.head())
print(cleaned_train_dataset.describe())

true_news_df["label"] = 1
fake_news_df["label"] = 0

vocab = build_vocab_from_iterator(yield_tokens(
    cleaned_train_dataset['text']), specials=["<pad>", "<unk>"])

vocab.set_default_index(vocab["<unk>"])

MODEL_PATH = config["model_path"]

print(len(vocab))

VOCAB_SIZE = len(vocab)
# TODO: remove this line
VOCAB_SIZE = 38368
EMBEDDING_DIM = config["embedding_dim"]
HIDDEN_DIM = config["hidden_dim"]
OUTPUT_DIM = config["output_dim"]
MAX_LENGTH = config["max_length"]

pad_idx = vocab["<pad>"]

model = load_model(MODEL_PATH, VOCAB_SIZE, EMBEDDING_DIM,
                   HIDDEN_DIM, OUTPUT_DIM, pad_idx)


def make_prediction(user_input: str):
    processed_text = preprocess_text(user_input, vocab, MAX_LENGTH, pad_idx)
    print(f"""processing the text: {user_input}\nvocab:{

          vocab}\nmax length:{MAX_LENGTH}\npad index:{pad_idx}""")
    with torch.no_grad():
        print("Feeding the text to the model: ", processed_text)
        prediction = model(processed_text)
        predicted_label = torch.argmax(prediction, dim=1).item()
        label_name = "Real News" if predicted_label == 0 else "Fake News"

        probabilities = torch.nn.functional.softmax(prediction, dim=1)
        first_pred = probabilities[0, 0].item()
        second_pred = probabilities[0, 1].item()
        print(user_input, probabilities)
        return f"{label_name}\nProbabilty real:{first_pred}\nProbability fake:{second_pred}"
