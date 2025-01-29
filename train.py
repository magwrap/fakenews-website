import re
import torch
from torch.nn.functional import pad
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


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
    model = NewsLSTM(vocab_size, embedding_dim,
                     hidden_dim, output_dim, pad_idx)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


MODEL_PATH = "/content/drive/MyDrive/private-projects/fake-news-detection/models/news_lstm.pth"
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = 2
pad_idx = vocab["<pad>"]
max_length = 100

model = load_model(MODEL_PATH, vocab_size, embedding_dim,
                   hidden_dim, output_dim, pad_idx)


print("Enter news text to classify (type 'exit' to quit):")
while True:
    user_input = input("News: ")
    if user_input.lower() == "exit":
        print("Exiting...")
        break
    processed_text = preprocess_text(user_input, vocab, max_length, pad_idx)
    with torch.no_grad():
        prediction = model(processed_text)
        predicted_label = torch.argmax(prediction, dim=1).item()
        label_name = "Real News" if predicted_label == 0 else "Fake News"
        print(f"Prediction: {label_name}")
