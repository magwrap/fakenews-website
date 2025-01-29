import torch
import torch.nn as nn


class NewsLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):

        super(NewsLSTM, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):
        embedded = self.embedding(text)

        lstm_output, (hidden, cell) = self.lstm(embedded)

        final_hidden_state = hidden[-1]

        output = self.fc(final_hidden_state)

        output = self.softmax(output)

        return output
