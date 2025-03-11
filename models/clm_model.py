import torch
import torch.nn as nn

class CLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, n_layers=3, dropout=0.2):
        super(CLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(weight.device),
                  weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(weight.device))
        return hidden