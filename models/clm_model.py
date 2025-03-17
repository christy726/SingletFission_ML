# File: models/clm_model.py
import torch
import torch.nn as nn

class CLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        predictions = self.fc(output)
        return predictions, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(next(self.parameters()).device),
                torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(next(self.parameters()).device))