import torch
import torch.nn as nn


# DKTplus model definition


class DKTplus(nn.Module):
    def __init__(self, num_c, emb_size, hidden_size, lambda_r, lambda_w1, lambda_w2, dropout=0.3):
        super(DKTplus, self).__init__()

        # Model Hyper-parameters
        self.model_name = "dkt+"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.lambda_r, self.lambda_w1, self.lambda_w2 = lambda_r, lambda_w1, lambda_w2

        # Model Embedding Layers
        self.interaction_emb = nn.Embedding((self.num_c + 1) * 2, self.emb_size)
        self.time_emb = nn.Linear(1, self.emb_size)

        # Model Structural layers
        self.lstm_layers = nn.LSTM(self.emb_size * 2, self.hidden_size, num_layers=2, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_c),
        )

    def forward(self, q, r, t):

        x = q + (self.num_c + 1) * r

        x_emb = self.interaction_emb(x)
        t_emb = torch.sigmoid(self.time_emb(t.unsqueeze(-1)))

        combined_emb = torch.cat([x_emb, t_emb], dim=-1)

        y, _ = self.lstm_layers(combined_emb)
        y = self.output_layers(self.dropout_layer(self.relu(y)))

        return y
