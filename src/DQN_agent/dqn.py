import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, num_c, hidden_size):
        super().__init__()

        self.num_c = num_c
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(self.num_c, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_c),
        )

    def forward(self, x):
        return self.net(x)
