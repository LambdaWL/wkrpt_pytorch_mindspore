import math

import torch
import torch.nn as nn

from config import Config


class CustomGRUCell(nn.Module):
    """ For a more fair comparison with the Mindspore implementation, we do not use
    nn.GRUCell that has a significant performance gain (using nn.GRUCell is ~3 times
    faster per epoch). """
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        num_chunks = 3
        stdv = 1.0 / math.sqrt(hidden_size)

        self.weight_ih = nn.Parameter(torch.Tensor(num_chunks, hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_chunks, hidden_size, input_size))
        self.bias_ih = nn.Parameter(torch.Tensor(num_chunks, hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(num_chunks, hidden_size))

        nn.init.uniform_(self.weight_ih, -stdv, stdv)
        nn.init.uniform_(self.weight_hh, -stdv, stdv)
        nn.init.uniform_(self.bias_ih, -stdv, stdv)
        nn.init.uniform_(self.bias_hh, -stdv, stdv)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hx):
        r = self.sigmoid(
            torch.matmul(input, self.weight_ih[0]) + self.bias_ih[0] +
            torch.matmul(hx, self.weight_hh[0]) + self.bias_hh[0]
        )
        z = self.sigmoid(
            torch.matmul(input, self.weight_ih[1]) + self.bias_ih[1] + 
            torch.matmul(hx, self.weight_hh[1]) + self.bias_hh[1]
        )
        n = self.tanh(
            torch.matmul(input, self.weight_ih[2]) + self.bias_ih[2] + 
            r * (torch.matmul(hx, self.weight_hh[2]) + self.bias_hh[2])
        )
        return (1 - z) * n + z * hx


class TorchModel(nn.Module):
    def __init__(self, config: Config):
        super(TorchModel, self).__init__()

        self.pool_feature = nn.Linear(config.feature_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.output1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output2 = nn.Linear(config.hidden_dim, config.num_class)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, features):
        """
        Args:
            features: Tensor of shape (B, qlen, D)
        """
        qlen = features.size()[1]
        inputs = self.relu(self.pool_feature(features))  # (B, qlen, H)

        # Handle the first utterance separately
        prev_h = torch.zeros_like(inputs[:, 0, :])
        prev_h = self.dropout(self.gru(inputs[:, 0, :], prev_h))
        hidden_states = prev_h.unsqueeze(1)  # (B, 1, H)
        for t in range(1, qlen):
            prev_h = self.dropout(self.gru(inputs[:, t, :], prev_h))
            hidden_states = torch.cat((hidden_states, prev_h.unsqueeze(1)), 1)  # (B, t+1, H)
        
        outputs = self.relu(self.output1(hidden_states))  # (B, qlen, H)
        outputs = self.output2(self.dropout(outputs))  # (B, qlen, C)
        return outputs
