# Practical 4

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_len):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len  # maximum sequence length
        # register positional encoding
        self.register_buffer("positional_encoding", self._generate_positional_encoding())

    def _generate_positional_encoding(self):
        encoding = torch.zeros(self.seq_len, self.embedding_dim)  # positional encoding matrix
        position_indices = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        frequency_terms = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-math.log(10000.0) / self.embedding_dim))
        encoding[:, 0::2] = torch.sin(position_indices * frequency_terms)  # sinusoidal for even indices
        encoding[:, 1::2] = torch.cos(position_indices * frequency_terms)  # cosine for odd indices
        return encoding

    def forward(self, input_tensor):
        batch_size, seq_len, embedding_dim = input_tensor.size()
        if seq_len > self.seq_len:  # check input sequence length
            raise ValueError(f"Input sequence length ({seq_len}) exceeds maximum ({self.seq_len})")
        pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0)  # Adjust positional encoding to input length
        return input_tensor + pos_encoding