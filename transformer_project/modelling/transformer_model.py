# Practical 7

import torch
import torch.nn as nn
from transformer_project.modelling.attention import MultiHeadAttention
from transformer_project.modelling.functional import BaseTransformerLayer, TransformerDecoderLayer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, dropout, max_len):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        # Encoder
        self.encoder_layers = nn.ModuleList([
            BaseTransformerLayer(d_model, n_heads, dim_feedforward, dropout)
            for i in range(num_encoder_layers)
        ])
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for i in range(num_decoder_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Embedding and positional encoding for source and target
        src_emb = self.positional_encoding(self.embedding(src))
        tgt_emb = self.positional_encoding(self.embedding(tgt))
        # Pass through encoder layers
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)
        # Pass through decoder layers
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, src_mask, tgt_mask)
        # Final linear transformation to vocabulary size
        logits = self.output_layer(output)
        return logits

#Test of the implementation
if __name__ == "__main__":
    model = Transformer(vocab_size=10000, d_model=64, n_heads=2, num_encoder_layers=4,
                        num_decoder_layers=4, dim_feedforward=64, dropout=0.0001, max_len=64)
    src = torch.randint(0, 10000, (32, 64))  # source input
    tgt = torch.randint(0, 10000, (32, 64))  # target input
    output = model(src, tgt)
    print(output.shape)  # Should be (batch_size, tgt_len, vocab_size)