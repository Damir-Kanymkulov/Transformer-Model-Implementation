import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .attention import MultiHeadAttention
from .functional import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Masked self-attention layer
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=True)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.dropout_1 = nn.Dropout(dropout)
        
        # Position-wise feed-forward
        self.feature_transformation = PositionwiseFeedForward(input_dim, feature_dim, dropout)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, self_attention_mask=None):
        # Masked self-attention with residual connection and normalization
        attn_output = self.self_attention(x, x, x, self_attention_mask)
        x = self.layer_norm_1(x + self.dropout_1(attn_output))
        
        # Feed-forward network with residual connection and normalization
        ff_output = self.feature_transformation(x)
        x = self.layer_norm_2(x + self.dropout_2(ff_output))
        
        return x
    

class Decoder(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        d_model,
        n_heads, 
        num_decoder_layers, 
        dim_feedforward, 
        dropout, 
        max_len
    ):
        super(Decoder, self).__init__()
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Final linear layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        # Embedding + Positional Encoding
        tgt = self.dropout(self.positional_encoding(self.embedding(tgt)))

        # Decoder layers
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, tgt_mask)

        # Linear transformation to logits
        output = self.output_layer(output)

        return output

    def decode(self, tgt, tgt_mask=None):
        output = self.dropout(self.positional_encoding(self.embedding(tgt)))
        for layer in self.decoder_layers:
            output = layer(output, tgt_mask)
        logits = self.output_layer(output)
        return logits



