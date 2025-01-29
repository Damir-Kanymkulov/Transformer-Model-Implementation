# Practical 5, 6
 
import torch
import torch.nn as nn
from transformer_project.modelling.attention import MultiHeadAttention

"""""
Equation for position-wise Feed-Forward layer:
FFN(x) = max(0, xW1 + b1)*W2 + b2

The equation involves a linear transformation followed by a ReLU activation and a second linear 
transformation. This layer independently processes each position in the sequence while capturing 
complex relationships among the input features.
"""""

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # First linear transformation
        self.relu = nn.ReLU()   # ReLU activation        
        self.dropout = nn.Dropout(dropout)  # Dropout
        self.linear2 = nn.Linear(d_ff, d_model) # Second linear transformation

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

"""""
Equation for normalization layer:
LayerNorm(x) = (x-µ)/sqrt(sigma^2 + epsilon) * gamma + beta

This layer normalizes the inputs across the features for each data point. Layer normalization 
stabilizes the training process. The learnable parameters gamma and beta allow the model to 
flexibly undo normalization if needed.
"""""

class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.1):
        super(BaseTransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(input_dim, num_heads)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.feature_transformation = PositionwiseFeedForward(input_dim, feature_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, x, x, mask)    # self-attention layer
        x = self.layer_norm_1(x + self.dropout(attn_output))    # residual connection with normalization
        ff_output = self.feature_transformation(x)  # feed-forward layer
        x = self.layer_norm_2(x + self.dropout(ff_output))  # residual connection with normalization
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=True)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.dropout_1 = nn.Dropout(dropout)        
        self.encoder_attention = MultiHeadAttention(input_dim, num_heads, mask_future=False)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.dropout_2 = nn.Dropout(dropout)
        self.feature_transformation = PositionwiseFeedForward(input_dim, feature_dim, dropout)
        self.layer_norm_3 = nn.LayerNorm(input_dim)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, encoder_attention_mask=None, self_attention_mask=None):
        attn_output = self.self_attention(x, x, x, self_attention_mask) # Masked self-attention
        x = self.layer_norm_1(x + self.dropout_1(attn_output))  # residual connection with normalization
        # Encoder-decoder cross-attention layer
        cross_attn_output = self.encoder_attention(x, encoder_output, encoder_output, encoder_attention_mask)
        x = self.layer_norm_2(x + self.dropout_2(cross_attn_output))    # residual connection with normalization
        ff_output = self.feature_transformation(x)  #Position-wise feed-forward network
        x = self.layer_norm_3(x + self.dropout_3(ff_output))    #residual connection with normalization
        return x