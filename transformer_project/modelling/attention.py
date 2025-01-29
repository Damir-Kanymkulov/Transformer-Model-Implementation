# Practical 2

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, mask_future=False):
        super(Attention, self).__init__()
        self.mask_future = mask_future

    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # Compute attention scores
        attention_scores /= (key.size(-1) ** 0.5)   # Scale scores by key dimension
        
        if self.mask_future:     # Apply masking for future positions
            future_mask = torch.triu(torch.ones_like(attention_scores), diagonal=1)
            attention_scores = attention_scores.masked_fill(future_mask.bool(), float('-inf'))

        if attention_mask is not None:  # apply attention mask
            while attention_mask.dim() < attention_scores.dim():
                attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.expand_as(attention_scores) # Match shapes
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1) # Normalize scores
        output = torch.matmul(attention_weights, value) # Weighted sum of values
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mask_future=False):
        super(MultiHeadAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mask_future = mask_future
        self.attention = Attention(mask_future=mask_future)
        # linear transformations for query, key, value, and output
        self.query_transform = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_transform = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_transform = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_transform = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, query, key, value, attention_mask=None):
        batch_size, query_seq_len, embed_dim = query.size() # Extract query dimensions
        _, key_seq_len, _ = key.size()  # Extract key dimensions
        # Transform query, key, and value
        query = self.query_transform(query)
        key = self.key_transform(key)
        value = self.value_transform(value)
        # Split query, key and value into multiple heads
        query = query.view(batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        output = self.attention(query, key, value, attention_mask)  # Scaled dot-product attention per head
        output = output.transpose(1, 2).contiguous().view(batch_size, query_seq_len, embed_dim) # Combine heads
        return self.output_transform(output)    # Final linear transformation