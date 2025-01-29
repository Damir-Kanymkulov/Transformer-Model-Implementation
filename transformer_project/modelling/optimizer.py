# Practical 8

import torch.optim as optim

def get_param_groups(model):
    no_decay = ['bias', 'layer_norm.weight']
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0     # No weight decay for biases and LayerNorm weights
        }
    ]
    return param_groups

def initialize_optimizer(model, lr=0.001):
    return optim.AdamW(get_param_groups(model), lr=lr)
