import sys
import os

# Add the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
model_save_path = os.path.join(project_root, "final_project", "encoder_decoder_model_2.pth")

from functions import train

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from modelling.training import TransformerLRScheduler, get_optimizer
from modelling.transformer import Transformer
import data_generation


train_data, val_data, _ = data_generation.create_train_val_test_split(50000, split_ratio=(0.7, 0.2, 0.1), save=True, distribution='normal')

train_data = data_generation.EncoderDecoderDataset(train_data, data_generation.input_tokenizer, data_generation.output_tokenizer)
val_data = data_generation.EncoderDecoderDataset(val_data, data_generation.input_tokenizer, data_generation.output_tokenizer)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

# Configuration Model
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-2
D_MODEL = 64
N_HEADS = 2
ENCODER_LAYERS = 2
DECODER_LAYERS = 2
DIM_FEEDFORWARD = 64
DROPOUT = 0.1
VOCAB_SIZE = 127
WARMUP_STEPS = 2000
EPOCHS = 25

# Initialize Model
model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    num_encoder_layers=ENCODER_LAYERS,
    num_decoder_layers=DECODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    max_len=128,
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Loss, Optimizer, Scheduler
loss_fn = nn.CrossEntropyLoss(ignore_index=2)
optimizer = get_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
scheduler = TransformerLRScheduler(optimizer, d_model=D_MODEL, warmup_steps=WARMUP_STEPS)

# Train the model
train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, EPOCHS, vocab_size=VOCAB_SIZE, model_save_path=model_save_path)