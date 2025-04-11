# Training a translation model for practical 9

import sys
import os

# Add the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from modelling.training import TransformerLRScheduler, get_optimizer
from modelling.transformer import Transformer
from transformers import GPT2Tokenizer
from tqdm import tqdm

# Data
from dataset import TranslationDataset
from dataset import get_cleaned_data
from dataset import build_tokenizer

# Define paths
model_save_path = os.path.join(project_root, "modelling", "trained_transformer.pth")
tokenizer_path = os.path.join(os.getcwd(), "tokenizer_data")

# Configuration Data
BATCH_SIZE = 16
MAX_LEN = 64
SUBSET_RATIO = 0.05

# Load Tokenizer
if os.path.exists(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'})
else:
    print('No tokenizer available')

print("Collecting data...")

data = get_cleaned_data(SUBSET_RATIO)

dataset = TranslationDataset(data, tokenizer, max_length=MAX_LEN)

# Train-validation-test split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Configuration Model
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
D_MODEL = 64
N_HEADS = 2
ENCODER_LAYERS = 4
DECODER_LAYERS = 4
DIM_FEEDFORWARD = 64
DROPOUT = 0.00001
VOCAB_SIZE = len(tokenizer)
WARMUP_STEPS = 500
EPOCHS = 5

# Initialize Model
model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    num_encoder_layers=ENCODER_LAYERS,
    num_decoder_layers=DECODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    max_len=MAX_LEN,
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Loss, Optimizer, Scheduler
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = get_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
scheduler = TransformerLRScheduler(optimizer, d_model=D_MODEL, warmup_steps=WARMUP_STEPS)

# Training Loop
def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Training
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            src = batch['source_ids'].to(device)
            src_mask = batch['source_mask'].to(device)
            tgt = batch['target_ids'].to(device)
            tgt_mask = batch['target_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            #print(input_ids)
            #print(input_ids.shape)
            output = model(src, tgt, src_mask, tgt_mask)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), labels.view(-1))
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}"):
                src = batch['source_ids'].to(device)
                src_mask = batch['source_mask'].to(device)
                tgt = batch['target_ids'].to(device)
                tgt_mask = batch['target_mask'].to(device)
                labels = batch['labels'].to(device)

                output = model(src, tgt, src_mask, tgt_mask)
                loss = loss_fn(output.view(-1, VOCAB_SIZE), labels.view(-1))
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss/len(train_dataloader)}, Val Loss = {val_loss/len(val_dataloader)}")

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Train the model
train(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, EPOCHS)
