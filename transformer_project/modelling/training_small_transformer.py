# Practical 9

import sys, os, torch
import torch.nn as nn
from tqdm import tqdm
from transformer_project.modelling.dataset import (get_cleaned_data, build_tokenizer, 
                                                   TranslationDataset)
from transformer_project.modelling.optimizer import initialize_optimizer
from transformer_project.modelling.lr_scheduler import CustomLRScheduler
from transformer_project.modelling.transformer_model import Transformer

# Project directory
project_directory = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(project_directory)
model_save_path = os.path.join(project_directory, "modelling", "trained_transformer.pth")
tokenizer_path = os.path.join(os.getcwd(), "tokenizer_data")

data = get_cleaned_data(0.05) #subset percentage is 5%
tokenizer = build_tokenizer(data, tokenizer_path)   # build tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'})
dataset = TranslationDataset(data, tokenizer, max_length=64)

# train/val/test ratios = 70% / 20% / 10%
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

model = Transformer(vocab_size=len(tokenizer), d_model=64, n_heads=2, num_encoder_layers=4,
                    num_decoder_layers=4, dim_feedforward=64, dropout=0.0001, max_len=64)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  #loss function
optimizer = initialize_optimizer(model, lr=0.001)    #optimiser
scheduler = CustomLRScheduler(optimizer, d_model=64, warmup_steps=300)  #learning rate scheduler

# training loop
def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, num_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        # Training
        for batch in tqdm(train_dataloader, desc=f"Train_epoch {epoch+1}"):
            optimizer.zero_grad()
            src, src_mask = batch['source_ids'].to(device), batch['source_mask'].to(device)
            tgt, tgt_mask = batch['target_ids'].to(device), batch['target_mask'].to(device)
            labels = batch['labels'].to(device)
            # Forward pass
            output = model(src, tgt, src_mask, tgt_mask)
            loss = loss_fn(output.view(-1, len(tokenizer)), labels.view(-1))
            train_loss += loss.item()
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Val_epoch {epoch+1}"):
                src, src_mask = batch['source_ids'].to(device), batch['source_mask'].to(device)
                tgt, tgt_mask = batch['target_ids'].to(device), batch['target_mask'].to(device)
                labels = batch['labels'].to(device)
                output = model(src, tgt, src_mask, tgt_mask)
                loss = loss_fn(output.view(-1, len(tokenizer)), labels.view(-1))
                val_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss/len(train_dataloader)}, Val Loss = {val_loss/len(val_dataloader)}")
    torch.save(model.state_dict(), model_save_path) # Save the model

train(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, num_epochs=5)