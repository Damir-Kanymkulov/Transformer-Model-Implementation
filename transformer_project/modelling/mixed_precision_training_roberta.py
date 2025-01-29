# Practical 11

import torch
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import load_dataset
import tqdm
from torch.amp import autocast, GradScaler
from transformer_project.modelling.gpu_training_roberta import (convert_example_to_features, collate, 
                                            optimizer_grouped_parameters, evaluate)

# Configuration
MODEL_NAME_OR_PATH = 'roberta-base'
BATCH_SIZE = 16
TRAINING_EPOCHS = 2
LEARNING_RATE = 2e-5
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION_TRAINING = True if torch.cuda.is_available() else False

scaler = GradScaler()   # Initialize GradScaler
qnli_dataset = load_dataset('glue', 'qnli')

# Apply tokenization to the datasets
# train_dataset = qnli_dataset['train'].map(convert_example_to_features)
train_dataset = qnli_dataset['train'].select(range(100)).map(convert_example_to_features) #small subset with .select(range(100)) 
# validation_dataset = qnli_dataset['validation'].map(convert_example_to_features)
validation_dataset = qnli_dataset['validation'].select(range(100)).map(convert_example_to_features) #small subset with .select(range(100))

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, collate_fn = collate)

def training_step(batch):
    # Move batch to device
    batch = {key: value.to(DEVICE) for key, value in batch.items()}
    # Enable mixed precision
    with autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(**batch)
        loss = outputs.loss
    # Scale the loss for mixed precision
    scaler.scale(loss).backward()
    # Clip gradients
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    # Update weights using scaled gradients
    scaler.step(optimizer)
    scaler.update()
    lr_scheduler.step()
    model.zero_grad()
    return loss

model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH)
model = model.to(DEVICE)    # Move model to GPU
# Initialise the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# Initialise the learning rate scheduler
num_training_steps = len(train_dataloader) * TRAINING_EPOCHS
num_warmup_steps = WARMUP_PROPORTION * num_training_steps
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                               num_training_steps=num_training_steps)

# Training Loop
model.train()
model.zero_grad()
optimizer.zero_grad()
for e in range(TRAINING_EPOCHS):
    iterator = tqdm.tqdm(train_dataloader, desc=f"Epoch {e+1}/{TRAINING_EPOCHS}")
    # Perform an epoch of training
    for batch in iterator:
        loss = training_step(batch)
        iterator.set_postfix({'Loss': loss.item()})
    # Evaluate the model and report F1 Score
    f1 = evaluate(validation_dataloader)
    print(f"Validation F1 Score: {f1}")