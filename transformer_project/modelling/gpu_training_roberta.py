# Practical 11

import torch
from transformers import (RobertaForSequenceClassification, RobertaTokenizer,
                          get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader
from datasets import load_dataset
import tqdm
from sklearn.metrics import f1_score

# Configuration
MODEL_NAME_OR_PATH = 'roberta-base'
MAX_INPUT_LENGTH = 256
BATCH_SIZE = 16
TRAINING_EPOCHS = 2
WEIGHT_DECAY = 0.01
LEARNING_RATE = 2e-5
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION_TRAINING = True if torch.cuda.is_available() else False

model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH)
model = model.to(DEVICE)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
qnli_dataset = load_dataset('glue', 'qnli')

def convert_example_to_features(example: dict) -> dict:
    features = tokenizer(example['question'], example['sentence'], max_length=MAX_INPUT_LENGTH,
                         padding='max_length', truncation='longest_first')

    features['labels'] = example['label']

    return features

def collate(batch: list) -> dict:
    features = {
        'input_ids': torch.tensor([itm['input_ids'] for itm in batch]),
        'attention_mask': torch.tensor([itm['attention_mask'] for itm in batch]),
        'labels': torch.tensor([itm['labels'] for itm in batch]),
    }

    return features

# Apply tokenization to the datasets
# train_dataset = qnli_dataset['train'].map(convert_example_to_features)
train_dataset = qnli_dataset['train'].select(range(100)).map(convert_example_to_features) #small subset with .select(range(100)) 
# validation_dataset = qnli_dataset['validation'].map(convert_example_to_features)
validation_dataset = qnli_dataset['validation'].select(range(100)).map(convert_example_to_features) #small subset with .select(range(100))

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, collate_fn = collate)

# Exercise 2: Update the initialisation to incorporate GPU training
# Specify the weight decay for each parameter
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": WEIGHT_DECAY,
        "lr": LEARNING_RATE
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": LEARNING_RATE
    },
]

# Initialise the optimizer
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
# Initialise the learning rate scheduler
num_training_steps = len(train_dataloader) * TRAINING_EPOCHS
num_warmup_steps = WARMUP_PROPORTION * num_training_steps
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                               num_training_steps=num_training_steps)


# Exercise 2: Update the functions to incorporate GPU training
# Exercise 3: Update the functions to incorporate mixed precision training
def training_step(batch):
    batch = {key: value.to(DEVICE) for key, value in batch.items()}
    loss = model(**batch).loss
    loss.backward()
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    # Update step
    optimizer.step()
    lr_scheduler.step()
    model.zero_grad()
    return loss

def evaluate(dataloader):
    # Set model to evaluation mode
    model.eval()
    predictions = list()
    labels = list()
    for batch in tqdm.tqdm(dataloader, desc="Eval"):
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        # Forward pass data
        with torch.no_grad():
            logits = model(**batch).logits.detach().cpu()
        pred = logits.argmax(-1)
        predictions.append(pred.cpu().reshape(-1))
        labels.append(batch['labels'].cpu().reshape(-1))
    # Reset model to training mode
    model.zero_grad()
    model.train()
    # Compute the F1 Score
    predictions = torch.concat(predictions, 0)
    labels = torch.concat(labels, 0)
    # Convert to NumPy arrays for sklearn
    predictions = predictions.numpy()
    labels = labels.numpy()
    f1 = f1_score(labels, predictions)
    return f1

# Move model to GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)
# Reinitialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                               num_training_steps=num_training_steps)

# Prepare model for training
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