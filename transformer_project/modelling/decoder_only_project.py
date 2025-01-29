import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformer_project.modelling.functional import TransformerDecoderLayer
from transformer_project.modelling.optimizer import initialize_optimizer
from transformer_project.modelling.lr_scheduler import CustomLRScheduler

# Custom Dataset for Fibonacci Sequence
def generate_fibonacci_data(max_len=10, num_samples=1000):
    data = []
    for _ in range(num_samples):
        seq_len = np.random.randint(3, max_len + 1)
        fib = [0, 1]
        for i in range(2, seq_len):
            fib.append(fib[-1] + fib[-2])
        input_seq = fib[:-1]  # Input sequence
        target_seq = fib[1:]  # Target sequence (shifted)
        data.append((input_seq, target_seq))
    return data

class FibonacciDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return {
            'input': torch.tensor(input_seq, dtype=torch.float32),
            'target': torch.tensor(target_seq, dtype=torch.float32)
        }

# Custom collate function for padding sequences
def collate_fn(batch):
    inputs = [item['input'] for item in batch]
    targets = [item['target'] for item in batch]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0.0)
    return {
        'input': padded_inputs,
        'target': padded_targets
    }

# Decoder Model
class DecoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, feature_dim, num_heads, dropout):
        super(DecoderModel, self).__init__()
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(input_dim, num_heads, feature_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, encoder_output=None, encoder_attention_mask=None, self_attention_mask=None):
        for layer in self.decoder_layers:
            if encoder_output is not None:
                x = layer(x, encoder_output, encoder_attention_mask, self_attention_mask)
            else:
                x = layer(x, x, None, self_attention_mask)  # Self-attention only
        x = self.fc(x)
        return x

# Training Loop
def train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Training
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            inputs = batch['input'].unsqueeze(-1).expand(-1, -1, input_dim).to(device)
            targets = batch['target'].unsqueeze(-1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}"):
                inputs = batch['input'].unsqueeze(-1).expand(-1, -1, input_dim).to(device)
                targets = batch['target'].unsqueeze(-1).to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss/len(train_dataloader):.4f}, Val Loss = {val_loss/len(val_dataloader):.4f}")

# Test the model
def test_model(model, test_dataset, batch_size=16):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch['input'].unsqueeze(-1).expand(-1, -1, input_dim).to(device)
            targets = batch['target'].to(device)

            outputs = model(inputs).squeeze(-1)

            # Print first 5 samples
            for i in range(min(5, len(inputs))):  # Safeguard for smaller batches
                input_seq = inputs[i].squeeze().tolist()
                target_seq = targets[i].tolist()
                output_seq = outputs[i].tolist()

                print("Input Sequence: ", input_seq)
                print("Target Sequence: ", target_seq)
                print("Predicted Sequence: ", output_seq)
                print("-")


# Generate Data
data = generate_fibonacci_data()
train_data = data[:800]
test_data = data[800:]

train_dataset = FibonacciDataset(train_data)
test_dataset = FibonacciDataset(test_data)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Initialize Model and Training Components
input_dim = 64
hidden_dim = 32
output_dim = 1
num_layers = 2
feature_dim = 64
num_heads = 2
dropout = 0.1

model = DecoderModel(input_dim, hidden_dim, output_dim, num_layers, feature_dim, num_heads, dropout)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

optimizer = initialize_optimizer(model, lr=0.001)    #optimiser
scheduler = CustomLRScheduler(optimizer, d_model=64, warmup_steps=300)  #learning rate scheduler

# Train and Test
train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, epochs=20)
test_model(model, test_dataset)
