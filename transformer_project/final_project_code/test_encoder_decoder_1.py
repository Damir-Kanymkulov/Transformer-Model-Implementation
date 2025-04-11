import os
import sys

# Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
model_load_path = os.path.join(project_root, "final_project", "encoder_decoder_model_1.pth")
data_path = os.path.join(os.getcwd(), "test_data.pt")

import matplotlib.pyplot as plt
import numpy as np

import data_generation

import torch
from torch.utils.data import DataLoader

import data_generation
from functions import generate_examples, compute_mse
from modelling.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    vocab_size=127,
    d_model=64,
    n_heads=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=64,
    dropout=0.0,
    max_len=128,
).to('cuda' if torch.cuda.is_available() else 'cpu')


# Load the state_dict
model.load_state_dict(torch.load(model_load_path))
model.eval()
model.to(device)
print("Model loaded successfully.")


#####################################################
# Get Data from unifrom distribution
test_data = data_generation.generate_polynomial_dataset(num_samples=5000)

test_data = data_generation.EncoderDecoderDataset(test_data, data_generation.input_tokenizer, data_generation.output_tokenizer)

test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# Evaluate results
mse, valid, invalid = compute_mse(model, test_loader)

print(mse, valid, invalid)

data_iter = iter(test_loader)
batch = next(data_iter)

generate_examples(model, batch, 3)
#####################################################