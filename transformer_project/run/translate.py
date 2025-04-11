# Autoregressive generation for practical 10
import os
import sys
import types

# Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
tokenizer_path = os.path.join(os.getcwd(), "tokenizer_data")
model_load_path = os.path.join(project_root, "modelling", "trained_transformer.pth")

from modelling.transformer import Transformer
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

# Data
from dataset import TranslationDataset
from dataset import get_cleaned_data
from datasets import load_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Tokenizer
if os.path.exists(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'})
else:
    print("The tokenizer needs to be built")

# Configuration Data
BATCH_SIZE = 16
MAX_LEN = 64
SUBSET_RATIO = 0.01

data = get_cleaned_data(SUBSET_RATIO, mode='test')

dataset = TranslationDataset(data, tokenizer, max_length=MAX_LEN)

dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

print('Dataloader built')


# Load model
model = Transformer(
    vocab_size=len(tokenizer),
    d_model=64,
    n_heads=2,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=64,
    dropout=0.00001,
    max_len=64,
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Load the state_dict into the model
model.load_state_dict(torch.load(model_load_path))
model.eval()
model.to(device)
print("Model loaded successfully.")


def greedy_decode(model, src, tokenizer, max_len=64, device="cpu"):
    # Encode source
    src = src.to(device, dtype=torch.long)
    src_mask = (src != tokenizer.pad_token_id).unsqueeze(1).to(device)

    memory = model.encode(src)  # Memory from the encoder stack
    
    # Initialize decoder input
    batch_size = src.size(0)
    tgt = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long).to(device)
    
    translations = [[] for _ in range(batch_size)]
    for _ in range(max_len):
        logits = model.decode(tgt, memory, src_mask=src_mask, tgt_mask=None)
        next_token_logits = logits[:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1).unsqueeze(1)
        
        for i, token in enumerate(next_tokens.squeeze(1).tolist()):
            if token == tokenizer.eos_token_id:
                continue
            translations[i].append(token)
        
        # Update target with the newly generated token
        tgt = torch.cat([tgt, next_tokens], dim=1)
        
        # Stop if all sentences have reached <eos>
        if all(tokenizer.eos_token_id in seq for seq in translations):
            break
    
    # Convert token IDs back to words
    translations = [tokenizer.decode(seq) for seq in translations]
    return translations


# Print exemplary translations
i = 0 # Number of batches to be printed
for batch in dataloader:
    if i <= 0:
        output = greedy_decode(model, batch['source_ids'], tokenizer)
        for i in range(len(batch)):
            print('######')
            print('Input:', tokenizer.decode(batch['source_ids'][i]))
            print('Labels: ', tokenizer.decode(batch['labels'][i]))
            print('Output:', output[i])
        i += 1
    else:
        break


def calculate_bleu_score(model, dataloader, tokenizer, max_len=64, device="cpu"):
    bleu_metric = load_metric("bleu", trust_remote_code=True)

    all_predictions = []
    all_references = []

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            output = greedy_decode(model, batch['source_ids'], tokenizer, max_len=max_len, device=device)

            for j in range(len(batch['source_ids'])):
                reference_text = tokenizer.decode(batch['labels'][j], skip_special_tokens=True)
                prediction_text = output[j]

                all_predictions.append(prediction_text.split())
                all_references.append([reference_text.split()])

    bleu_score = bleu_metric.compute(predictions=all_predictions, references=all_references)

    return bleu_score["bleu"]


bleu_score = calculate_bleu_score(model, dataloader, tokenizer, max_len=64, device=device)
print(f"BLEU Score: {bleu_score}")
