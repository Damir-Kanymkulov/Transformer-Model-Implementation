# Practical 10

import torch
from evaluate import load
import sys, os, torch
from transformer_project.modelling.dataset import (get_cleaned_data, build_tokenizer, 
                                                   TranslationDataset)
from transformer_project.modelling.optimizer import initialize_optimizer
from transformer_project.modelling.lr_scheduler import CustomLRScheduler
from transformer_project.modelling.transformer_model import Transformer

# Greedy Decoding Function
def greedy_decode(model, tokenizer, src_sentence, max_len=50, device='cuda'):
    model.eval()
    with torch.no_grad():
        # Tokenize the source sentence
        src_ids = tokenizer.encode(src_sentence, return_tensors="pt").to(device)
        src_ids = src_ids.long()
        src_mask = torch.ones_like(src_ids).to(device)
        memory = model.embedding(src_ids)
        for layer in model.encoder_layers:
            memory = layer(memory, src_mask)
        # Initialize target sequence with BOS token
        tgt_ids = torch.tensor([[tokenizer.bos_token_id]], device=device).long()
        # Generate target sequence token by token
        for _ in range(max_len):
            tgt_mask = torch.ones_like(tgt_ids).to(device)
            output = model(src=memory, tgt=tgt_ids, src_mask=src_mask, tgt_mask=tgt_mask)
            # Get the token with the highest probability
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            # If EOS token is generated, stop the generation
            if next_token.item() == tokenizer.eos_token_id:
                break
            # Append the generated token to the target sequence
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
        return tokenizer.decode(tgt_ids.squeeze().tolist(), skip_special_tokens=True)

def generate_translations(model, tokenizer, test_dataset, max_len=50, device='cuda'):
    translations = []
    for example in test_dataset:
        src_sentence = example["source"]
        reference = example["target"]
        translation = greedy_decode(model, tokenizer, src_sentence, max_len, device)
        translations.append({"source": src_sentence, "reference": reference, "translation": translation})
    return translations

def evaluate_bleu(translations):
    bleu = load("bleu")
    references = [[t["reference"].split()] for t in translations]
    predictions = [t["translation"].split() for t in translations]
    results = bleu.compute(predictions=predictions, references=references)
    return results["bleu"]


# Project directory
project_directory = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(project_directory)
tokenizer_path = os.path.join(os.getcwd(), "tokenizer_data")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = get_cleaned_data(0.05) #subset percentage is 5%
tokenizer = build_tokenizer(data, tokenizer_path)   # build tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'})
dataset = TranslationDataset(data, tokenizer, max_length=64)

model = Transformer(vocab_size=len(tokenizer), d_model=64, n_heads=2, num_encoder_layers=4,
                    num_decoder_layers=4, dim_feedforward=64, dropout=0.0001, max_len=64).to(device)
optimizer = initialize_optimizer(model, lr=0.001)    #optimiser
scheduler = CustomLRScheduler(optimizer, d_model=64, warmup_steps=300)  #learning rate scheduler

# Generate translations for the dataset
translations = generate_translations(model, tokenizer, dataset, max_len=64, device=device)

# Evaluate BLEU score
bleu_score = evaluate_bleu(translations)
print(f"BLEU score: {bleu_score:.4f}")

# Display first 5 translations
for translation in translations[:5]:
    print(f"Source: {translation['source']}")
    print(f"Reference: {translation['reference']}")
    print(f"Translation: {translation['translation']}")