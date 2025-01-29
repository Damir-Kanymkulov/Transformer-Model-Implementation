# Practical 4

import os, torch, re
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
from datasets import load_dataset

# Cleaning function
def clean_text(text, min_length=5, max_length=64):
    # allowed characters
    whitelist = "abcdefghijklmnopqrstuvwxyzÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-UTF8 characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)    #Remove URLs and HTML tags
    text = ''.join(c for c in text if c in whitelist)
    text = text.lower() # To lowercase
    # Tokenize and check length
    tokens = text.split()
    if min_length <= len(tokens) <= max_length:
        return ' '.join(tokens)
    else:
        return None

def clean_dataset(dataset, mode='train'):
    cleaned_data = []
    for example in dataset[mode]:
        source = clean_text(example['translation']['de'])
        target = clean_text(example['translation']['en'])
        if source and target and len(source.split()) / len(target.split()) <= 2.0:
            cleaned_data.append({'source': source, 'target': target})
    return cleaned_data

def get_cleaned_data(subset_percentage=None, mode='train'):
    dataset = load_dataset("wmt17", "de-en")
    dataset = dataset.shuffle(seed=1)
    num_samples = int(len(dataset[mode]) * subset_percentage)
    num_samples = 50 ########################################################################
    dataset[mode] = dataset[mode].select(range(num_samples))
    cleaned_data = clean_dataset(dataset, mode)
    return cleaned_data

save_directory = os.path.join(os.getcwd(), "tokenizer_data")

def build_tokenizer(cleaned_data, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator([example['source'] for example in cleaned_data] + 
                                      [example['target'] for example in cleaned_data], 
                                      vocab_size=50000, min_frequency=2)
        tokenizer.save_model(save_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(save_directory)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'})
    return tokenizer

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Special token IDs
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    # Add special tokens to tokenized sequences
    def add_special_tokens(self, token_ids):
        new_max_length = self.max_length - 2    # Adjust max length for special tokens
        if len(token_ids) > new_max_length:
            token_ids = token_ids[:new_max_length]  # Truncate if necessary
        token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]   # Add BOS and EOS tokens
        padding_length = self.max_length - len(token_ids)   # Calculate padding length
        if padding_length > 0:
            token_ids = token_ids + [self.pad_token_id] * padding_length    # Add padding tokens
        return token_ids

    def build_attention_mask(self, token_ids):
        return [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]

    def __getitem__(self, idx):
        source = self.data[idx]["source"]
        target = self.data[idx]["target"]
        # Encode source and target texts
        encoded_source = self.tokenizer.encode(source, add_special_tokens=False)
        encoded_target = self.tokenizer.encode(target, add_special_tokens=False)
        # Add special tokens to source and target and create attention masks
        source_ids = self.add_special_tokens(encoded_source)
        source_mask = self.build_attention_mask(source_ids)
        target_input_ids = self.add_special_tokens(encoded_target)
        target_mask = self.build_attention_mask(target_input_ids)
        # Shift the labels
        target_labels = target_input_ids[1:] + [self.pad_token_id]
        return {
            "source": source, "target": target,   # Original source and target text
            "source_ids": torch.tensor(source_ids), # Tokenized source IDs
            "source_mask": torch.tensor(source_mask),   # Attention mask for source
            "target_ids": torch.tensor(target_input_ids),   # Tokenized target IDs
            "target_mask": torch.tensor(target_mask),   # Attention mask for target
            "labels": torch.tensor(target_labels),  # Shifted labels for training
        }