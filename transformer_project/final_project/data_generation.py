import numpy as np
import random
import torch
from torch.utils.data import Dataset
import os
    

def generate_coefficient(distribution="uniform", coefficient_range=(-9.9, 9.9), tail_percentage=0, sample_from_tail=False):
    if distribution == 'uniform':
        return random.uniform(coefficient_range[0], coefficient_range[1]) 
    elif distribution == 'normal': 
        if tail_percentage == 0:
            # Normal distribution with rejection of values outside the coefficient range
            while True:
                sample = random.gauss(0, 2)
                if coefficient_range[0] <= sample <= coefficient_range[1]:
                    return sample
        else:
            # Calculate tail cutoffs using the normal distribution
            tail_cutoff = np.percentile(np.random.normal(0, 2, 100000), [tail_percentage / 2 * 100, (1 - tail_percentage / 2) * 100])
            lower_tail, upper_tail = tail_cutoff

            if sample_from_tail:
                while True:
                    sample = random.gauss(0, 2)
                    if (sample < lower_tail or sample > upper_tail) and coefficient_range[0] <= sample <= coefficient_range[1]:
                        return sample
            else:
                while True:
                    sample = random.gauss(0, 2)
                    if lower_tail <= sample <= upper_tail and coefficient_range[0] <= sample <= coefficient_range[1]:
                        return sample

    else:
        raise ValueError("Invalid distribution type. Choose 'uniform' or 'normal'.")


def generate_polynomial_dataset(num_samples=50000, x_range=(-5, 5), num_points=11, distribution="uniform", tail_percentage=0, sample_from_tail=False):
    dataset = []
    
    for _ in range(num_samples):
        # Generate coefficients the polynomial f(x) = ax^2 + bx + c
        a = generate_coefficient(distribution=distribution, tail_percentage=tail_percentage, sample_from_tail=sample_from_tail)
        b = generate_coefficient(distribution=distribution, tail_percentage=tail_percentage, sample_from_tail=sample_from_tail)
        c = generate_coefficient(distribution=distribution, tail_percentage=tail_percentage, sample_from_tail=sample_from_tail)
        
        # Create the polynomial equation string
        if a < 0:
            a_term = f"{a:.2f}x^2"
        else:
            a_term = f"{a:.3f}x^2"
        if b < 0:
            b_term = f"{b:.2f}x"
        else:
            b_term = f"{b:.3f}x"
        if c < 0:
            c_term = f"{c:.2f}"
        else:
            c_term = f"{c:.3f}"


        polynomial_str = "f(x) = " + a_term + " + " + b_term + " + " + c_term

        
        # Calculate the values of f(x) for x
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        y_values = a * x_values**2 + b * x_values + c  
        
        # Append the input-output pair
        dataset.append((polynomial_str, y_values))
    
    return dataset


# Simple character-level tokenizer
def input_tokenizer(polynomial_str):
    # Tokenize based on characters
    return [char for char in polynomial_str if char != ' ']

# Simple character-level tokenizer and enshuring coherent lengths
def output_tokenizer(values):
    result = ""
    for value in values:
        digits = 4
        if value < 0:
            digits -= 1
        if (abs(value) >= 10) and (abs(value) < 100):
            digits -= 1
        elif abs(value) >= 100:
            digits -= 2
        number = str(round(value, digits))
        if len(number) == 5:
            number += '0'
        if len(number) == 4:
            number += '00'
        if len(number) == 3:
            number += '000'
        result += ' & ' + number
    tokens = [char for char in result if char != ' ']

    return tokens


def decode(tokens):
    # Define the special token mappings
    BOS_TOKEN = 0
    PAD_TOKEN = 1
    EOS_TOKEN = 2
    SEP_TOKEN = 3
    
    # Map each token to its corresponding character or handle the special tokens
    decoded_string = ''
    
    for token in tokens:
        if token == BOS_TOKEN:
            decoded_string += '[BOS]'
        elif token == EOS_TOKEN:
            decoded_string += '[EOS]'
        elif token == PAD_TOKEN:
            decoded_string += '[PAD]'
        elif token == SEP_TOKEN:
            decoded_string += '[SEP]'
        else:
            # For other tokens, convert the token to a character
            decoded_string += chr(token)
    
    return decoded_string


class EncoderDecoderDataset(Dataset):
    def __init__(self, dataset, input_tokenizer, output_tokenizer):
        self.dataset = dataset
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        self.BOS_TOKEN = 0  
        self.PAD_TOKEN = 1  
        self.EOS_TOKEN = 2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        polynomial_str, y_values = self.dataset[idx]
        
        # Tokenize the input and output
        input_tokens = self.input_tokenizer(polynomial_str)
        output_tokens = self.output_tokenizer(y_values)
        
        input_ids = torch.tensor([ord(token) for token in input_tokens]) 
        target_ids = torch.tensor([ord(val) for val in output_tokens]) 
        target_ids = torch.cat([torch.tensor([self.BOS_TOKEN]), target_ids], dim=0)
        labels = torch.cat([target_ids[1:], torch.tensor([self.EOS_TOKEN])], dim=0)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "target_ids": torch.tensor(target_ids),
            "labels": torch.tensor(labels)
        }


class DecoderDataset(Dataset):
    def __init__(self, dataset, input_tokenizer, output_tokenizer):
        self.dataset = dataset
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        self.BOS_TOKEN = 0  
        self.PAD_TOKEN = 1  
        self.EOS_TOKEN = 2
        self.SEP_TOKEN = 3

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        polynomial_str, y_values = self.dataset[idx]
        
        # Tokenize the input and output
        input_tokens = self.input_tokenizer(polynomial_str)
        output_tokens = self.output_tokenizer(y_values)
        
        # Convert tokens to integer representation
        input_ids = torch.tensor([ord(token) for token in input_tokens])
        output_ids = torch.tensor([ord(val) for val in output_tokens])
        
        # Concatenate sequences 
        input_ids = torch.cat([
            torch.tensor([self.BOS_TOKEN]),
            input_ids,
            torch.tensor([self.SEP_TOKEN]), 
            output_ids
        ], dim=0)
        
        labels = torch.cat([input_ids[1:], torch.tensor([self.EOS_TOKEN])], dim=0)

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)

        return {
            "input_ids": input_ids,
            "labels": labels
        }
    

def create_train_val_test_split(num_samples, split_ratio=(0.7, 0.2, 0.1), save=False, distribution="uniform"):
    assert round(sum(split_ratio), 15) == 1.0, "Split ratios must sum to 1.0"

    # Generate the dataset
    dataset = generate_polynomial_dataset(num_samples=num_samples, distribution=distribution)

    # Shuffle 
    random.shuffle(dataset)

    # Calculate split indices
    train_size = int(split_ratio[0] * num_samples)
    val_size = int(split_ratio[1] * num_samples)
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]

    if save:
        torch.save(train_data, "train_data.pt")
        torch.save(val_data, "val_data.pt")
        torch.save(test_data, "test_data.pt")

    return train_data, val_data, test_data
