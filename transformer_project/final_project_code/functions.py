# Helper functions for conducting the experiments

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import data_generation


def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, epochs, vocab_size=127, model_save_path=False):
    # Train function for the encoder-decoder model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    train_loss_curve = []
    val_loss_curve = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Training
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            src = batch['input_ids'].to(device)
            tgt = batch['target_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            output = model(src, tgt)
            loss = loss_fn(output.view(-1, vocab_size), labels.view(-1))
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
                src = batch['input_ids'].to(device)
                tgt = batch['target_ids'].to(device)
                labels = batch['labels'].to(device)

                output = model(src, tgt)
                loss = loss_fn(output.view(-1, vocab_size), labels.view(-1))
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss/len(train_dataloader)}, Val Loss = {val_loss/len(val_dataloader)}")

        train_loss_curve.append(train_loss / len(train_dataloader))
        val_loss_curve.append(val_loss / len(val_dataloader))

    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_loss_curve, label='Train Loss')
    plt.plot(range(epochs), val_loss_curve, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curve')
    plt.show()
  
    # Save the model
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")


def greedy_decode(model, src, max_len=77, device="cpu"):
    # Greedy decode function for the encoder decoder model  
    # Encode source
    src = src.to(device, dtype=torch.long)

    memory = model.encode(src)
    
    # Initialize decoder input
    batch_size = src.size(0)
    tgt = torch.full((batch_size, 1), 0, dtype=torch.long).to(device)
    
    translations = [[] for _ in range(batch_size)]
    for _ in range(max_len):
        logits = model.decode(tgt, memory)
        next_token_logits = logits[:, -1, :] 
        next_tokens = next_token_logits.argmax(dim=-1).unsqueeze(1)  
        
        # Append generated token to translations
        for i, token in enumerate(next_tokens.squeeze(1).tolist()):
            translations[i].append(token)
        
        # Update target with the newly generated token
        tgt = torch.cat([tgt, next_tokens], dim=1)

    return translations


def extract_values_from_string(value_string):
    """
    Extracts float values from a string sequence, expecting the format: '&value1&value2&...'
    """
    value_string = value_string.replace('[EOS]', '')
    values = value_string.split('&')[1:]
    return [float(val) for val in values]

def plot(perfect_str, predicted_str, x_range=(-5, 5), num_points=11):
    # Extract the f(x) values from the input strings
    perfect_values = extract_values_from_string(perfect_str)
    predicted_values = extract_values_from_string(predicted_str)
    
    # Generate the x values for the plot
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    
    # Ensure that the number of values matches the number of x values
    assert len(perfect_values) == len(x_values), "The number of f(x) values does not match the number of x values"
    assert len(predicted_values) == len(x_values), "The number of predicted f(x) values does not match the number of x values"
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, perfect_values, label="Perfect f(x)", color='b', linestyle='-', marker='o')
    plt.plot(x_values, predicted_values, label="Predicted f(x)", color='r', linestyle='--', marker='x')
    
    # Adding labels and title
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Perfect vs Predicted f(x) Values')
    plt.legend()
    
    # Plot
    plt.grid(True)
    plt.show()


def generate_examples(model, batch, num_examples):
    # Print the inputs and outputs generated from the trained model 
    # Plot the predefined number of example predictions 
    x = 0
    output = greedy_decode(model, batch['input_ids'])
    for i in range(batch['input_ids'].size(0)):
        print('######')
        print('Input:', data_generation.decode(batch['input_ids'][i]))
        print('Labels: ', data_generation.decode(batch['labels'][i]))
        print('Output:', data_generation.decode(output[i]))
        if x < num_examples:
            try:
                plot(data_generation.decode(batch['labels'][i]), data_generation.decode(output[i]))
                x += 1
            except:
                print('Outputs generated in wrong format.')
                print(data_generation.decode(output[i]))
    

def compute_mse(model, test_loader, device="cpu"):
    """
    Computes the MSE of the given model on the test set.
    If the predicted string is in the wrong format, assigns an MSE of -1 for that example.
    
    Returns:
    - overall_mse: The computed mean squared error over the test set, with format validation.
    - num_valid: Number of valid samples
    - num_invalid: Number of invalid samples
    """
    model.eval()
    mse_values = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing MSE"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Generate predictions
            predictions = greedy_decode(model, input_ids, device=device)

            # Decode predictions and ground truth to extract numerical values
            for i in range(len(labels)):
                try:
                    true_values = extract_values_from_string(data_generation.decode(labels[i]))
                    predicted_values = extract_values_from_string(data_generation.decode(predictions[i]))

                    # Ensure both lists have the same length before computing MSE
                    if len(true_values) == len(predicted_values):
                        mse = np.mean((np.array(true_values) - np.array(predicted_values)) ** 2)
                    else:
                        # If lengths don't match, mark as incorrect format
                        mse = -1
                except (ValueError, IndexError) as e:
                    # If decoding fails or values are invalid, assign MSE of -1
                    mse = -1
                
                mse_values.append(mse)

    # Compute the overall mean of the MSE values, ignoring invalid cases
    valid_mse_values = [value for value in mse_values if value != -1]
    invalid_mse_values = [value for value in mse_values if value == -1]
    overall_mse = np.mean(valid_mse_values) if valid_mse_values else -1

    num_valid = len(valid_mse_values)
    num_invalid = len(invalid_mse_values)


    return overall_mse, num_valid, num_invalid


def count_parameters(model):
    # Count the number of trainable parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


################# Decoder Functions #################

def train_decoder(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, epochs, model_save_path=False, source_length=27):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    train_loss_curve = []
    val_loss_curve = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Training
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            src = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            output = model(src)

            # Create a mask to ignore the first tokens
            mask = torch.ones_like(labels, dtype=torch.bool)
            mask[:, :source_length] = False

            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)
            mask = mask.view(-1)

            loss = loss_fn(output[mask], labels[mask])
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
                src = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                output = model(src)

                mask = torch.ones_like(labels, dtype=torch.bool)
                mask[:, :source_length] = False

                output = output.view(-1, output.size(-1))
                labels = labels.view(-1)
                mask = mask.view(-1)

                loss = loss_fn(output[mask], labels[mask])
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss/len(train_dataloader)}, Val Loss = {val_loss/len(val_dataloader)}")

        train_loss_curve.append(train_loss / len(train_dataloader))
        val_loss_curve.append(val_loss / len(val_dataloader))

    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_loss_curve, label='Train Loss')
    plt.plot(range(epochs), val_loss_curve, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curve')
    plt.show()

    # Save the model
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")


def greedy_decode_decoder(model, tgt, max_len=77, device="cpu"):

    batch_size = tgt.size(0)

    
    tokens = [[] for _ in range(batch_size)]
    for _ in range(max_len):
        logits = model.decode(tgt)
        next_token_logits = logits[:, -1, :] 
        next_tokens = next_token_logits.argmax(dim=-1).unsqueeze(1)
        
        # Append generated token to translations
        for i, token in enumerate(next_tokens.squeeze(1).tolist()):
            tokens[i].append(token)
        
        # Update target with the newly generated token
        tgt = torch.cat([tgt, next_tokens], dim=1)

    return tokens


def generate_examples_decoder(model, batch, num_examples, source_length=27):
    x = 0
    input = batch['input_ids'][:, :(source_length + 1)]
    output = greedy_decode_decoder(model, input)
    for i in range(batch['input_ids'].size(0)):
        print('######')
        print('Input:', data_generation.decode(batch['input_ids'][i]))
        print('Labels: ', data_generation.decode(batch['labels'][i]))
        print('Output:', data_generation.decode(output[i]))
        if x < num_examples:
            try: 
                plot(data_generation.decode(batch['labels'][i][source_length:]), data_generation.decode(output[i]))
                x += 1
            except:
                print('Outputs generated in wrong format.')
                print(data_generation.decode(output[i]))


def compute_mse_decoder(model, test_loader, device="cpu"):
    model.eval()
    mse_values = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing MSE"):
            input_ids = batch['input_ids'][:, :28].to(device)
            labels = batch['labels'].to(device)

            # Generate predictions
            predictions = greedy_decode_decoder(model, input_ids, device=device)

            # Decode predictions and ground truth to extract numerical values
            for i in range(len(labels)):
                try:
                    true_values = extract_values_from_string(data_generation.decode(labels[i]))
                    predicted_values = extract_values_from_string(data_generation.decode(predictions[i]))

                    # Ensure both lists have the same length before computing MSE
                    if len(true_values) == len(predicted_values):
                        mse = np.mean((np.array(true_values) - np.array(predicted_values)) ** 2)
                    else:
                        # If lengths don't match, mark as incorrect format
                        print('Length error')
                        print(data_generation.decode(labels[i]))
                        print(data_generation.decode(predictions[i]))
                        mse = -1
                except (ValueError, IndexError) as e:
                    # If decoding fails or values are invalid, assign MSE of -1
                    print('Value error')
                    print(data_generation.decode(labels[i]))
                    print(data_generation.decode(predictions[i]))
                    mse = -1
                
                mse_values.append(mse)

    # Compute the overall mean of the MSE values, ignoring invalid cases
    valid_mse_values = [value for value in mse_values if value != -1]
    invalid_mse_values = [value for value in mse_values if value == -1]
    overall_mse = np.mean(valid_mse_values) if valid_mse_values else -1

    num_valid = len(valid_mse_values)
    num_invalid = len(invalid_mse_values)


    return overall_mse, num_valid, num_invalid
