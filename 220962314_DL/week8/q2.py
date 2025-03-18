import os
import random
import string
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json

# Path to the extracted data folder (replace with your actual path)
data_path = './data/names'  # Replace with the correct path to your data folder

# Load the dataset
with open(os.path.join(data_path, 'names.json'), 'r') as f:
    all_data = json.load(f)

# Display the first few languages and names
for language in list(all_data.keys())[:5]:
    print(f"Language: {language}, Number of names: {len(all_data[language])}")

# Create a list of all characters that will be used for encoding
all_characters = string.ascii_letters + " .,;'"
n_characters = len(all_characters)

# Create a dictionary to map characters to indices
char_to_index = {char: index for index, char in enumerate(all_characters)}
index_to_char = {index: char for index, char in enumerate(all_characters)}


# Convert names to tensor format
def name_to_tensor(name):
    indices = [char_to_index[char] for char in name if char in char_to_index]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension


# Create a dictionary for language to index
languages = list(all_data.keys())
language_to_index = {language: index for index, language in enumerate(languages)}
index_to_language = {index: language for index, language in enumerate(languages)}


# Convert language name to index
def language_to_tensor(language):
    return torch.tensor([language_to_index[language]], dtype=torch.long)


# Custom Dataset class
class NamesDataset(Dataset):
    def __init__(self, all_data):
        self.names = []
        self.languages = []
        for language, name_list in all_data.items():
            for name in name_list:
                self.names.append(name)
                self.languages.append(language)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        language = self.languages[idx]
        input_tensor = name_to_tensor(name)
        target_tensor = language_to_tensor(language)
        return input_tensor, target_tensor


# Create the dataset and dataloaders
dataset = NamesDataset(all_data)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)


# Define the RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # RNN output
        out, _ = self.rnn(x, h0)

        # Output from the last time step
        out = self.fc(out[:, -1, :])
        return out


# Initialize the model
model = RNNModel(input_size=n_characters, hidden_size=128, output_size=len(languages))

# Define the Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        # Inputs should be one-hot encoded: [batch_size, seq_len, n_characters]
        inputs_one_hot = torch.zeros(inputs.size(0), inputs.size(1), n_characters).to(inputs.device)
        for i, name in enumerate(inputs):
            for j, char_idx in enumerate(name):
                inputs_one_hot[i, j, char_idx] = 1

        # Move inputs and labels to the device
        inputs_one_hot = inputs_one_hot.to(model.device)
        labels = labels.to(model.device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs_one_hot)

        # Calculate loss
        loss = criterion(outputs, labels.view(-1))
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')


# Evaluate the Model
def predict_language(name):
    input_tensor = name_to_tensor(name)
    input_tensor = torch.zeros(1, input_tensor.size(0), n_characters).to(model.device)
    for i, char_idx in enumerate(input_tensor[0]):
        input_tensor[0, i, char_idx] = 1

    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    return index_to_language[predicted.item()]


# Example usage:
test_name = "Bianchi"
predicted_language = predict_language(test_name)
print(f"The predicted language for '{test_name}' is: {predicted_language}")
