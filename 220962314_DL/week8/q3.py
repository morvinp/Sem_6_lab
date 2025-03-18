import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
import random

# Define the corpus (you can replace this with any text dataset)
text = "hello world this is an example of rnn text generation"

# Create a list of all characters used in the text
all_characters = string.ascii_lowercase + " "  # Include space as a character
n_characters = len(all_characters)

# Create a dictionary to map characters to indices and vice versa
char_to_index = {char: index for index, char in enumerate(all_characters)}
index_to_char = {index: char for index, char in enumerate(all_characters)}


# Convert a sequence of characters to tensor (integer encoding)
def char_to_tensor(text):
    indices = [char_to_index[char] for char in text]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension


# Create training data (pairs of input sequence and next character)
seq_length = 5  # Length of input sequence
input_data = []
target_data = []

for i in range(len(text) - seq_length):
    input_seq = text[i:i + seq_length]
    target_char = text[i + seq_length]

    input_data.append(char_to_tensor(input_seq))
    target_data.append(char_to_tensor(target_char))

# Convert to tensor
X = torch.cat(input_data, dim=0)  # Shape: (num_samples, seq_length)
Y = torch.cat(target_data, dim=0)  # Shape: (num_samples, 1)


# Define the RNN Model for next character prediction
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size

        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # Define the output layer to predict the next character
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize the hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # RNN output
        out, _ = self.rnn(x, h0)

        # Output from the last time step
        out = self.fc(out[:, -1, :])  # (batch_size, hidden_size) -> (batch_size, output_size)
        return out


# Initialize the model
model = RNNModel(input_size=n_characters, hidden_size=128, output_size=n_characters)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()

    # One-hot encode the input
    X_one_hot = torch.zeros(X.size(0), X.size(1), n_characters).to(X.device)
    for i, seq in enumerate(X):
        for j, char_idx in enumerate(seq):
            X_one_hot[i, j, char_idx] = 1

    # Forward pass
    output = model(X_one_hot)

    # Calculate the loss
    loss = criterion(output, Y)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


# Generate text after training
def generate_text(start_string, length=100):
    model.eval()  # Set the model to evaluation mode
    input_seq = char_to_tensor(start_string).to(model.device)

    # One-hot encode the input
    input_one_hot = torch.zeros(1, input_seq.size(0), n_characters).to(model.device)
    for i, char_idx in enumerate(input_seq[0]):
        input_one_hot[0, i, char_idx] = 1

    generated_text = start_string
    current_input = input_one_hot

    # Generate characters one by one
    for _ in range(length):
        output = model(current_input)
        _, predicted_idx = torch.max(output, dim=1)

        # Convert predicted index to character
        predicted_char = index_to_char[predicted_idx.item()]
        generated_text += predicted_char

        # Update the input for the next prediction (shift by one character)
        current_input = torch.zeros(1, 1, n_characters).to(model.device)
        current_input[0, 0, predicted_idx] = 1

    return generated_text


# Test the model by generating text
generated_text = generate_text("hello", length=50)
print(f"Generated Text: {generated_text}")
