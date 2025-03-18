import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

# Step 1: Load and preprocess the data
df = pd.read_csv("./data/NaturalGasPrice/daily.csv")

# Drop NA values
df = df.dropna()

# Extract price column
y = df['Price'].values

# Normalize the prices to a range of 0 to 1
minm = y.min()
maxm = y.max()
y = (y - minm) / (maxm - minm)

# Define sequence length (last 10 days to predict the 11th)
Sequence_Length = 10

# Prepare input and output sequences
X = []
Y = []

for i in range(len(y) - Sequence_Length):
    X.append(y[i:i + Sequence_Length])
    Y.append(y[i + Sequence_Length])

# Convert lists to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Step 2: Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle=False)

# Step 3: Define the Dataset class for PyTorch
class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

# Create Dataset and DataLoader
train_dataset = NGTimeSeries(x_train, y_train)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)

# Step 4: Define the RNN Model
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = output[:, -1, :]  # Get the last output of the sequence
        output = self.fc1(torch.relu(output))
        return output

# Instantiate the model
model = RNNModel()

# Step 5: Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Step 6: Train the model
epochs = 1500

for epoch in range(epochs):
    for data in train_loader:
        inputs, targets = data
        inputs = inputs.view(-1, Sequence_Length, 1)  # Reshape inputs for RNN (batch_size, seq_len, input_size)
        optimizer.zero_grad()
        y_pred = model(inputs).view(-1)
        loss = criterion(y_pred, targets)
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Step 7: Evaluate the model on the test set
test_dataset = NGTimeSeries(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()  # Set model to evaluation mode

test_pred = []
test_actual = []

with torch.no_grad():
    for data in test_loader:
        inputs, targets = data
        inputs = inputs.view(-1, Sequence_Length, 1)
        y_pred = model(inputs).view(-1)
        test_pred.append(y_pred.item())
        test_actual.append(targets.item())

# Convert lists to arrays for plotting
test_pred = np.array(test_pred)
test_actual = np.array(test_actual)

# Step 8: Plot actual vs predicted prices
plt.plot(test_pred, label='Predicted')
plt.plot(test_actual, label='Actual')
plt.legend()
plt.title('Natural Gas Price Prediction: Actual vs Predicted')
plt.show()

# Step 9: Undo normalization for final results
test_pred_unnormalized = test_pred * (maxm - minm) + minm
test_actual_unnormalized = test_actual * (maxm - minm) + minm

# Plot the unnormalized prices
plt.plot(test_actual_unnormalized, label='Actual')
plt.plot(test_pred_unnormalized, label='Predicted')
plt.legend()
plt.title('Unnormalized Natural Gas Price Prediction')
plt.show()
