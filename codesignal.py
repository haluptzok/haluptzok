import torch

tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Original Tensor:\n{tensor_a}\n")

reshaped_tensor = tensor_a.view(3, 2)
print(f"Reshaped Tensor:\n{reshaped_tensor}\n")

flattened_tensor = tensor_a.view(-1)
print(f"Flattened Tensor:\n{flattened_tensor}")

tensor_a[0][0] = 7
reshaped_tensor[1][1] = 8
flattened_tensor[5] = 9

print(f"Original Tensor:\n{tensor_a}\n")
print(f"Reshaped Tensor:\n{reshaped_tensor}\n")
print(f"Flattened Tensor:\n{flattened_tensor}")

# Creating a tensor for manipulation
tensor_example = torch.tensor([[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [9, 10, 11, 12],
                               [13, 14, 15, 16]])

# Reshape the tensor into an 8x2 tensor
reshaped_tensor = tensor_example.view(8, 2)

# Displaying the original and reshaped tensors
print(f"Original Tensor: \n{tensor_example}\n")
print(f"Reshaped Tensor: \n{reshaped_tensor}")

import numpy as np

# Define a simple array as input data
X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
# Define the target outputs for our dataset
y = np.array([0, 1, 0, 1])
print("X:\n", X, X.dtype, X.shape)
print("Y:\n", y, y.dtype, y.shape)

# Convert X and y into PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.int32)
print(X_tensor, X_tensor.dtype, X_tensor.shape)
print(y_tensor, y_tensor.dtype, y_tensor.shape)

from torch.utils.data import TensorDataset

# Create a tensor dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Print x and y of the TensorDataset
for i in range(len(dataset)):
    X_sample, y_sample = dataset[i]
    print(f"X[{i}]: {X_sample}, y[{i}]: {y_sample}")

from torch.utils.data import DataLoader

# Create a data loader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_index, (batch_X, batch_y) in enumerate(dataloader):
    print(f"{batch_index=} Batch X:\n{batch_X}")
    print(f"{batch_index=} Batch y:\n{batch_y}\n")

import torch.nn as nn

# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self):
        # Initialize the parent class
        super(SimpleNN, self).__init__()
        # First fully connected layer: input size 2, output size 10
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        # ReLU activation function to be used after layer1
        self.relu = nn.ReLU()
        # Second fully connected layer: input size 10, output size 1
        self.layer2 = nn.Linear(in_features=10, out_features=1)
        # Sigmoid activation function to be used after layer2
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply layer1 (input: 2, output: 10)
        x = self.layer1(x)
        # Apply ReLU activation function (output: 10)
        x = self.relu(x)
        # Apply layer2 (input: 10, output: 1)
        x = self.layer2(x)
        # Apply Sigmoid activation function (output: 1)
        x = self.sigmoid(x)
        # Return the output
        return x

# Instantiate the model
model = SimpleNN()

# Print the model's architecture
print(model)

# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self):
        # Initialize the parent class
        super(SimpleNN, self).__init__()
        # First fully connected layer: input size 2, output size 10
        self.layer1 = nn.Linear(in_features=4, out_features=10)
        # ReLU activation function to be used after layer1
        self.relu = nn.ReLU()
        # Second fully connected layer: input size 10, output size 1
        self.layer2 = nn.Linear(in_features=10, out_features=2)  
        # Sigmoid activation function to be used after layer2
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply layer1 (input: 2, output: 10)
        x = self.layer1(x)
        # Apply ReLU activation function (output: 10)
        x = self.relu(x)
        # Apply layer2 (input: 10, output: 1)
        x = self.layer2(x)
        # Apply Sigmoid activation function (output: 1)
        x = self.sigmoid(x)
        # Return the output
        return x

# Instantiate the model
model = SimpleNN()

# Print the model's architecture
print(model)

# Define the neural network class
class MyOwnNN(nn.Module):
    def __init__(self):
        # Initialize the parent class
        super(MyOwnNN, self).__init__()
        # First fully connected layer: input size 5, output size 10
        self.layer1 = nn.Linear(in_features=5, out_features=10)
        # ReLU activation function to be used after layer1
        self.relu1 = nn.ReLU()
        # Second fully connected layer: input size 10, output size 20
        self.layer2 = nn.Linear(in_features=10, out_features=20)
        # Sigmoid activation function to be used after layer2
        self.sigmoid1 = nn.Sigmoid()
        # Incorrectly created third fully connected layer
        self.layer3 = nn.Linear(in_features=20, out_features=10)  
        # Sigmoid activation function to be used after layer3
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        # Apply layer1 (input: 5, output: 10)
        x = self.layer1(x)
        # Apply ReLU activation function
        x = self.relu1(x)
        # Apply layer2 (input: 10, output: 20)
        x = self.layer2(x)
        # Apply Sigmoid activation function
        x = self.sigmoid1(x)
        # Apply layer3 (input: 2, output: 10) — Error here due to incorrect input size
        x = self.layer3(x)
        # Apply Sigmoid activation function
        x = self.sigmoid2(x)
        # Return the output
        return x

# Instantiate the model
model = MyOwnNN()

# Print the model's architecture
print(model)

# TODO: Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self):
        # TODO: Initialize the parent class
        super(SimpleNN, self).__init__()
        # TODO: Define first fully connected layer: input size 3, output size 16
        self.layer1 = nn.Linear(in_features = 3, out_features = 16)
        # TODO: Define ReLU activation function (for layer1)
        self.relu1 = nn.ReLU()
        # TODO: Define second fully connected layer: input size 16, output size 8
        self.layer2 = nn.Linear(in_features = 16, out_features = 8)
        # TODO: Define ReLU activation function (for layer2)
        self.relu2 = nn.ReLU()
        # TODO: Define third fully connected layer: input size 8, output size 1
        self.layer3 = nn.Linear(in_features = 8, out_features = 1)
        # TODO: Define Sigmoid activation function (for layer3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply layer1 (input: 3, output: 16)
        x = self.layer1(x)
        # Apply ReLU activation function (output: 16)
        x = self.relu1(x)
        # Apply layer2 (input: 16, output: 8)
        x = self.layer2(x)
        # Apply ReLU activation function (output: 8)
        x = self.relu2(x)
        # Apply layer3 (input: 8, output: 1)
        x = self.layer3(x)
        # Apply Sigmoid activation function (output: 1)
        x = self.sigmoid(x)
        # Return the output
        return x

# TODO: Instantiate the model

model = SimpleNN()

# TODO: Display model's architecture

print(model)

# TODO: Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # TODO: First fully connected layer (input: 10, output: 20)
        self.layer1 = nn.Linear(in_features=10, out_features=20)
        # TODO: ReLU activation function 
        self.relu1 = nn.ReLU()
        # TODO: Second fully connected layer (input: 20, output: 15)
        self.layer2 = nn.Linear(in_features=20, out_features=15)
        # TODO: ReLU activation function
        self.relu2 = nn.ReLU()
        # TODO: Third fully connected layer (input: 15, output: 5)
        self.layer3 = nn.Linear(in_features=15, out_features=5)
        # TODO: Sigmoid activation function to be used after layer3
        self.sigmoid = nn.Sigmoid()

    # TODO: Define forward method
    def forward(self, x):
        # TODO: Apply layer1
        x = self.layer1(x)
        # TODO: Apply ReLU activation function
        x = self.relu1(x)
        # TODO: Apply layer2
        x = self.layer2(x)
        # TODO: Apply ReLU activation function
        x = self.relu2(x)
        # TODO: Apply layer3
        x = self.layer3(x)
        # TODO: Apply Sigmoid activation function
        x = self.sigmoid(x)
        # TODO: Return the output
        return(x)

# TODO: Instantiate the model
model = SimpleNN()
# TODO: Print the model's architecture
print(model)

# Defining an input tensor
input_tensor = torch.tensor([[5.0, 6.0]], dtype=torch.float32)

# Creating a linear layer with 2 input features and 3 output features
layer = nn.Linear(in_features=2, out_features=3)

# Defining a ReLU activation function
relu = nn.ReLU()

# Defining a Sigmoid activation function
sigmoid = nn.Sigmoid()

# Processing the input through the linear layer 
output_tensor = layer(input_tensor)

# Applying the ReLU function to the output of the linear layer
activated_output_relu = relu(output_tensor)

# Applying the Sigmoid function to the output of the linear layer
activated_output_sigmoid = sigmoid(output_tensor)

# Displaying the original input tensor
print(f"Input Tensor:\n{input_tensor}\n")

# Displaying the output before activation to see the linear transformation effect
print(f"Output Tensor Before Activation:\n{output_tensor}\n")

# Displaying the output after activation to observe the effect of ReLU
print(f"Output Tensor After ReLU Activation:\n{activated_output_relu}\n")

# Displaying the output after activation to observe the effect of Sigmoid
print(f"Output Tensor After Sigmoid Activation:\n{activated_output_sigmoid}")

import torch.optim as optim

# Input features [Average Goals Scored, Average Goals Conceded by Opponent]
X = torch.tensor([
    [3.0, 0.5], [1.0, 1.0], [0.5, 2.0], [2.0, 1.5],
    [3.5, 3.0], [2.0, 2.5], [1.5, 1.0], [0.5, 0.5],
    [2.5, 0.8], [2.1, 2.0], [1.2, 0.5], [0.7, 1.5]
], dtype=torch.float32)

# Target outputs [1 if the team is likely to win, 0 otherwise]
y = torch.tensor([[1], [0], [0], [1], [1], [0], [1], [0], [1], [0], [1], [0]], dtype=torch.float32)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)  

# Train the model for 50 epochs
for epoch in range(50):  
    model.train()  # Set the model to training mode

    optimizer.zero_grad()  # Zero the gradients for this iteration

    outputs = model(X)  # Forward pass: compute predictions

    loss = criterion(outputs, y)  # Compute the loss

    loss.backward()  # Backward pass: compute the gradient of the loss

    optimizer.step()  # Optimize the model parameters based on the gradients

    if (epoch+1) % 10 == 0:  # Print every 10 epochs
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")  # Print epoch loss


import torch
import torch.nn as nn
import torch.optim as optim

# Input features [Average Goals Scored, Average Goals Conceded by Opponent]
X = torch.tensor([
    [3.0, 0.5], [1.0, 1.0], [0.5, 2.0], [2.0, 1.5],
    [3.5, 3.0], [2.0, 2.5], [1.5, 1.0], [0.5, 0.5],
    [2.5, 0.8], [2.1, 2.0], [1.2, 0.5], [0.7, 1.5]
], dtype=torch.float32)

# Target outputs [1 if the team is likely to win, 0 otherwise]
y = torch.tensor([[1], [0], [0], [1], [1], [0], [1], [0], [1], [0], [1], [0]], dtype=torch.float32)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)  

# Train the model for 50 epochs
for epoch in range(50):  
    model.train()  # Set the model to training mode

    optimizer.zero_grad()  # Zero the gradients for this iteration

    outputs = model(X)  # Forward pass: compute predictions

    loss = criterion(outputs, y)  # Compute the loss

    loss.backward()  # Backward pass: compute the gradient of the loss

    optimizer.step()  # Optimize the model parameters based on the gradients

    if (epoch+1) % 10 == 0:  # Print every 10 epochs
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")  # Print epoch loss

