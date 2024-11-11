import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

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

# Input features [Average Goals Scored, Average Goals Conceded by Opponent]
X = torch.tensor([
    [3.0, 0.5], [1.0, 1.0], [0.5, 2.0], [2.0, 1.5],
    [3.5, 3.0], [2.0, 2.5], [1.5, 1.0], [0.5, 0.5],
    [2.5, 0.8], [2.1, 2.0], [1.2, 0.5], [0.7, 1.5]
], dtype=torch.float32)

# Target outputs [1 if the team is likely to win, 0 otherwise]
y = torch.tensor([[1], [0], [0], [1], [1], [0], [1], [0], [1], [0], [1], [0]], dtype=torch.float32)

# Define the model using nn.Sequential with 10 hidden units
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  

# Train the model for 50 epochs
for epoch in range(50):  
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients for iteration
    outputs = model(X)  # Compute predictions
    loss = criterion(outputs, y)  # Compute the loss
    loss.backward()  # Compute the gradient of the loss
    optimizer.step()  # Optimize the model parameters

# Create a new input tensor
new_input = torch.tensor([[2.0, 2.0]], dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# Disable gradient calculation for inference
with torch.no_grad():
    # Make a prediction for the new input
    prediction = model(new_input)

# Print the raw output from the model
print("Raw output:", prediction)

# Convert the probability to a binary class label
print("Prediction:", (prediction > 0.3).int().item())


# Training Features
X_train = torch.tensor([
    [3.0, 0.5], [1.0, 1.0], [0.5, 2.0], [2.0, 1.5],
    [3.5, 3.0], [2.0, 2.5], [1.5, 1.0], [0.5, 0.5],
    [2.5, 0.8], [2.1, 2.0], [1.2, 0.5], [0.7, 1.5]
], dtype=torch.float32)
# Training Targets
y_train = torch.tensor([[1], [0], [0], [1], [1], [0], [1], [0], [1], [0], [1], [0]], dtype=torch.float32)

# Test Features
X_test = torch.tensor([[2.5, 1.0], [0.8, 0.8], [1.0, 2.0], [3.0, 2.5]], dtype=torch.float32)
# Test Targets
y_test = torch.tensor([[1], [0], [0], [1]], dtype=torch.float32)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  

# Train the model for 50 epochs
for epoch in range(50):  
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients for iteration
    outputs = model(X_train)  # Compute predictions
    loss = criterion(outputs, y_train)  # Compute the loss
    loss.backward()  # Compute the gradient of the loss
    optimizer.step()  # Optimize the model parameters

# Set evaluation mode and disable gradient
model.eval()
with torch.no_grad():
    # Make Predictions
    outputs = model(X_test) 
    # Convert to binary classes
    predicted_classes = (outputs > 0.5).int() 
    # Calculate the loss on the test data
    test_loss = criterion(outputs, y_test).item()
    # Calculate the accuracy on the test data
    test_accuracy = accuracy_score(y_test.numpy(), predicted_classes.numpy())

# Print the test accuracy and loss
print(f'\nTest accuracy: {test_accuracy}, Test loss: {test_loss}')

from sklearn.datasets import load_wine

# Load the Wine dataset
wine = load_wine()

# Explore dataset features and target classes
print("Features:", wine.feature_names)
print("Target classes:", wine.target_names)

from sklearn.model_selection import train_test_split

X, y = wine.data, wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Display the shapes of the resulting splits
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

from sklearn.preprocessing import StandardScaler

# Initialize the scaler and fit it to the training data
scaler = StandardScaler().fit(X_train)

# Transform both the training and testing datasets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display scaled and unscaled samples
print("Unscaled X sample:\n", X_train[0])
print("Scaled X sample:\n", X_train_scaled[0])

# Convert scaled data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Display example tensors
print("Sample of X_train_tensor:", X_train_tensor[0])
print("Sample of y_train_tensor:", y_train_tensor[0])

def load_preprocessed_data():    
    # 1. Load the Wine dataset
    wine = load_wine()
    # 2. Split the dataset into training and testing sets
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    # 3. Scale the features
    # Initialize the scaler and fit it to the training data
    scaler = StandardScaler().fit(X_train)
    # Transform both the training and testing datasets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # 4. Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)    
    # 5. Return the processed x and y tensors for training and testing 
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

# TODO: Call load_preprocessed_data and print the shapes of the returned tensors

X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_preprocessed_data()
print(X_train_tensor.shape, X_test_tensor.shape, y_train_tensor.shape, y_test_tensor.shape)

X_train, X_test, y_train, y_test = load_preprocessed_data()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(13, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

# Display model's architecture
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 150
history = {'loss': [], 'val_loss': []}
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    history['loss'].append(loss.item())
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test).item()
        history['val_loss'].append(test_loss)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {test_loss:.4f}')

# Set the model to evaluation mode
model.eval()
# Disables gradient calculation
with torch.no_grad():
    # Input the test data into the model
    outputs = model(X_test)
    # print(f"{outputs=}")
    # Calculate the Cross Entropy Loss
    test_loss = criterion(outputs, y_test).item()
    # Choose the class with the highest value as the predicted output
    _, predicted = torch.max(outputs, 1)
    # print(f"{predicted=}")
    # print(f"{y_test=}")
    # Calculate the accuracy
    test_accuracy = accuracy_score(y_test, predicted)

    print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}')

import matplotlib.pyplot as plt

# Plotting actual training and validation loss
epochs = range(1, num_epochs + 1)
train_loss = history['loss']
val_loss = history['val_loss']

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
# plt.show()

torch.save(model, 'wine_model.pth')
# Load the entire model
loaded_model = torch.load('wine_model.pth')
loaded_model.eval()

# Verify the loaded model by evaluating it on test data
with torch.no_grad():
    # Make predictions for both models
    model.eval()
    original_outputs = model(X_test)
    loaded_outputs = loaded_model(X_test)
    # Format predictions
    _, original_predicted = torch.max(original_outputs, 1)
    _, loaded_predicted = torch.max(loaded_outputs, 1)
    # Calculate accuracies
    original_accuracy = accuracy_score(y_test, original_predicted)
    loaded_accuracy = accuracy_score(y_test, loaded_predicted)

# Display accuracies for both models
print(f'Original Model Accuracy: {original_accuracy:.4f}')
print(f'Loaded Model Accuracy: {loaded_accuracy:.4f}')
print(f"{original_accuracy=:.4f} {loaded_accuracy=:.4f}")

# Load dataset
wine = load_wine()
X = torch.tensor(wine.data, dtype=torch.float32)
y = torch.tensor(wine.target, dtype=torch.long)

# Split dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(13, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load dataset
wine = load_wine()
X = torch.tensor(wine.data, dtype=torch.float32)
y = torch.tensor(wine.target, dtype=torch.long)

# Split dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(13, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_loss = float('inf')
checkpoint_path = "best_model.pth"

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Validate the model
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid)
        val_loss = criterion(val_outputs, y_valid).item()
    
    # Save the model if the validation loss has decreased
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model, checkpoint_path)
        print(f"Model saved at epoch {epoch} with validation loss {val_loss:.4f}!")

batch_size = 32
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model training with mini-batches
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        print(f'Batch Loss: {loss.item():.4f} {batch_X.size(0)=}')
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(data_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')

import torch.optim.lr_scheduler as lr_scheduler

# Learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Model training with learning rate scheduling
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid)
        val_loss = criterion(val_outputs, y_valid)
    
    scheduler.step(val_loss)  # Update learning rate based on validation loss

    if (epoch + 1) % 10 == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch [{epoch + 1}/{num_epochs}], LR: {lr:.6f} Loss: {val_loss.item():.4f}')

# Load dataset
wine = load_wine()
X = torch.tensor(wine.data, dtype=torch.float32)
y = torch.tensor(wine.target, dtype=torch.long)

# Split dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = nn.Sequential(
    nn.Linear(13, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# TODO: Complete the learning rate scheduler                            
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Model training with learning rate scheduling
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid)
        val_loss = criterion(val_outputs, y_valid)
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)  

    if (epoch + 1) % 10 == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch [{epoch + 1}/{num_epochs}], LR: {lr:.6f},  Loss: {val_loss.item():.4f}')        

# Load and select the data
wine = load_wine()
X = wine.data
y = wine.target

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Define the model with dropout
model = nn.Sequential(
    nn.Linear(13, 10),
    nn.ReLU(),
    nn.Dropout(0.2),  # Dropout applied to the previous layer
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Dropout(0.1),  # Dropout applied to the previous layer
    nn.Linear(10, 3)
)

# Print the model summary
print(model)        

# Defining criterion and optimizer without weight decay
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  

for i in range(100):
    model.train()
    optimizer.zero_grad()  # Zero the gradient buffers
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, y)  # Compute loss
    loss.backward()  # Backward pass

    if(i==50):
        # Introducing weight decay from 50th epoch on
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01) 
        print("\nRegularization added to optimizer\n")

    if (i+1) % 10 ==0:    
        # L2 norm of weights of the first linear layer 
        first_layer_weights = model[0].weight.norm(2).item()
        print(f'{i+1} - L2 norm of weights: {first_layer_weights}')

    optimizer.step()  # Update weights