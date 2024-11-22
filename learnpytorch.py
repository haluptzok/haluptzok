import torch
import random
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Check PyTorch version
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.current_device())

# Part02 PyTorch Neural Networks Classification

from sklearn.datasets import make_circles


# Make 1000 samples 
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values

print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")

# Make DataFrame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
circles.head(10)
# Check different labels
circles.label.value_counts()
# Visualize with a plot
import matplotlib.pyplot as plt
if False:
    plt.scatter(x=X[:, 0], 
                y=X[:, 1], 
                c=y, 
                cmap=plt.cm.RdYlBu);
    # plt.show()

# Check the shapes of our features and labels
print(f"{X.shape=}, {y.shape=}")

# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

# Turn data into tensors
# Otherwise this causes issues with computations later on

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# View the first five samples
print(f"{X[:5]=}, {y[:5]=}")
# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

print(len(X_train), len(X_test), len(y_train), len(y_test))
# Standard PyTorch imports
import torch
from torch import nn

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device=}")

# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features, produces 1 feature (y)
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(self.layer_1(x)) # computation goes through layer_1 first then the output of layer_1 goes through layer_2

# 4. Create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)
print("model_0\n", model_0)
# Replicate CircleModelV0 with nn.Sequential
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
print("model_0\n", model_0)
# Make predictions with the model
untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")

# Create a loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    # print(f"{len(y_true)=}, {len(y_pred)=}")
    # print(f"{y_true[:5]=}, {y_pred[:5]=}")
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    # print(f"{correct=}")
    acc = (correct / len(y_pred))
    return acc

# View the frist 5 outputs of the forward pass on the test data
y_logits = model_0(X_test.to(device))[:5]
print(f"{y_logits=}")
# Use sigmoid on model logits
y_pred_probs = torch.sigmoid(y_logits)
print(f"{y_pred_probs=}")

# Find the predicted labels (round the prediction probabilities)
y_preds = torch.round(y_pred_probs)

# In full
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Check for equality
print(f"{y_preds.squeeze()=}, {y_pred_labels.squeeze()=}")
print(f"{torch.eq(y_preds.squeeze(), y_pred_labels.squeeze())=}")

y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device)))).cpu().type(torch.float32)
print(f"{accuracy_fn(y_test.squeeze(), y_pred_labels.squeeze())=}", f"numsamples = {len(y_test)} {len(y_pred_labels)}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the number of epochs
epochs = 100

# Put data to target device
print(f"{device=}")
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
  
    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train) 
    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   y_train) 
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    if epoch % 10 == 0:    
        model_0.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model_0(X_test).squeeze() 
            test_pred = torch.round(torch.sigmoid(test_logits))
            # 2. Caculate loss/accuracy
            test_loss = loss_fn(test_logits,
                                y_test)
            test_acc = accuracy_fn(y_true=y_test,
                                y_pred=test_pred)

        # Print out what's happening every 10 epochs
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# Plot decision boundaries for training and test sets
if False:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model_0, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_0, X_test, y_test)
    plt.show()

# Build model with non-linear activation function
from torch import nn
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # <- add in ReLU activation function
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to(device)
print(model_3)
# Setup loss and optimizer 
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)
# Fit the model
torch.manual_seed(42)
torch
epochs = 2000

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

if False:

    for epoch in range(epochs):
        # 1. Forward pass
        y_logits = model_3(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
        # y_pred = torch.round(y_logits) # logits -> prediction probabilities -> prediction labels
        
        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
        acc = accuracy_fn(y_true=y_train, 
                        y_pred=y_pred)
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        if epoch % 200 == 0:
            model_3.eval()
            with torch.inference_mode():
                # 1. Forward pass
                test_logits = model_3(X_test).squeeze()
                test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
                # test_pred = torch.round(test_logits) # logits -> prediction probabilities -> prediction labels
                # 2. Calcuate loss and accuracy
                test_loss = loss_fn(test_logits, y_test)
                test_acc = accuracy_fn(y_true=y_test,
                                        y_pred=test_pred)

            # Print out what's happening        
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

    # Make predictions
    model_3.eval()
    with torch.inference_mode():
        y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()
    y_preds[:10], y[:10] # want preds in same format as truth labels

    # Plot decision boundaries for training and test sets
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model_0, X_train, y_train) # model_1 = no non-linearity
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_3, X_test, y_test) # model_3 = has non-linearity
    plt.show()

# Create a toy tensor (similar to the data going into our model(s))
A = torch.arange(-10, 10, 1, dtype=torch.float32)
A = torch.arange(-10, 10, 1, dtype=torch.float32)

if False:
    print(f"{A=}")
    # Visualize the toy tensor
    plt.plot(A);

    def relu(x):
        return torch.max(torch.tensor(0), x)

    print(f"{relu(A)=}")
    plt.plot(relu(A))

    plt.show()

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

print(f"{sigmoid(A)=}")

if False:
    plt.plot(sigmoid(A))
    plt.show()

exit()
# Part01 PyTorch Workflow Fundamentals

def myfunc(a, b):
  return a + b

x = map(myfunc, ('apple', 'banana', 'cherry'), ('orange', 'lemon', 'pineapple'))
print(next(x))
print(next(x))
print(next(x))
print(next(x))
print(next(x))

print(x)

#convert the map into a list, for readability:
print(list(x))
exit()

what_were_covering = {1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}

print(what_were_covering)

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.1
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
print(f"{X=}\n{y=}")

# Create train/test split
train_split = int(0.5 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(f"{len(X_train)=}, {len(y_train)=}, {len(X_test)=}, {len(y_test)=}")

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None, show=False):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});
  if show:
    plt.show()

plot_predictions()

# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

# Set manual seed since nn.Parameter are randomly initialzied
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

# Check the nn.Parameter(s) within the nn.Module subclass we created
print(f"{list(model_0.parameters())=}")
# List named parameters 
print(f"{model_0.state_dict()=}")

# Make predictions with model
with torch.inference_mode():
    y_preds = model_0(X_test)

# Note: in older PyTorch code you might also see torch.no_grad()
with torch.no_grad():
    y_preds = model_0(X_test)

print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

for y_pred_i, y_i in zip(y_preds, y_test):
    print(f"Pred: {y_pred_i.item():.4f}, Actual: {y_i.item():.4f}")

plot_predictions(predictions=y_preds, show=False)

print("MSE is:", torch.square(y_test - y_preds))
print("MSE is:", torch.square(y_test - y_preds).mean())
print("MSE is:", (y_test - y_preds).square().mean())
loss_fn = nn.MSELoss()
mse = loss_fn(y_test, y_preds)
print(f"{mse=}")

# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss
l1_loss = loss_fn(y_test, y_preds)
print(f"{l1_loss=}")
print("L1 Loss is:", torch.abs(y_test - y_preds).mean())
print("L1 Loss is:", (y_test - y_preds).abs().mean())

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))


torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 101

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    if epoch % 10 == 0:
        with torch.inference_mode():
            # 1. Forward pass on test data
            test_pred = model_0(X_test)

            # 2. Caculate loss on test data
            test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

    ### Training
    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    # Print out what's happening
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# Plot the loss curves
if False:
    plt.figure(figsize=(10, 7))
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

exit()
# Part00 Pytorch Fundamentals part1
random.seed(42) # set the seed for reproducibility 
torch.manual_seed(42) # set the seed for reproducibility
torch.cuda.manual_seed(42) # set the seed for reproducibility

x = torch.arange(1,8)
print(x)
print(f"{x=}")
print(x.shape, x.size(), x.ndim,x.dtype, x.device, x.layout)

x_reshaped = x.reshape(1,7)
print(x_reshaped, x_reshaped.shape)

z = x.view(1, 7)
print(z, z.shape)
z[:, 0] = 5

print(x, z, x_reshaped)

x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked, x_stacked.shape)

x_stacked = torch.stack([x, x, x, x], dim=1)
print(f"{x_stacked=}, {x_stacked.shape=}")
print(f"{x_stacked[:][0]=}")
print(f"{x_stacked[:,0]=}")
print(f"{x_stacked[0,:]=}")

x_cat = torch.cat([x, x, x, x], dim=0)
print(x_cat, x_cat.shape)

x_cat = torch.cat([x_reshaped, x_reshaped, x_reshaped], dim=0)
print(x_cat, x_cat.shape)

x_cat = torch.cat([x_reshaped, x_reshaped, x_reshaped], dim=1)
print(x_cat, x_cat.shape)

print(f"{x_reshaped=}")
print(f"{x_reshaped.shape=}")
print(f"{x_reshaped.unsqueeze(dim=0)=}")
print(f"{x_reshaped.unsqueeze(dim=0).shape=}")
print(f"{x_reshaped.unsqueeze(dim=1)=}")
print(f"{x_reshaped.unsqueeze(dim=1).shape=}")
print(f"{x_reshaped.unsqueeze(dim=2)=}")
print(f"{x_reshaped.unsqueeze(dim=2).shape=}")
print(f"{x_reshaped.squeeze()=}")
print(f"{x_reshaped.squeeze().shape=}")

x = torch.arange(1, 10).reshape(1, 3, 3)
print(f"{x=}")
print(f"{x.shape=}")

# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]=}") 
print(f"Second square bracket: {x[0][0]=}") 
print(f"Third square bracket: {x[0][0][0]=}")

print(f"example: {x[0,:,1]=}")
print(f"example: {x[:,1,1]=}")
print(f"example: {x[:,1,:]=}")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")    

print(f"{device=}")

x = torch.tensor([1, 2, 3], device=device)
x_cpu = torch.tensor([1, 2, 3])

print(f"{x=}")
print(f"{x_cpu=}")
x_cpu = x_cpu.to(device)
print(f"{x_cpu=}")
y = x + x_cpu
print(f"{y=}")
y = y.to("cpu")
print(f"{y.numpy()=}")


# NumPy array to tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
tensor16 = torch.from_numpy(array).type(torch.float32)
print(f"{array=}, {tensor=} {tensor16=}")

array += 1
print(f"{array=}, {tensor=} {tensor16=}")

tensor += 1
print(f"{array=}, {tensor=} {tensor16=}")

random.seed(42) # set the seed for reproducibility 
torch.manual_seed(42) # set the seed for reproducibility
torch.cuda.manual_seed(42) # set the seed for reproducibility

x = torch.arange(1,50).reshape(7,7)
print(x)
print(f"{x=}")
print(x.shape, x.size(), x.ndim,x.dtype, x.device, x.layout)
y = torch.arange(1,8).reshape(1,7)
print(f"{y=}")
print(y.shape, y.size(), y.ndim,y.dtype, y.device, y.layout)
print(f"{x+y=}")
print(f"{x-y=}")
print(f"{x*y=}")
print(f"{x/y=}")
print(f"{y@x=}")
print(f"{torch.matmul(y,x)=}")
print(f"{torch.matmul(x,y.T)=}")
print(f"{torch.matmul(x,y[0])=}")

exit()
# Part00 Pytorch Fundamentals part0
a = torch.tensor([1, 2, 3], dtype=torch.float32)
print(a)
print(f"{a=}") 
a = torch.zeros([3, 3], dtype=torch.float32)
a = torch.zeros([3, 3], dtype=torch.int32)
print(a)

cuda0 = torch.device('cuda:0')
print(cuda0, f"{cuda0=}")
a = torch.tensor([1, 2, 3], dtype=torch.int32, device=cuda0)
print(a)
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)
print(x)
print(x[1][2])
print(x[1, 2])
x[0][1] = 8
print(x)
print(f"{x=}")
print(f"{x[1,2]=}")
print(f"{x[1,2].item()=}")

x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
out = x.pow(2).sum()
out.backward()
print(f"{x.grad=}")
z = torch.tensor(7, dtype=torch.float32, requires_grad=True)
print(f"{z=}")
z_out = z.pow(2)
y_out = z_out.pow(2)
z_out.retain_grad()
y_out.retain_grad()
print(f"{z_out=}")
print(f"{y_out=}")
y_out.backward()
print(f"{y_out.grad=}")
print(f"{z_out.grad=}")
print(f"{z.grad=}")
z = torch.tensor(7.0001, dtype=torch.float32, requires_grad=True)
print(f"{z=}")
z_out = z.pow(2)
y_out = z_out.pow(2)
z_out.retain_grad()
y_out.retain_grad()
print(f"{z_out=}")
print(f"{y_out=}")
y_out.backward()
print(f"{y_out.grad=}")
print(f"{z_out.grad=}")
print(f"{z.grad=}")
z = torch.arange(1, 5, dtype=torch.float32, requires_grad=True).reshape(2, 2)
print(f"{z=}")
scalar = torch.tensor(7, dtype=torch.float32, requires_grad=True)
print(f"{scalar=} {scalar.grad=} {scalar.ndim=} {scalar.shape=}")
arr = torch.tensor([7, 7, 7], dtype=torch.float32, requires_grad=True)
print(f"{arr=} {arr.grad=} {arr.ndim=} {arr.shape=}")
mat = torch.tensor([[7, 7], [7, 7]], dtype=torch.float32)
print(f"{mat=} {mat.grad=} {mat.ndim=} {mat.shape=}")
random_tensor = torch.rand(size=(3,4))
print(f"{random_tensor=} {random_tensor.grad=} {random_tensor.ndim=} {random_tensor.shape=}")
zeros = torch.zeros(size=(3,4))
print(f"{zeros=} {zeros.grad=} {zeros.ndim=} {zeros.shape=}")
ones = torch.ones(size=(3,4))
print(f"{ones=} {ones.grad=} {ones.ndim=} {ones.shape=}")
sudoku = torch.arange(1, 10).reshape(3, 3)
print(f"{sudoku=} {sudoku.grad=} {sudoku.ndim=} {sudoku.shape=}")
zeros = torch.zeros_like(sudoku)
print(f"{zeros=} {zeros.grad=} {zeros.ndim=} {zeros.shape=}")
zeros = torch.zeros(sudoku.size())
print(f"{zeros=} {zeros.grad=} {zeros.ndim=} {zeros.shape=} {zeros.size()=}")
float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work
print(f"{float_16_tensor.dtype=}")
# Create a tensor
some_tensor = torch.rand([3, 4])
# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU
print(f"Layout of tensor: {some_tensor.layout}")
some_tensor += 10
print(some_tensor)
some_tensor = torch.multiply(some_tensor, 10)
print(some_tensor)
another_tensor = torch.rand([4, 3])
print(another_tensor)
prod = torch.matmul(some_tensor, another_tensor)
print(prod)
print(f"{prod=}")
prod = some_tensor @ another_tensor
print(prod)
print(f"{prod=}")
vector = torch.tensor([1, 2, 3, 4])
print(vector, vector * vector, vector @ vector)
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

print(torch.matmul(tensor_A, tensor_B.T))

# The operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\nOutput shape: {output.shape}")

# Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
torch.manual_seed(42)
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 
x = tensor_A
output = linear(x)
print("nn.Linear\n")
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")

# Create a tensor
x = torch.arange(0, 100, 10)
print(f"{x=}\n{x.shape=}\n{x.ndim=}\n{x.size()=}\n{x.dtype=}\n{x.device=}\n{x.layout=}")
print(f"Minimum: {x.min()}")
max_val, max_idx = x.max(0)
print(f"Maximum: {max_val}")
print(f"Maximum index: {max_idx}")
print(f"Maximum: {x.max()=}")
print(f"Maximum: {x.max(0)=}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")
y = x.type(torch.float32)
print(f"{y=}\n{y.shape=}\n{y.ndim=}\n{y.size()=}\n{y.dtype=}\n{y.device=}\n{y.layout=}")
# Create a tensor and check its datatype
tensor = torch.arange(0., 1001., 100.)
print(f"{tensor=}")
print(f"{tensor.dtype=}")
# Convert the tensor to another datatype
tensor = tensor.type(torch.int32)
print(f"{tensor=}")
print(f"{tensor.dtype=}")
# Convert the tensor to another datatype
tensor = tensor.type(torch.uint8)
print(f"{tensor=}")
print(f"{tensor.dtype=}")
# Convert the tensor to another datatype
tensor = tensor.type(torch.int8)
print(f"{tensor=}")
print(f"{tensor.dtype=}")
torch.manual_seed(42)

# Create two random tensors
random_tensor_A = torch.rand(3, 4)

torch.manual_seed(42) # set the seed again (try commenting this out and see what happens)
random_tensor_B = torch.rand(3, 4)

print(f"Tensor A:\n{random_tensor_A}\n")
print(f"Tensor B:\n{random_tensor_B}\n")
print(f"Does Tensor A equal Tensor B? (anywhere)")
print("Are they equal:", random_tensor_A == random_tensor_B)

tensor_A = torch.arange(0, 12, dtype=torch.float32)
print(f"{tensor_A=}")
tensor_A = tensor_A.reshape(3, 4)
print(f"{tensor_A=}")
tensor_C = torch.nn.LayerNorm(4)(tensor_A)
print(f"{tensor_C=}")

print(f"{torch.mean(tensor_A, dim=0)=}")
print(f"{torch.mean(tensor_A, dim=1)=}")
print(f"{torch.mean(tensor_A, dim=1, keepdim=True)=}")
print(f"{torch.var(tensor_A, dim=1, keepdim=True)=}")

tensor_mean = torch.mean(tensor_A, dim=1, keepdim=True)
tensor_var = torch.var(tensor_A, dim=1, correction=0, keepdim=True)
tensor_std = torch.std(tensor_A, dim=1, correction=0, keepdim=True)

tensor_CC = (tensor_A - tensor_mean) / torch.sqrt(tensor_var)
tensor_CC = (tensor_A - tensor_mean) / tensor_std
print(f"{tensor_C=}")
print(f"{tensor_CC=}")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")    

print(f"{device=}")

x = torch.tensor([1, 2, 3], device=device)
x_cpu = torch.tensor([1, 2, 3])

print(f"{x=}")
print(f"{x_cpu=}")
x_cpu = x_cpu.to(device)
print(f"{x_cpu=}")
y = x + x_cpu
print(f"{y=}")
y = y.to("cpu")
print(f"{y.numpy()=}")

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) # data
y = weight * X + bias # labels (want model to learn from data to predict these)

print(f"{X[:10]=}, {y[:10]=}")

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(f"{len(X_train)=}, {len(y_train)=}, {len(X_test)=}, {len(y_test)=}")

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, 
                                      out_features=1)

    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

model_0 = LinearRegressionModel()
print(f"{model_0=}, {model_0.state_dict()=}")

model_1 = torch.nn.Sequential(
    nn.Linear(in_features=1,
              out_features=1))

print(f"{model_1=}, {model_1.state_dict()=}")

# Create loss function
loss_fn = nn.L1Loss()

# Create optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), # optimize newly created model's parameters
                            lr=0.001)
optimizer_0 = torch.optim.SGD(params=model_0.parameters(), # optimize newly created model's parameters
                            lr=0.001)

torch.manual_seed(42)

# Set the number of epochs 
epochs = 1001

# Put data on the available device
# Without this, an error will happen (not all data on target device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# Put model on the available device
# With this, an error will happen (the model is not on target device)
model_1 = model_1.to(device)
model_0 = model_0.to(device)

for epoch in range(epochs):
    ### Training
    model_1.train() # train mode is on by default after construction
    model_0.train() # train mode is on by default after construction

    # 1. Forward pass
    y_pred = model_1(X_train)
    y_pred_0 = model_0(X_train)
    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)
    loss_0 = loss_fn(y_pred_0, y_train)

    # 3. Zero grad optimizer
    optimizer.zero_grad()
    optimizer_0.zero_grad()

    # 4. Loss backward
    loss.backward()
    loss_0.backward()

    # 5. Step the optimizer
    optimizer.step()
    optimizer_0.step()

    ### Testing
    model_1.eval() # put the model in evaluation mode for testing (inference)
    model_0.eval() # put the model in evaluation mode for testing (inference)
    # 1. Forward pass
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_pred_0 = model_0(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)
        test_loss_0 = loss_fn(test_pred_0, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")                            
        print(f"{model_0.state_dict()=}")
        print(f"{model_1.state_dict()=}")
        # print(f"{model_1.state_dict()['0.weight']=}")
        # print(f"{model_1.state_dict()['0.bias']=}")
        # print(f"{list(model_1.parameters())=}")