import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# import random
# import os
# import time
# import sys

tensor_example = torch.tensor([[1,2,3],[4,5,6]])
print(tensor_example)
print(tensor_example[0])
print(tensor_example[0][:])
print(tensor_example[0, :])
print(tensor_example[:, :1])

print(f"Shape of tensor: {tensor_example.shape}")
print(f"Data type of tensor: {tensor_example.dtype}")

# Creating two tensors
tensor_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
tensor_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)

# Tensor addition
tensor_sum = torch.add(tensor_a, tensor_b)
print(f"Tensor Addition:\n{tensor_sum}")

# Element-wise Multiplication
tensor_product = torch.mul(tensor_a, tensor_b)
print(f"Element-wise Multiplication:\n{tensor_product}")

# Matrix Multiplication
tensor_c = torch.tensor([[1], [2]], dtype=torch.int32) # 2x1 tensor
tensor_matmul = torch.matmul(tensor_a, tensor_c)
print(f"Matrix Multiplication:\n{tensor_matmul}")

# Broadcasted Addition (Tensor + scalar)
tensor_add_scalar = tensor_a + 5
print(f"Broadcasted Addition (Adding scalar value):\n{tensor_add_scalar}")

# Broadcasted Addition between tensors of different shapes (same as torch.add)
broadcasted_sum = tensor_a + tensor_c
print(f"Broadcasted Addition:\n{broadcasted_sum}")

# Broadcasted Multiplication between tensors of different shapes (same as torch.mul)
broadcasted_mul = tensor_a * tensor_c
print(f"Broadcasted Multiplication:\n{broadcasted_mul}")