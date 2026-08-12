"""add more data
reduce the complexity of NN architecture
use regularization =>in adam /optimiser add a weight decay
L1

Adds λ∑∣w∣ → encourages sparse weights.

L2

Adds λ∑w
2
 → discourages large weights.
dropout  =>prob of a neuron being dropped
batchnormalisation => linear layerbatchnormalisation =>relu
data augmentation
early stopping
"""

1. Dropout

Dropout randomly disables neurons during training.

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(p=0.5),   # 50% dropout
    nn.Linear(64, 10)
)

Important: Dropout is active during training and automatically disabled during evaluation when you use:

model.train()   # Dropout ON
model.eval()  # Dropout OFF
2. Batch Normalization

For a fully connected network:

import torch.nn as nn
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),

    nn.Linear(64, 10)
)

A common pattern is:

Linear → BatchNorm → ReLU

For a CNN, you would typically use:

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3),
    nn.BatchNorm2d(32),
    nn.ReLU()
)

Again, remember:

model.train()  # BatchNorm uses batch statistics
model.eval()   # BatchNorm uses running statistics
3. L2 Regularization / Weight Decay

A very common way to apply L2-style regularization is through the optimizer.

import torch
import torch.nn as nn
model = nn.Linear(128, 10)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

Here:
weight_decay=1e-4 controls the strength of weight decay.
Higher value → stronger regularization.


4. L1 Regularization
L1 regularization can be added explicitly to the loss.

import torch
criterion = nn.CrossEntropyLoss()
outputs = model(x)
loss = criterion(outputs, y)
l1_lambda = 1e-5
l1_penalty = sum(
    param.abs().sum()
    for param in model.parameters()
)
loss = loss + l1_lambda * l1_penalty
loss.backward()
optimizer.step()

The important part is:
loss = loss + l1_lambda * l1_penalty
Mathematically:
Ltotal=L+λi∑∣wi∣
