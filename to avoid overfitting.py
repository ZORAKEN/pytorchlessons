# Practical Techniques to Reduce Overfitting in Neural Networks

Overfitting occurs when a neural network performs very well on the training data but poorly on unseen validation or test data.

Common ways to reduce overfitting include:

* Adding more training data
* Data augmentation
* Reducing neural-network complexity
* L1/L2 regularization
* Weight decay
* Dropout
* Batch normalization
* Early stopping

---

## 1. Add More Data

One of the most effective ways to reduce overfitting is to provide the model with more representative training data.

More data gives the network more examples from which to learn general patterns instead of memorizing the training set.

```text
Small dataset
     ↓
Model memorizes examples
     ↓
Overfitting

More representative data
     ↓
Model learns general patterns
     ↓
Better generalization
```

### Practical note

If collecting more real data is difficult, **data augmentation** can create additional training examples from existing data.

---

## 2. Reduce the Complexity of the Neural Network

A model with too many parameters can memorize the training data.

For example:

```text
Too complex:
Input → 1024 → 1024 → 512 → 256 → Output

Simpler:
Input → 256 → 128 → Output
```

Reducing the number of layers or neurons can help the model generalize better.

### Practical note

If the model is overfitting, consider:

* Fewer layers
* Fewer neurons per layer
* Smaller convolutional filters/channels
* Removing unnecessary components

However, reducing the model too much can cause **underfitting**.

---

# 3. Regularization

Regularization adds a penalty to the learning objective to discourage overly complex models.

The general idea is:

[
L_{total} = L_{data} + L_{regularization}
]

where:

* (L_{data}) = original training loss
* (L_{regularization}) = penalty on the model parameters

Two common forms are **L1** and **L2** regularization.

---

## 3.1 L1 Regularization

L1 regularization adds the absolute values of the weights to the loss:

[
L_{total}
=========

L + \lambda\sum_i |w_i|
]

where:

* (w_i) = model weights
* (\lambda) = regularization strength

### Main effect

L1 regularization encourages some weights to become exactly or approximately zero.

Therefore, it can produce a **sparse model**.

```text
Before:

[0.8, 0.03, -0.7, 0.01, 0.5]

After L1:

[0.8, 0, -0.7, 0, 0.5]
```

### PyTorch example

L1 regularization is commonly added explicitly to the loss:

```python
import torch
import torch.nn as nn

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
```

The key operation is:

```python
loss = loss + l1_lambda * l1_penalty
```

### Practical note

Use L1 when sparsity or feature selection is desirable. The value of `l1_lambda` should be tuned; a value that is too large can cause underfitting.

---

# 3.2 L2 Regularization

L2 regularization adds the squared weights to the loss:

[
L_{total}
=========

L + \lambda\sum_i w_i^2
]

L2 discourages the model from developing very large weights.

### Main effect

```text
Very large weights
       ↓
L2 penalty
       ↓
Smaller, more controlled weights
       ↓
Better generalization
```

In modern PyTorch code, L2-style regularization is often implemented using **weight decay** in the optimizer.

---

# 4. Weight Decay

Weight decay penalizes large model weights during optimization.

For example, using AdamW:

```python
import torch

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
```

The important parameter is:

```python
weight_decay=1e-4
```

Increasing the value generally increases the regularization strength.

### Practical starting point

For example:

```python
weight_decay = 1e-4
```

can be a reasonable starting point, but the appropriate value depends on the dataset, architecture, and optimizer.

### Important distinction

L2 regularization and weight decay are closely related, but they are not mathematically identical for every optimizer.

For adaptive optimizers such as **Adam**, using **AdamW** provides decoupled weight decay and is generally preferred when weight decay is intended.

---

# 5. Dropout

Dropout is a regularization technique that randomly disables neurons during training.

For example, with:

```python
nn.Dropout(p=0.5)
```

each neuron has a probability of `0.5` of being dropped during a training forward pass.

```text
Before Dropout:

● ● ● ● ● ●

After Dropout:

● ✕ ● ● ✕ ●
```

The dropped neurons are selected randomly, so a different subset may be disabled during each iteration.

### PyTorch example

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(64, 10)
)
```

### Training vs evaluation

Dropout is active during training:

```python
model.train()
```

and disabled during evaluation:

```python
model.eval()
```

For example:

```python
model.train()

# Dropout is active
output = model(x)
```

During evaluation:

```python
model.eval()

# Dropout is disabled
with torch.no_grad():
    output = model(x)
```

### Practical notes

* `p=0.5` means a 50% dropout probability.
* `p=0.2` means a 20% dropout probability.
* Too much dropout can cause underfitting.
* Dropout is often useful in fully connected layers.
* It is not always necessary when other regularization methods are already effective.

---

# 6. Batch Normalization

Batch Normalization (BatchNorm) normalizes activations during training.

For a fully connected network, a common pattern is:

```text
Linear → BatchNorm → ReLU
```

### PyTorch example

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),

    nn.Linear(64, 10)
)
```

Here:

```python
nn.Linear(128, 64)
```

produces 64 features, so BatchNorm uses:

```python
nn.BatchNorm1d(64)
```

### CNN example

For convolutional neural networks:

```python
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3),
    nn.BatchNorm2d(32),
    nn.ReLU()
)
```

### Training vs evaluation

BatchNorm behaves differently during training and evaluation.

During training:

```python
model.train()
```

BatchNorm uses statistics calculated from the current mini-batch and updates its running statistics.

During evaluation:

```python
model.eval()
```

BatchNorm uses its stored running statistics.

### Practical note

BatchNorm is primarily used to improve **training stability and optimization**. Although it can have a regularizing effect, it should not be thought of as simply another form of regularization like L1, L2, or dropout.

---

# 7. Data Augmentation

Data augmentation creates modified versions of training examples.

For image classification, common techniques include:

* Random cropping
* Horizontal flipping
* Rotation
* Scaling
* Translation
* Color changes
* Random erasing

For example:

```text
Original image
      ↓
 ┌────┼─────┬──────┐
 ↓    ↓     ↓      ↓
Crop Flip Rotate  Color change
```

The model sees more variation and is encouraged to learn features that generalize.

### PyTorch example

Using `torchvision`:

```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
```

### Important practical rule

Apply augmentation to the **training data**, not normally to the validation/test data.

The validation and test sets should represent the actual evaluation distribution.

---

# 8. Early Stopping

Early stopping prevents the model from continuing to train after its validation performance starts getting worse.

Example:

```text
Epoch       Validation Loss

1           0.80
2           0.65
3           0.52
4           0.45
5           0.42
6           0.41   ← Best
7           0.43
8           0.46
9           0.49
```

The model should preferably be restored to the checkpoint from epoch 6.

### Patience

A common approach is to use a patience value.

```text
patience = 3
```

This means training can continue for three epochs without improvement before stopping.

### Practical implementation idea

```python
best_val_loss = float("inf")
patience = 3
epochs_without_improvement = 0

for epoch in range(num_epochs):

    # Training
    model.train()

    # ... training code ...

    # Validation
    model.eval()

    # ... validation code ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping")
        break
```

---

# 9. Putting Everything Together

A practical neural network might combine several of these techniques:

```python
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


model = Model()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
```

This example uses:

* **BatchNorm** → helps stabilize training
* **Dropout** → reduces over-reliance on individual neurons
* **AdamW + weight decay** → discourages overly large weights

---

# 10. Quick Reference

| Technique               | Main purpose             | Practical implementation              |
| ----------------------- | ------------------------ | ------------------------------------- |
| More data               | Improve generalization   | Collect more representative examples  |
| Reduce model complexity | Prevent memorization     | Fewer layers/neurons                  |
| L1                      | Encourage sparsity       | Add `λ * abs(weights)` to loss        |
| L2                      | Discourage large weights | Commonly implemented via weight decay |
| Weight decay            | Control weight magnitude | `AdamW(..., weight_decay=...)`        |
| Dropout                 | Reduce overfitting       | `nn.Dropout(p=0.3)`                   |
| BatchNorm               | Stabilize optimization   | `BatchNorm → ReLU`                    |
| Data augmentation       | Increase input variation | Random crops/flips/etc.               |
| Early stopping          | Stop before overfitting  | Monitor validation loss               |

---


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
