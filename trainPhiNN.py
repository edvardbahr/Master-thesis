import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMLP(nn.Module):

    #the class structure of the GaussianMLP is as follows:
    # - It inherits from nn.Module, which is the base class for all neural network modules
    def __init__(self, input_dim, hidden_dims=(64, 64), activation=nn.ReLU):
        super().__init__()

        layers = []
        dims = [input_dim, *hidden_dims]

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))

            # nn.Linear applies a linear transformation to the incoming data: y = xA^T + b.
            # Given the parameters d_in and d_out, it returns a "function" that takes
            # an input of shape (batch_size, d_in) and produces an output of shape (batch_size, d_out). 

            layers.append(activation())

        self.net = nn.Sequential(*layers)
        # nn.Sequential is a container module that sequences the layers in the order they are added.
        self.mu_head = nn.Linear(dims[-1], 1)
        self.var_head = nn.Linear(dims[-1], 1)

    def forward(self, z):
        h = self.net(z)

        #the structure of h 

        mean = self.mu_head(h)

        # GaussianNLLLoss wants variance, not standard deviation.
        var = F.softplus(self.var_head(h)) + 1e-6
        # Softplus is a smooth approximation to the ReLU function, defined as softplus(x) = log(1 + exp(x)).

        return mean, var


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

data = np.load("training_data.npz")

z = data["z"].astype(np.float32)
theta = data["theta"].astype(np.float32)

mu_true = theta[:, 0]
phi = theta[:, 1]
sigma = theta[:, 2]

# Transform phi from (-1, 1) to R
eps = 1e-6
phi_clipped = np.clip(phi, -1 + eps, 1 - eps)
psi = 2 * np.arctanh(phi_clipped)

# Standardize z
z_mean = z.mean(axis=0, keepdims=True)
z_std = z.std(axis=0, keepdims=True)
z_std[z_std == 0] = 1.0

z_scaled = (z - z_mean) / z_std


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Train/validation split
# ------------------------------------------------------------

n = z_scaled.shape[0]
idx = np.random.permutation(n)

n_train = int(0.8 * n)

train_idx = idx[:n_train]
val_idx = idx[n_train:]

z_train = torch.tensor(z_scaled[train_idx], dtype=torch.float32)
target_train = torch.tensor(psi[train_idx], dtype=torch.float32).reshape(-1, 1)

z_val = torch.tensor(z_scaled[val_idx], dtype=torch.float32)
target_val = torch.tensor(psi[val_idx], dtype=torch.float32).reshape(-1, 1)


# ------------------------------------------------------------
# Model, loss, optimizer
# ------------------------------------------------------------

hidden_dims = (128, 64, 32)

model = GaussianMLP(
    input_dim=z_scaled.shape[1],
    hidden_dims=hidden_dims
)

loss_fn = nn.GaussianNLLLoss(full=True)
# full=True means that the loss is averaged over all elements in the batch, rather than summed.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------
# Training loop with early stopping
# ------------------------------------------------------------

n_epochs = 1000
patience = 50
min_delta = 1e-4

train_loss_history = []
val_loss_history = []

best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0

for epoch in range(n_epochs):
    # Training step
    model.train()

    optimizer.zero_grad()

    mean_train, var_train = model(z_train)
    train_loss = loss_fn(mean_train, target_train, var_train)

    train_loss.backward()
    optimizer.step()

    # Validation step
    model.eval()

    with torch.no_grad():
        mean_val, var_val = model(z_val)
        val_loss = loss_fn(mean_val, target_val, var_val)

    train_loss_value = train_loss.item()
    val_loss_value = val_loss.item()

    train_loss_history.append(train_loss_value)
    val_loss_history.append(val_loss_value)

    # Check improvement
    if val_loss_value < best_val_loss - min_delta:
        best_val_loss = val_loss_value
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"train loss = {train_loss_value:.4f}, "
            f"val loss = {val_loss_value:.4f}"
        )

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break


if best_state is not None:
    model.load_state_dict(best_state)
else:
    print("Warning: best_state is None, saving current model instead.")

model.eval()

checkpoint = {
    "model_state_dict": model.state_dict(),
    "input_dim": z.shape[1],
    "hidden_dims": hidden_dims,
    "z_mean": z_mean,
    "z_std": z_std,
    "target_transform": "psi = 2 * atanh(phi)"
}

torch.save(checkpoint, "gaussian_mlp_phi.pt")
print("Model saved to gaussian_mlp_phi.pt")



plt.plot(train_loss_history, label="train")
plt.plot(val_loss_history, label="validation")
plt.xlabel("Epoch")
plt.ylabel("Gaussian NLL")
plt.legend()
plt.show()

