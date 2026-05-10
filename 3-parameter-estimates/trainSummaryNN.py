import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Model definition
# ============================================================

def make_mlp(
    input_dim,
    hidden_dims,
    output_dim=None,
    activation=nn.ReLU,
    dropout=0.0,
    layer_norm=False,
):
    layers = []
    d_prev = input_dim

    for d_hidden in hidden_dims:
        layers.append(nn.Linear(d_prev, d_hidden))

        if layer_norm:
            layers.append(nn.LayerNorm(d_hidden))

        layers.append(activation())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        d_prev = d_hidden


    if output_dim is not None:
        layers.append(nn.Linear(d_prev, output_dim))
        d_prev = output_dim

    if len(layers) == 0:
        return nn.Identity(), input_dim

    return nn.Sequential(*layers), d_prev

class SVPosteriorNN(nn.Module):
    """
    Neural network for amortized inference in the standard SV model.

    Input:
        z: summary statistics, shape (batch_size, input_dim)

    Output:
        mean: shape (batch_size, 3)
              columns: [mu_mean, psi_mean, log_sigma_mean]

        var: shape (batch_size, 3)
             columns: [mu_var, psi_var, log_sigma_var]

    where:
        psi = 2 * atanh(phi)
        log_sigma = log(sigma)
    """

    param_names = ("mu", "psi", "log_sigma")

    def __init__(
        self,
        input_dim,
        hidden_dims_shared_trunk=(128, 128),
        hidden_dims_head=(64,),
        activation=nn.ReLU,
        min_var=1e-6,
        dropout=0.0,
        layer_norm=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims_shared_trunk = hidden_dims_shared_trunk
        self.hidden_dims_head = hidden_dims_head
        self.min_var = min_var

        self.shared_trunk, trunk_output_dim = make_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims_shared_trunk,
            output_dim=None,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )

        self.heads = nn.ModuleDict()

        for name in self.param_names:
            head, _ = make_mlp(
                input_dim=trunk_output_dim,
                hidden_dims=hidden_dims_head,
                output_dim=2,
                activation=activation,
                dropout=dropout,
                layer_norm=layer_norm,
            )
            self.heads[name] = head

    def forward(self, z):
        h = self.shared_trunk(z)

        means = []
        variances = []

        for name in self.param_names:
            out = self.heads[name](h)

            mean = out[:, 0:1]    # Preserves shape (batch_size, 1) which is
            raw_var = out[:, 1:2] # important for torch.cat later.

            var = F.softplus(raw_var) + self.min_var

            means.append(mean)
            variances.append(var)

        mean = torch.cat(means, dim=1)
        var = torch.cat(variances, dim=1)

        # The mean and var posterior parameters for each main parameter is returned as we assume a diagonal Gaussian posterior.
        return mean, var  
    

def theta_to_target_numpy(theta, eps=1e-6):
    """
    Converts original SV parameters to transformed training targets.

    theta has shape (n_samples, 3), with columns:
        theta[:, 0] = mu
        theta[:, 1] = phi
        theta[:, 2] = sigma

    Returns target with columns:
        mu
        psi = 2 * atanh(phi)
        log_sigma = log(sigma)
    """
    mu = theta[:, 0]
    phi = theta[:, 1]
    sigma = theta[:, 2]

    phi = np.clip(phi, -1.0 + eps, 1.0 - eps)
    sigma = np.clip(sigma, eps, None)

    psi = 2 * np.arctanh(phi)
    log_sigma = np.log(sigma)

    target = np.column_stack([mu, psi, log_sigma])

    return target.astype(np.float32)


def diagonal_gaussian_nll(mean, var, target):
    """
    Computes the mean negative joint log score under a diagonal Gaussian.

    mean:
        shape (batch_size, 3)

    var:
        shape (batch_size, 3)

    target:
        shape (batch_size, 3)

    Returns:
        scalar loss
    """
    elementwise_nll = F.gaussian_nll_loss(
        input=mean,
        target=target,
        var=var,
        full=True,
        reduction="none",
    )

    # Sum over the three transformed parameters.
    # Then average over the batch.
    loss = elementwise_nll.sum(dim=1).mean()

    return loss



# ============================================================
# Load data
# ============================================================

data = np.load("sv_dataset_1Mill.npz", allow_pickle=True)

Z = data["summaries"].astype(np.float32)
theta = data["params"].astype(np.float32)

print("Z shape:", Z.shape)
print("theta shape:", theta.shape)

# Transform to unconstrained space for NN training.
target = theta_to_target_numpy(theta) 

print("target shape:", target.shape)


# ============================================================
# Train-validation split
# ============================================================

rng = np.random.default_rng(seed=1)

N = len(Z)
indices = rng.permutation(N)

val_fraction = 0.2
n_val = int(val_fraction * N)

val_idx = indices[:n_val]
train_idx = indices[n_val:]

Z_train = Z[train_idx]
Z_val = Z[val_idx]

target_train = target[train_idx]
target_val = target[val_idx]


# ============================================================
# Standardize summary statistics using training set only
# ============================================================

z_mean = Z_train.mean(axis=0, keepdims=True)
z_std = Z_train.std(axis=0, keepdims=True)

# Avoid division by zero if one summary statistic is constant
z_std = np.where(z_std < 1e-8, 1.0, z_std)

Z_train_scaled = (Z_train - z_mean) / z_std
Z_val_scaled = (Z_val - z_mean) / z_std


# ============================================================
# Create PyTorch datasets and dataloaders
# ============================================================

train_dataset = TensorDataset(
    torch.from_numpy(Z_train_scaled),
    torch.from_numpy(target_train)
)

val_dataset = TensorDataset(
    torch.from_numpy(Z_val_scaled),
    torch.from_numpy(target_val)
)

batch_size = 1024

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)


# ============================================================
# Initialize model
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input_dim = Z.shape[1]

model = SVPosteriorNN(
    input_dim=input_dim,
    hidden_dims_shared_trunk=(128, 128),
    hidden_dims_head=(64,64,),
    activation=nn.ReLU,
    min_var=1e-6,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-3)


# ============================================================
# Evaluation helper
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_n = 0

    for z_batch, target_batch in loader:
        z_batch = z_batch.to(device)
        target_batch = target_batch.to(device)

        mean, var = model(z_batch)
        loss = diagonal_gaussian_nll(mean, var, target_batch)

        batch_n = z_batch.shape[0]

        total_loss += loss.item() * batch_n
        total_n += batch_n

    return total_loss / total_n





# ============================================================
# Training loop with early stopping
# ============================================================

n_epochs = 1000
patience = 50
min_delta = 1e-4

train_loss_history = []
val_loss_history = []

best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0

for epoch in range(n_epochs):
    model.train()

    total_train_loss = 0.0
    total_train_n = 0

    for z_batch, target_batch in train_loader:
        z_batch = z_batch.to(device)
        target_batch = target_batch.to(device)

        mean, var = model(z_batch)
        train_loss = diagonal_gaussian_nll(mean, var, target_batch)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        batch_n = z_batch.shape[0]

        total_train_loss += train_loss.item() * batch_n
        total_train_n += batch_n

    train_loss_value = total_train_loss / total_train_n
    val_loss_value = evaluate(model, val_loader, device)

    train_loss_history.append(train_loss_value)
    val_loss_history.append(val_loss_value)

    if val_loss_value < best_val_loss - min_delta:
        best_val_loss = val_loss_value
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:4d}: "
            f"train NLL = {train_loss_value:.4f}, "
            f"val NLL = {val_loss_value:.4f}"
        )

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break


# ============================================================
# Restore best model
# ============================================================

if best_state is not None:
    model.load_state_dict(best_state)
else:
    print("Warning: best_state is None, using current model.")

model.eval()

final_val_loss = evaluate(model, val_loader, device)

print()
print(f"Best validation mean negative joint log score: {best_val_loss:.6f}")
print(f"Final validation mean negative joint log score: {final_val_loss:.6f}")





# ============================================================
# Save checkpoint
# ============================================================

checkpoint = {
    "model_state_dict": model.state_dict(),

    "input_dim": input_dim,
    "hidden_dims_shared_trunk": (128, 128),
    "hidden_dims_head": (64,),

    "z_mean": z_mean.astype(np.float32),
    "z_std": z_std.astype(np.float32),

    "target_names": ["mu", "psi", "log_sigma"],
    "target_transform": {
        "mu": "mu",
        "psi": "2 * atanh(phi)",
        "log_sigma": "log(sigma)",
    },

    "loss": "mean negative joint Gaussian log score, diagonal covariance",
    "best_val_loss": best_val_loss,
    "final_val_loss": final_val_loss,

    "train_loss_history": train_loss_history,
    "val_loss_history": val_loss_history,
}

torch.save(checkpoint, "sv_posterior_nn6464.pt")

print("Model saved to sv_posterior_nn6464.pt")





plt.plot(train_loss_history, label="train")
plt.plot(val_loss_history, label="validation")
plt.xlabel("Epoch")
plt.ylabel("Mean negative joint log score")
plt.legend()
plt.show()