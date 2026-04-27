import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64), activation=nn.ReLU):
        super().__init__()

        layers = []
        dims = [input_dim, *hidden_dims]

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(activation())

        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(dims[-1], 1)
        self.var_head = nn.Linear(dims[-1], 1)

    def forward(self, z):
        h = self.net(z)
        mean = self.mu_head(h)
        var = F.softplus(self.var_head(h)) + 1e-6
        return mean, var


# ------------------------------------------------------------
# 1. Load trained model
# ------------------------------------------------------------

checkpoint = torch.load("gaussian_mlp_phi.pt", weights_only=False)

model = GaussianMLP(
    input_dim=checkpoint["input_dim"],
    hidden_dims=checkpoint["hidden_dims"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

z_mean = checkpoint["z_mean"]
z_std = checkpoint["z_std"]


# ------------------------------------------------------------
# 2. Load test data
# ------------------------------------------------------------

test_data = np.load("test_data.npz")

z_test = test_data["z"].astype(np.float32)
theta_test = test_data["theta"].astype(np.float32)

phi_test = theta_test[:, 1]


# ------------------------------------------------------------
# 3. Apply same transform as during training
# ------------------------------------------------------------

eps = 1e-6
phi_clipped = np.clip(phi_test, -1 + eps, 1 - eps)

psi_test = 2 * np.arctanh(phi_clipped)


# ------------------------------------------------------------
# 4. Standardize z using training-set mean and sd
# ------------------------------------------------------------

z_test_scaled = (z_test - z_mean) / z_std


# ------------------------------------------------------------
# 5. Convert to torch tensors
# ------------------------------------------------------------

z_test_tensor = torch.tensor(z_test_scaled, dtype=torch.float32)
psi_test_tensor = torch.tensor(psi_test, dtype=torch.float32).reshape(-1, 1)


# ------------------------------------------------------------
# 6. Compute mean log score
# ------------------------------------------------------------

with torch.no_grad():
    mu_hat, var_hat = model(z_test_tensor)

    sd_hat = torch.sqrt(var_hat)

    dist = torch.distributions.Normal(mu_hat, sd_hat)

    log_scores = dist.log_prob(psi_test_tensor)

    mean_log_score = log_scores.mean()

print("Mean log score:", mean_log_score.item())