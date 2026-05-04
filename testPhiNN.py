import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


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




# ------------------------------------------------------------
# Compare torch with MCMC
# ------------------------------------------------------------


def acf_r_style(x, acf_lags=8):
    """
    Approximate R's acf(x, lag.max = acf_lags, plot = FALSE)$acf[-1]
    for a univariate series x.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    denom = np.sum(x ** 2)

    if denom <= 1e-14:
        return np.zeros(acf_lags, dtype=np.float64)

    ac = np.empty(acf_lags, dtype=np.float64)

    for lag in range(1, acf_lags + 1):
        ac[lag - 1] = np.sum(x[:-lag] * x[lag:]) / denom

    return ac


def summary_stats_python(y, k=1e-12, acf_lags=8):
    """
    Python version of your R summary_stats():

    summary_stats <- function(y, k = 1e-12, acf_lags = 8L) {
      y <- y - mean(y)
      x <- log(y^2 + k)
      ac <- as.numeric(acf(x, lag.max = acf_lags, plot = FALSE)$acf)[-1]
      qs <- quantile(x, probs = c(0.05, 0.5, 0.95))
      c(mean(x), var(x), ac, qs)
    }
    """
    y = np.asarray(y, dtype=np.float64)

    if len(y) <= acf_lags:
        raise ValueError("length(y) must be larger than acf_lags")

    y = y - np.mean(y)
    x = np.log(y ** 2 + k)

    ac = acf_r_style(x, acf_lags=acf_lags)

    qs = np.quantile(x, [0.05, 0.5, 0.95])

    out = np.concatenate([
        np.array([
            np.mean(x),
            np.var(x, ddof=1)   # R's var() uses sample variance
        ]),
        ac,
        qs
    ])

    return out.astype(np.float32)




# ------------------------------------------------------------
# Extract one standardized summary vector z for each MCMC dataset
# ------------------------------------------------------------

summary = pd.read_csv("mcmc_summary.csv")
simulated_data = pd.read_csv("simulated_data.csv")
phi_draws = pd.read_csv("phi_posterior_draws.csv")
psi_draws = pd.read_csv("psi_posterior_draws.csv")


dataset_indices = sorted(simulated_data["index"].unique())

z_list = []
phi_true_list = []
psi_true_list = []

for idx in dataset_indices:
    df_i = simulated_data[simulated_data["index"] == idx].sort_values("time")

    y_i = df_i["y"].to_numpy()

    z_i = summary_stats_python(
        y_i,
        k=1e-12,
        acf_lags=8
    )

    z_list.append(z_i)
    phi_true_list.append(df_i["phi_true"].iloc[0])
    psi_true_list.append(df_i["psi_true"].iloc[0])

z_mcmc_data = np.vstack(z_list).astype(np.float32)

phi_true = np.asarray(phi_true_list, dtype=np.float32)
psi_true = np.asarray(psi_true_list, dtype=np.float32)

# Standardize using the training-set mean and sd saved in the checkpoint
z_mcmc_scaled = (z_mcmc_data - z_mean) / z_std

z_mcmc_tensor = torch.tensor(z_mcmc_scaled, dtype=torch.float32)




from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

alpha = 0.05
zcrit = norm.ppf(1 - alpha / 2)

# ------------------------------------------------------------
# Amortized posterior predictions for psi
# ------------------------------------------------------------

model.eval()

with torch.no_grad():
    psi_mu_torch, psi_var_torch = model(z_mcmc_tensor)

psi_mu_torch = psi_mu_torch.numpy().reshape(-1)
psi_var_torch = psi_var_torch.numpy().reshape(-1)
psi_sd_torch = np.sqrt(psi_var_torch)

# CI on psi scale
psi_ci_lower_torch = psi_mu_torch - zcrit * psi_sd_torch
psi_ci_upper_torch = psi_mu_torch + zcrit * psi_sd_torch

# Convert back to phi scale
phi_mean_torch = np.tanh(psi_mu_torch / 2)
phi_ci_lower_torch = np.tanh(psi_ci_lower_torch / 2)
phi_ci_upper_torch = np.tanh(psi_ci_upper_torch / 2)


# ------------------------------------------------------------
# Extract MCMC summaries
# ------------------------------------------------------------

summary = summary.sort_values("index").reset_index(drop=True)

phi_mean_mcmc = summary["phi_mean"].to_numpy()
phi_ci_lower_mcmc = summary["phi_ci_lower"].to_numpy()
phi_ci_upper_mcmc = summary["phi_ci_upper"].to_numpy()

psi_mean_mcmc = summary["psi_mean"].to_numpy()
psi_var_mcmc = summary["psi_var"].to_numpy()
psi_sd_mcmc = np.sqrt(psi_var_mcmc)


# ------------------------------------------------------------
# Comparison table
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "phi_true": phi_true,
    "psi_true": psi_true,

    "phi_mean_MCMC": phi_mean_mcmc,
    "phi_ci_lower_MCMC": phi_ci_lower_mcmc,
    "phi_ci_upper_MCMC": phi_ci_upper_mcmc,

    "phi_mean_torch": phi_mean_torch,
    "phi_ci_lower_torch": phi_ci_lower_torch,
    "phi_ci_upper_torch": phi_ci_upper_torch,

    "psi_mean_MCMC": psi_mean_mcmc,
    "psi_sd_MCMC": psi_sd_mcmc,

    "psi_mean_torch": psi_mu_torch,
    "psi_sd_torch": psi_sd_torch,
})

#print("\nComparison of MCMC and amortized posterior estimates:\n")
#print(comparison.to_string(index=False))


# ------------------------------------------------------------
# Mean log scores evaluated at true psi
# ------------------------------------------------------------

eps = 1e-8

log_score_torch = norm.logpdf(
    psi_true,
    loc=psi_mu_torch,
    scale=np.sqrt(psi_var_torch + eps)
)

log_score_mcmc_normal = norm.logpdf(
    psi_true,
    loc=psi_mean_mcmc,
    scale=np.sqrt(psi_var_mcmc + eps)
)

print("\nMean log scores evaluated at true psi:")
print("Torch amortized method:", np.mean(log_score_torch))
print("MCMC normal approximation:", np.mean(log_score_mcmc_normal))


# ------------------------------------------------------------
# Plot CIs on phi scale
# ------------------------------------------------------------

x = np.arange(len(phi_true))
offset = 0.13

plt.figure(figsize=(11, 6))

plt.errorbar(
    x - offset,
    phi_mean_mcmc,
    yerr=[
        phi_mean_mcmc - phi_ci_lower_mcmc,
        phi_ci_upper_mcmc - phi_mean_mcmc
    ],
    fmt="o",
    capsize=15,
    label="MCMC"
)

plt.errorbar(
    x + offset,
    phi_mean_torch,
    yerr=[
        phi_mean_torch - phi_ci_lower_torch,
        phi_ci_upper_torch - phi_mean_torch
    ],
    fmt="o",
    capsize=15,
    label="Amortized"
)

plt.scatter(
    x,
    phi_true,
    marker="x",
    s=80,
    label="True phi"
)

#Increase the font size of x-axis labels and rotate them by 45 degrees
plt.xticks(x, [f"{v:.3f}" for v in phi_true], rotation=45, fontsize=16)
plt.xlabel("True phi", fontsize=16)
plt.ylabel("Posterior estimate and 95% CI on phi scale", fontsize=16)
plt.title("MCMC vs amortized posterior estimates", fontsize=16)
plt.yticks(fontsize=16)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

comparison.to_csv("mcmc_vs_amortized_comparison.csv", index=False)
print("\nSaved comparison table to mcmc_vs_amortized_comparison.csv")