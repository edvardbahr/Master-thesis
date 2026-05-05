# TODO:
# Test mini batches with fresh simulated data (try to implement fast data generation in Python)
# Implement 1D CNN on log(y^2 + k) data which encodes into a shared trunk
# Test out ARMA(1, 1) coefficients in summary_stats()
# If mini batches is a success, do a mathematical analysis in the thesis to justify approach
# Clean up in the scaling and standarization nightmare
# Do some prompt engineering to speed up the briefing phase of GPT


# When creating a joint posterior, assume independence
# Prioritize runtime efficiency and simplicity as we need a baseline model first

"""
I am working on a project in amortized inference where I use a NN to
estimate the joint posterior parameters given some data y. The data
follows a standard time discrete stochastic volatility model and so
the goal is to estimate the parameters of the model (mu, phi, sigma).
The posterior is assumed to be Gaussian and so we have 6 parameters
that must be estimated (we assume independence between the parameters for simplicity):
mu_mean, phi_mean, sigma_mean, mu_var, phi_var, sigma_var.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class summaryNN(nn.Module):

    def __init__(self, input_dim, hidden_dims=(64, 64), activation=nn.ReLU):
        super().__init__()

        layers = []
        dims = [input_dim, *hidden_dims]

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(activation())

        self.net = nn.Sequential(*layers)
        self.mu_mean_head = nn.Linear(dims[-1], 1)
        self.phi_mean_head = nn.Linear(dims[-1], 1)
        self.sigma_mean_head = nn.Linear(dims[-1], 1)

        self.mu_var_head = nn.Linear(dims[-1], 1)
        self.phi_var_head = nn.Linear(dims[-1], 1)
        self.sigma_var_head = nn.Linear(dims[-1], 1)

    def forward(self, z):
        h = self.net(z)

        mu_mean = self.mu_mean_head(h)
        phi_mean = self.phi_mean_head(h)
        sigma_mean = self.sigma_mean_head(h)

        mu_var = F.softplus(self.mu_var_head(h)) + 1e-6
        phi_var = F.softplus(self.phi_var_head(h)) + 1e-6
        sigma_var = F.softplus(self.sigma_var_head(h)) + 1e-6

        return mu_mean, phi_mean, sigma_mean, mu_var, phi_var, sigma_var