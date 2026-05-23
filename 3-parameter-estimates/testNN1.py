import argparse
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
import pandas as pd
import simulateData as sim
import torch
import torch.nn as nn
from trainLiveCNN import LiveSVPosteriorTCN
from trainSummaryNN import SVPosteriorNN


# TODO: Need to add layer_norm and model_class to summaryNN checkpoint


HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "stochvolMCMC.R"

DEFAULT_SUMMARY_CHECKPOINT_PATH = "sv_posterior_nn_1M_ARIMA_finance.pt"
DEFAULT_OUTPUT_DIR = "nn_vs_mcmc_comparison"
DEFAULT_ALPHA = 0.05
DEFAULT_K = 1e-12

PARAMETER_NAMES = ("mu", "phi", "sigma")
TRANSFORMED_TARGET_NAMES = ["mu", "psi", "log_sigma"]
DEFAULT_BASELINE = {
    "mu": -9.0,
    "phi": 0.98,
    "sigma": 0.20,
}
DEFAULT_SWEEP_DELTAS = {
    "mu": 2.0,
    "phi": 0.015,
    "sigma": 0.10,
}


def find_rscript():
    """
    Try to find Rscript in the system. First check PATH,
    then look in common installation directories on Windows.
    """
    rscript = shutil.which("Rscript")

    if rscript is not None:
        return rscript

    program_files = Path("C:/Program Files")
    candidates = sorted(
        program_files.glob("R/R-*/bin/Rscript.exe"),
        reverse=True,
    )

    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(
        "Could not find Rscript. Add R's bin folder to PATH, or install R."
    )


def activation_from_checkpoint(checkpoint):
    """
    Get the activation function class from the checkpoint. Default to ReLU if not specified.
    """
    activation_name = checkpoint.get("activation", -1)
    if activation_name == -1:
        print("Warning: checkpoint does not specify activation. Defaulting to ReLU.")
        activation_name = "ReLU"
    activation = getattr(nn, activation_name, None)

    if activation is None:
        raise ValueError(f"Unknown activation in checkpoint: {activation_name}")

    return activation


def build_summary_nn(checkpoint, state_dict, device):
    activation = activation_from_checkpoint(checkpoint)

    model = SVPosteriorNN(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims_shared_trunk=tuple(checkpoint["hidden_dims_shared_trunk"]),
            hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
            activation=activation,
            min_var=float(checkpoint.get("min_var", 1e-12)),
            dropout=float(checkpoint.get("dropout", 0.0)),
            layer_norm=checkpoint["layer_norm"],
        ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()

    return model


def build_cnn(checkpoint, state_dict, device):
    activation = activation_from_checkpoint(checkpoint)

    model = LiveSVPosteriorTCN(
        sequence_length=int(checkpoint["sequence_length"]),
        tcn_channels=tuple(checkpoint["tcn_channels"]),
        kernel_size=int(checkpoint["kernel_size"]),
        dilations=tuple(checkpoint["dilations"]),
        hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
        activation=activation,
        dropout=float(checkpoint.get("dropout", 0.0)),
        use_batch_norm=bool(checkpoint.get("use_batch_norm", True)),
        min_var=float(checkpoint.get("min_var", 1e-12)),
        input_mean=float(checkpoint.get("input_mean", 0.0)),
        input_std=float(checkpoint.get("input_std", 1.0)),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.no_grad()
def predict_transformed_gaussian(model, x, device, batch_size=4096):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    model.eval()

    means = []
    variances = []

    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        x_batch = torch.from_numpy(x[start:stop]).float().to(device)
        mean_batch, var_batch = model(x_batch)

        means.append(mean_batch.detach().cpu().float().numpy())
        variances.append(var_batch.detach().cpu().float().numpy())

    return np.vstack(means), np.vstack(variances)


def transformed_gaussian_to_parameter_frame(
    transformed_mean,
    transformed_var,
    alpha=DEFAULT_ALPHA,
    prefix="model",
):
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    transformed_mean = np.asarray(transformed_mean, dtype=np.float64)
    transformed_var = np.asarray(transformed_var, dtype=np.float64)

    if transformed_mean.shape != transformed_var.shape:
        raise ValueError("transformed_mean and transformed_var must have the same shape.")

    if transformed_mean.ndim != 2 or transformed_mean.shape[1] != 3:
        raise ValueError("transformed_mean and transformed_var must have shape (m, 3).")

    if np.any(transformed_var < 0.0):
        raise ValueError("transformed_var must be non-negative.")

    zcrit = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    transformed_sd = np.sqrt(transformed_var)

    mu_mean = transformed_mean[:, 0]
    mu_sd = transformed_sd[:, 0]
    mu_lower = mu_mean - zcrit * mu_sd
    mu_upper = mu_mean + zcrit * mu_sd

    psi_mean = transformed_mean[:, 1]
    psi_sd = transformed_sd[:, 1]
    phi_median = np.tanh(psi_mean / 2.0)
    phi_lower = np.tanh((psi_mean - zcrit * psi_sd) / 2.0)
    phi_upper = np.tanh((psi_mean + zcrit * psi_sd) / 2.0)

    log_sigma_mean = transformed_mean[:, 2]
    log_sigma_sd = transformed_sd[:, 2]
    sigma_median = np.exp(log_sigma_mean)
    sigma_lower = np.exp(log_sigma_mean - zcrit * log_sigma_sd)
    sigma_upper = np.exp(log_sigma_mean + zcrit * log_sigma_sd)

    return pd.DataFrame({
        f"{prefix}_alpha": alpha,
        f"{prefix}_credible_level": 1.0 - alpha,
        f"{prefix}_mu_median": mu_mean,
        f"{prefix}_mu_ci_lower": mu_lower,
        f"{prefix}_mu_ci_upper": mu_upper,
        f"{prefix}_mu_mean_transformed": mu_mean,
        f"{prefix}_mu_sd_transformed": mu_sd,
        f"{prefix}_phi_median": phi_median,
        f"{prefix}_phi_ci_lower": phi_lower,
        f"{prefix}_phi_ci_upper": phi_upper,
        f"{prefix}_psi_mean": psi_mean,
        f"{prefix}_psi_sd": psi_sd,
        f"{prefix}_sigma_median": sigma_median,
        f"{prefix}_sigma_ci_lower": sigma_lower,
        f"{prefix}_sigma_ci_upper": sigma_upper,
        f"{prefix}_log_sigma_mean": log_sigma_mean,
        f"{prefix}_log_sigma_sd": log_sigma_sd,
    })


def run_stochvol_mcmc(
    y,
    prior="default",
    draws=2000,
    burnin=500,
    thinpara=1,
    alpha=DEFAULT_ALPHA,
):
    
    if draws < 1:
        raise ValueError("draws must be at least 1.")

    if burnin < 0:
        raise ValueError("burnin must be non-negative.")

    if thinpara < 1:
        raise ValueError("thinpara must be at least 1.")

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    if not R_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find MCMC R script: {R_SCRIPT}")

    prior_constants = sim.get_stochvol_prior_constants(prior)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "y.csv"
        output_path = tmpdir / "sv_result.csv"

        np.savetxt(input_path, y, delimiter=",")

        subprocess.run(
            [
                find_rscript(),
                str(R_SCRIPT),
                str(input_path),
                str(output_path),
                str(int(draws)),
                str(int(burnin)),
                str(int(thinpara)),
                str(float(prior_constants.mu_mean)),
                str(float(prior_constants.mu_sd)),
                str(float(prior_constants.phi_a0)),
                str(float(prior_constants.phi_b0)),
                str(float(prior_constants.Bsigma)),
                str(float(alpha)),
            ],
            check=True,
        )

        result = pd.read_csv(output_path)

    expected_rows = y.shape[0]

    if len(result) != expected_rows:
        raise RuntimeError(
            f"stochvol MCMC returned {len(result)} rows for {expected_rows} series."
        )

    return result


def simulate_single_parameter_sweep_datasets(
    baseline=None,
    sweeps=None,
    sweep_deltas=None,
    sweep_size=9,
    n=253,
    rng=None,
    random_init=True,
):
    """
    Return one simulated dataset per parameter sweep.

    datasets[parameter] contains simulations where only that parameter changes
    across rows; the other SV parameters are held fixed at the baseline.
    """
    baseline = DEFAULT_BASELINE | (baseline or {})
    sweep_deltas = DEFAULT_SWEEP_DELTAS | (sweep_deltas or {})

    if rng is None:
        rng = np.random.default_rng()

    if sweeps is None:
        sweeps = {
            parameter: np.linspace(
                baseline[parameter] - sweep_deltas[parameter],
                baseline[parameter] + sweep_deltas[parameter],
                sweep_size,
            )
            for parameter in PARAMETER_NAMES
        }

    datasets = {}
    for swept_parameter in PARAMETER_NAMES:
        m = len(sweeps[swept_parameter])
        params = {
            parameter: np.full(m, baseline[parameter])
            for parameter in PARAMETER_NAMES
        }
        params[swept_parameter] = np.asarray(sweeps[swept_parameter])

        datasets[swept_parameter] = sim.simulate_sv_chunk(
            mu=params["mu"],
            phi=params["phi"],
            sigma=params["sigma"],
            n=n,
            rng=rng,
            random_init=random_init,
        )

    return datasets, sweeps, baseline


def main():


    
    n = 253
    rng = np.random.default_rng(12345)
    datasets, sweeps, baseline = simulate_single_parameter_sweep_datasets(n=n, rng=rng)





if __name__ == "__main__":
    main()
