"""Credible-interval sweeps and SBC for the five-parameter SV-GHST TCN.

The default settings mirror the standardized configurations in
``five_param_estimates/test_NN_models.py``:

* 95% credible intervals over ten values per parameter;
* 20,000 stochvol draws, 500 burn-in draws, and no thinning;
* 5,000 simulations for the calibration experiment.

The stochvol comparison is available only for ``mu``, ``phi``, and ``s``.
The TCN estimates all five parameters, including GHST skewness ``r`` and
degrees of freedom ``nu``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
FIVE_PARAMETER_DIR = PROJECT_DIR / "five_param_estimates"
if str(FIVE_PARAMETER_DIR) not in sys.path:
    sys.path.insert(0, str(FIVE_PARAMETER_DIR))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sim_5_param_data as sim
from R_to_py_interface import run_stochvol_mcmc, validate_series_matrix
from train_live_CNN import (
    SVGHST_TARGET_NAMES,
    SVPosteriorTCN,
    theta_to_target_numpy,
)


ALPHA = 0.05
SEQUENCE_LENGTH = 253 * 10
MCMC_DRAWS = 20_000
MCMC_BURNIN = 500
MCMC_THINPARA = 1
MCMC_MAX_CORES = -2
CI_SWEEP_SIZE = 10
CI_SEED = 2
SBC_SIMULATIONS = 5_000
SBC_SEED = 3
SBC_BINS = 50
PREDICTION_BATCH_SIZE = 128
POSTERIOR_VAR_EPS = 1e-12

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "weights"
    / "svghst_posterior_tcn_live_default_n2530_multiscale_topk.pt"
)
DEFAULT_OUTPUT_DIR = HERE / "five_param_tcn_results"

PARAMETERS = ("mu", "phi", "s", "r", "nu")
PLOT_PARAMETERS = ("phi", "s", "r", "nu")
TRANSFORMED_PARAMETERS = tuple(SVGHST_TARGET_NAMES)
MCMC_PARAMETERS = ("mu", "phi", "s")
METHODS = ("stochvol", "TCN")

BASELINE = {
    "mu": -9.0,
    "phi": 0.95,
    "s": 0.25,
    "r": 0.50,
    "nu": 15.0,
}

# These are the same ranges used by test.TCN.py. The leading three also match
# the standardized ranges in test_NN_models.py.
SWEEPS = {
    "mu": np.linspace(-12.0, -6.0, CI_SWEEP_SIZE),
    "phi": np.linspace(0.905, 0.995, CI_SWEEP_SIZE),
    "s": np.linspace(0.05, 0.45, CI_SWEEP_SIZE),
    "r": np.linspace(0.20, 0.80, CI_SWEEP_SIZE),
    "nu": np.linspace(8.0, 22.0, CI_SWEEP_SIZE),
}

COLORS = {
    "stochvol": "#0000ff",
    "TCN": "#ff0000",
}
SBC_COLOR = "#4C78A8"

MARKERS = {
    "stochvol": "o",
    "TCN": "^",
}

PARAMETER_LABELS = {
    "mu": r"$\mu$",
    "phi": r"$\phi$",
    "s": r"$\sigma$",
    "r": r"$r$",
    "nu": r"$\nu$",
}

TRANSFORMED_PARAMETER_LABELS = {
    "mu": r"$\mu$",
    "psi": r"$\psi$",
    "log_s": r"$\log s$",
    "logit_r": r"$\operatorname{logit}(r)$",
    "log_nu": r"$\log(\nu-\nu_{\min})$",
}


@dataclass
class LoadedTCN:
    """A reconstructed TCN and the metadata needed for preprocessing."""

    model: nn.Module
    checkpoint: dict
    device: torch.device


def torch_load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load checkpoints under both new and older PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_tcn_model(
    checkpoint_path: Path = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> LoadedTCN:
    """Load and validate the full five-parameter TCN checkpoint."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TCN checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    required_keys = (
        "model_class",
        "model_state_dict",
        "sequence_length",
        "tcn_channels",
        "dilations",
        "hidden_dims_head",
        "activation",
        "min_var",
        "input_mean",
        "input_std",
        "target_names",
        "k",
    )
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(
            f"{checkpoint_path} is missing checkpoint keys: "
            + ", ".join(missing_keys)
        )

    if checkpoint["model_class"] != "SVPosteriorTCN":
        raise ValueError(
            f"Expected an SVPosteriorTCN checkpoint, got "
            f"{checkpoint['model_class']!r}."
        )

    checkpoint_length = int(checkpoint["sequence_length"])
    if checkpoint_length != SEQUENCE_LENGTH:
        raise ValueError(
            f"The analysis uses sequence length {SEQUENCE_LENGTH}, but the "
            f"checkpoint expects {checkpoint_length}."
        )

    target_names = tuple(checkpoint["target_names"])
    if target_names != TRANSFORMED_PARAMETERS:
        raise ValueError(
            f"The checkpoint targets {target_names}; expected all five targets "
            f"{TRANSFORMED_PARAMETERS}."
        )

    activation = getattr(nn, checkpoint["activation"], None)
    if activation is None:
        raise ValueError(
            f"Unknown activation in checkpoint: {checkpoint['activation']!r}"
        )

    kernel_sizes = checkpoint.get(
        "kernel_sizes",
        checkpoint.get("kernel_size"),
    )
    if kernel_sizes is None:
        raise KeyError("Checkpoint is missing kernel_size/kernel_sizes.")

    model = SVPosteriorTCN(
        tcn_channels=tuple(checkpoint["tcn_channels"]),
        kernel_size=kernel_sizes,
        dilations=tuple(checkpoint["dilations"]),
        hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
        topk_pool_fraction=checkpoint.get("topk_pool_fraction"),
        activation=activation,
        param_names=target_names,
        min_var=float(checkpoint["min_var"]),
        input_mean=float(checkpoint["input_mean"]),
        input_std=float(checkpoint["input_std"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LoadedTCN(model=model, checkpoint=checkpoint, device=device)


def prepare_tcn_input(y: np.ndarray, checkpoint: dict) -> np.ndarray:
    """Apply the same centering and log-square transform used in training."""
    y = validate_series_matrix(y)
    expected_length = int(checkpoint["sequence_length"])
    if y.shape[1] != expected_length:
        raise ValueError(
            f"Checkpoint expects sequences of length {expected_length}, "
            f"but received length {y.shape[1]}."
        )

    centered_y = y - np.mean(y, axis=1, keepdims=True)
    return np.log(
        centered_y**2 + float(checkpoint["k"])
    ).astype(np.float32, copy=False)


@torch.inference_mode()
def predict_transformed_gaussian(
    loaded_model: LoadedTCN,
    y: np.ndarray,
    batch_size: int = PREDICTION_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict transformed posterior means and marginal variances."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    x = prepare_tcn_input(y, loaded_model.checkpoint)
    means = np.empty(
        (len(x), len(TRANSFORMED_PARAMETERS)),
        dtype=np.float64,
    )
    variances = np.empty_like(means)

    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        x_batch = torch.from_numpy(x[start:stop]).to(loaded_model.device)
        mean_batch, variance_batch = loaded_model.model(x_batch)
        means[start:stop] = mean_batch.cpu().numpy()
        variances[start:stop] = variance_batch.cpu().numpy()

    return means, variances


def inverse_logit(x: np.ndarray) -> np.ndarray:
    """Numerically stable inverse-logit transform."""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    positive = x >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def transformed_gaussian_ci(
    means: np.ndarray,
    variances: np.ndarray,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Transform marginal Gaussian credible intervals to model parameters."""
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    expected_shape = (len(means), len(TRANSFORMED_PARAMETERS))
    if means.shape != expected_shape or variances.shape != expected_shape:
        raise ValueError(
            f"means and variances must both have shape {expected_shape}."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")

    critical_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    standard_deviations = np.sqrt(
        np.clip(variances, POSTERIOR_VAR_EPS, None)
    )
    lower = means - critical_value * standard_deviations
    upper = means + critical_value * standard_deviations
    nu_min = sim.get_gh_skew_t_prior_constants("default").nu_min

    return pd.DataFrame(
        {
            "mu_median": means[:, 0],
            "mu_ci_lower": lower[:, 0],
            "mu_ci_upper": upper[:, 0],
            "phi_median": np.tanh(means[:, 1] / 2.0),
            "phi_ci_lower": np.tanh(lower[:, 1] / 2.0),
            "phi_ci_upper": np.tanh(upper[:, 1] / 2.0),
            "s_median": np.exp(means[:, 2]),
            "s_ci_lower": np.exp(lower[:, 2]),
            "s_ci_upper": np.exp(upper[:, 2]),
            "r_median": inverse_logit(means[:, 3]),
            "r_ci_lower": inverse_logit(lower[:, 3]),
            "r_ci_upper": inverse_logit(upper[:, 3]),
            "nu_median": nu_min + np.exp(means[:, 4]),
            "nu_ci_lower": nu_min + np.exp(lower[:, 4]),
            "nu_ci_upper": nu_min + np.exp(upper[:, 4]),
        }
    )


def simulate_ci_series(
    n: int = SEQUENCE_LENGTH,
    seed: int = CI_SEED,
) -> dict[str, np.ndarray]:
    """Simulate the five one-at-a-time parameter sweeps."""
    rng = np.random.default_rng(seed)
    datasets = {}

    for swept_parameter, true_values in SWEEPS.items():
        parameters = {
            name: np.full(CI_SWEEP_SIZE, value, dtype=np.float64)
            for name, value in BASELINE.items()
        }
        parameters[swept_parameter] = true_values
        datasets[swept_parameter] = sim.simulate_sv_chunk(
            mu=parameters["mu"],
            phi=parameters["phi"],
            s=parameters["s"],
            r=parameters["r"],
            nu=parameters["nu"],
            n=n,
            rng=rng,
            random_init=True,
        )

    return datasets


def add_ci_rows(
    rows: list[dict],
    parameter: str,
    method: str,
    ci: pd.DataFrame,
    ci_parameter: str | None = None,
) -> None:
    """Append one method's interval estimates for a parameter sweep."""
    if "index" in ci:
        ci = ci.sort_values("index").reset_index(drop=True)
    else:
        ci = ci.reset_index(drop=True)

    ci_parameter = parameter if ci_parameter is None else ci_parameter
    if len(ci) != CI_SWEEP_SIZE:
        raise ValueError(
            f"Expected {CI_SWEEP_SIZE} CI rows for {parameter}, got {len(ci)}."
        )

    for value_index, true_value in enumerate(SWEEPS[parameter]):
        rows.append(
            {
                "parameter": parameter,
                "value_index": value_index,
                "true_value": float(true_value),
                "method": method,
                "prior": "default",
                "median": float(
                    ci.loc[value_index, f"{ci_parameter}_median"]
                ),
                "ci_lower": float(
                    ci.loc[value_index, f"{ci_parameter}_ci_lower"]
                ),
                "ci_upper": float(
                    ci.loc[value_index, f"{ci_parameter}_ci_upper"]
                ),
            }
        )


def calculate_credible_intervals(
    loaded_model: LoadedTCN,
    *,
    include_mcmc: bool = True,
    batch_size: int = PREDICTION_BATCH_SIZE,
) -> pd.DataFrame:
    """Calculate TCN intervals and the available standard-SV comparisons."""
    datasets = simulate_ci_series()
    rows: list[dict] = []

    for parameter in PARAMETERS:
        y = datasets[parameter]

        if include_mcmc and parameter in MCMC_PARAMETERS:
            print(
                f"Running stochvol (default) for the {parameter} sweep "
                f"with {MCMC_DRAWS:,} draws."
            )
            mcmc_ci = run_stochvol_mcmc(
                y,
                prior="default",
                draws=MCMC_DRAWS,
                burnin=MCMC_BURNIN,
                thinpara=MCMC_THINPARA,
                alpha=ALPHA,
                transforms=None,
                max_cores=MCMC_MAX_CORES,
            )
            mcmc_parameter = "sigma" if parameter == "s" else parameter
            add_ci_rows(
                rows,
                parameter,
                "stochvol",
                mcmc_ci,
                ci_parameter=mcmc_parameter,
            )

        print(f"Predicting TCN intervals for the {parameter} sweep.")
        means, variances = predict_transformed_gaussian(
            loaded_model,
            y,
            batch_size=batch_size,
        )
        add_ci_rows(
            rows,
            parameter,
            "TCN",
            transformed_gaussian_ci(means, variances),
        )

    return pd.DataFrame(rows)


def apply_plot_style() -> None:
    """Use the plotting convention from test_NN_models.py."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "legend.fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
        }
    )


def plot_credible_intervals(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the four most informative credible-interval sweeps."""
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    legend_handles = {}

    for axis, parameter in zip(axes.flat, PLOT_PARAMETERS):
        true_values = SWEEPS[parameter]
        spacing = float(np.min(np.diff(true_values)))
        available_methods = tuple(
            method
            for method in METHODS
            if method in set(
                comparison.loc[
                    comparison["parameter"] == parameter,
                    "method",
                ]
            )
        )
        if len(available_methods) > 1:
            offsets = dict(
                zip(
                    available_methods,
                    np.linspace(-0.12, 0.12, len(available_methods))
                    * spacing,
                )
            )
        else:
            offsets = {method: 0.0 for method in available_methods}

        axis.plot(
            true_values,
            true_values,
            color="0.35",
            linestyle="--",
        )

        for method in available_methods:
            data = comparison[
                (comparison["parameter"] == parameter)
                & (comparison["method"] == method)
            ].sort_values("value_index")
            median = data["median"].to_numpy()
            lower = data["ci_lower"].to_numpy()
            upper = data["ci_upper"].to_numpy()
            legend_handles[method] = axis.errorbar(
                true_values + offsets[method],
                median,
                yerr=np.vstack((median - lower, upper - median)),
                fmt=MARKERS[method],
                color=COLORS[method],
                capsize=3,
                markersize=4.5,
                label=method,
            )

        parameter_label = PARAMETER_LABELS[parameter]
        axis.set_xlabel(f"True {parameter_label}")
        axis.set_ylabel(f"{parameter_label} posterior median")
        axis.grid(alpha=0.25)

    legend_order = tuple(
        method for method in METHODS if method in legend_handles
    )
    fig.legend(
        [legend_handles[method] for method in legend_order],
        legend_order,
        loc="lower center",
        ncol=len(legend_order),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    """Evaluate the standard-normal CDF without adding a SciPy dependency."""
    z = np.asarray(z, dtype=np.float64)
    normal = NormalDist()
    flat_cdf = np.fromiter(
        (normal.cdf(float(value)) for value in z.ravel()),
        dtype=np.float64,
        count=z.size,
    )
    return flat_cdf.reshape(z.shape)


def compute_sbc_cdf_values(
    transformed_theta: np.ndarray,
    posterior_means: np.ndarray,
    posterior_variances: np.ndarray,
) -> np.ndarray:
    """Evaluate each marginal posterior CDF at its true transformed value."""
    transformed_theta = np.asarray(transformed_theta, dtype=np.float64)
    posterior_means = np.asarray(posterior_means, dtype=np.float64)
    posterior_variances = np.asarray(
        posterior_variances,
        dtype=np.float64,
    )
    if not (
        transformed_theta.shape
        == posterior_means.shape
        == posterior_variances.shape
    ):
        raise ValueError(
            "SBC targets, means, and variances must have equal shapes."
        )

    posterior_sd = np.sqrt(
        np.clip(posterior_variances, POSTERIOR_VAR_EPS, None)
    )
    standardized_error = (
        transformed_theta - posterior_means
    ) / posterior_sd
    return standard_normal_cdf(standardized_error)


def ks_uniform_statistic(values: np.ndarray) -> float:
    """One-sample Kolmogorov--Smirnov distance from Uniform(0, 1)."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    if len(values) < 1:
        raise ValueError("values must contain at least one entry.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("SBC CDF values must lie in [0, 1].")

    n = len(values)
    empirical_upper = np.arange(1, n + 1, dtype=np.float64) / n
    empirical_lower = np.arange(0, n, dtype=np.float64) / n
    return float(
        max(
            np.max(empirical_upper - values),
            np.max(values - empirical_lower),
        )
    )


def asymptotic_ks_uniform_pvalue(
    ks_distance: float,
    n: int,
    n_terms: int = 100,
) -> float:
    """Approximate the two-sided uniform KS p-value."""
    if n < 1:
        raise ValueError("n must be at least 1.")
    if ks_distance <= 0.0:
        return 1.0

    scaled_distance = (
        np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)
    ) * ks_distance
    terms = [
        (-1.0) ** (index - 1)
        * np.exp(-2.0 * (index * scaled_distance) ** 2)
        for index in range(1, n_terms + 1)
    ]
    return float(np.clip(2.0 * np.sum(terms), 0.0, 1.0))


def sbc_uniformity_metrics(
    cdf_values: np.ndarray,
    bins: int = SBC_BINS,
) -> pd.DataFrame:
    """Summarize uniformity of the five marginal SBC distributions."""
    cdf_values = np.asarray(cdf_values, dtype=np.float64)
    if (
        cdf_values.ndim != 2
        or cdf_values.shape[1] != len(PARAMETERS)
    ):
        raise ValueError(
            f"cdf_values must have shape (n, {len(PARAMETERS)})."
        )
    if bins < 2:
        raise ValueError("bins must be at least 2.")

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []

    for index, (parameter, transformed_parameter) in enumerate(
        zip(PARAMETERS, TRANSFORMED_PARAMETERS)
    ):
        values = cdf_values[:, index]
        counts, _ = np.histogram(values, bins=bin_edges)
        observed_bin_mass = counts.astype(np.float64) / len(values)
        expected_bin_mass = 1.0 / bins
        bin_deviation = observed_bin_mass - expected_bin_mass
        ks_distance = ks_uniform_statistic(values)

        rows.append(
            {
                "parameter": parameter,
                "transformed_parameter": transformed_parameter,
                "n": len(values),
                "cdf_mean": float(np.mean(values)),
                "cdf_variance": (
                    float(np.var(values, ddof=1))
                    if len(values) > 1
                    else np.nan
                ),
                "ks_distance": ks_distance,
                "ks_pvalue_asymptotic": asymptotic_ks_uniform_pvalue(
                    ks_distance,
                    len(values),
                ),
                "histogram_l1_distance": float(
                    np.sum(np.abs(bin_deviation))
                ),
                "histogram_rmse": float(
                    np.sqrt(np.mean(bin_deviation**2))
                ),
                "max_abs_bin_deviation": float(
                    np.max(np.abs(bin_deviation))
                ),
            }
        )

    return pd.DataFrame(rows)


def simulate_sbc_dataset(
    n_simulations: int = SBC_SIMULATIONS,
    seed: int = SBC_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw parameters from the default prior and simulate observations."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")

    rng = np.random.default_rng(seed)
    mu, phi, s, r, nu = sim.sample_stochvol_prior(
        n_simulations,
        rng=rng,
        prior="default",
        return_s2=False,
        dtype=np.float64,
    )
    y = sim.simulate_sv_chunk(
        mu=mu,
        phi=phi,
        s=s,
        r=r,
        nu=nu,
        n=SEQUENCE_LENGTH,
        rng=rng,
        random_init=True,
    )
    theta = np.column_stack((mu, phi, s, r, nu))
    transformed_theta = theta_to_target_numpy(
        theta,
        target_names=TRANSFORMED_PARAMETERS,
    ).astype(np.float64)
    return y, theta, transformed_theta


def run_sbc(
    loaded_model: LoadedTCN,
    *,
    n_simulations: int = SBC_SIMULATIONS,
    batch_size: int = PREDICTION_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the full joint-prior SBC experiment."""
    print(
        f"Simulating {n_simulations:,} default-prior SBC datasets "
        f"of length {SEQUENCE_LENGTH:,}."
    )
    y, theta, transformed_theta = simulate_sbc_dataset(
        n_simulations=n_simulations,
    )
    print("Predicting the five transformed posterior marginals for SBC.")
    posterior_means, posterior_variances = predict_transformed_gaussian(
        loaded_model,
        y,
        batch_size=batch_size,
    )
    cdf_values = compute_sbc_cdf_values(
        transformed_theta,
        posterior_means,
        posterior_variances,
    )
    return (
        theta,
        transformed_theta,
        posterior_means,
        posterior_variances,
        cdf_values,
    )


def build_sbc_prediction_frame(
    theta: np.ndarray,
    transformed_theta: np.ndarray,
    posterior_means: np.ndarray,
    posterior_variances: np.ndarray,
    cdf_values: np.ndarray,
) -> pd.DataFrame:
    """Combine true parameters, predictions, and SBC values in one table."""
    frame = pd.DataFrame(
        theta,
        columns=[f"theta_{name}" for name in PARAMETERS],
    )

    for index, (parameter, transformed_parameter) in enumerate(
        zip(PARAMETERS, TRANSFORMED_PARAMETERS)
    ):
        frame[f"target_{transformed_parameter}"] = transformed_theta[:, index]
        frame[f"posterior_mean_{transformed_parameter}"] = (
            posterior_means[:, index]
        )
        frame[f"posterior_sd_{transformed_parameter}"] = np.sqrt(
            np.clip(
                posterior_variances[:, index],
                POSTERIOR_VAR_EPS,
                None,
            )
        )
        frame[f"sbc_cdf_{parameter}"] = cdf_values[:, index]

    return frame


def plot_sbc_histograms(
    cdf_values: np.ndarray,
    metrics: pd.DataFrame,
    output_path: Path,
    bins: int = SBC_BINS,
) -> None:
    """Plot the four most informative SBC histograms in a 2-by-2 layout."""
    apply_plot_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 9.0),
        sharex=True,
        sharey=True,
    )
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    metrics_by_parameter = metrics.set_index("parameter")

    for axis, parameter in zip(axes.flat, PLOT_PARAMETERS):
        parameter_index = PARAMETERS.index(parameter)
        metric_row = metrics_by_parameter.loc[parameter]
        axis.hist(
            cdf_values[:, parameter_index],
            bins=bin_edges,
            density=True,
            color=SBC_COLOR,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.axhline(
            1.0,
            color="0.35",
            linestyle="--",
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Posterior CDF at the true parameter")
        axis.set_ylabel("Density")
        axis.set_title(
            f"{PARAMETER_LABELS[parameter]}: "
            f"KS = {metric_row['ks_distance']:.3f}"
        )
        axis.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_sbc_metrics(metrics: pd.DataFrame) -> None:
    """Print the main SBC diagnostics in a compact table."""
    columns = (
        "parameter",
        "transformed_parameter",
        "cdf_mean",
        "cdf_variance",
        "ks_distance",
        "ks_pvalue_asymptotic",
        "histogram_rmse",
        "max_abs_bin_deviation",
    )
    print("\nSBC uniformity diagnostics:")
    with pd.option_context("display.width", 180):
        print(
            metrics.loc[:, columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.6g}",
            )
        )


def parse_args() -> argparse.Namespace:
    """Parse full-run and smoke-test controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures and CSV files.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device, such as cpu or cuda. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=PREDICTION_BATCH_SIZE,
        help="TCN prediction batch size.",
    )
    parser.add_argument(
        "--sbc-simulations",
        type=int,
        default=SBC_SIMULATIONS,
        help="Number of joint-prior simulations used for SBC.",
    )
    parser.add_argument(
        "--skip-mcmc",
        action="store_true",
        help="Generate TCN-only CI sweeps without running stochvol.",
    )
    parser.add_argument(
        "--skip-ci",
        action="store_true",
        help="Skip the credible-interval sweep.",
    )
    parser.add_argument(
        "--skip-sbc",
        action="store_true",
        help="Skip simulation-based calibration.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the standardized CI sweep and SBC analysis."""
    args = parse_args()
    if args.skip_ci and args.skip_sbc:
        raise ValueError("At least one of CI or SBC must be enabled.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.sbc_simulations < 1:
        raise ValueError("--sbc-simulations must be at least 1.")

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading {CHECKPOINT_PATH.name} on {device}.")
    loaded_model = load_tcn_model(CHECKPOINT_PATH, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_ci:
        comparison = calculate_credible_intervals(
            loaded_model,
            include_mcmc=not args.skip_mcmc,
            batch_size=args.batch_size,
        )
        ci_csv_path = args.output_dir / "five_parameter_ci_sweeps.csv"
        ci_plot_path = args.output_dir / "five_parameter_ci_sweeps.pdf"
        comparison.to_csv(ci_csv_path, index=False)
        plot_credible_intervals(comparison, ci_plot_path)
        print(f"Saved CI estimates to {ci_csv_path}")
        print(f"Saved the 5-by-1 CI plot to {ci_plot_path}")

    if not args.skip_sbc:
        (
            theta,
            transformed_theta,
            posterior_means,
            posterior_variances,
            cdf_values,
        ) = run_sbc(
            loaded_model,
            n_simulations=args.sbc_simulations,
            batch_size=args.batch_size,
        )
        metrics = sbc_uniformity_metrics(cdf_values, bins=SBC_BINS)
        predictions = build_sbc_prediction_frame(
            theta,
            transformed_theta,
            posterior_means,
            posterior_variances,
            cdf_values,
        )
        suffix = f"n{args.sbc_simulations}"
        prediction_path = (
            args.output_dir / f"five_parameter_sbc_predictions_{suffix}.csv"
        )
        cdf_path = (
            args.output_dir / f"five_parameter_sbc_cdf_values_{suffix}.csv"
        )
        metrics_path = (
            args.output_dir / f"five_parameter_sbc_metrics_{suffix}.csv"
        )
        sbc_plot_path = (
            args.output_dir / f"five_parameter_sbc_histograms_{suffix}.pdf"
        )

        predictions.to_csv(prediction_path, index_label="simulation")
        pd.DataFrame(cdf_values, columns=PARAMETERS).to_csv(
            cdf_path,
            index_label="simulation",
        )
        metrics.to_csv(metrics_path, index=False)
        plot_sbc_histograms(
            cdf_values,
            metrics,
            sbc_plot_path,
            bins=SBC_BINS,
        )
        print_sbc_metrics(metrics)
        print(f"Saved SBC predictions to {prediction_path}")
        print(f"Saved SBC CDF values to {cdf_path}")
        print(f"Saved SBC metrics to {metrics_path}")
        print(f"Saved the 5-by-1 SBC plot to {sbc_plot_path}")


if __name__ == "__main__":
    main()
