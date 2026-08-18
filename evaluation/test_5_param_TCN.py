"""CI, SBC, validation loss, and scoring for the five-parameter SV-GHST TCN."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from scipy.integrate import quad
from scipy.special import digamma, expit, polygamma
from scipy.stats import kstest, norm

from simulation import sim_5_param_data as sim
from simulation.stochvol_mcmc import (
    log_positive_transform,
    psi_transform,
    run_stochvol_mcmc,
    validate_series_matrix,
)
from training.sbt_tcn import (
    SVGHST_TARGET_NAMES,
    TCN,
    theta_to_target_numpy,
)


# Core configuration ---------------------------------------------------------
CHECKPOINT_PATH = (
    PROJECT_DIR
    / "weights"
    / "svghst_posterior_tcn_live_default_n2530_multiscale_topk.pt"
)
DEFAULT_OUTPUT_DIR = HERE / "five_param_tcn_results"

SEQUENCE_LENGTH = 253 * 10
ALPHA = 0.05
PREDICTION_BATCH_SIZE = 128
POSTERIOR_VAR_EPS = 1e-12
DEVICE_NAME: str | None = None

RUN_PLOTS = False
RUN_METRICS = True

# CI configuration
CI_SWEEP_SIZE = 10
CI_SEED = 2
MCMC_DRAWS = 20_000
MCMC_BURNIN = 500
MCMC_THINPARA = 1
MCMC_MAX_CORES = 3

BASELINE: dict[str, float] = {
    "mu": -9.0,
    "phi": 0.95,
    "s": 0.25,
    "r": 0.50,
    "nu": 15.0,
}
SWEEPS: dict[str, np.ndarray] = {
    "mu": np.linspace(-12.0, -6.0, CI_SWEEP_SIZE),
    "phi": np.linspace(0.905, 0.995, CI_SWEEP_SIZE),
    "s": np.linspace(0.05, 0.45, CI_SWEEP_SIZE),
    "r": np.linspace(0.20, 0.80, CI_SWEEP_SIZE),
    "nu": np.linspace(8.0, 22.0, CI_SWEEP_SIZE),
}

# Shared SBC and metric-benchmark configuration
BENCHMARK_SIZE = 5_000
BENCHMARK_SEED = 3
SBC_BINS = 50

# Names and plotting
PARAMETERS = ("mu", "phi", "s", "r", "nu")
TRANSFORMED_PARAMETERS = tuple(SVGHST_TARGET_NAMES)
METRIC_PARAMETERS = (
    "mu",
    "psi",
    "rho",
    "logit_r",
    "log_nu_shifted",
)
MCMC_PARAMETERS = ("mu", "phi", "s")
METHODS = ("stochvol", "TCN")
NU_MIN = sim.get_gh_skew_t_prior_constants("default").nu_min

COLORS = {"stochvol": "#0000ff", "TCN": "#ff0000", "prior": "#008000"}
MARKERS = {"stochvol": "o", "TCN": "^", "prior": "s"}
SBC_COLOR = "#4C78A8"
PARAMETER_LABELS = {
    "mu": r"$\mu$",
    "phi": r"$\phi$",
    "s": r"$\sigma$",
    "r": r"$r$",
    "nu": r"$\nu$",
}
LOSS_LABELS = {
    "mu": r"$\mu$",
    "psi": r"$\psi$",
    "rho": r"$\rho$",
    "logit_r": r"$\operatorname{logit}(r)$",
    "log_nu_shifted": r"$\log(\nu-\nu_{\min})$",
    "mean_standard": "Mean",
    "mean_ghst": "Mean",
    "mean_all": "Mean",
}
METHOD_LABELS = {
    "stochvol": "stochvol",
    "TCN": "TCN",
    "prior": "Prior baseline",
}
MCMC_TRANSFORMS = {
    "phi": {"psi": psi_transform},
    "sigma": {"rho": log_positive_transform},
}


@dataclass
class LoadedTCN:
    model: nn.Module
    checkpoint: dict
    device: torch.device


@dataclass
class BenchmarkResult:
    y: np.ndarray
    theta: np.ndarray
    transformed_theta: np.ndarray
    posterior_means: np.ndarray
    posterior_variances: np.ndarray
    cdf_values: np.ndarray


# Model loading and prediction ----------------------------------------------
def load_tcn_model(
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedTCN:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if int(checkpoint["sequence_length"]) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Checkpoint length is {checkpoint['sequence_length']}, "
            f"expected {SEQUENCE_LENGTH}."
        )
    if tuple(checkpoint["target_names"]) != TRANSFORMED_PARAMETERS:
        raise ValueError("Checkpoint does not contain all five parameter heads.")

    activation = getattr(nn, checkpoint["activation"])
    kernel_sizes = checkpoint.get("kernel_sizes") or checkpoint["kernel_size"]
    model = TCN(
        tcn_channels=tuple(checkpoint["tcn_channels"]),
        kernel_size=kernel_sizes,
        dilations=tuple(checkpoint["dilations"]),
        hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
        topk_pool_fraction=checkpoint.get("topk_pool_fraction"),
        activation=activation,
        param_names=tuple(checkpoint["target_names"]),
        min_var=float(checkpoint["min_var"]),
        input_mean=float(checkpoint["input_mean"]),
        input_std=float(checkpoint["input_std"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return LoadedTCN(model, checkpoint, device)


def prepare_tcn_input(y: np.ndarray, checkpoint: dict) -> np.ndarray:
    y = validate_series_matrix(y)
    if y.shape[1] != int(checkpoint["sequence_length"]):
        raise ValueError(f"Expected sequences of length {SEQUENCE_LENGTH}.")
    centered_y = y - np.mean(y, axis=1, keepdims=True)
    return np.log(centered_y**2 + checkpoint["k"]).astype(np.float32)


@torch.inference_mode()
def predict_transformed_gaussian(
    loaded_model: LoadedTCN,
    y: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = prepare_tcn_input(y, loaded_model.checkpoint)
    output_shape = (len(x), len(PARAMETERS))
    means = np.empty(output_shape, dtype=np.float64)
    variances = np.empty(output_shape, dtype=np.float64)

    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        x_batch = torch.from_numpy(x[start:stop]).to(loaded_model.device)
        mean_batch, variance_batch = loaded_model.model(x_batch)
        means[start:stop] = mean_batch.cpu().numpy()
        variances[start:stop] = variance_batch.cpu().numpy()

    return means, variances


def inverse_transform(values: np.ndarray) -> np.ndarray:
    """Map (mu, psi, rho, logit_r, log_nu) to model parameters."""
    values = np.asarray(values, dtype=np.float64)
    parameters = np.empty_like(values)
    parameters[:, 0] = values[:, 0]
    parameters[:, 1] = np.tanh(values[:, 1] / 2.0)
    parameters[:, 2] = np.exp(values[:, 2])
    parameters[:, 3] = expit(values[:, 3])
    parameters[:, 4] = NU_MIN + np.exp(values[:, 4])
    return parameters


# Credible-interval sweeps ---------------------------------------------------
def transformed_gaussian_ci(
    means: np.ndarray,
    variances: np.ndarray,
) -> pd.DataFrame:
    z = NormalDist().inv_cdf(1.0 - ALPHA / 2.0)
    sd = np.sqrt(np.clip(variances, POSTERIOR_VAR_EPS, None))
    median = inverse_transform(means)
    lower = inverse_transform(means - z * sd)
    upper = inverse_transform(means + z * sd)

    columns: dict[str, np.ndarray] = {}
    for index, parameter in enumerate(PARAMETERS):
        columns[f"{parameter}_median"] = median[:, index]
        columns[f"{parameter}_ci_lower"] = lower[:, index]
        columns[f"{parameter}_ci_upper"] = upper[:, index]
    return pd.DataFrame(columns)


def simulate_ci_series() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(CI_SEED)
    datasets: dict[str, np.ndarray] = {}

    for swept_parameter, true_values in SWEEPS.items():
        values = {
            name: np.full(CI_SWEEP_SIZE, baseline)
            for name, baseline in BASELINE.items()
        }
        values[swept_parameter] = true_values
        datasets[swept_parameter] = sim.simulate_sv_chunk(
            mu=values["mu"],
            phi=values["phi"],
            s=values["s"],
            r=values["r"],
            nu=values["nu"],
            n=SEQUENCE_LENGTH,
            rng=rng,
            random_init=True,
        )

    return datasets


def append_ci_rows(
    rows: list[dict[str, object]],
    parameter: str,
    method: str,
    ci: pd.DataFrame,
    ci_parameter: str | None = None,
) -> None:
    ci = ci.sort_values("index").reset_index(drop=True) if "index" in ci else ci
    ci_parameter = ci_parameter or parameter

    for index, true_value in enumerate(SWEEPS[parameter]):
        rows.append(
            {
                "parameter": parameter,
                "value_index": index,
                "true_value": true_value,
                "method": method,
                "median": ci.loc[index, f"{ci_parameter}_median"],
                "ci_lower": ci.loc[index, f"{ci_parameter}_ci_lower"],
                "ci_upper": ci.loc[index, f"{ci_parameter}_ci_upper"],
            }
        )


def calculate_credible_intervals(
    loaded_model: LoadedTCN,
    include_mcmc: bool,
    batch_size: int,
) -> pd.DataFrame:
    datasets = simulate_ci_series()
    rows: list[dict[str, object]] = []

    for parameter, y in datasets.items():
        if include_mcmc and parameter in MCMC_PARAMETERS:
            print(f"Running stochvol for the {parameter} sweep.")
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
            mcmc_name = "sigma" if parameter == "s" else parameter
            append_ci_rows(rows, parameter, "stochvol", mcmc_ci, mcmc_name)

        print(f"Predicting TCN intervals for the {parameter} sweep.")
        means, variances = predict_transformed_gaussian(
            loaded_model,
            y,
            batch_size,
        )
        append_ci_rows(
            rows,
            parameter,
            "TCN",
            transformed_gaussian_ci(means, variances),
        )

    return pd.DataFrame(rows)


# Plotting -------------------------------------------------------------------
def apply_plot_style() -> None:
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


def plot_ci_axis(
    axis: Axes,
    comparison: pd.DataFrame,
    parameter: str,
) -> dict[str, object]:
    true_values = SWEEPS[parameter]
    spacing = np.min(np.diff(true_values))
    methods = tuple(
        method
        for method in METHODS
        if method
        in comparison.loc[
            comparison["parameter"] == parameter,
            "method",
        ].values
    )
    offsets = (
        dict(zip(methods, np.linspace(-0.12, 0.12, len(methods)) * spacing))
        if len(methods) > 1
        else {method: 0.0 for method in methods}
    )

    axis.plot(true_values, true_values, color="0.35", linestyle="--")
    handles: dict[str, object] = {}
    for method in methods:
        data = comparison[
            (comparison["parameter"] == parameter)
            & (comparison["method"] == method)
        ].sort_values("value_index")
        median = data["median"].to_numpy()
        lower = data["ci_lower"].to_numpy()
        upper = data["ci_upper"].to_numpy()
        handles[method] = axis.errorbar(
            true_values + offsets[method],
            median,
            yerr=np.vstack((median - lower, upper - median)),
            fmt=MARKERS[method],
            color=COLORS[method],
            capsize=3,
            markersize=4.5,
            label=method,
        )

    label = PARAMETER_LABELS[parameter]
    axis.set_xlabel(f"True {label}")
    axis.set_ylabel(f"{label} posterior median")
    axis.grid(alpha=0.25)
    return handles


def plot_sbc_axis(
    axis: Axes,
    cdf_values: np.ndarray,
    metrics: pd.DataFrame,
    parameter: str,
) -> None:
    parameter_index = PARAMETERS.index(parameter)
    ks_distance = metrics.set_index("parameter").loc[parameter, "ks_distance"]
    axis.hist(
        cdf_values[:, parameter_index],
        bins=np.linspace(0.0, 1.0, SBC_BINS + 1),
        density=True,
        color=SBC_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )
    axis.axhline(1.0, color="0.35", linestyle="--")
    axis.text(
        0.97,
        0.93,
        f"{PARAMETER_LABELS[parameter]}, KS = {ks_distance:.3f}",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Posterior CDF at the true parameter")
    axis.set_ylabel("Density")
    axis.grid(axis="y", alpha=0.25)


def plot_ci_and_sbc(
    comparison: pd.DataFrame,
    cdf_values: np.ndarray,
    metrics: pd.DataFrame,
    parameters: tuple[str, ...],
    output_path: Path,
) -> None:
    apply_plot_style()
    figure_height = 13.0 if len(parameters) == 3 else 8.6
    fig, axes = plt.subplots(
        len(parameters),
        2,
        figsize=(11.5, figure_height),
        squeeze=False,
    )
    legend_handles: dict[str, object] = {}

    for row, parameter in enumerate(parameters):
        legend_handles.update(
            plot_ci_axis(axes[row, 0], comparison, parameter)
        )
        plot_sbc_axis(axes[row, 1], cdf_values, metrics, parameter)

    axes[0, 0].set_title("Credible intervals")
    axes[0, 1].set_title("Simulation-based calibration")
    legend_order = tuple(method for method in METHODS if method in legend_handles)
    fig.legend(
        [legend_handles[method] for method in legend_order],
        legend_order,
        loc="lower center",
        ncol=len(legend_order),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_validation_loss(checkpoint: dict, output_path: Path) -> None:
    """Plot mean marginal validation loss, following test_NN_models.py."""
    apply_plot_style()
    marginal_losses = np.asarray(
        checkpoint["val_marginal_loss_history"],
        dtype=np.float64,
    )
    losses = np.mean(marginal_losses, axis=1)
    epochs = np.arange(1, len(losses) + 1)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    tcn_handle = axis.plot(
        epochs,
        losses,
        color=COLORS["TCN"],
        label="TCN",
    )[0]
    final_handle = axis.plot(
        epochs[-1],
        losses[-1],
        color="black",
        marker="x",
        markersize=8,
        markeredgewidth=1.8,
        linestyle="none",
        label="Final epoch",
    )[0]
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean marginal Gaussian loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)

    fig.legend(
        [tcn_handle, final_handle],
        ["TCN", "Final epoch"],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Simulation-based calibration ----------------------------------------------
def run_benchmark(
    loaded_model: LoadedTCN,
    n_simulations: int,
    batch_size: int,
) -> BenchmarkResult:
    print(f"Simulating {n_simulations:,} benchmark datasets.")
    rng = np.random.default_rng(BENCHMARK_SEED)
    theta_columns = sim.sample_stochvol_prior(
        n_simulations,
        rng=rng,
        prior="default",
        return_s2=False,
        dtype=np.float64,
    )
    theta = np.column_stack(theta_columns)
    y = sim.simulate_sv_chunk(
        mu=theta[:, 0],
        phi=theta[:, 1],
        s=theta[:, 2],
        r=theta[:, 3],
        nu=theta[:, 4],
        n=SEQUENCE_LENGTH,
        rng=rng,
        random_init=True,
    )
    transformed_theta = theta_to_target_numpy(
        theta,
        target_names=TRANSFORMED_PARAMETERS,
    ).astype(np.float64)
    posterior_means, posterior_variances = predict_transformed_gaussian(
        loaded_model,
        y,
        batch_size,
    )
    posterior_sd = np.sqrt(
        np.clip(posterior_variances, POSTERIOR_VAR_EPS, None)
    )
    cdf_values = norm.cdf(
        (transformed_theta - posterior_means) / posterior_sd
    )
    return BenchmarkResult(
        y,
        theta,
        transformed_theta,
        posterior_means,
        posterior_variances,
        cdf_values,
    )


def sbc_uniformity_metrics(cdf_values: np.ndarray) -> pd.DataFrame:
    bin_edges = np.linspace(0.0, 1.0, SBC_BINS + 1)
    rows: list[dict[str, object]] = []

    for index, (parameter, transformed_parameter) in enumerate(
        zip(PARAMETERS, TRANSFORMED_PARAMETERS)
    ):
        values = cdf_values[:, index]
        bin_mass = np.histogram(values, bins=bin_edges)[0] / len(values)
        bin_deviation = bin_mass - 1.0 / SBC_BINS
        ks_result = kstest(values, "uniform", method="asymp")
        rows.append(
            {
                "parameter": parameter,
                "transformed_parameter": transformed_parameter,
                "n": len(values),
                "cdf_mean": np.mean(values),
                "cdf_variance": np.var(values, ddof=1),
                "ks_distance": ks_result.statistic,
                "ks_pvalue_asymptotic": ks_result.pvalue,
                "histogram_l1_distance": np.sum(np.abs(bin_deviation)),
                "histogram_rmse": np.sqrt(np.mean(bin_deviation**2)),
                "max_abs_bin_deviation": np.max(np.abs(bin_deviation)),
            }
        )

    return pd.DataFrame(rows)


def build_sbc_prediction_frame(result: BenchmarkResult) -> pd.DataFrame:
    frame = pd.DataFrame(
        result.theta,
        columns=[f"theta_{name}" for name in PARAMETERS],
    )
    for index, (parameter, transformed_parameter) in enumerate(
        zip(PARAMETERS, TRANSFORMED_PARAMETERS)
    ):
        frame[f"target_{transformed_parameter}"] = result.transformed_theta[:, index]
        frame[f"posterior_mean_{transformed_parameter}"] = (
            result.posterior_means[:, index]
        )
        frame[f"posterior_sd_{transformed_parameter}"] = np.sqrt(
            np.clip(
                result.posterior_variances[:, index],
                POSTERIOR_VAR_EPS,
                None,
            )
        )
        frame[f"sbc_cdf_{parameter}"] = result.cdf_values[:, index]
    return frame


def print_sbc_metrics(metrics: pd.DataFrame) -> None:
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
    print(
        metrics.loc[:, columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )


def save_sbc_results(
    result: BenchmarkResult,
    output_dir: Path,
    n_simulations: int,
) -> pd.DataFrame:
    suffix = f"n{n_simulations}"
    metrics = sbc_uniformity_metrics(result.cdf_values)
    build_sbc_prediction_frame(result).to_csv(
        output_dir / f"five_parameter_sbc_predictions_{suffix}.csv",
        index_label="simulation",
    )
    pd.DataFrame(result.cdf_values, columns=PARAMETERS).to_csv(
        output_dir / f"five_parameter_sbc_cdf_values_{suffix}.csv",
        index_label="simulation",
    )
    metrics.to_csv(
        output_dir / f"five_parameter_sbc_metrics_{suffix}.csv",
        index=False,
    )
    print_sbc_metrics(metrics)
    return metrics


# Marginal Gaussian loss benchmark ------------------------------------------
def transformed_prior_moments() -> tuple[np.ndarray, np.ndarray]:
    """Exact moments of the five transformed parameters under the prior."""
    prior = sim.get_gh_skew_t_prior_constants("default")
    if prior.r_a0 is not None or prior.r_b0 is not None:
        raise ValueError("The r-moment calculation expects a uniform prior.")

    psi_mean = digamma(prior.phi_a0) - digamma(prior.phi_b0)
    psi_variance = polygamma(1, prior.phi_a0) + polygamma(1, prior.phi_b0)
    rho_mean = 0.5 * (
        np.log(prior.Bs) + digamma(0.5) + np.log(2.0)
    )
    rho_variance = 0.25 * polygamma(1, 0.5)

    def squared_logit(r: float) -> float:
        return float((np.log(r) - np.log1p(-r)) ** 2)

    r_max = prior.r_max
    logit_r_mean = (
        r_max * np.log(r_max)
        + (1.0 - r_max) * np.log1p(-r_max)
    ) / r_max
    logit_r_second_moment = (
        quad(squared_logit, 0.0, r_max, limit=200)[0] / r_max
    )
    logit_r_variance = logit_r_second_moment - logit_r_mean**2

    log_nu_mean = digamma(1.0) - np.log(prior.nu_rate)
    log_nu_variance = polygamma(1, 1.0)

    means = np.array(
        [
            prior.mu_mean,
            psi_mean,
            rho_mean,
            logit_r_mean,
            log_nu_mean,
        ],
        dtype=np.float64,
    )
    variances = np.array(
        [
            prior.mu_sd**2,
            psi_variance,
            rho_variance,
            logit_r_variance,
            log_nu_variance,
        ],
        dtype=np.float64,
    )
    return means, variances


def gaussian_loss(
    targets: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Elementwise negative log density of a marginal Gaussian."""
    variances = np.clip(variances, POSTERIOR_VAR_EPS, None)
    return 0.5 * (
        np.log(2.0 * np.pi * variances)
        + (targets - means) ** 2 / variances
    )


def mcmc_posterior_moments(
    y: np.ndarray,
    draws: int,
    max_cores: int,
) -> tuple[np.ndarray, np.ndarray]:
    print(
        f"Running {len(y):,} stochvol chains with {draws:,} draws each."
    )
    summary = run_stochvol_mcmc(
        y,
        prior="default",
        draws=draws,
        burnin=MCMC_BURNIN,
        thinpara=MCMC_THINPARA,
        alpha=ALPHA,
        transforms=MCMC_TRANSFORMS,
        max_cores=max_cores,
    ).sort_values("index")
    means = summary[["mu_mean", "psi_mean", "rho_mean"]].to_numpy()
    variances = summary[["mu_var", "psi_var", "rho_var"]].to_numpy()
    return means, variances


def loss_components(
    parameter_names: tuple[str, ...],
    losses: np.ndarray,
) -> dict[str, np.ndarray]:
    components = {
        parameter: losses[:, index]
        for index, parameter in enumerate(parameter_names)
    }
    if all(parameter in parameter_names for parameter in METRIC_PARAMETERS[:3]):
        indices = [parameter_names.index(name) for name in METRIC_PARAMETERS[:3]]
        components["mean_standard"] = np.mean(losses[:, indices], axis=1)
    if all(parameter in parameter_names for parameter in METRIC_PARAMETERS[3:]):
        indices = [parameter_names.index(name) for name in METRIC_PARAMETERS[3:]]
        components["mean_ghst"] = np.mean(losses[:, indices], axis=1)
    if parameter_names == METRIC_PARAMETERS:
        components["mean_all"] = np.mean(losses, axis=1)
    return components


def summarize_method_losses(
    method: str,
    parameter_names: tuple[str, ...],
    losses: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    sample_frames: list[pd.DataFrame] = []

    for parameter, values in loss_components(parameter_names, losses).items():
        sample_sd = np.std(values, ddof=1) if len(values) > 1 else np.nan
        metric_rows.append(
            {
                "method": method,
                "parameter": parameter,
                "benchmark_size": len(values),
                "mean_loss": np.mean(values),
                "sample_sd": sample_sd,
                "standard_error": sample_sd / np.sqrt(len(values)),
            }
        )
        sample_frames.append(
            pd.DataFrame(
                {
                    "method": method,
                    "benchmark_index": np.arange(len(values)),
                    "parameter": parameter,
                    "marginal_loss": values,
                }
            )
        )

    return pd.DataFrame(metric_rows), pd.concat(sample_frames, ignore_index=True)


def calculate_metric_comparison(
    benchmark: BenchmarkResult,
    include_mcmc: bool,
    mcmc_draws: int = MCMC_DRAWS,
    mcmc_max_cores: int = MCMC_MAX_CORES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = benchmark.transformed_theta
    prior_means, prior_variances = transformed_prior_moments()
    estimates = pd.DataFrame({"benchmark_index": np.arange(len(targets))})
    for index, parameter in enumerate(METRIC_PARAMETERS):
        estimates[f"target_{parameter}"] = targets[:, index]
        estimates[f"tcn_mean_{parameter}"] = benchmark.posterior_means[:, index]
        estimates[f"tcn_var_{parameter}"] = benchmark.posterior_variances[:, index]
        estimates[f"prior_mean_{parameter}"] = prior_means[index]
        estimates[f"prior_var_{parameter}"] = prior_variances[index]

    method_losses: list[tuple[str, tuple[str, ...], np.ndarray]] = [
        (
            "TCN",
            METRIC_PARAMETERS,
            gaussian_loss(
                targets,
                benchmark.posterior_means,
                benchmark.posterior_variances,
            ),
        ),
        (
            "prior",
            METRIC_PARAMETERS,
            gaussian_loss(targets, prior_means, prior_variances),
        ),
    ]

    if include_mcmc:
        mcmc_means, mcmc_variances = mcmc_posterior_moments(
            benchmark.y,
            draws=mcmc_draws,
            max_cores=mcmc_max_cores,
        )
        method_losses.insert(
            0,
            (
                "stochvol",
                METRIC_PARAMETERS[:3],
                gaussian_loss(
                    targets[:, :3],
                    mcmc_means,
                    mcmc_variances,
                ),
            ),
        )
        for index, parameter in enumerate(METRIC_PARAMETERS[:3]):
            estimates[f"stochvol_mean_{parameter}"] = mcmc_means[:, index]
            estimates[f"stochvol_var_{parameter}"] = mcmc_variances[:, index]

    metric_frames = []
    sample_frames = []
    for method, parameter_names, losses in method_losses:
        metrics, samples = summarize_method_losses(
            method,
            parameter_names,
            losses,
        )
        metric_frames.append(metrics)
        sample_frames.append(samples)
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(sample_frames, ignore_index=True),
        estimates,
    )


def plot_metric_comparison(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    legend_handles: dict[str, object] = {}
    panels = (
        (
            axes[0],
            ("mu", "psi", "rho", "mean_standard"),
            ("stochvol", "TCN", "prior"),
            "Standard SV parameters",
        ),
        (
            axes[1],
            ("logit_r", "log_nu_shifted", "mean_ghst"),
            ("TCN", "prior"),
            "GHST parameters",
        ),
    )

    for axis, parameters, methods, title in panels:
        x = np.arange(len(parameters), dtype=np.float64)
        available_methods = tuple(
            method for method in methods if method in metrics["method"].values
        )
        offsets = dict(
            zip(
                available_methods,
                np.linspace(-0.18, 0.18, len(available_methods)),
            )
        )

        for method in available_methods:
            data = (
                metrics[
                    (metrics["method"] == method)
                    & (metrics["parameter"].isin(parameters))
                ]
                .set_index("parameter")
                .loc[list(parameters)]
            )
            legend_handles[method] = axis.errorbar(
                x + offsets[method],
                data["mean_loss"],
                yerr=data["standard_error"],
                fmt=MARKERS[method],
                color=COLORS[method],
                capsize=3,
                markersize=5,
                linestyle="none",
                label=METHOD_LABELS[method],
            )

        axis.set_xticks(x, [LOSS_LABELS[name] for name in parameters])
        axis.set_title(title)
        axis.set_ylabel("Mean marginal Gaussian loss")
        axis.grid(axis="y", alpha=0.25)

    legend_order = tuple(
        method
        for method in ("stochvol", "TCN", "prior")
        if method in legend_handles
    )
    fig.legend(
        [legend_handles[method] for method in legend_order],
        [METHOD_LABELS[method] for method in legend_order],
        loc="lower center",
        ncol=len(legend_order),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def paired_loss_differences(samples: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired score differences; negative means the method is better."""
    wide = samples.pivot(
        index=["benchmark_index", "parameter"],
        columns="method",
        values="marginal_loss",
    )
    rows: list[dict[str, object]] = []

    for method, reference in (
        ("TCN", "stochvol"),
        ("TCN", "prior"),
        ("stochvol", "prior"),
    ):
        if method not in wide or reference not in wide:
            continue
        differences = (wide[method] - wide[reference]).dropna()
        for parameter, values in differences.groupby(level="parameter"):
            values = values.to_numpy()
            standard_error = np.std(values, ddof=1) / np.sqrt(len(values))
            rows.append(
                {
                    "method": method,
                    "reference": reference,
                    "parameter": parameter,
                    "benchmark_size": len(values),
                    "mean_loss_difference": np.mean(values),
                    "standard_error": standard_error,
                    "ci_lower": np.mean(values) - 1.96 * standard_error,
                    "ci_upper": np.mean(values) + 1.96 * standard_error,
                }
            )
    return pd.DataFrame(rows)


def save_metric_results(
    benchmark: BenchmarkResult,
    output_dir: Path,
) -> None:
    metrics, samples, estimates = calculate_metric_comparison(
        benchmark,
        include_mcmc=True,
    )
    metrics.to_csv(
        output_dir / "marginal_gaussian_loss_metrics.csv",
        index=False,
    )
    samples.to_csv(
        output_dir / "marginal_gaussian_loss_samples.csv",
        index=False,
    )
    paired_loss_differences(samples).to_csv(
        output_dir / "marginal_gaussian_loss_differences.csv",
        index=False,
    )
    estimates.to_csv(
        output_dir / "marginal_gaussian_moments.csv",
        index=False,
    )
    prior_means, prior_variances = transformed_prior_moments()
    pd.DataFrame(
        {
            "parameter": METRIC_PARAMETERS,
            "mean": prior_means,
            "variance": prior_variances,
        }
    ).to_csv(
        output_dir / "transformed_prior_moments.csv",
        index=False,
    )
    plot_metric_comparison(
        metrics,
        output_dir / "marginal_gaussian_loss_comparison.pdf",
    )
    print("\nMarginal Gaussian loss comparison:")
    print(
        metrics.pivot(
            index="parameter",
            columns="method",
            values="mean_loss",
        ).to_string(float_format=lambda value: f"{value:.6g}")
    )


def main() -> None:
    if not RUN_PLOTS and not RUN_METRICS:
        return

    device = torch.device(
        DEVICE_NAME or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading {CHECKPOINT_PATH.name} on {device}.")
    loaded_model = load_tcn_model(CHECKPOINT_PATH, device)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = None
    if RUN_PLOTS:
        loss_path = DEFAULT_OUTPUT_DIR / "five_parameter_validation_loss_history.pdf"
        plot_validation_loss(loaded_model.checkpoint, loss_path)
        print(f"Saved validation loss history to {loss_path}")
        comparison = calculate_credible_intervals(
            loaded_model,
            include_mcmc=True,
            batch_size=PREDICTION_BATCH_SIZE,
        )
        comparison.to_csv(
            DEFAULT_OUTPUT_DIR / "five_parameter_ci_sweeps.csv",
            index=False,
        )

    benchmark = run_benchmark(
        loaded_model,
        n_simulations=BENCHMARK_SIZE,
        batch_size=PREDICTION_BATCH_SIZE,
    )
    if RUN_PLOTS:
        if comparison is None:
            raise RuntimeError("CI results are unavailable.")
        sbc_metrics = save_sbc_results(
            benchmark,
            DEFAULT_OUTPUT_DIR,
            BENCHMARK_SIZE,
        )
        plot_ci_and_sbc(
            comparison,
            benchmark.cdf_values,
            sbc_metrics,
            PARAMETERS[:3],
            DEFAULT_OUTPUT_DIR / "standard_sv_ci_sbc.pdf",
        )
        plot_ci_and_sbc(
            comparison,
            benchmark.cdf_values,
            sbc_metrics,
            PARAMETERS[3:],
            DEFAULT_OUTPUT_DIR / "ghst_sv_ci_sbc.pdf",
        )
    if RUN_METRICS:
        save_metric_results(benchmark, DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
