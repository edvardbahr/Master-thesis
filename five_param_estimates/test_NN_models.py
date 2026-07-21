"""Test the three standard-SV posterior estimators.

There are deliberately only two entry points:

``main0()`` creates the fixed 3-by-2 credible-interval figure.
``main1()`` runs the fixed 2,000-sequence metric and runtime benchmark.

Both functions always load the same four neural checkpoints and compare them
with stochvol under the default and finance priors.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from time import perf_counter


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sim_5_param_data as sim
from R_to_py_interface import (
    centered_square_transform,
    log_positive_sq_transform,
    log_positive_transform,
    psi_sq_transform,
    psi_transform,
    run_stochvol_mcmc,
)
from train_live_CNN import SVPosteriorTCN
from train_live_summary_nn import SVPosteriorNN


ALPHA = 0.05
SEQUENCE_LENGTH = 253
MCMC_DRAWS = 20_000
MCMC_BURNIN = 500
MCMC_THINPARA = 1
MCMC_MAX_CORES = -2
BENCHMARK_SIZE = 2_000
CI_SWEEP_SIZE = 10
CI_SEED = 2
METRIC_SEED = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = HERE.parents[2] / "weights"
OUTPUT_DIR = HERE / "nn_model_tests"

PRIORS = ("default", "finance")
PARAMETERS = ("mu", "phi", "sigma2")
TRANSFORMED_PARAMETERS = ("mu", "psi", "rho")
METHODS = ("stochvol", "TCN", "Summary NN")

CHECKPOINT_NAMES = {
    ("Summary NN", "default"): "sv_posterior_summary_nn_live_default_arima.pt",
    ("Summary NN", "finance"): "sv_posterior_summary_nn_live_finance_arima.pt",
    ("TCN", "default"): "sv_posterior_tcn_live_default_n253_multiscale_topk.pt",
    ("TCN", "finance"): "sv_posterior_tcn_live_finance_n253_multiscale_topk.pt",
}

BASELINE = {
    "mu": -9.0,
    "phi": 0.95,
    "sigma2": 0.25**2,
}

SWEEPS = {
    "mu": np.linspace(-12.0, -6.0, CI_SWEEP_SIZE),
    "phi": np.linspace(0.905, 0.995, CI_SWEEP_SIZE),
    "sigma2": np.linspace(0.05**2, 0.45**2, CI_SWEEP_SIZE),
}

COLORS = {
    "default": {
        "stochvol": "#0000ff",  # blue
        "TCN": "#bf00bf",  # purple
        "Summary NN": "#bfbf00",  # yellow
    },
    "finance": {
        "stochvol": "#008000",  # green
        "TCN": "#ff0000",  # red
        "Summary NN": "#00bfbf",  # teal
    },
}

MARKERS = {
    "stochvol": "o",
    "TCN": "^",
    "Summary NN": "s",
}

PARAMETER_LABELS = {
    "mu": r"$\mu$",
    "phi": r"$\phi$",
    "sigma2": r"$\sigma^2$",
}


@dataclass
class LoadedModel:
    architecture: str
    prior: str
    model: nn.Module
    checkpoint: dict


def load_models() -> dict[tuple[str, str], LoadedModel]:
    models = {}

    for prior in PRIORS:
        summary_checkpoint = torch.load(
            WEIGHTS_DIR / CHECKPOINT_NAMES[("Summary NN", prior)],
            map_location=DEVICE,
            weights_only=False,
        )
        summary_model = SVPosteriorNN(
            input_dim=summary_checkpoint["input_dim"],
            hidden_dims_shared_trunk=tuple(
                summary_checkpoint["hidden_dims_shared_trunk"]
            ),
            hidden_dims_head=tuple(summary_checkpoint["hidden_dims_head"]),
            activation=getattr(nn, summary_checkpoint["activation"]),
            min_var=summary_checkpoint["min_var"],
            dropout=summary_checkpoint.get("dropout", 0.0),
            layer_norm=summary_checkpoint["layer_norm"],
        ).to(DEVICE)
        summary_model.load_state_dict(summary_checkpoint["model_state_dict"])
        summary_model.eval()
        models[("Summary NN", prior)] = LoadedModel(
            "Summary NN", prior, summary_model, summary_checkpoint
        )

        tcn_checkpoint = torch.load(
            WEIGHTS_DIR / CHECKPOINT_NAMES[("TCN", prior)],
            map_location=DEVICE,
            weights_only=False,
        )
        tcn_model = SVPosteriorTCN(
            tcn_channels=tuple(tcn_checkpoint["tcn_channels"]),
            kernel_size=tcn_checkpoint["kernel_sizes"],
            dilations=tuple(tcn_checkpoint["dilations"]),
            hidden_dims_head=tuple(tcn_checkpoint["hidden_dims_head"]),
            topk_pool_fraction=tcn_checkpoint["topk_pool_fraction"],
            activation=getattr(nn, tcn_checkpoint["activation"]),
            param_names=tuple(tcn_checkpoint["target_names"]),
            min_var=tcn_checkpoint["min_var"],
            input_mean=tcn_checkpoint["input_mean"],
            input_std=tcn_checkpoint["input_std"],
        ).to(DEVICE)
        tcn_model.load_state_dict(tcn_checkpoint["model_state_dict"])
        tcn_model.eval()
        models[("TCN", prior)] = LoadedModel(
            "TCN", prior, tcn_model, tcn_checkpoint
        )

    print(f"Loaded the four neural checkpoints on {DEVICE}.")
    return models


def print_parameter_counts(models) -> pd.DataFrame:
    rows = []
    for architecture in ("Summary NN", "TCN"):
        for prior in PRIORS:
            model = models[(architecture, prior)].model
            rows.append(
                {
                    "architecture": architecture,
                    "prior": prior,
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                }
            )

    counts = pd.DataFrame(rows)
    print("\nNeural-network parameter counts:")
    print(counts.to_string(index=False))

    summary_count = int(counts.iloc[0]["total_parameters"])
    tcn_count = int(counts.iloc[2]["total_parameters"])
    print(
        f"\nTCN has {tcn_count - summary_count:,} more parameters than Summary NN "
        f"({tcn_count / summary_count:.2f} times as many)."
    )
    return counts


def prepare_summary_input(y, checkpoint) -> np.ndarray:
    summaries = np.empty((len(y), checkpoint["input_dim"]), dtype=np.float32)
    for index, series in enumerate(y):
        summaries[index] = sim.summary_stats_sv(
            series,
            k=checkpoint["k"],
            n_acvf_ratios=checkpoint["n_acvf_ratios"],
            n_quantiles=checkpoint["n_quantiles"],
            eps=checkpoint["eps"],
            compute_arima_coeff=checkpoint["compute_arima_coeff"],
            center_y=checkpoint["center_y"],
            remove_NaNs=checkpoint["remove_NaNs"],
        )

    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float32)
    z_std = np.asarray(checkpoint["z_std"], dtype=np.float32)
    return ((summaries - z_mean) / z_std).astype(np.float32)


def prepare_tcn_input(y, checkpoint) -> np.ndarray:
    centered_y = y - np.mean(y, axis=1, keepdims=True)
    return np.log(centered_y**2 + checkpoint["k"]).astype(np.float32)


def prepare_model_input(loaded_model, y) -> np.ndarray:
    if loaded_model.architecture == "Summary NN":
        return prepare_summary_input(y, loaded_model.checkpoint)
    return prepare_tcn_input(y, loaded_model.checkpoint)


@torch.inference_mode()
def predict(loaded_model, y) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(prepare_model_input(loaded_model, y)).to(DEVICE)
    mean, variance = loaded_model.model(x)
    return (
        mean.cpu().numpy().astype(np.float64),
        variance.cpu().numpy().astype(np.float64),
    )


def synchronize_device() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


@torch.inference_mode()
def predict_with_runtimes(loaded_model, y):
    """Time preprocessing plus mean/variance prediction for every sequence."""
    if loaded_model.architecture == "Summary NN":
        dummy_shape = (1, loaded_model.checkpoint["input_dim"])
    else:
        dummy_shape = (1, SEQUENCE_LENGTH)
    loaded_model.model(torch.zeros(dummy_shape, dtype=torch.float32, device=DEVICE))
    synchronize_device()

    means = np.empty((len(y), 3), dtype=np.float64)
    variances = np.empty((len(y), 3), dtype=np.float64)
    runtimes = np.empty(len(y), dtype=np.float64)

    for index, series in enumerate(y):
        synchronize_device()
        started_at = perf_counter()

        x = prepare_model_input(loaded_model, series[None, :])
        x = torch.from_numpy(x).to(DEVICE)
        mean, variance = loaded_model.model(x)
        means[index] = mean[0].cpu().numpy()
        variances[index] = variance[0].cpu().numpy()

        synchronize_device()
        runtimes[index] = perf_counter() - started_at

    return means, variances, runtimes


def simulate_ci_series():
    rng = np.random.default_rng(CI_SEED)
    datasets = {}

    for swept_parameter, values in SWEEPS.items():
        parameters = {
            name: np.full(CI_SWEEP_SIZE, value)
            for name, value in BASELINE.items()
        }
        parameters[swept_parameter] = values
        datasets[swept_parameter] = sim.simulate_sv_chunk(
            mu=parameters["mu"],
            phi=parameters["phi"],
            s=np.sqrt(parameters["sigma2"]),
            r=np.zeros(CI_SWEEP_SIZE),
            nu=np.full(CI_SWEEP_SIZE, np.inf),
            n=SEQUENCE_LENGTH,
            rng=rng,
            random_init=True,
        )

    return datasets


def transformed_gaussian_ci(mean, variance) -> pd.DataFrame:
    z = NormalDist().inv_cdf(1.0 - ALPHA / 2.0)
    sd = np.sqrt(variance)
    lower = mean - z * sd
    upper = mean + z * sd

    return pd.DataFrame(
        {
            "mu_median": mean[:, 0],
            "mu_ci_lower": lower[:, 0],
            "mu_ci_upper": upper[:, 0],
            "phi_median": np.tanh(mean[:, 1] / 2.0),
            "phi_ci_lower": np.tanh(lower[:, 1] / 2.0),
            "phi_ci_upper": np.tanh(upper[:, 1] / 2.0),
            "sigma2_median": np.exp(2.0 * mean[:, 2]),
            "sigma2_ci_lower": np.exp(2.0 * lower[:, 2]),
            "sigma2_ci_upper": np.exp(2.0 * upper[:, 2]),
        }
    )


def sigma_squared_transform(sigma):
    return np.asarray(sigma) ** 2


CI_MCMC_TRANSFORMS = {
    "sigma": {"sigma2": sigma_squared_transform},
}


def add_ci_rows(rows, parameter, method, prior, ci):
    if "index" in ci:
        ci = ci.sort_values("index").reset_index(drop=True)

    for value_index, true_value in enumerate(SWEEPS[parameter]):
        rows.append(
            {
                "parameter": parameter,
                "value_index": value_index,
                "true_value": true_value,
                "method": method,
                "prior": prior,
                "median": ci.loc[value_index, f"{parameter}_median"],
                "ci_lower": ci.loc[value_index, f"{parameter}_ci_lower"],
                "ci_upper": ci.loc[value_index, f"{parameter}_ci_upper"],
            }
        )


def calculate_credible_intervals(models) -> pd.DataFrame:
    datasets = simulate_ci_series()
    rows = []

    for prior in PRIORS:
        for parameter in PARAMETERS:
            y = datasets[parameter]

            print(f"Running stochvol ({prior}) for the {parameter} sweep.")
            mcmc_ci = run_stochvol_mcmc(
                y,
                prior=prior,
                draws=MCMC_DRAWS,
                burnin=MCMC_BURNIN,
                thinpara=MCMC_THINPARA,
                alpha=ALPHA,
                transforms=CI_MCMC_TRANSFORMS,
                max_cores=MCMC_MAX_CORES,
            )
            add_ci_rows(rows, parameter, "stochvol", prior, mcmc_ci)

            for architecture in ("TCN", "Summary NN"):
                mean, variance = predict(models[(architecture, prior)], y)
                ci = transformed_gaussian_ci(mean, variance)
                add_ci_rows(rows, parameter, architecture, prior, ci)

    return pd.DataFrame(rows)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
        }
    )


def plot_credible_intervals(comparison, output_path) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 13.0), sharex="row", sharey="row")
    legend_handles = {}

    for row, parameter in enumerate(PARAMETERS):
        true_values = SWEEPS[parameter]
        spacing = np.min(np.diff(true_values))
        offsets = dict(zip(METHODS, np.linspace(-0.18, 0.18, 3) * spacing))

        for column, prior in enumerate(PRIORS):
            ax = axes[row, column]
            ax.plot(true_values, true_values, color="0.35", linestyle="--")

            for method in METHODS:
                data = comparison[
                    (comparison["parameter"] == parameter)
                    & (comparison["prior"] == prior)
                    & (comparison["method"] == method)
                ].sort_values("value_index")
                median = data["median"].to_numpy()
                lower = data["ci_lower"].to_numpy()
                upper = data["ci_upper"].to_numpy()
                label = f"{method} ({prior})"
                legend_handles[label] = ax.errorbar(
                    true_values + offsets[method],
                    median,
                    yerr=np.vstack((median - lower, upper - median)),
                    fmt=MARKERS[method],
                    color=COLORS[prior][method],
                    capsize=3,
                    markersize=4.5,
                    label=label,
                )

            parameter_label = PARAMETER_LABELS[parameter]
            ax.set_title(f"{parameter_label} — {prior.capitalize()} prior")
            ax.set_xlabel(f"True {parameter_label}")
            ax.set_ylabel(f"Posterior {parameter_label}")
            ax.grid(alpha=0.25)

    labels = [f"{method} ({prior})" for prior in PRIORS for method in METHODS]
    fig.suptitle("95% credible intervals for the three-parameter SV model")
    fig.legend(
        [legend_handles[label] for label in labels],
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


METRIC_MCMC_TRANSFORMS = {
    "mu": {
        "mu_centered_sq": centered_square_transform,
    },
    "phi": {
        "psi": psi_transform,
        "psi_centered_sq": psi_sq_transform,
    },
    "sigma": {
        "rho": log_positive_transform,
        "rho_centered_sq": log_positive_sq_transform,
    },
}


def simulate_benchmark(prior, seed):
    rng = np.random.default_rng(seed)
    mu, phi, sigma, r, nu = sim.sample_stochvol_prior(
        BENCHMARK_SIZE,
        rng=rng,
        prior=prior,
        fixed_r=0.0,
        fixed_nu=np.inf,
        return_s2=False,
        dtype=np.float64,
    )
    y = sim.simulate_sv_chunk(
        mu=mu,
        phi=phi,
        s=sigma,
        r=r,
        nu=nu,
        n=SEQUENCE_LENGTH,
        rng=rng,
        random_init=True,
    )
    targets = np.column_stack(
        (mu, psi_transform(phi), log_positive_transform(sigma))
    )
    return y, targets


def mcmc_moments(summary):
    summary = summary.sort_values("index")
    means = summary[["mu_mean", "psi_mean", "rho_mean"]].to_numpy()
    variances = summary[
        [
            "mu_centered_sq_mean",
            "psi_centered_sq_mean",
            "rho_centered_sq_mean",
        ]
    ].to_numpy()
    runtimes = summary["runtime_seconds"].to_numpy()
    return means, variances, runtimes


def gaussian_loss(targets, means, variances):
    return 0.5 * (np.log(variances) + (targets - means) ** 2 / variances)


def metric_row(
    method,
    prior,
    targets,
    means,
    variances,
    runtimes,
    mcmc_loss,
    mcmc_variances,
):
    loss_difference = np.mean(
        gaussian_loss(targets, means, variances) - mcmc_loss,
        axis=0,
    )
    rmse = np.sqrt(np.mean((targets - means) ** 2, axis=0))
    log_variance_ratio = np.mean(np.log(variances / mcmc_variances), axis=0)

    row = {"model": f"{method} ({prior})"}
    for index, parameter in enumerate(TRANSFORMED_PARAMETERS):
        row[f"delta_loss_{parameter}"] = loss_difference[index]
        row[f"rmse_{parameter}"] = rmse[index]
        row[f"log_var_{parameter}"] = log_variance_ratio[index]
    row["mean_runtime_seconds"] = np.mean(runtimes)
    row["sd_runtime_seconds"] = np.std(runtimes, ddof=1)
    return row


def calculate_metrics(models) -> pd.DataFrame:
    rows = []

    for prior_index, prior in enumerate(PRIORS):
        print(f"\nGenerating {BENCHMARK_SIZE} benchmark series for the {prior} prior.")
        y, targets = simulate_benchmark(prior, METRIC_SEED + prior_index)

        print(
            f"Running stochvol ({prior}) with {MCMC_DRAWS:,} draws per sequence."
        )
        mcmc_summary = run_stochvol_mcmc(
            y,
            prior=prior,
            draws=MCMC_DRAWS,
            burnin=MCMC_BURNIN,
            thinpara=MCMC_THINPARA,
            alpha=ALPHA,
            transforms=METRIC_MCMC_TRANSFORMS,
            max_cores=MCMC_MAX_CORES,
        )
        mcmc_mean, mcmc_variance, mcmc_runtimes = mcmc_moments(mcmc_summary)
        mcmc_loss = gaussian_loss(targets, mcmc_mean, mcmc_variance)

        rows.append(
            metric_row(
                "stochvol",
                prior,
                targets,
                mcmc_mean,
                mcmc_variance,
                mcmc_runtimes,
                mcmc_loss,
                mcmc_variance,
            )
        )

        for architecture in ("TCN", "Summary NN"):
            print(f"Timing {architecture} ({prior}) one sequence at a time.")
            mean, variance, runtimes = predict_with_runtimes(
                models[(architecture, prior)], y
            )
            display_name = "summary NN" if architecture == "Summary NN" else "TCN"
            rows.append(
                metric_row(
                    display_name,
                    prior,
                    targets,
                    mean,
                    variance,
                    runtimes,
                    mcmc_loss,
                    mcmc_variance,
                )
            )

    columns = ["model"]
    columns += [f"delta_loss_{name}" for name in TRANSFORMED_PARAMETERS]
    columns += [f"rmse_{name}" for name in TRANSFORMED_PARAMETERS]
    columns += [f"log_var_{name}" for name in TRANSFORMED_PARAMETERS]
    columns += ["mean_runtime_seconds", "sd_runtime_seconds"]
    return pd.DataFrame(rows)[columns]


def main0() -> None:
    """Create the fixed 3-by-2 credible-interval plot."""
    models = load_models()
    counts = print_parameter_counts(models)
    comparison = calculate_credible_intervals(models)

    OUTPUT_DIR.mkdir(exist_ok=True)
    comparison_path = OUTPUT_DIR / "three_parameter_sigma2_credible_intervals.csv"
    plot_path = OUTPUT_DIR / "three_parameter_credible_intervals.pdf"
    counts_path = OUTPUT_DIR / "neural_network_parameter_counts.csv"

    comparison.to_csv(comparison_path, index=False)
    counts.to_csv(counts_path, index=False)
    plot_credible_intervals(comparison, plot_path)

    print(f"\nSaved credible intervals to {comparison_path}")
    print(f"Saved the 3-by-2 plot to {plot_path}")


def main1() -> None:
    """Run the fixed 2,000-sequence metrics and runtime benchmark."""
    models = load_models()
    counts = print_parameter_counts(models)
    metrics = calculate_metrics(models)

    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics_path = OUTPUT_DIR / "transformed_posterior_benchmark_metrics.csv"
    counts_path = OUTPUT_DIR / "neural_network_parameter_counts.csv"

    metrics.to_csv(metrics_path, index=False)
    counts.to_csv(counts_path, index=False)

    print("\nMetric and runtime comparison:")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main1()
