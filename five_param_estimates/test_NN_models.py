"""Test the three standard-SV posterior estimators.

There are deliberately only three entry points:

``main0()`` creates the credible-interval and loss-history figures.
``main1()`` runs the fixed 2,000-sequence metric, coverage and runtime benchmark.
``main2()`` formats the saved metrics and estimates mean-loss uncertainty.

The comparisons use the same four neural checkpoints and stochvol under the
default and finance priors.
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
COVERAGE_ALPHAS = np.arange(5, 51, 5) / 100.0
MCMC_DATA_PERCENTAGES = (100, 95, 90, 85, 80)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = HERE.parent / "weights"
OUTPUT_DIR = HERE / "nn_model_tests"
LATEX_TABLES_DIR = OUTPUT_DIR / "latex_tables"

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
    "stochvol": "#0000ff",
    "TCN": "#ff0000",
    "Summary NN": "#008000",
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

TRANSFORMED_PARAMETER_LABELS = {
    "mu": r"$\mu$",
    "psi": r"$\psi$",
    "rho": r"$\rho$",
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
            row = {
                "architecture": architecture,
                "prior": prior,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "total_parameters": sum(p.numel() for p in model.parameters()),
            }
            for parameter, head in zip(TRANSFORMED_PARAMETERS, model.heads.values()):
                row[f"{parameter}_head_parameters"] = sum(
                    p.numel() for p in head.parameters() if p.requires_grad
                )
            rows.append(row)

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
                legend_handles[method] = ax.errorbar(
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
            if row == 0:
                ax.set_title(f"{prior.capitalize()} prior")
            ax.set_xlabel(f"True {parameter_label}")
            if column == 0:
                ax.set_ylabel(f"{parameter_label} posterior mean")
            ax.grid(alpha=0.25)

    fig.legend(
        [legend_handles[method] for method in METHODS],
        METHODS,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_histories(models, output_path) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    legend_handles = {}

    for column, prior in enumerate(PRIORS):
        ax = axes[column]

        for architecture in ("TCN", "Summary NN"):
            checkpoint = models[(architecture, prior)].checkpoint
            marginal_losses = np.asarray(
                checkpoint["val_marginal_loss_history"], dtype=np.float64
            )
            losses = np.mean(marginal_losses, axis=1)
            epochs = np.arange(1, len(losses) + 1)

            legend_handles[architecture] = ax.plot(
                epochs,
                losses,
                color=COLORS[architecture],
                label=architecture,
            )[0]
            legend_handles["Final epoch"] = ax.plot(
                epochs[-1],
                losses[-1],
                color="black",
                marker="x",
                markersize=8,
                markeredgewidth=1.8,
                linestyle="none",
                label="Final epoch",
            )[0]

        ax.set_title(f"{prior.capitalize()} prior")
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Mean marginal Gaussian loss")
    legend_order = ("TCN", "Summary NN", "Final epoch")
    fig.legend(
        [legend_handles[name] for name in legend_order],
        legend_order,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
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


def mean_loss_samples(method, prior, targets, means, variances):
    losses = gaussian_loss(targets, means, variances)
    return pd.DataFrame(
        {
            "method": method,
            "prior": prior,
            "benchmark_index": np.arange(len(targets)),
            **{
                f"loss_{parameter}": losses[:, index]
                for index, parameter in enumerate(TRANSFORMED_PARAMETERS)
            },
            "mean_loss": np.mean(losses, axis=1),
        }
    )


def empirical_coverage_rows(method, prior, targets, means, variances):
    standardized_errors = np.abs(targets - means) / np.sqrt(variances)
    rows = []

    for alpha in COVERAGE_ALPHAS:
        nominal_coverage = 1.0 - alpha
        critical_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        empirical_coverage = np.mean(
            standardized_errors <= critical_value,
            axis=0,
        )

        for index, parameter in enumerate(TRANSFORMED_PARAMETERS):
            rows.append(
                {
                    "method": method,
                    "prior": prior,
                    "parameter": parameter,
                    "alpha": alpha,
                    "nominal_coverage": nominal_coverage,
                    "empirical_coverage": empirical_coverage[index],
                }
            )

    return rows


def metric_row(
    method,
    prior,
    data_percentage,
    n_observations,
    targets,
    means,
    variances,
    runtimes,
    reference_variances,
):
    marginal_loss = np.mean(gaussian_loss(targets, means, variances), axis=0)
    rmse = np.sqrt(np.mean((targets - means) ** 2, axis=0))
    log_variance_ratio = np.mean(
        np.log(variances / reference_variances), axis=0
    )

    row = {
        "method": method,
        "prior": prior,
        "data_percentage": data_percentage,
        "n_observations": n_observations,
    }
    for index, parameter in enumerate(TRANSFORMED_PARAMETERS):
        row[f"marginal_loss_{parameter}"] = marginal_loss[index]
        row[f"rmse_{parameter}"] = rmse[index]
        row[f"log_var_ratio_{parameter}"] = log_variance_ratio[index]
    row["mean_runtime_seconds"] = np.mean(runtimes)
    row["sd_runtime_seconds"] = np.std(runtimes, ddof=1)
    return row


def run_mcmc_benchmarks(y, prior, data_percentages):
    results = {}

    for data_percentage in data_percentages:
        n_observations = y.shape[1] * data_percentage // 100
        print(
            f"Running stochvol ({prior}, {data_percentage}% data, "
            f"n={n_observations}) with {MCMC_DRAWS:,} draws per sequence."
        )
        summary = run_stochvol_mcmc(
            y[:, :n_observations],
            prior=prior,
            draws=MCMC_DRAWS,
            burnin=MCMC_BURNIN,
            thinpara=MCMC_THINPARA,
            alpha=ALPHA,
            transforms=METRIC_MCMC_TRANSFORMS,
            max_cores=MCMC_MAX_CORES,
        )
        results[data_percentage] = (
            n_observations,
            *mcmc_moments(summary),
        )

    return results


def metric_columns():
    columns = ["method", "prior", "data_percentage", "n_observations"]
    columns += [f"marginal_loss_{name}" for name in TRANSFORMED_PARAMETERS]
    columns += [f"rmse_{name}" for name in TRANSFORMED_PARAMETERS]
    columns += [f"log_var_ratio_{name}" for name in TRANSFORMED_PARAMETERS]
    columns += ["mean_runtime_seconds", "sd_runtime_seconds"]
    return columns


def calculate_metrics(models) -> tuple[pd.DataFrame, pd.DataFrame]:
    npe_rows = []
    mcmc_rows = []
    coverage_rows = []

    for prior_index, prior in enumerate(PRIORS):
        print(f"\nGenerating {BENCHMARK_SIZE} benchmark series for the {prior} prior.")
        y, targets = simulate_benchmark(prior, METRIC_SEED + prior_index)

        mcmc_results = run_mcmc_benchmarks(y, prior, MCMC_DATA_PERCENTAGES)

        _, mcmc_mean, mcmc_variance, _ = mcmc_results[100]
        coverage_rows.extend(
            empirical_coverage_rows(
                "stochvol",
                prior,
                targets,
                mcmc_mean,
                mcmc_variance,
            )
        )

        for architecture in ("TCN", "Summary NN"):
            print(f"Timing {architecture} ({prior}) one sequence at a time.")
            mean, variance, runtimes = predict_with_runtimes(
                models[(architecture, prior)], y
            )
            npe_rows.append(
                metric_row(
                    architecture,
                    prior,
                    100,
                    y.shape[1],
                    targets,
                    mean,
                    variance,
                    runtimes,
                    mcmc_variance,
                )
            )
            coverage_rows.extend(
                empirical_coverage_rows(
                    architecture,
                    prior,
                    targets,
                    mean,
                    variance,
                )
            )

        for data_percentage in MCMC_DATA_PERCENTAGES:
            n_observations, mean, variance, runtimes = mcmc_results[
                data_percentage
            ]
            mcmc_rows.append(
                metric_row(
                    "stochvol",
                    prior,
                    data_percentage,
                    n_observations,
                    targets,
                    mean,
                    variance,
                    runtimes,
                    mcmc_variance,
                )
            )

    metrics = pd.DataFrame(npe_rows + mcmc_rows)[metric_columns()]
    coverage = pd.DataFrame(coverage_rows)
    return metrics, coverage


def calculate_full_data_loss_samples(models) -> pd.DataFrame:
    frames = []

    for prior_index, prior in enumerate(PRIORS):
        print(f"\nRegenerating {BENCHMARK_SIZE} series for the {prior} prior.")
        y, targets = simulate_benchmark(prior, METRIC_SEED + prior_index)
        mcmc_results = run_mcmc_benchmarks(y, prior, (100,))
        _, mean, variance, _ = mcmc_results[100]
        frames.append(
            mean_loss_samples("stochvol", prior, targets, mean, variance)
        )

        for architecture in ("TCN", "Summary NN"):
            print(f"Predicting {architecture} ({prior}) for the loss diagnostic.")
            mean, variance = predict(models[(architecture, prior)], y)
            frames.append(
                mean_loss_samples(
                    architecture,
                    prior,
                    targets,
                    mean,
                    variance,
                )
            )

    return pd.concat(frames, ignore_index=True)


def loss_component_columns():
    return {
        **{
            parameter: f"loss_{parameter}"
            for parameter in TRANSFORMED_PARAMETERS
        },
        "mean": "mean_loss",
    }


def uncertainty_row(
    estimate_type,
    method,
    reference_method,
    prior,
    component,
    values,
):
    values = np.asarray(values, dtype=np.float64)
    sample_variance = np.var(values, ddof=1)
    variance_of_mean = sample_variance / len(values)
    return {
        "estimate_type": estimate_type,
        "method": method,
        "reference_method": reference_method,
        "prior": prior,
        "component": component,
        "benchmark_size": len(values),
        "estimate": np.mean(values),
        "sample_sd": np.sqrt(sample_variance),
        "estimated_variance_of_mean": variance_of_mean,
        "estimated_sd_of_mean": np.sqrt(variance_of_mean),
    }


def summarize_loss_sampling_uncertainty(loss_samples) -> pd.DataFrame:
    rows = []
    components = loss_component_columns()

    for prior in PRIORS:
        method_samples = {}
        for method in METHODS:
            selected = loss_samples[
                (loss_samples["method"] == method)
                & (loss_samples["prior"] == prior)
            ].sort_values("benchmark_index")
            if (
                len(selected) != BENCHMARK_SIZE
                or selected["benchmark_index"].nunique() != BENCHMARK_SIZE
            ):
                raise ValueError(
                    f"Expected {BENCHMARK_SIZE} loss samples for {method} "
                    f"({prior}), found {len(selected)}."
                )
            if not np.array_equal(
                selected["benchmark_index"].to_numpy(),
                np.arange(BENCHMARK_SIZE),
            ):
                raise ValueError(
                    f"Unexpected benchmark indices for {method} ({prior})."
                )

            method_samples[method] = selected
            for component, column in components.items():
                values = selected[column].to_numpy(dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Non-finite {component} loss found for {method} ({prior})."
                    )
                rows.append(
                    uncertainty_row(
                        "raw_loss",
                        method,
                        "",
                        prior,
                        component,
                        values,
                    )
                )

        reference = method_samples["stochvol"]
        for method in ("TCN", "Summary NN"):
            for component, column in components.items():
                differences = (
                    method_samples[method][column].to_numpy(dtype=np.float64)
                    - reference[column].to_numpy(dtype=np.float64)
                )
                rows.append(
                    uncertainty_row(
                        "paired_loss_difference",
                        method,
                        "stochvol",
                        prior,
                        component,
                        differences,
                    )
                )

    return pd.DataFrame(rows)


def mean_loss_uncertainty_summary(uncertainty) -> pd.DataFrame:
    selected = uncertainty[
        (uncertainty["estimate_type"] == "raw_loss")
        & (uncertainty["component"] == "mean")
    ].copy()
    return selected[
        [
            "method",
            "prior",
            "benchmark_size",
            "estimate",
            "estimated_variance_of_mean",
            "estimated_sd_of_mean",
        ]
    ].rename(
        columns={
            "estimate": "mean_loss",
            "estimated_variance_of_mean": "estimated_variance_of_mean_loss",
            "estimated_sd_of_mean": "estimated_sd_of_mean_loss",
        }
    )


def plot_empirical_coverage(coverage, output_path) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 13.0), sharex=True, sharey="row")
    legend_handles = {}

    nominal_limits = (
        1.0 - np.max(COVERAGE_ALPHAS),
        1.0 - np.min(COVERAGE_ALPHAS),
    )

    for row, parameter in enumerate(TRANSFORMED_PARAMETERS):
        for column, prior in enumerate(PRIORS):
            ax = axes[row, column]
            ax.plot(nominal_limits, nominal_limits, color="0.35", linestyle="--")

            for method in METHODS:
                data = coverage[
                    (coverage["parameter"] == parameter)
                    & (coverage["prior"] == prior)
                    & (coverage["method"] == method)
                ].sort_values("nominal_coverage")
                legend_handles[method] = ax.plot(
                    data["nominal_coverage"],
                    data["empirical_coverage"],
                    color=COLORS[method],
                    marker=MARKERS[method],
                    markersize=4.5,
                    label=method,
                )[0]

            parameter_label = TRANSFORMED_PARAMETER_LABELS[parameter]
            if row == 0:
                ax.set_title(f"{prior.capitalize()} prior")
            ax.set_xlabel("Nominal coverage level")
            if column == 0:
                ax.set_ylabel(f"Empirical {parameter_label} coverage")
            ax.grid(alpha=0.25)

    fig.legend(
        [legend_handles[method] for method in METHODS],
        METHODS,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def latex_number(value) -> str:
    value = float(value)
    if abs(value) < 0.00005:
        value = 0.0
    return rf"\({value:.4f}\)"


def latex_scientific_number(value) -> str:
    value = float(value)
    if value == 0.0:
        return r"\(0\)"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10.0**exponent
    return rf"\({coefficient:.3f}\times 10^{{{exponent}}}\)"


def latex_estimate_sd(estimate, sd) -> str:
    estimate = float(estimate)
    sd = float(sd)
    if abs(estimate) < 0.00005:
        estimate = 0.0
    if abs(sd) < 0.00005:
        sd = 0.0
    return rf"\({estimate:.4f}\mathbin{{\pm}}{sd:.4f}\)"


def latex_row(values) -> str:
    return " & ".join(values) + r" \\"


def latex_metric_row(label, values) -> str:
    return latex_row([label, *(latex_number(value) for value in values)])


def select_metric_row(metrics, method, prior, data_percentage=100):
    selected = metrics[
        (metrics["method"] == method)
        & (metrics["prior"] == prior)
        & (metrics["data_percentage"] == data_percentage)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {method} ({prior}, {data_percentage}%) metric row, "
            f"found {len(selected)}."
        )
    return selected.iloc[0]


def loss_table_body(metrics, prior) -> str:
    loss_columns = [
        f"marginal_loss_{parameter}" for parameter in TRANSFORMED_PARAMETERS
    ]
    reference = select_metric_row(metrics, "stochvol", prior)
    reference_loss = reference[loss_columns].to_numpy(dtype=np.float64)

    npe_rows = [
        select_metric_row(metrics, method, prior)
        for method in ("TCN", "Summary NN")
    ]
    npe_losses = np.vstack(
        [row[loss_columns].to_numpy(dtype=np.float64) for row in npe_rows]
    )
    loss_differences = npe_losses - reference_loss
    best_index = int(np.argmin(np.mean(loss_differences, axis=1)))
    best_npe = npe_rows[best_index]
    best_npe_loss = npe_losses[best_index]

    mcmc_candidates = metrics[
        (metrics["method"] == "stochvol")
        & (metrics["prior"] == prior)
        & (metrics["data_percentage"] < 100)
    ]
    if mcmc_candidates.empty:
        raise ValueError(f"No reduced-data MCMC rows found for the {prior} prior.")

    candidate_mean_losses = mcmc_candidates[loss_columns].mean(axis=1)
    closest_index = (candidate_mean_losses - np.mean(best_npe_loss)).abs().idxmin()
    closest_mcmc = mcmc_candidates.loc[closest_index]
    closest_mcmc_loss = closest_mcmc[loss_columns].to_numpy(dtype=np.float64)
    closest_percentage = int(closest_mcmc["data_percentage"])

    rows = [
        r"\hline",
        latex_row(
            [
                "Method",
                r"\(\Delta_{\mathrm{loss},\mu}\)",
                r"\(\Delta_{\mathrm{loss},\psi}\)",
                r"\(\Delta_{\mathrm{loss},\rho}\)",
                r"\(\overline{\Delta}_{\mathrm{loss}}\)",
            ]
        ),
        r"\hline",
        latex_metric_row(
            r"MCMC (\(100\%\), raw loss)",
            [*reference_loss, np.mean(reference_loss)],
        ),
        r"\hline",
    ]
    for row, differences in zip(npe_rows, loss_differences):
        rows.append(
            latex_metric_row(
                f"{row['method']} (loss difference)",
                [*differences, np.mean(differences)],
            )
        )
    rows += [
        r"\hline",
        latex_metric_row(
            f"MCMC (\\({closest_percentage}\\%\\), raw loss)",
            [*closest_mcmc_loss, np.mean(closest_mcmc_loss)],
        ),
        latex_metric_row(
            f"{best_npe['method']} (raw loss)",
            [*best_npe_loss, np.mean(best_npe_loss)],
        ),
        r"\hline",
    ]
    return "\n".join(rows) + "\n"


def rmse_log_ratio_table_body(metrics, prior) -> str:
    columns = [
        *(f"rmse_{parameter}" for parameter in TRANSFORMED_PARAMETERS),
        *(f"log_var_ratio_{parameter}" for parameter in TRANSFORMED_PARAMETERS),
    ]
    rows = [
        r"\hline",
        latex_row(
            [
                "Method",
                r"\(\mathrm{RMSE}_{\mu}\)",
                r"\(\mathrm{RMSE}_{\psi}\)",
                r"\(\mathrm{RMSE}_{\rho}\)",
                r"\(\overline{r}_{\mathrm{var},\mu}\)",
                r"\(\overline{r}_{\mathrm{var},\psi}\)",
                r"\(\overline{r}_{\mathrm{var},\rho}\)",
            ]
        ),
        r"\hline",
    ]
    for method in ("stochvol", "TCN", "Summary NN"):
        row = select_metric_row(metrics, method, prior)
        label = r"MCMC (\(100\%\))" if method == "stochvol" else method
        rows.append(latex_metric_row(label, row[columns]))
    rows.append(r"\hline")
    return "\n".join(rows) + "\n"


def runtime_table_body(metrics) -> str:
    rows = [
        r"\hline",
        latex_row(
            ["Method", "Mean runtime (seconds)", "SD runtime (seconds)"]
        ),
        r"\hline",
    ]
    for prior in PRIORS:
        for method in ("stochvol", "TCN", "Summary NN"):
            row = select_metric_row(metrics, method, prior)
            method_name = "MCMC" if method == "stochvol" else method
            label = f"{method_name} ({prior})"
            rows.append(
                latex_metric_row(
                    label,
                    [row["mean_runtime_seconds"], row["sd_runtime_seconds"]],
                )
            )
        if prior != PRIORS[-1]:
            rows.append(r"\hline")
    rows.append(r"\hline")
    return "\n".join(rows) + "\n"


def mean_loss_uncertainty_table_body(uncertainty) -> str:
    rows = [
        r"\hline",
        latex_row(
            [
                "Method",
                r"\(\overline{\ell}\)",
                r"\(\widehat{\mathrm{Var}}(\overline{\ell})\)",
                r"\(\widehat{\mathrm{SD}}(\overline{\ell})\)",
            ]
        ),
        r"\hline",
    ]
    for prior in PRIORS:
        for method in METHODS:
            selected = uncertainty[
                (uncertainty["method"] == method)
                & (uncertainty["prior"] == prior)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one mean-loss summary for {method} ({prior})."
                )
            selected = selected.iloc[0]
            method_name = "MCMC" if method == "stochvol" else method
            rows.append(
                latex_row(
                    [
                        f"{method_name} ({prior})",
                        latex_number(selected["mean_loss"]),
                        latex_scientific_number(
                            selected["estimated_variance_of_mean_loss"]
                        ),
                        latex_number(selected["estimated_sd_of_mean_loss"]),
                    ]
                )
            )
        if prior != PRIORS[-1]:
            rows.append(r"\hline")
    rows.append(r"\hline")
    return "\n".join(rows) + "\n"


def select_uncertainty_row(
    uncertainty,
    estimate_type,
    method,
    prior,
    component,
):
    selected = uncertainty[
        (uncertainty["estimate_type"] == estimate_type)
        & (uncertainty["method"] == method)
        & (uncertainty["prior"] == prior)
        & (uncertainty["component"] == component)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {estimate_type} uncertainty row for {method}, "
            f"{prior}, {component}; found {len(selected)}."
        )
    return selected.iloc[0]


def loss_uncertainty_table_body(uncertainty, prior, differences=False) -> str:
    estimate_type = "paired_loss_difference" if differences else "raw_loss"
    components = (*TRANSFORMED_PARAMETERS, "mean")
    parameter_labels = {
        "mu": r"\mu",
        "psi": r"\psi",
        "rho": r"\rho",
    }
    if differences:
        column_labels = [
            rf"\overline{{\Delta}}_{{\mathrm{{loss}},{parameter_labels[component]}}}"
            for component in TRANSFORMED_PARAMETERS
        ]
        column_labels.append(r"\overline{\Delta}_{\mathrm{loss}}")
    else:
        column_labels = [
            rf"\overline{{\ell}}_{{{parameter_labels[component]}}}"
            for component in TRANSFORMED_PARAMETERS
        ]
        column_labels.append(r"\overline{\ell}")

    rows = [
        r"\hline",
        latex_row(
            [
                "Method",
                *(
                    rf"\({column_label}"
                    rf"\mathbin{{\pm}}\widehat{{\mathrm{{SD}}}}\)"
                    for column_label in column_labels
                ),
            ]
        ),
        r"\hline",
    ]

    methods = ("TCN", "Summary NN") if differences else METHODS
    for method in methods:
        values = []
        for component in components:
            selected = select_uncertainty_row(
                uncertainty,
                estimate_type,
                method,
                prior,
                component,
            )
            values.append(
                latex_estimate_sd(
                    selected["estimate"],
                    selected["estimated_sd_of_mean"],
                )
            )

        if differences:
            label = f"{method} \\(-\\) MCMC"
        elif method == "stochvol":
            label = r"MCMC (\(100\%\))"
        else:
            label = method
        rows.append(latex_row([label, *values]))

    rows.append(r"\hline")
    return "\n".join(rows) + "\n"


def main0() -> None:
    """Create the credible-interval and Gaussian-loss-history plots."""
    models = load_models()
    counts = print_parameter_counts(models)
    comparison = calculate_credible_intervals(models)

    OUTPUT_DIR.mkdir(exist_ok=True)
    comparison_path = OUTPUT_DIR / "three_parameter_sigma2_credible_intervals.csv"
    plot_path = OUTPUT_DIR / "three_parameter_credible_intervals.pdf"
    loss_plot_path = OUTPUT_DIR / "gaussian_loss_history.pdf"
    counts_path = OUTPUT_DIR / "neural_network_parameter_counts.csv"

    comparison.to_csv(comparison_path, index=False)
    counts.to_csv(counts_path, index=False)
    plot_credible_intervals(comparison, plot_path)
    plot_loss_histories(models, loss_plot_path)

    print(f"\nSaved credible intervals to {comparison_path}")
    print(f"Saved the 3-by-2 plot to {plot_path}")
    print(f"Saved the Gaussian loss histories to {loss_plot_path}")


def main1() -> None:
    """Run the fixed 2,000-sequence metrics, coverage and runtime benchmark."""
    models = load_models()
    counts = print_parameter_counts(models)
    metrics, coverage = calculate_metrics(models)

    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics_path = OUTPUT_DIR / "transformed_posterior_benchmark_metrics.csv"
    coverage_path = OUTPUT_DIR / "transformed_posterior_empirical_coverage.csv"
    coverage_plot_path = OUTPUT_DIR / "transformed_posterior_empirical_coverage.pdf"
    counts_path = OUTPUT_DIR / "neural_network_parameter_counts.csv"

    metrics.to_csv(metrics_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    counts.to_csv(counts_path, index=False)
    plot_empirical_coverage(coverage, coverage_plot_path)

    print("\nMetric and runtime comparison:")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved empirical coverage to {coverage_path}")
    print(f"Saved the 3-by-2 coverage plot to {coverage_plot_path}")


def main2() -> None:
    """Format metrics and estimate uncertainty in full-data marginal losses."""
    metrics_path = OUTPUT_DIR / "transformed_posterior_benchmark_metrics.csv"
    loss_samples_path = OUTPUT_DIR / "full_data_mean_loss_samples.csv"
    mean_uncertainty_path = OUTPUT_DIR / "mean_loss_sampling_uncertainty.csv"
    uncertainty_path = OUTPUT_DIR / "loss_sampling_uncertainty.csv"
    metrics = pd.read_csv(metrics_path)
    missing_columns = sorted(set(metric_columns()).difference(metrics.columns))
    if missing_columns:
        raise ValueError(
            "Metric CSV is missing column(s): " + ", ".join(missing_columns)
        )

    loss_sample_columns = {
        "method",
        "prior",
        "benchmark_index",
        *loss_component_columns().values(),
    }
    if loss_samples_path.exists():
        loss_samples = pd.read_csv(loss_samples_path)
    else:
        loss_samples = pd.DataFrame()

    missing_loss_columns = sorted(
        loss_sample_columns.difference(loss_samples.columns)
    )
    if missing_loss_columns:
        if not loss_samples.empty:
            print(
                "The saved loss samples predate the parameter-specific "
                "diagnostic and will be regenerated."
            )
        models = load_models()
        loss_samples = calculate_full_data_loss_samples(models)
        loss_samples.to_csv(loss_samples_path, index=False)
        print(f"Saved per-sequence marginal losses to {loss_samples_path}")

    uncertainty = summarize_loss_sampling_uncertainty(loss_samples)
    mean_uncertainty = mean_loss_uncertainty_summary(uncertainty)
    uncertainty.to_csv(uncertainty_path, index=False)
    mean_uncertainty.to_csv(mean_uncertainty_path, index=False)

    tables = {
        "loss_default_tabular.txt": loss_table_body(metrics, "default"),
        "loss_finance_tabular.txt": loss_table_body(metrics, "finance"),
        "rmse_log_ratio_default_tabular.txt": rmse_log_ratio_table_body(
            metrics, "default"
        ),
        "rmse_log_ratio_finance_tabular.txt": rmse_log_ratio_table_body(
            metrics, "finance"
        ),
        "runtime_tabular.txt": runtime_table_body(metrics),
        "mean_loss_uncertainty_tabular.txt": mean_loss_uncertainty_table_body(
            mean_uncertainty
        ),
        "raw_loss_uncertainty_default_tabular.txt": loss_uncertainty_table_body(
            uncertainty, "default"
        ),
        "raw_loss_uncertainty_finance_tabular.txt": loss_uncertainty_table_body(
            uncertainty, "finance"
        ),
        "loss_difference_uncertainty_default_tabular.txt": (
            loss_uncertainty_table_body(uncertainty, "default", differences=True)
        ),
        "loss_difference_uncertainty_finance_tabular.txt": (
            loss_uncertainty_table_body(uncertainty, "finance", differences=True)
        ),
    }

    LATEX_TABLES_DIR.mkdir(exist_ok=True)
    for filename, table in tables.items():
        output_path = LATEX_TABLES_DIR / filename
        output_path.write_text(table, encoding="utf-8")
        print(f"Saved LaTeX tabular body to {output_path}")

    print("\nLoss sampling uncertainty:")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(
            uncertainty.to_string(
                index=False,
                float_format=lambda value: f"{value:.6g}",
            )
        )
    print(f"\nSaved detailed loss uncertainty to {uncertainty_path}")
    print(f"Saved mean-loss uncertainty to {mean_uncertainty_path}")


if __name__ == "__main__":
    main2()
