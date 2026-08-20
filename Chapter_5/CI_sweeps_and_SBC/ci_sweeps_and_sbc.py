"""Credible-interval sweeps and SBC for the five-parameter SV-GHST TCN."""

import os
import sys
import tempfile
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from matplotlib.axes import Axes
from scipy.special import expit
from scipy.stats import kstest, norm

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.test_ghst_sv_tcn import load_model, predict
from simulation import sim_5_param_data as sim
from simulation.stochvol_mcmc import run_stochvol_mcmc
from training.sbt_tcn import SVGHST_TARGET_NAMES, theta_to_target_numpy


CHECKPOINT_PATH = PROJECT_DIR / "weights" / "svghst_tcn_default.pt"
SEQUENCE_LENGTH = 253 * 10
PREDICTION_BATCH_SIZE = 128
DEVICE_NAME: str | None = None
POSTERIOR_VAR_EPS = 1e-12

ALPHA = 0.05
CI_SWEEP_SIZE = 10
CI_SEED = 2
MCMC_DRAWS = 20_000
MCMC_BURNIN = 500
MCMC_THINPARA = 1
MCMC_MAX_CORES = 3

BENCHMARK_SIZE = 5_000
BENCHMARK_SEED = 3
SBC_BINS = 50

PARAMETERS = ("mu", "phi", "s", "r", "nu")
TRANSFORMED_PARAMETERS = tuple(SVGHST_TARGET_NAMES)
MCMC_PARAMETERS = ("mu", "phi", "s")
METHODS = ("stochvol", "TCN")
NU_MIN = sim.get_gh_skew_t_prior_constants("default").nu_min

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

COLORS = {"stochvol": "#0000ff", "TCN": "#ff0000"}
MARKERS = {"stochvol": "o", "TCN": "^"}
SBC_COLOR = "#4C78A8"
PARAMETER_LABELS = {
    "mu": r"$\mu$",
    "phi": r"$\phi$",
    "s": r"$\sigma$",
    "r": r"$r$",
    "nu": r"$\nu$",
}

CI_PATH = HERE / "five_parameter_ci_sweeps.csv"
SBC_PREDICTIONS_PATH = (
    HERE / f"five_parameter_sbc_predictions_n{BENCHMARK_SIZE}.csv"
)
SBC_CDF_PATH = HERE / f"five_parameter_sbc_cdf_values_n{BENCHMARK_SIZE}.csv"
SBC_METRICS_PATH = HERE / f"five_parameter_sbc_metrics_n{BENCHMARK_SIZE}.csv"
STANDARD_SV_FIGURE_PATH = HERE / "standard_sv_ci_sbc.pdf"
GHST_SV_FIGURE_PATH = HERE / "ghst_sv_ci_sbc.pdf"


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


def transformed_gaussian_ci(
    means: np.ndarray,
    variances: np.ndarray,
) -> pd.DataFrame:
    critical_value = NormalDist().inv_cdf(1.0 - ALPHA / 2.0)
    sd = np.sqrt(np.clip(variances, POSTERIOR_VAR_EPS, None))
    median = inverse_transform(means)
    lower = inverse_transform(means - critical_value * sd)
    upper = inverse_transform(means + critical_value * sd)

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


def ci_rows(
    parameter: str,
    method: str,
    intervals: pd.DataFrame,
    interval_parameter: str | None = None,
) -> list[dict[str, object]]:
    if "index" in intervals:
        intervals = intervals.sort_values("index").reset_index(drop=True)
    interval_parameter = interval_parameter or parameter

    return [
        {
            "parameter": parameter,
            "value_index": value_index,
            "true_value": true_value,
            "method": method,
            "median": intervals.loc[
                value_index,
                f"{interval_parameter}_median",
            ],
            "ci_lower": intervals.loc[
                value_index,
                f"{interval_parameter}_ci_lower",
            ],
            "ci_upper": intervals.loc[
                value_index,
                f"{interval_parameter}_ci_upper",
            ],
        }
        for value_index, true_value in enumerate(SWEEPS[parameter])
    ]


def calculate_credible_intervals(
    model: nn.Module,
    checkpoint: dict[str, object],
    include_mcmc: bool = True,
) -> pd.DataFrame:
    datasets = simulate_ci_series()
    rows: list[dict[str, object]] = []

    for parameter, y in datasets.items():
        if include_mcmc and parameter in MCMC_PARAMETERS:
            print(f"Running stochvol for the {parameter} sweep.")
            mcmc_intervals = run_stochvol_mcmc(
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
            rows.extend(
                ci_rows(
                    parameter,
                    "stochvol",
                    mcmc_intervals,
                    mcmc_parameter,
                )
            )

        print(f"Predicting TCN intervals for the {parameter} sweep.")
        means, variances = predict(
            model,
            checkpoint,
            y,
            batch_size=PREDICTION_BATCH_SIZE,
        )
        rows.extend(
            ci_rows(
                parameter,
                "TCN",
                transformed_gaussian_ci(means, variances),
            )
        )

    return pd.DataFrame(rows)


def run_sbc(
    model: nn.Module,
    checkpoint: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Simulating {BENCHMARK_SIZE:,} benchmark datasets.")
    rng = np.random.default_rng(BENCHMARK_SEED)
    theta = np.column_stack(
        sim.sample_stochvol_prior(
            BENCHMARK_SIZE,
            rng=rng,
            prior="default",
            return_s2=False,
            dtype=np.float64,
        )
    )
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
    posterior_means, posterior_variances = predict(
        model,
        checkpoint,
        y,
        batch_size=PREDICTION_BATCH_SIZE,
    )
    posterior_sd = np.sqrt(
        np.clip(posterior_variances, POSTERIOR_VAR_EPS, None)
    )
    cdf_values = norm.cdf(
        (transformed_theta - posterior_means) / posterior_sd
    )

    predictions = build_sbc_prediction_frame(
        theta,
        transformed_theta,
        posterior_means,
        posterior_sd,
        cdf_values,
    )
    cdf_frame = pd.DataFrame(cdf_values, columns=PARAMETERS)
    metrics = sbc_uniformity_metrics(cdf_values)
    return predictions, cdf_frame, metrics


def build_sbc_prediction_frame(
    theta: np.ndarray,
    transformed_theta: np.ndarray,
    posterior_means: np.ndarray,
    posterior_sd: np.ndarray,
    cdf_values: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        theta,
        columns=[f"theta_{parameter}" for parameter in PARAMETERS],
    )
    for index, (parameter, transformed_parameter) in enumerate(
        zip(PARAMETERS, TRANSFORMED_PARAMETERS)
    ):
        frame[f"target_{transformed_parameter}"] = transformed_theta[:, index]
        frame[f"posterior_mean_{transformed_parameter}"] = posterior_means[:, index]
        frame[f"posterior_sd_{transformed_parameter}"] = posterior_sd[:, index]
        frame[f"sbc_cdf_{parameter}"] = cdf_values[:, index]
    return frame


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


def main() -> None:
    print(f"Loading {CHECKPOINT_PATH.name}.")
    model, checkpoint = load_model(CHECKPOINT_PATH, device=DEVICE_NAME)
    if int(checkpoint["sequence_length"]) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Checkpoint length is {checkpoint['sequence_length']}, "
            f"expected {SEQUENCE_LENGTH}."
        )

    comparison = calculate_credible_intervals(model, checkpoint)
    comparison.to_csv(CI_PATH, index=False)

    predictions, cdf_frame, sbc_metrics = run_sbc(model, checkpoint)
    predictions.to_csv(SBC_PREDICTIONS_PATH, index_label="simulation")
    cdf_frame.to_csv(SBC_CDF_PATH, index_label="simulation")
    sbc_metrics.to_csv(SBC_METRICS_PATH, index=False)
    print_sbc_metrics(sbc_metrics)

    cdf_values = cdf_frame.to_numpy()
    plot_ci_and_sbc(
        comparison,
        cdf_values,
        sbc_metrics,
        PARAMETERS[:3],
        STANDARD_SV_FIGURE_PATH,
    )
    plot_ci_and_sbc(
        comparison,
        cdf_values,
        sbc_metrics,
        PARAMETERS[3:],
        GHST_SV_FIGURE_PATH,
    )

    print(f"\nSaved credible intervals to {CI_PATH}")
    print(f"Saved SBC predictions to {SBC_PREDICTIONS_PATH}")
    print(f"Saved SBC CDF values to {SBC_CDF_PATH}")
    print(f"Saved SBC metrics to {SBC_METRICS_PATH}")
    print(f"Saved standard-SV plot to {STANDARD_SV_FIGURE_PATH}")
    print(f"Saved GHST-SV plot to {GHST_SV_FIGURE_PATH}")


if __name__ == "__main__":
    main()
