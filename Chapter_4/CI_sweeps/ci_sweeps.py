"""Compare credible intervals across standard-SV posterior estimators."""

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

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.test_sv_nn_model import load_model, predict
from simulation import sim_5_param_data as sim
from simulation.stochvol_mcmc import run_stochvol_mcmc


ALPHA = 0.05
SEQUENCE_LENGTH = 253
MCMC_DRAWS = 20_000
MCMC_BURNIN = 500
MCMC_THINPARA = 1
MCMC_MAX_CORES = -2
SWEEP_SIZE = 10
SEED = 2

PRIORS = ("default", "finance")
PARAMETERS = ("mu", "phi", "sigma")
METHODS = ("stochvol", "TCN", "Summary NN")

WEIGHTS_DIR = PROJECT_DIR / "weights"
CHECKPOINT_NAMES = {
    ("Summary NN", "default"): "summary_nn_default_arima.pt",
    ("Summary NN", "finance"): "summary_nn_finance_arima.pt",
    ("TCN", "default"): "tcn_default.pt",
    ("TCN", "finance"): "tcn_finance.pt",
}

BASELINE = {
    "mu": -9.0,
    "phi": 0.95,
    "sigma": 0.25,
}
SWEEPS = {
    "mu": np.linspace(-12.0, -6.0, SWEEP_SIZE),
    "phi": np.linspace(0.905, 0.995, SWEEP_SIZE),
    "sigma": np.linspace(0.05, 0.45, SWEEP_SIZE),
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
    "sigma": r"$\sigma$",
}

CSV_PATH = HERE / "three_parameter_sigma_credible_intervals.csv"
FIGURE_PATH = HERE / "three_parameter_credible_intervals.pdf"
PARAMETER_COUNT_PATH = HERE / "neural_network_parameter_counts.csv"


def simulate_sweep_series() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    datasets = {}

    for swept_parameter, values in SWEEPS.items():
        # Naive way of doing this, but doesnt matter much when SWEEP_SIZE is small.
        parameters = {
            name: np.full(SWEEP_SIZE, value)
            for name, value in BASELINE.items()
        }
        parameters[swept_parameter] = values
        datasets[swept_parameter] = sim.simulate_sv_chunk(
            mu=parameters["mu"],
            phi=parameters["phi"],
            s=parameters["sigma"],
            r=np.zeros(SWEEP_SIZE),
            nu=np.full(SWEEP_SIZE, np.inf),
            n=SEQUENCE_LENGTH,
            rng=rng,
            random_init=True,
        )

    return datasets


def transformed_gaussian_ci(
    mean: np.ndarray,
    variance: np.ndarray,
) -> pd.DataFrame:
    critical_value = NormalDist().inv_cdf(1.0 - ALPHA / 2.0)
    sd = np.sqrt(variance)
    lower = mean - critical_value * sd
    upper = mean + critical_value * sd

    return pd.DataFrame(
        {
            "mu_median": mean[:, 0],
            "mu_ci_lower": lower[:, 0],
            "mu_ci_upper": upper[:, 0],
            "phi_median": np.tanh(mean[:, 1] / 2.0),
            "phi_ci_lower": np.tanh(lower[:, 1] / 2.0),
            "phi_ci_upper": np.tanh(upper[:, 1] / 2.0),
            "sigma_median": np.exp(mean[:, 2]),
            "sigma_ci_lower": np.exp(lower[:, 2]),
            "sigma_ci_upper": np.exp(upper[:, 2]),
        }
    )


def ci_rows(
    parameter: str,
    method: str,
    prior: str,
    intervals: pd.DataFrame,
) -> list[dict[str, object]]:
    if "index" in intervals:
        intervals = intervals.sort_values("index").reset_index(drop=True)

    return [
        {
            "parameter": parameter,
            "value_index": value_index,
            "true_value": true_value,
            "method": method,
            "prior": prior,
            "median": intervals.loc[value_index, f"{parameter}_median"],
            "ci_lower": intervals.loc[value_index, f"{parameter}_ci_lower"],
            "ci_upper": intervals.loc[value_index, f"{parameter}_ci_upper"],
        }
        for value_index, true_value in enumerate(SWEEPS[parameter])
    ]


def parameter_count_row(
    model: nn.Module,
    architecture: str,
    prior: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "architecture": architecture,
        "prior": prior,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    for parameter_name, head in zip(("mu", "psi", "rho"), model.heads.values()):
        row[f"{parameter_name}_head_parameters"] = sum(
            parameter.numel()
            for parameter in head.parameters()
            if parameter.requires_grad
        )
    return row


def calculate_credible_intervals() -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = simulate_sweep_series()
    intervals: dict[tuple[str, str, str], pd.DataFrame] = {}

    for prior in PRIORS:
        for parameter in PARAMETERS:
            print(f"Running stochvol ({prior}) for the {parameter} sweep.")
            intervals[("stochvol", prior, parameter)] = run_stochvol_mcmc(
                datasets[parameter],
                prior=prior,
                draws=MCMC_DRAWS,
                burnin=MCMC_BURNIN,
                thinpara=MCMC_THINPARA,
                alpha=ALPHA,
                transforms=None,
                max_cores=MCMC_MAX_CORES,
            )

    count_rows = []
    for architecture in ("Summary NN", "TCN"):
        for prior in PRIORS:
            checkpoint_path = WEIGHTS_DIR / CHECKPOINT_NAMES[(architecture, prior)]
            print(f"Loading {architecture} ({prior}) from {checkpoint_path.name}.")
            model, checkpoint = load_model(checkpoint_path)
            count_rows.append(parameter_count_row(model, architecture, prior))

            for parameter in PARAMETERS:
                mean, variance = predict(model, checkpoint, datasets[parameter])
                intervals[(architecture, prior, parameter)] = transformed_gaussian_ci(
                    mean,
                    variance,
                )

    rows = []
    for prior in PRIORS:
        for parameter in PARAMETERS:
            for method in METHODS:
                rows.extend(
                    ci_rows(
                        parameter,
                        method,
                        prior,
                        intervals[(method, prior, parameter)],
                    )
                )

    return pd.DataFrame(rows), pd.DataFrame(count_rows)


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


def plot_credible_intervals(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 13.0), sharex="row", sharey="row")
    legend_handles = {}

    for row, parameter in enumerate(PARAMETERS):
        true_values = SWEEPS[parameter]
        spacing = np.min(np.diff(true_values))
        offsets = dict(zip(METHODS, np.linspace(-0.18, 0.18, 3) * spacing))

        for column, prior in enumerate(PRIORS):
            axis = axes[row, column]
            axis.plot(true_values, true_values, color="0.35", linestyle="--")

            for method in METHODS:
                data = comparison[
                    (comparison["parameter"] == parameter)
                    & (comparison["prior"] == prior)
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
            if row == 0:
                axis.set_title(f"{prior.capitalize()} prior")
            axis.set_xlabel(f"True {parameter_label}")
            if column == 0:
                axis.set_ylabel(f"{parameter_label} posterior median")
            axis.grid(alpha=0.25)

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


def main() -> None:
    comparison, parameter_counts = calculate_credible_intervals()
    comparison.to_csv(CSV_PATH, index=False)
    parameter_counts.to_csv(PARAMETER_COUNT_PATH, index=False)
    plot_credible_intervals(comparison, FIGURE_PATH)

    print("\nNeural-network parameter counts:")
    print(parameter_counts.to_string(index=False))
    summary_count = int(
        parameter_counts[
            (parameter_counts["architecture"] == "Summary NN")
            & (parameter_counts["prior"] == "default")
        ]["total_parameters"].iloc[0]
    )
    tcn_count = int(
        parameter_counts[
            (parameter_counts["architecture"] == "TCN")
            & (parameter_counts["prior"] == "default")
        ]["total_parameters"].iloc[0]
    )
    print(
        f"\nTCN has {tcn_count - summary_count:,} more parameters than Summary NN "
        f"({tcn_count / summary_count:.2f} times as many)."
    )
    print(f"\nSaved credible intervals to {CSV_PATH}")
    print(f"Saved the 3-by-2 plot to {FIGURE_PATH}")
    print(f"Saved parameter counts to {PARAMETER_COUNT_PATH}")


if __name__ == "__main__":
    main()
