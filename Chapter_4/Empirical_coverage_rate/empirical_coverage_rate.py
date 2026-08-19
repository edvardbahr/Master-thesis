"""Calculate and plot empirical coverage from saved posterior moments."""

import os
import tempfile
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
POSTERIOR_MOMENTS_PATH = (
    PROJECT_DIR
    / "Chapter_4"
    / "RSD_and_metrics"
    / "full_data_posterior_moments.csv"
)
CSV_PATH = HERE / "transformed_posterior_empirical_coverage.csv"
FIGURE_PATH = HERE / "transformed_posterior_empirical_coverage.pdf"

BENCHMARK_SIZE = 5_000
PRIORS = ("default", "finance")
PARAMETERS = ("mu", "psi", "rho")
METHODS = ("stochvol", "TCN", "Summary NN")
COVERAGE_ALPHAS = np.arange(5, 51, 5) / 100.0

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
    "psi": r"$\psi$",
    "rho": r"$\rho$",
}


def select_moments(
    moments: pd.DataFrame,
    method: str,
    prior: str,
) -> pd.DataFrame:
    selected = moments[
        (moments["method"] == method)
        & (moments["prior"] == prior)
    ].sort_values("benchmark_index")
    expected_indices = np.arange(BENCHMARK_SIZE)
    if not np.array_equal(selected["benchmark_index"].to_numpy(), expected_indices):
        raise ValueError(
            f"Expected benchmark indices 0 through {BENCHMARK_SIZE - 1} for "
            f"{method} ({prior})."
        )
    return selected


def calculate_empirical_coverage(moments: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for prior in PRIORS:
        for method in METHODS:
            selected = select_moments(moments, method, prior)
            targets = selected[
                [f"target_{parameter}" for parameter in PARAMETERS]
            ].to_numpy(dtype=np.float64)
            means = selected[
                [f"mean_{parameter}" for parameter in PARAMETERS]
            ].to_numpy(dtype=np.float64)
            variances = selected[
                [f"variance_{parameter}" for parameter in PARAMETERS]
            ].to_numpy(dtype=np.float64)
            standardized_errors = np.abs(targets - means) / np.sqrt(variances)

            for alpha in COVERAGE_ALPHAS:
                nominal_coverage = 1.0 - alpha
                critical_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
                empirical_coverage = np.mean(
                    standardized_errors <= critical_value,
                    axis=0,
                )
                for index, parameter in enumerate(PARAMETERS):
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


def plot_empirical_coverage(
    coverage: pd.DataFrame,
    output_path: Path,
) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 13.0), sharex=True, sharey="row")
    legend_handles = {}
    nominal_limits = (
        1.0 - np.max(COVERAGE_ALPHAS),
        1.0 - np.min(COVERAGE_ALPHAS),
    )

    for row, parameter in enumerate(PARAMETERS):
        for column, prior in enumerate(PRIORS):
            axis = axes[row, column]
            axis.plot(nominal_limits, nominal_limits, color="0.35", linestyle="--")

            for method in METHODS:
                data = coverage[
                    (coverage["parameter"] == parameter)
                    & (coverage["prior"] == prior)
                    & (coverage["method"] == method)
                ].sort_values("nominal_coverage")
                legend_handles[method] = axis.plot(
                    data["nominal_coverage"],
                    data["empirical_coverage"],
                    color=COLORS[method],
                    marker=MARKERS[method],
                    markersize=4.5,
                    label=method,
                )[0]

            if row == 0:
                axis.set_title(f"{prior.capitalize()} prior")
            axis.set_xlabel("Nominal coverage level")
            if column == 0:
                axis.set_ylabel(
                    f"Empirical {PARAMETER_LABELS[parameter]} coverage"
                )
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
    if not POSTERIOR_MOMENTS_PATH.is_file():
        raise FileNotFoundError(
            "Run Chapter_4/RSD_and_metrics/rsd_and_metrics.py first: "
            f"{POSTERIOR_MOMENTS_PATH}"
        )

    moments = pd.read_csv(POSTERIOR_MOMENTS_PATH)
    coverage = calculate_empirical_coverage(moments)
    coverage.to_csv(CSV_PATH, index=False)
    plot_empirical_coverage(coverage, FIGURE_PATH)
    print(f"Saved empirical coverage rates to {CSV_PATH}")
    print(f"Saved the 3-by-2 coverage plot to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
