"""Assess the weakest effective sample size in each fitted MCMC chain."""

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation import sim_5_param_data as sim
from simulation.stochvol_mcmc import (
    centered_square_transform,
    log_positive_sq_transform,
    log_positive_transform,
    psi_sq_transform,
    psi_transform,
    run_stochvol_mcmc,
)


N_SERIES = 2_000
SEQUENCE_LENGTH = 253
DRAWS = 20_000
BURNIN = 500
ALPHA = 0.05
SEED = 2
MAX_CORES = -2
PRIORS = ("default", "finance")
COLORS = {"default": "#0000ff", "finance": "#008000"}
QUANTILE_COLOR = "#ff0000"
ESS_TRANSFORMS = {
    "mu": {"mu_centered_sq": centered_square_transform},
    "phi": {
        "psi": psi_transform,
        "psi_centered_sq": psi_sq_transform,
    },
    "sigma": {
        "rho": log_positive_transform,
        "rho_centered_sq": log_positive_sq_transform,
    },
}
ESS_COLUMNS = (
    "mu_ESS",
    "psi_ESS",
    "rho_ESS",
    "mu_centered_sq_ESS",
    "psi_centered_sq_ESS",
    "rho_centered_sq_ESS",
)
PARAMETER_LABELS = (
    r"$\mu$",
    r"$\psi$",
    r"$\rho$",
    r"$(\mu-\bar{\mu})^2$",
    r"$(\psi-\bar{\psi})^2$",
    r"$(\rho-\bar{\rho})^2$",
)
FIGURE_PATH = HERE / "minimum_ess_histograms.pdf"
SUMMARY_PATH = HERE / "minimum_ess_summary.csv"
RATIO_PATH = HERE / "minimum_ess_parameter_ratios.csv"


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


def main() -> None:
    apply_plot_style()
    rng = np.random.default_rng(SEED)
    parameter_labels = np.asarray(PARAMETER_LABELS)
    min_ess_by_prior: dict[str, np.ndarray] = {}
    lower_quantile_by_prior: dict[str, float] = {}
    ratio_by_prior: dict[str, pd.Series] = {}
    summary_rows = []

    for prior in PRIORS:
        print(
            f"Running {N_SERIES:,} {prior} MCMC chains with "
            f"{DRAWS:,} draws each."
        )
        parameters = sim.sample_stochvol_prior(
            N_SERIES,
            prior=prior,
            fixed_r=0.0,
            fixed_nu=np.inf,
            rng=rng,
        )
        y = sim.simulate_sv_chunk(
            *parameters,
            n=SEQUENCE_LENGTH,
            rng=rng,
        )
        summary = run_stochvol_mcmc(
            y=y,
            prior=prior,
            draws=DRAWS,
            burnin=BURNIN,
            alpha=ALPHA,
            estimate_ess=True,
            transforms=ESS_TRANSFORMS,
            max_cores=MAX_CORES,
        )

        ess_values = summary.loc[:, ESS_COLUMNS].to_numpy()
        valid_chains = np.all(np.isfinite(ess_values), axis=1)
        n_invalid = int(np.count_nonzero(~valid_chains))
        if not np.any(valid_chains):
            raise RuntimeError(f"ESS could not be estimated for any {prior} chain.")
        if n_invalid:
            print(f"Excluding {n_invalid:,} {prior} chain(s) with undefined ESS.")

        valid_ess_values = ess_values[valid_chains]
        weakest_parameter = np.argmin(valid_ess_values, axis=1)
        min_ess = valid_ess_values[
            np.arange(len(valid_ess_values)),
            weakest_parameter,
        ]
        lower_quantile = float(np.quantile(min_ess, ALPHA))
        parameter_ratio = (
            pd.Series(parameter_labels[weakest_parameter])
            .value_counts(normalize=True)
            .reindex(parameter_labels, fill_value=0.0)
        )

        min_ess_by_prior[prior] = min_ess
        lower_quantile_by_prior[prior] = lower_quantile
        ratio_by_prior[prior] = parameter_ratio
        summary_rows.append(
            {
                "prior": prior,
                "valid_chains": len(min_ess),
                "invalid_chains": n_invalid,
                "mean_min_ess": float(np.mean(min_ess)),
                "median_min_ess": float(np.median(min_ess)),
                f"q{100 * ALPHA:.0f}_min_ess": lower_quantile,
            }
        )

    summary_table = pd.DataFrame(summary_rows)
    ratio_table = pd.DataFrame(ratio_by_prior).rename_axis("parameter")
    summary_table.to_csv(SUMMARY_PATH, index=False)
    ratio_table.to_csv(RATIO_PATH)

    print("\nMinimum ESS summary")
    print(summary_table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nShare of chains for which each quantity has the lowest ESS")
    print(ratio_table.to_string(float_format=lambda x: f"{x:.3f}"))

    all_min_ess = np.concatenate([min_ess_by_prior[prior] for prior in PRIORS])
    bins = np.histogram_bin_edges(all_min_ess, bins="fd")
    fig, axes = plt.subplots(
        1,
        len(PRIORS),
        figsize=(11.5, 5.2),
        sharex=True,
        sharey=True,
    )

    for ax, prior in zip(axes, PRIORS):
        min_ess = min_ess_by_prior[prior]
        lower_quantile = lower_quantile_by_prior[prior]
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", linewidth=0.8)
        ax.hist(
            min_ess,
            bins=bins,
            color=COLORS[prior],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.axvline(
            lower_quantile,
            color=QUANTILE_COLOR,
            linestyle="--",
            linewidth=2,
            label=rf"{100 * ALPHA:.0f}% quantile = {lower_quantile:.1f}",
        )
        ax.set_title(f"{prior.capitalize()} prior")
        ax.set_xlabel("Minimum effective sample size")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)

    axes[0].set_ylabel("Number of MCMC chains")
    fig.suptitle("Minimum ESS across parameters and centered-square transforms")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    print(f"Saved {FIGURE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
