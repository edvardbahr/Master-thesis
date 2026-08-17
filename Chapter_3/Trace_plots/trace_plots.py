"""Plot representative transformed-parameter traces for both SV priors."""

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
    log_positive_transform,
    psi_transform,
    run_stochvol_mcmc,
)


SEQUENCE_LENGTH = 253
DRAWS = 1_000
BURNIN = 500
PRIORS = ("default", "finance")
PARAMETER_NAMES = ("mu", "psi", "rho")
PARAMETER_LABELS = (
    r"$\mu$",
    r"$\psi = 2\operatorname{atanh}(\phi)$",
    r"$\rho = \log(\sigma)$",
)
COLORS = {"default": "#08519c", "finance": "#006d2c"}
OUTPUT_PATH = HERE / "mcmc_parameter_trace_plots.pdf"


def transform_draws(draws: pd.DataFrame) -> pd.DataFrame:
    """Select the unconstrained parameters used in the Chapter 3 analyses."""
    return pd.DataFrame(
        {
            "draw_index": draws["draw_index"].to_numpy(),
            "mu": draws["mu"].to_numpy(),
            "psi": psi_transform(draws["phi"]),
            "rho": log_positive_transform(draws["sigma"]),
        }
    )


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
        }
    )


def main() -> None:
    apply_plot_style()
    rng = np.random.default_rng()
    chains: dict[str, pd.DataFrame] = {}
    true_values: dict[str, dict[str, float]] = {}

    for prior in PRIORS:
        parameters = sim.sample_stochvol_prior(
            1,
            prior=prior,
            fixed_r=0.0,
            fixed_nu=np.inf,
            rng=rng,
        )
        mu, phi, sigma, _, _ = parameters
        y = sim.simulate_sv_chunk(
            *parameters,
            n=SEQUENCE_LENGTH,
            rng=rng,
        )

        print(f"Running the representative {prior} MCMC chain.")
        _, parameter_draws = run_stochvol_mcmc(
            y=y,
            prior=prior,
            draws=DRAWS,
            burnin=BURNIN,
            transforms=None,
            max_cores=1,
            return_draws=True,
        )
        chains[prior] = transform_draws(parameter_draws)
        true_values[prior] = {
            "mu": float(mu[0]),
            "psi": float(psi_transform(phi)[0]),
            "rho": float(log_positive_transform(sigma)[0]),
        }

    fig, axes = plt.subplots(
        len(PARAMETER_NAMES),
        len(PRIORS),
        figsize=(12, 8),
        sharex=True,
        sharey="row",
    )

    for column, prior in enumerate(PRIORS):
        draws = chains[prior]
        for row, (parameter, label) in enumerate(
            zip(PARAMETER_NAMES, PARAMETER_LABELS)
        ):
            ax = axes[row, column]
            ax.plot(
                draws["draw_index"],
                draws[parameter],
                color=COLORS[prior],
                linewidth=0.55,
                rasterized=True,
            )
            ax.axhline(
                true_values[prior][parameter],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Simulated value",
            )
            ax.spines[["top", "right"]].set_visible(False)

            if row == 0:
                ax.set_title(f"{prior.capitalize()} prior")
                ax.legend(frameon=False)
            if row == len(PARAMETER_NAMES) - 1:
                ax.set_xlabel("MCMC draw")
            if column == 0:
                ax.set_ylabel(label)

    fig.suptitle(
        f"Representative MCMC traces for sequences of length {SEQUENCE_LENGTH}"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
