"""Compare prior and posterior normality as the sequence length increases."""

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from simulation import sim_5_param_data as sim
from simulation.stochvol_mcmc import DEFAULT_TRANSFORMS, run_stochvol_mcmc


DRAWS = 20_000
BURNIN = 500
SHORT_SEQUENCE_LENGTH = 253
LONG_SEQUENCE_LENGTH = 4 * SHORT_SEQUENCE_LENGTH
QQ_POINTS = 500
SEED = 4
PRIORS = ("default", "finance")
PARAMETER_NAMES = ("mu", "psi", "rho")
PARAMETER_LABELS = (
    r"$\mu$",
    r"$\psi = 2\operatorname{atanh}(\phi)$",
    r"$\rho = \log(\sigma)$",
)
QQ_COLORS = {
    "default": ("#9ecae1", "#4292c6", "#08519c"),
    "finance": ("#a1d99b", "#41ab5d", "#006d2c"),
}
OUTPUT_PATH = HERE / "bernstein_von_mises_qq_plots.pdf"


def select_transformed_parameters(draws: pd.DataFrame) -> pd.DataFrame:
    """Return mu, psi, and rho draws using the central MCMC transforms."""
    draws = draws.copy()
    for parameter, parameter_transforms in DEFAULT_TRANSFORMS.items():
        for transformed_name, transform_fn in parameter_transforms.items():
            draws[transformed_name] = transform_fn(draws[parameter])

    return draws.loc[:, PARAMETER_NAMES]


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


def main() -> None:
    apply_plot_style()
    rng = np.random.default_rng(SEED)
    chains: dict[str, dict[int, pd.DataFrame]] = {}

    for prior in PRIORS:
        prior_mu, prior_phi, prior_sigma, _, _ = sim.sample_stochvol_prior(
            DRAWS,
            prior=prior,
            fixed_r=0.0,
            fixed_nu=np.inf,
            rng=rng,
        )
        prior_draws = pd.DataFrame(
            {"mu": prior_mu, "phi": prior_phi, "sigma": prior_sigma}
        )
        chains[prior] = {0: select_transformed_parameters(prior_draws)}

        true_parameters = sim.sample_stochvol_prior(
            1,
            prior=prior,
            fixed_r=0.0,
            fixed_nu=np.inf,
            rng=rng,
        )
        y = sim.simulate_sv_chunk(
            *true_parameters,
            n=LONG_SEQUENCE_LENGTH,
            rng=rng,
        )

        for sequence_length in (SHORT_SEQUENCE_LENGTH, LONG_SEQUENCE_LENGTH):
            print(
                f"Running {prior} MCMC with n={sequence_length:,} "
                f"and {DRAWS:,} draws."
            )
            _, parameter_draws = run_stochvol_mcmc(
                y=y[..., :sequence_length],
                prior=prior,
                draws=DRAWS,
                burnin=BURNIN,
                transforms=None,
                max_cores=1,
                return_draws=True,
            )
            chains[prior][sequence_length] = select_transformed_parameters(
                parameter_draws
            )

    probabilities = (np.arange(1, QQ_POINTS + 1) - 0.5) / QQ_POINTS
    theoretical_quantiles = stats.norm.ppf(probabilities)
    sample_sizes = (
        (0, r"$n=0$"),
        (SHORT_SEQUENCE_LENGTH, rf"$n={SHORT_SEQUENCE_LENGTH}$"),
        (LONG_SEQUENCE_LENGTH, rf"$n={LONG_SEQUENCE_LENGTH}$"),
    )

    fig, axes = plt.subplots(
        len(PARAMETER_NAMES),
        len(PRIORS),
        figsize=(11.5, 13.0),
        sharex=True,
        sharey=True,
    )

    for column, prior in enumerate(PRIORS):
        for row, (parameter, parameter_label) in enumerate(
            zip(PARAMETER_NAMES, PARAMETER_LABELS)
        ):
            ax = axes[row, column]
            ax.plot(
                theoretical_quantiles,
                theoretical_quantiles,
                color="black",
                linestyle="--",
                label="Gaussian reference",
                zorder=1,
            )

            for curve_index, (sequence_length, label) in enumerate(sample_sizes):
                values = chains[prior][sequence_length][parameter].to_numpy()
                standardized_values = (values - values.mean()) / values.std(ddof=1)
                empirical_quantiles = np.quantile(
                    standardized_values,
                    probabilities,
                )
                ax.scatter(
                    theoretical_quantiles,
                    empirical_quantiles,
                    color=QQ_COLORS[prior][curve_index],
                    s=16,
                    alpha=0.85,
                    edgecolors="none",
                    label=label,
                    rasterized=True,
                    zorder=2 + curve_index,
                )

            ax.set_axisbelow(True)
            ax.grid(linestyle=":", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)

            if row == 0:
                ax.set_title(f"{prior.capitalize()} prior")
                ax.legend(frameon=False)
            if row == len(PARAMETER_NAMES) - 1:
                ax.set_xlabel("Theoretical standard-normal quantiles")
            if column == 0:
                ax.set_ylabel(f"{parameter_label}\nEmpirical quantiles")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
