"""Compare Gaussian and GH skew-t stochastic-volatility paths."""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from simulation.sim_5_param_data import sample_centered_gh_skew_t_innovations


N_TIME_STEPS = 200
MU = -9.0
PHI = 0.975
SIGMA = 0.25
INNOVATION_SEED = 200
OBSERVATION_SEED = 11
OUTPUT_PATH = HERE / "SV_plots.pdf"

DISTRIBUTIONS = (
    {
        "nu": np.inf,
        "r": 0.0,
        "title": r"Gaussian innovations ($\nu=\infty$, $r=0$)",
    },
    {
        "nu": 6.0,
        "r": 0.999,
        "title": r"GH skew-$t$ innovations ($\nu=6$, $r=0.999$)",
    },
)


def apply_plot_style() -> None:
    """Match the empirical-coverage figures in ``test_NN_models.py``."""
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


def simulate_sv_path(
    n: int,
    nu: float,
    r: float,
    *,
    innovation_seed: int = INNOVATION_SEED,
    observation_seed: int = OBSERVATION_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the latent log volatility and the observed log return.

    Both distributional specifications are called with the same
    ``innovation_seed``. Observation noise also uses a separate common seed,
    so differences between columns come from the log-volatility innovations.
    The recursion starts at its unconditional location, ``h_{-1} = mu``.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    sigma = np.full(n, SIGMA, dtype=np.float64)
    skewness = np.full(n, r, dtype=np.float64)
    degrees_of_freedom = np.full(n, nu, dtype=np.float64)
    innovations = sample_centered_gh_skew_t_innovations(
        sigma,
        skewness,
        degrees_of_freedom,
        rng=np.random.default_rng(innovation_seed),
    )

    log_volatility = np.empty(n, dtype=np.float64)
    previous_log_volatility = MU
    for time_index, innovation in enumerate(innovations):
        log_volatility[time_index] = (
            MU
            + PHI * (previous_log_volatility - MU)
            + innovation
        )
        previous_log_volatility = log_volatility[time_index]

    observation_noise = np.random.default_rng(
        observation_seed
    ).standard_normal(n)
    log_return = np.exp(0.5 * log_volatility) * observation_noise

    return log_volatility, log_return


def make_figure(n: int = N_TIME_STEPS) -> plt.Figure:
    """Create the 2-by-2 comparison figure."""
    apply_plot_style()
    time = np.arange(1, n + 1)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 8.0),
        sharex=True,
        sharey="row",
    )

    for column, distribution in enumerate(DISTRIBUTIONS):
        log_volatility, log_return = simulate_sv_path(
            n=n,
            nu=distribution["nu"],
            r=distribution["r"],
        )

        axes[0, column].plot(time, log_volatility, color="#0000ff")
        axes[1, column].plot(time, log_return, color="#ff0000", linewidth=1.0)
        axes[0, column].set_title(distribution["title"])
        axes[1, column].set_xlabel(r"Time $t$")
        axes[1, column].axhline(
            0.0,
            color="0.35",
            linestyle="--",
            linewidth=0.8,
        )

    axes[0, 0].set_ylabel(r"Log volatility $h_t$")
    axes[1, 0].set_ylabel(r"Observed log return $y_t$")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.margins(x=0)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    figure = make_figure()
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to {OUTPUT_PATH}")