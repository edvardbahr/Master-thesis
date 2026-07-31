"""Compare standardized GH skew-t and sinh-arcsinh densities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
from scipy.special import gammaln, kve
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "ghst_shash_comparison.pdf"


# ---------------------------------------------------------------------
# Standardized GHST density under the (r, nu) parameterization
# ---------------------------------------------------------------------

def ghst_parameters(r, nu):
    """Return (m, delta, beta) for a centered, unit-variance GHST."""
    if not 0.0 <= r < 1.0:
        raise ValueError("r must satisfy 0 <= r < 1.")
    if nu <= 4.0:
        raise ValueError("nu must exceed 4 for the variance to exist.")

    delta = np.sqrt((nu - 2.0) * (1.0 - r))
    beta = np.sqrt(r * (nu - 4.0) / 2.0) / (1.0 - r)
    m = -np.sqrt(r * (nu - 4.0) / 2.0)

    return m, delta, beta


def ghst_pdf(x, r, nu):
    """Density of the centered, unit-variance GHST distribution."""
    x = np.asarray(x, dtype=float)

    # The Gaussian limiting case is plotted exactly.
    if np.isinf(nu):
        return norm.pdf(x)

    m, delta, beta = ghst_parameters(r, nu)

    # Symmetric boundary case: scaled Student's t distribution.
    if np.isclose(beta, 0.0):
        log_pdf = (
            gammaln((nu + 1.0) / 2.0)
            - 0.5 * np.log(np.pi)
            - np.log(delta)
            - gammaln(nu / 2.0)
            - ((nu + 1.0) / 2.0)
            * np.log1p(((x - m) / delta) ** 2)
        )
        return np.exp(log_pdf)

    q = np.sqrt(delta**2 + (x - m) ** 2)
    order = (nu + 1.0) / 2.0
    argument = beta * q

    # kve(v, x) = exp(x) K_v(x), which avoids numerical underflow.
    log_bessel_k = np.log(kve(order, argument)) - argument

    log_pdf = (
        ((1.0 - nu) / 2.0) * np.log(2.0)
        + nu * np.log(delta)
        + order * np.log(beta)
        - 0.5 * np.log(np.pi)
        - gammaln(nu / 2.0)
        + log_bessel_k
        - order * np.log(q)
        + beta * (x - m)
    )

    return np.exp(log_pdf)


# ---------------------------------------------------------------------
# Standardized SHASH density
# ---------------------------------------------------------------------

# Gauss--Hermite quadrature is used to calculate the SHASH mean and SD.
_HERMITE_NODES, _HERMITE_WEIGHTS = hermgauss(120)
_GAUSSIAN_NODES = np.sqrt(2.0) * _HERMITE_NODES
_GAUSSIAN_WEIGHTS = _HERMITE_WEIGHTS / np.sqrt(np.pi)


def shash_mean_sd(epsilon, tau):
    """Mean and SD of the unstandardized SHASH variable."""
    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    raw_values = np.sinh(
        (np.arcsinh(_GAUSSIAN_NODES) + epsilon) / tau
    )

    mean = np.sum(_GAUSSIAN_WEIGHTS * raw_values)
    variance = np.sum(
        _GAUSSIAN_WEIGHTS * (raw_values - mean) ** 2
    )

    return mean, np.sqrt(variance)


def shash_pdf_raw(x, epsilon, tau):
    """Density of the unstandardized Gaussian-generated SHASH."""
    transformed = tau * np.arcsinh(x) - epsilon
    s_value = np.sinh(transformed)
    c_value = np.cosh(transformed)

    return (
        tau
        * c_value
        / np.sqrt(2.0 * np.pi * (1.0 + x**2))
        * np.exp(-0.5 * s_value**2)
    )


def standardized_shash_pdf(x, epsilon, tau):
    """Density after centering the SHASH variable and scaling it to SD 1."""
    mean, sd = shash_mean_sd(epsilon, tau)

    # If Y = (X - mean) / sd, then f_Y(y) = sd f_X(mean + sd*y).
    raw_x = mean + sd * np.asarray(x)

    return sd * shash_pdf_raw(raw_x, epsilon, tau)


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

# Set this to True if the tail differences should be emphasized.
USE_LOG_SCALE = False

COLORS = ("#0000ff", "#008000", "#ff0000")

GHST_CASES = [
    {
        "r": 0.0,
        "nu": np.inf,
        "label": r"$r=0,\ \nu=\infty$",
    },
    {
        "r": 0.99,
        "nu": 12.0,
        "label": r"$r=0.99,\ \nu=12$",
    },
    {
        "r": 0.99,
        "nu": 6.0,
        "label": r"$r=0.99,\ \nu=6$",
    },
]

SHASH_CASES = [
    {
        "epsilon": 0.0,
        "tau": 1.0,
        "label": r"$\epsilon=0,\ \tau=1$",
    },
    {
        "epsilon": 0.5,
        "tau": 0.7,
        "label": r"$\epsilon=0.5,\ \tau=0.7$",
    },
    {
        "epsilon": 0.5,
        "tau": 0.5,
        "label": r"$\epsilon=0.5,\ \tau=0.5$",
    },
]


def apply_plot_style() -> None:
    """Use the plotting convention from test_NN_models.py."""
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


def make_figure(use_log_scale: bool = USE_LOG_SCALE) -> plt.Figure:
    """Create the two-panel standardized-density comparison."""
    apply_plot_style()
    x = np.linspace(-4.0, 10.0, 2500)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.2),
        sharex=True,
        sharey=True,
    )

    for case, color in zip(GHST_CASES, COLORS):
        axes[0].plot(
            x,
            ghst_pdf(x, case["r"], case["nu"]),
            color=color,
            label=case["label"],
        )

    for case, color in zip(SHASH_CASES, COLORS):
        axes[1].plot(
            x,
            standardized_shash_pdf(
                x,
                case["epsilon"],
                case["tau"],
            ),
            color=color,
            label=case["label"],
        )

    titles = (r"GH skew-$t$", "Sinh-arcsinh")
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel("Standardized innovation")
        axis.set_xlim(-4.0, 10.0)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", frameon=False)

    axes[0].set_ylabel("Density")

    if use_log_scale:
        for axis in axes:
            axis.set_yscale("log")
            axis.set_ylim(1e-8, 3.0)
        axes[0].set_ylabel("Density (log scale)")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    figure = make_figure()
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to {OUTPUT_PATH}")
