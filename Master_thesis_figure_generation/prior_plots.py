"""Plot the default and finance priors used for the SV model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, chi2, norm


HERE = Path(__file__).resolve().parent


# Prior hyperparameters -------------------------------------------------------
PRIORS = {
    "Default prior": {
        "mu_mean": 0.0,
        "mu_sd": 10.0,
        "phi_a": 5.0,
        "phi_b": 1.5,
        "sigma_sq_df": 1.0,
    },
    "Finance prior": {
        "mu_mean": -9.0,
        "mu_sd": 1.0,
        "phi_a": 20.0,
        "phi_b": 1.5,
        "sigma_sq_df": 1.0,
    },
}


def phi_pdf(phi, a, b):
    """Density induced by U ~ Beta(a, b) and phi = 2U - 1."""
    u = (phi + 1.0) / 2.0
    return 0.5 * beta.pdf(u, a, b)  # Jacobian du/dphi = 1/2


def make_figure():
    mu = np.linspace(-20.0, 20.0, 1500)
    phi = np.linspace(-0.9999, 0.9999, 1500)

    # The chi-squared density with one degree of freedom is unbounded at zero.
    # Start slightly above zero so that it can be displayed on a finite axis.
    sigma_sq = np.linspace(0.005, 5.0, 1500)

    # Compact styling resembling the classic Matplotlib plots in the Adam paper.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(6.0, 5.8))

    colors = (
        "#0000ff",
        "#008000",
        "#ff0000",
        "#00bfbf",
        "#bf00bf",
        "#bfbf00",
    )
    line_styles = ("-", "-")

    for (label, prior), color, line_style in zip(
        PRIORS.items(), colors[:2], line_styles
    ):
        densities = (
            norm.pdf(mu, loc=prior["mu_mean"], scale=prior["mu_sd"]),
            phi_pdf(phi, prior["phi_a"], prior["phi_b"]),
            chi2.pdf(sigma_sq, df=prior["sigma_sq_df"]),
        )

        for ax, x, density in zip(axes, (mu, phi, sigma_sq), densities):
            ax.plot(
                x,
                density,
                color=color,
                linestyle=line_style,
                label=label,
            )

    xlabels = (r"$\mu$", r"$\phi$", r"$\sigma^2$")
    for ax, xlabel in zip(axes, xlabels):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Prior density")
        ax.grid(True, linestyle=":", linewidth=0.7, color="0.45")
        ax.set_axisbelow(True)
        ax.margins(x=0)

    axes[0].set_xlim(-20.0, 20.0)
    axes[1].set_xlim(-1.0, 1.0)
    axes[1].set_xticks(np.linspace(-1.0, 1.0, 9))
    axes[2].set_xlim(0.0, 5.0)

    # Keep the legends inside the axes to reduce the total figure height.
    # Their locations are chosen to avoid the main mass of each density.
    legend_locations = ("upper right", "upper left", "upper right")
    for ax, location in zip(axes, legend_locations):
        ax.legend(
            loc=location,
            ncol=2,
            frameon=True,
        )

    fig.tight_layout(h_pad=0.4)
    return fig


if __name__ == "__main__":
    figure = make_figure()
    figure.savefig(HERE / "sv_prior_densities.pdf", bbox_inches="tight")
    # figure.savefig(HERE / "sv_prior_densities.png", dpi=300, bbox_inches="tight")
    plt.show()