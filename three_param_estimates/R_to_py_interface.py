import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sim_3_param_data as sim


HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "stochvol_MCMC.R"
PARAMETER_NAMES = ("mu", "phi", "sigma")


def plot_parameter_histograms_with_normal(
    draws,
    output_path,
    true_values=None,
    parameters=("mu", "phi", "sigma"),
    bins=50,
    transformations={},
):
    """
    Plot posterior draw histograms with fitted empirical normal overlays.

    Parameters
    ----------
    draws:
        Usually a pandas DataFrame with columns "mu", "phi", "sigma".
    output_path:
        Path where the figure is saved.
    true_values:
        Optional dict, e.g. {"mu": -9.0, "phi": 0.98, "sigma": 0.20}.
    parameters:
        Parameters to plot.
    bins:
        Number of histogram bins.
    transformations:
        Optional dict of parameter transformations, e.g. {"phi": lambda x: np.arctanh(x) * 2}.
    """

    def identity(x):
        return x

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(parameters), figsize=(5 * len(parameters), 4))

    if len(parameters) == 1:
        axes = [axes]

    for ax, param in zip(axes, parameters):
        transform = transformations.get(param, identity)
        values = transform(np.asarray(draws[param], dtype=float))
        values = values[np.isfinite(values)]

        mean_hat = np.mean(values)
        sd_hat = np.std(values, ddof=1)

        ax.hist(values, bins=bins, density=True, alpha=0.6)

        if sd_hat > 0:
            x_grid = np.linspace(values.min(), values.max(), 500)
            normal_density = (
                1.0 / (sd_hat * np.sqrt(2.0 * np.pi))
                * np.exp(-0.5 * ((x_grid - mean_hat) / sd_hat) ** 2)
            )

            ax.plot(
                x_grid,
                normal_density,
                linewidth=2,
                label=f"N({mean_hat:.3g}, {sd_hat:.3g}²)",
            )

        if true_values is not None and param in true_values:
            transformed_true_value = transform(true_values[param])
            if transform.__name__ == "identity":
                label = f"true {param} = {transformed_true_value:.3g}"
            else:
                label = f"true {transform.__name__}({param}) = {transformed_true_value:.3g}"
            ax.axvline(
                transformed_true_value,
                linestyle="--",
                linewidth=2,
                label=label,
            )
        
        if transform.__name__ == "identity":
            ax.set_title(f"{param}")
        else:
            ax.set_title(f"{transform.__name__}({param})")
        ax.set_xlabel("Posterior draw")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Posterior draws with empirical normal overlays")
    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def find_rscript():
    rscript = shutil.which("Rscript")

    if rscript is not None:
        return rscript

    program_files = Path("C:/Program Files")
    candidates = sorted(
        program_files.glob("R/R-*/bin/Rscript.exe"),
        reverse=True,
    )

    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(
        "Could not find Rscript. Add R's bin folder to PATH, or install R."
    )


def validate_series_matrix(y):
    y = np.asarray(y, dtype=np.float64)

    if y.ndim == 1:
        y = y.reshape(1, -1)
    elif y.ndim != 2:
        raise ValueError("y must have shape (n,) or (m, n).")

    if y.shape[0] < 1 or y.shape[1] < 1:
        raise ValueError("y must contain at least one series and one observation.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")

    return y


def run_stochvol_mcmc(
    y,
    prior="default",
    draws=2000,
    burnin=500,
    thinpara=1,
    alpha=0.05,
    return_draws=False,
):
    y = validate_series_matrix(y)

    if not R_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find MCMC R script: {R_SCRIPT}")

    prior_constants = sim.get_stochvol_prior_constants(prior)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "y.csv"
        summary_path = tmpdir / "summary.csv"
        draws_path = tmpdir / "parameter_draws.csv"

        np.savetxt(input_path, y, delimiter=",")

        command = [
            find_rscript(),
            str(R_SCRIPT),
            str(input_path),
            str(summary_path),
            str(int(draws)),
            str(int(burnin)),
            str(int(thinpara)),
            str(float(prior_constants.mu_mean)),
            str(float(prior_constants.mu_sd)),
            str(float(prior_constants.phi_a0)),
            str(float(prior_constants.phi_b0)),
            str(float(prior_constants.Bsigma)),
            str(float(alpha)),
        ]

        if return_draws:
            command.append(str(draws_path))

        subprocess.run(command, check=True)

        summary = pd.read_csv(summary_path)
        parameter_draws = pd.read_csv(draws_path) if return_draws else None

    if len(summary) != y.shape[0]:
        raise RuntimeError(
            f"stochvol MCMC returned {len(summary)} rows for {y.shape[0]} series."
        )

    if return_draws:
        return summary, parameter_draws

    return summary


def plot_parameter_trace(draws, output_path, series_index=1, true_values=None):

    draws = draws[draws["series_index"] == series_index]

    fig, axes = plt.subplots(
        len(PARAMETER_NAMES),
        1,
        figsize=(10, 7),
        sharex=True,
    )

    for ax, parameter in zip(axes, PARAMETER_NAMES):
        ax.plot(draws["draw_index"], draws[parameter], linewidth=0.7)
        if true_values is not None:
            ax.axhline(
                true_values[parameter],
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
        ax.set_ylabel(parameter)

    axes[-1].set_xlabel("MCMC draw")
    fig.suptitle(f"stochvol parameter traceplot, series {series_index}")
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def main():
    mu = [-9.0]
    phi = [0.98]
    sigma = [0.20]

    rng = np.random.default_rng(seed=1)

    simulated_data = sim.simulate_sv_chunk(mu, phi, sigma, n=int(np.floor(253*4)), rng=rng)[0]

    summary, draws = run_stochvol_mcmc(
        simulated_data,
        draws=10000,
        burnin=2000,
        thinpara=1,
        return_draws=True,
    )

    print(summary[["mu_mean", "phi_mean", "sigma_mean"]])

    true_values = {
        "mu": mu[0],
        "phi": phi[0],
        "sigma": sigma[0],
    }

    traceplot_path = plot_parameter_trace(
        draws,
        output_path=HERE / "recovered_plots" / "stochvol_traceplot.png",
        true_values=true_values,
    )
    print(f"Saved traceplot to {traceplot_path}")



    def logit(x):
        return np.arctanh(x) * 2
    
    from scipy.stats import norm

    transformations = {
        #"phi": lambda x: norm.ppf((x + 1.0) / 2.0),
        "phi": logit,
        "sigma": np.log,
    }

    hist_path = plot_parameter_histograms_with_normal(
        draws,
        output_path=HERE / "recovered_plots" / "stochvol_hist_normal_overlay_logit.png",
        true_values=true_values,
        parameters=("mu", "phi", "sigma"),
        bins=50,
        transformations=transformations,
    )
    print(f"Saved histogram plot to {hist_path}")


if __name__ == "__main__":
    main()
