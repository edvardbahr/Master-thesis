import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
import pandas as pd

import simulateData as sim


HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "stochvolMCMC.R"
PARAMETER_NAMES = ("mu", "phi", "sigma")


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
    import matplotlib.pyplot as plt

    draws = draws[draws["index"] == series_index]

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

    simulated_data = sim.simulate_sv_chunk(mu, phi, sigma, n=253*4, rng=rng)[0]

    summary, draws = run_stochvol_mcmc(
        simulated_data,
        draws=10000,
        burnin=2000,
        thinpara=1,
        return_draws=True,
    )

    print(summary[["mu_mean", "phi_mean", "sigma_mean"]])

    traceplot_path = plot_parameter_trace(
        draws,
        output_path=HERE / "recovered_plots" / "stochvol_traceplot.png",
        true_values={"mu": mu[0], "phi": phi[0], "sigma": sigma[0]},
    )
    print(f"Saved traceplot to {traceplot_path}")


if __name__ == "__main__":
    main()
