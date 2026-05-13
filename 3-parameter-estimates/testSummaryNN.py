import subprocess
import tempfile
import numpy as np
import simulateData as sim
import shutil

from pathlib import Path

# Identify the path to the R script that performs stochvol MCMC estimation.
# The R script should be located in the same directory as this Python script.
HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "stochvolMCMC.R"


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


def estimate_sv_with_r(y, draws=2000, burnin=500, thinpara=1):
    """
    Run stochvol MCMC in R for one or more simulated SV series.

    Parameters
    ----------
    y:
        One series with shape (n,) or many series with shape (m, n).

    Returns
    -------
    result:
        Structured NumPy array with one row per input series.
    """
    y = np.asarray(y, dtype=float)

    if y.ndim == 1:
        y = y.reshape(1, -1)
    elif y.ndim != 2:
        raise ValueError("y must have shape (n,) or (m, n).")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "y.csv"
        output_path = tmpdir / "sv_result.csv"

        np.savetxt(input_path, y, delimiter=",")

        subprocess.run(
            [
                find_rscript(),
                str(R_SCRIPT),
                str(input_path),
                str(output_path),
                str(draws),
                str(burnin),
                str(thinpara),
            ],
            check=True,
        )

        result = np.genfromtxt(
            output_path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )

    return result


if __name__ == "__main__":
    phi = np.linspace(0.5, 0.99, 11, endpoint=True)
    mu = -9 * np.ones(len(phi))
    sigma = 0.2 * np.ones(len(phi))

    rng = np.random.default_rng(1)

    y = sim.simulate_sv_chunk(
        mu=mu,
        phi=phi,
        sigma=sigma,
        n=253,
        rng=rng,
    )

    mcmc_summary = estimate_sv_with_r(
        y,
        draws=2000,
        burnin=500,
    )

    print(mcmc_summary)
