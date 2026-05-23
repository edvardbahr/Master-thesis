import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import simulateData as sim


HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "stochvolMCMC.R"
PARAMETER_NAMES = ("mu", "phi", "sigma")


@dataclass
class StochvolMCMCResult:
    summary: pd.DataFrame
    draws: pd.DataFrame | None = None


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
    prior="finance",
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
        return StochvolMCMCResult(summary=summary, draws=parameter_draws)

    return summary


def simulate_sanity_check_data(n_series=6, n_obs=253, seed=12345):
    rng = np.random.default_rng(seed)

    truth = pd.DataFrame({
        "mu_true": np.linspace(-10.0, -8.0, n_series),
        "phi_true": np.linspace(0.94, 0.985, n_series),
        "sigma_true": np.linspace(0.15, 0.35, n_series),
    })

    y = sim.simulate_sv_chunk(
        mu=truth["mu_true"].to_numpy(),
        phi=truth["phi_true"].to_numpy(),
        sigma=truth["sigma_true"].to_numpy(),
        n=n_obs,
        rng=rng,
    )

    return y, truth


def summarize_sanity_check(summary, truth):
    summary = summary.sort_values("index").reset_index(drop=True)
    truth = truth.reset_index(drop=True)

    rows = []
    for parameter in PARAMETER_NAMES:
        true_value = truth[f"{parameter}_true"]
        median = summary[f"{parameter}_median"]
        lower = summary[f"{parameter}_ci_lower"]
        upper = summary[f"{parameter}_ci_upper"]

        rows.append({
            "parameter": parameter,
            "coverage": np.mean((lower <= true_value) & (true_value <= upper)),
            "mean_error": np.mean(median - true_value),
            "mean_abs_error": np.mean(np.abs(median - true_value)),
            "mean_ci_width": np.mean(upper - lower),
        })

    return pd.DataFrame(rows)


def check_draw_summary_consistency(result):
    if result.draws is None:
        raise ValueError("Raw draws are needed for this check.")

    rows = []
    grouped = result.draws.groupby("index", sort=True)
    summary = result.summary.set_index("index")

    for index, draws_for_series in grouped:
        for parameter in PARAMETER_NAMES:
            rows.append({
                "index": index,
                "parameter": parameter,
                "mean_diff": (
                    draws_for_series[parameter].mean()
                    - summary.loc[index, f"{parameter}_mean"]
                ),
                "median_diff": (
                    draws_for_series[parameter].median()
                    - summary.loc[index, f"{parameter}_median"]
                ),
            })

    return pd.DataFrame(rows)


def run_sanity_check(
    n_series=6,
    n_obs=253,
    prior="finance",
    draws=2000,
    burnin=500,
    thinpara=1,
    alpha=0.05,
    seed=12345,
):
    y, truth = simulate_sanity_check_data(
        n_series=n_series,
        n_obs=n_obs,
        seed=seed,
    )

    result = run_stochvol_mcmc(
        y,
        prior=prior,
        draws=draws,
        burnin=burnin,
        thinpara=thinpara,
        alpha=alpha,
        return_draws=True,
    )

    sanity = summarize_sanity_check(result.summary, truth)
    consistency = check_draw_summary_consistency(result)

    return result, truth, sanity, consistency


def main():
    parser = argparse.ArgumentParser(
        description="Run small sanity checks for the Python-to-stochvol bridge."
    )
    parser.add_argument("--n-series", type=int, default=6)
    parser.add_argument("--n-obs", type=int, default=253)
    parser.add_argument("--prior", default="finance")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--thinpara", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    result, truth, sanity, consistency = run_sanity_check(
        n_series=args.n_series,
        n_obs=args.n_obs,
        prior=args.prior,
        draws=args.draws,
        burnin=args.burnin,
        thinpara=args.thinpara,
        alpha=args.alpha,
        seed=args.seed,
    )

    print("\nTrue parameters:")
    print(truth.to_string(index=False))

    print("\nMCMC posterior summaries:")
    print(result.summary.to_string(index=False))

    print("\nSanity-check metrics:")
    print(sanity.to_string(index=False))

    print("\nMax absolute summary-vs-raw-draw discrepancy:")
    print(
        consistency[["mean_diff", "median_diff"]]
        .abs()
        .max()
        .to_string()
    )

    print(f"\nRaw parameter draws returned: {len(result.draws)} rows")


if __name__ == "__main__":
    main()
