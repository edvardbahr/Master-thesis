import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from . import sim_5_param_data as sim

HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "stochvol_mcmc.R"
PARAMETER_NAMES = ("mu", "phi", "sigma")
Transform = Callable[[ArrayLike], ArrayLike]
TransformMap = dict[str, dict[str, Transform]]


def psi_transform(phi: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    phi = np.clip(np.asarray(phi, dtype=np.float64), -1.0 + eps, 1.0 - eps)
    return 2.0 * np.arctanh(phi)


def log_positive_transform(x: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), eps, None)
    return np.log(x)


def centered_square_transform(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x)) ** 2


def psi_sq_transform(phi: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    return centered_square_transform(psi_transform(phi, eps=eps))


def log_positive_sq_transform(x: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    return centered_square_transform(log_positive_transform(x, eps=eps))


DEFAULT_TRANSFORMS: TransformMap = {
    "phi": {"psi": psi_transform},
    "sigma": {"rho": log_positive_transform},
}


def find_rscript() -> str:
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


def validate_series_matrix(y: ArrayLike) -> np.ndarray:
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


def resolve_n_workers(max_cores: int, n_rows: int) -> int:
    """Resolve a worker count, capped by the number of input rows."""

    if type(max_cores) is not int:
        raise ValueError(
            "max_cores must be an integer. Use a positive worker count or a "
            "negative CPU offset, e.g. -2 means all available cores except 2."
        )

    available_cpus = os.cpu_count() or 1
    n_workers = max_cores

    if max_cores < 0:
        n_workers = available_cpus + max_cores

        if n_workers < 1:
            raise ValueError(
                f"max_cores={max_cores} leaves no worker processes available. "
                f"With {available_cpus} CPU core(s), use an integer from "
                f"{1 - available_cpus} to {available_cpus}, excluding 0."
            )

    if n_workers == 0:
        raise ValueError("max_cores must not be 0.")

    if n_workers > available_cpus:
        raise ValueError(
            f"max_cores={max_cores} exceeds the available CPU count "
            f"({available_cpus})."
        )

    return min(n_workers, n_rows)


def make_row_chunks(n_rows: int, n_workers: int) -> list[tuple[int, int]]:
    chunk_size = (n_rows + n_workers - 1) // n_workers
    return [
        (start, min(start + chunk_size, n_rows))
        for start in range(0, n_rows, chunk_size)
    ]


def validate_transforms(transforms: TransformMap | None, n_draws: int) -> TransformMap:
    """Check transform names and output shapes before starting the MCMC run."""
    if transforms is None:
        return {}

    unknown_parameters = sorted(set(transforms).difference(PARAMETER_NAMES))
    if unknown_parameters:
        raise ValueError(
            "Unknown raw parameter name(s) in transforms: "
            + ", ".join(unknown_parameters)
        )

    used_names = set(PARAMETER_NAMES)
    for parameter_name, parameter_transforms in transforms.items():
        test_values = np.zeros(n_draws)
        if parameter_name == "sigma":
            test_values.fill(1.0)

        for transformed_name, transform_fn in parameter_transforms.items():
            if transformed_name in used_names:
                raise ValueError(
                    f"Transform name {transformed_name!r} is already in use."
                )
            if not callable(transform_fn):
                raise TypeError(f"Transform {transformed_name!r} must be callable.")
            used_names.add(transformed_name)

            transformed_values = np.asarray(transform_fn(test_values))
            if transformed_values.shape != test_values.shape:
                raise ValueError(
                    f"Transform {transformed_name!r} for {parameter_name!r} "
                    f"returned shape {transformed_values.shape}; expected "
                    f"{test_values.shape}."
                )

    return transforms


def estimate_ess_fft(values: np.ndarray) -> float:
    """Estimate MCMC effective sample size using paired autocorrelations."""
    x = np.asarray(values, dtype=float).ravel()
    n = x.size

    if n < 3:
        return float(n)

    x = x - np.mean(x)
    variance = np.dot(x, x) / n

    if not np.isfinite(variance) or variance <= 0:
        return np.nan

    # FFT-based autocovariance calculation: O(n log n).
    fft_length = 1 << (2 * n - 1).bit_length()
    fft_values = np.fft.rfft(x, n=fft_length)
    autocovariance = np.fft.irfft(
        fft_values * np.conjugate(fft_values),
        n=fft_length,
    )[:n]

    # Biased normalization is generally more stable for ESS estimation.
    autocorrelation = autocovariance / autocovariance[0]

    # Geyer's initial positive sequence:
    # sum autocorrelations in adjacent pairs and stop when a pair is nonpositive.
    paired_sum_total = 0.0
    for lag in range(1, n - 1, 2):
        pair_sum = autocorrelation[lag] + autocorrelation[lag + 1]

        if not np.isfinite(pair_sum) or pair_sum <= 0:
            break

        paired_sum_total += pair_sum

    return n / (1.0 + 2.0 * paired_sum_total)


def summarize_values(
    values: ArrayLike,
    alpha: float,
    estimate_ess: bool = False,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "mean": np.nan,
            "var": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "ESS": np.nan,
        }

    return {
        "mean": float(np.mean(values)),
        "var": float(np.var(values, ddof=1)) if values.size > 1 else np.nan,
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "median": float(np.median(values)),
        "ci_lower": float(np.quantile(values, alpha / 2.0)),
        "ci_upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
        "ESS": estimate_ess_fft(values) if estimate_ess else np.nan,
    }


def add_summary_columns(
    row: dict[str, float | int],
    prefix: str,
    values: ArrayLike,
    alpha: float,
    estimate_ess: bool = False,
) -> None:
    summary = summarize_values(values, alpha, estimate_ess=estimate_ess)
    for statistic_name, value in summary.items():
        row[f"{prefix}_{statistic_name}"] = value


def summarize_parameter_draws(
    parameter_draws: pd.DataFrame,
    alpha: float = 0.05,
    estimate_ess: bool = False,
    transforms: TransformMap = DEFAULT_TRANSFORMS,
    runtime_by_series: dict[int, float] | None = None,
) -> pd.DataFrame:
    """
    Summarize raw and transformed draws separately for every series.

    ``transforms`` maps a raw parameter to any number of named transforms::

        {
            "phi": {"psi": psi_transform, "psi_centered_sq": psi_sq_transform},
            "sigma": {"rho": log_positive_transform},
        }

    Each transformed name becomes a summary-column prefix in the returned frame.
    """
    required_columns = {"series_index", "draw_index", *PARAMETER_NAMES}
    missing = sorted(required_columns.difference(parameter_draws.columns))

    if missing:
        raise ValueError(
            "Draw frame is missing required column(s): " + ", ".join(missing)
        )

    rows = []
    grouped = parameter_draws.groupby("series_index", sort=True)

    for series_index, group in grouped:
        summary_started_at = perf_counter()
        row = {
            "index": int(series_index),
            "alpha": float(alpha),
            "credible_level": 1.0 - float(alpha),
            "n_draws": int(len(group)),
        }

        for parameter in PARAMETER_NAMES:
            values = group[parameter].to_numpy(dtype=np.float64)
            add_summary_columns(
                row,
                parameter,
                values,
                alpha,
                estimate_ess=estimate_ess,
            )

        for parameter, parameter_transforms in transforms.items():
            values = group[parameter].to_numpy(dtype=np.float64)

            for transformed_name, transform_fn in parameter_transforms.items():
                transformed_values = transform_fn(values)
                transformed_values = np.asarray(transformed_values, dtype=np.float64)

                if transformed_values.shape != values.shape:
                    raise ValueError(
                        f"Transform {transformed_name!r} for {parameter!r} "
                        f"returned shape {transformed_values.shape}; expected "
                        f"{values.shape}."
                    )

                add_summary_columns(
                    row,
                    transformed_name,
                    transformed_values,
                    alpha,
                    estimate_ess=estimate_ess,
                )

        if runtime_by_series is not None:
            row["runtime_seconds"] = (
                runtime_by_series[int(series_index)]
                + perf_counter()
                - summary_started_at
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


def run_and_summarize_chunk(
    y_chunk: np.ndarray,
    chunk_start: int,
    chunk_id: int,
    tmpdir: Path,
    rscript: str,
    prior_constants: sim.GHSkewTPriorConstants,
    draws: int,
    burnin: int,
    thinpara: int,
    alpha: float,
    estimate_ess: bool,
    transforms: TransformMap,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunk_dir = tmpdir / f"chunk_{chunk_id:04d}"
    chunk_dir.mkdir()

    input_path = chunk_dir / "y.csv"
    draws_path = chunk_dir / "parameter_draws.csv"
    runtime_path = chunk_dir / "series_runtimes.csv"
    np.savetxt(input_path, y_chunk, delimiter=",")

    subprocess.run(
        [
            rscript,
            str(R_SCRIPT),
            str(input_path),
            str(draws_path),
            str(runtime_path),
            str(draws),
            str(burnin),
            str(thinpara),
            str(prior_constants.mu_mean),
            str(prior_constants.mu_sd),
            str(prior_constants.phi_a0),
            str(prior_constants.phi_b0),
            str(prior_constants.Bs),
        ],
        check=True,
    )

    parameter_draws = pd.read_csv(draws_path)
    series_runtimes = pd.read_csv(runtime_path)
    parameter_draws["series_index"] = (
        parameter_draws["series_index"].astype(int) + chunk_start
    )
    series_runtimes["series_index"] = (
        series_runtimes["series_index"].astype(int) + chunk_start
    )

    summary = summarize_parameter_draws(
        parameter_draws=parameter_draws,
        alpha=alpha,
        estimate_ess=estimate_ess,
        transforms=transforms,
        runtime_by_series=series_runtimes.set_index("series_index")[
            "runtime_seconds"
        ].to_dict(),
    )

    return summary, parameter_draws


def run_stochvol_mcmc(
    y: ArrayLike,
    prior: str = "default",
    draws: int = 2000,
    burnin: int = 500,
    thinpara: int = 1,
    alpha: float = 0.05,
    estimate_ess: bool = False,
    transforms: TransformMap | None = DEFAULT_TRANSFORMS,
    max_cores: int = 1,
    return_draws: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run stochvol MCMC independently for each row in ``y``.

    The R script only exports raw parameter draws. This Python wrapper chunks
    rows, runs chunks concurrently, summarizes raw and transformed draws, and
    optionally returns the full raw draw matrix.
    """
    y = validate_series_matrix(y)

    if not R_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find MCMC R script: {R_SCRIPT}")

    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer.")

    if type(burnin) is not int or burnin < 0:
        raise ValueError("burnin must be a non-negative integer.")

    if type(thinpara) is not int or thinpara < 1:
        raise ValueError("thinpara must be a positive integer.")

    if thinpara > draws:
        raise ValueError("thinpara must not exceed draws.")

    transforms = validate_transforms(transforms, draws // thinpara)

    n_workers = resolve_n_workers(max_cores, y.shape[0])
    chunks = make_row_chunks(y.shape[0], n_workers)
    prior_constants = sim.get_gh_skew_t_prior_constants(prior)
    rscript = find_rscript()

    summary_frames = []
    draw_frames = []

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    run_and_summarize_chunk,
                    y_chunk=y[start:stop],
                    chunk_start=start,
                    chunk_id=chunk_id,
                    tmpdir=tmpdir,
                    rscript=rscript,
                    prior_constants=prior_constants,
                    draws=draws,
                    burnin=burnin,
                    thinpara=thinpara,
                    alpha=alpha,
                    estimate_ess=estimate_ess,
                    transforms=transforms,
                )
                for chunk_id, (start, stop) in enumerate(chunks)
            ]

            for future in as_completed(futures):
                summary, parameter_draws = future.result()
                summary_frames.append(summary)
                if return_draws:
                    draw_frames.append(parameter_draws)

    summary = (
        pd.concat(summary_frames, ignore_index=True)
        .sort_values("index")
        .reset_index(drop=True)
    )

    if len(summary) != y.shape[0]:
        raise RuntimeError(
            f"stochvol MCMC returned {len(summary)} summary rows for "
            f"{y.shape[0]} series."
        )

    if return_draws:
        parameter_draws = (
            pd.concat(draw_frames, ignore_index=True)
            .sort_values(["series_index", "draw_index"])
            .reset_index(drop=True)
        )
        return summary, parameter_draws

    return summary
