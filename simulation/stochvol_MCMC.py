import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sim_5_param_data as sim
from numpy.typing import ArrayLike

HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "stochvol_MCMC.R"
PARAMETER_NAMES = ("mu", "phi", "sigma")
Transform = Callable[[ArrayLike], ArrayLike]
TransformMap = Mapping[str, Mapping[str, Transform]]


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


def normalize_transforms(transforms: TransformMap | None) -> TransformMap:
    """Check transform parameter names before starting the expensive MCMC run."""
    if transforms is None:
        return {}

    unknown_parameters = sorted(set(transforms).difference(PARAMETER_NAMES))
    if unknown_parameters:
        raise ValueError(
            "Unknown raw parameter name(s) in transforms: "
            + ", ".join(unknown_parameters)
        )

    used_names = set(PARAMETER_NAMES)
    for parameter_transforms in transforms.values():
        for transformed_name, transform_fn in parameter_transforms.items():
            if transformed_name in used_names:
                raise ValueError(
                    f"Transform name {transformed_name!r} is already in use."
                )
            if not callable(transform_fn):
                raise TypeError(f"Transform {transformed_name!r} must be callable.")
            used_names.add(transformed_name)

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
    runtime_by_series: Mapping[int, float] | None = None,
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
                        f"Transform {transformed_name!r} for {parameter!r} returned "
                        f"shape {transformed_values.shape}; expected {values.shape}."
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

    if draws < 1:
        raise ValueError("draws must be a positive integer.")

    if burnin < 0:
        raise ValueError("burnin must be a non-negative integer.")

    if thinpara < 1:
        raise ValueError("thinpara must be a positive integer.")

    transforms = normalize_transforms(transforms)

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
            f"stochvol MCMC returned {len(summary)} summary rows for {y.shape[0]} series."
        )

    if return_draws:
        parameter_draws = (
            pd.concat(draw_frames, ignore_index=True)
            .sort_values(["series_index", "draw_index"])
            .reset_index(drop=True)
        )
        return summary, parameter_draws

    return summary


def plot_parameter_histograms_with_normal(
    draws: pd.DataFrame,
    output_path: str | Path,
    true_values: Mapping[str, float] | None = None,
    parameters: Sequence[str] = PARAMETER_NAMES,
    bins: int = 50,
    transformations: Mapping[str, Transform] | None = None,
) -> Path:
    """Plot posterior histograms with empirical normal overlays."""
    transformations = {} if transformations is None else transformations
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(parameters), figsize=(5 * len(parameters), 4))
    axes = np.atleast_1d(axes)

    for ax, parameter in zip(axes, parameters):
        transform_fn = transformations.get(parameter)
        values = np.asarray(draws[parameter], dtype=float)
        if transform_fn is not None:
            values = np.asarray(transform_fn(values), dtype=float)
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
                label=f"N({mean_hat:.3g}, {sd_hat:.3g}^2)",
            )

        if true_values is not None and parameter in true_values:
            true_value = true_values[parameter]
            if transform_fn is not None:
                true_value = float(transform_fn(true_value))
            ax.axvline(
                true_value,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"true = {true_value:.3g}",
            )

        title = parameter
        if transform_fn is not None:
            transform_name = getattr(transform_fn, "__name__", "transformed")
            title = f"{transform_name}({parameter})"
        ax.set_title(title)
        ax.set_xlabel("Posterior draw")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Posterior draws with empirical normal overlays")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_parameter_trace(
    draws: pd.DataFrame,
    output_path: str | Path,
    series_index: int = 1,
    true_values: Mapping[str, float] | None = None,
) -> Path:
    draws = draws[draws["series_index"] == series_index]

    fig, axes = plt.subplots(
        len(PARAMETER_NAMES),
        1,
        figsize=(10, 7),
        sharex=True,
    )

    for ax, parameter in zip(axes, PARAMETER_NAMES):
        ax.plot(draws["draw_index"], draws[parameter], linewidth=0.7)
        if true_values is not None and parameter in true_values:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def main0():

    ### Plot QQ plots of transformed parameters for sequences from different priors.

    from scipy import stats

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

    draws = 20_000
    burnin = 500

    n_short = 253
    n_long = 4 * n_short

    priors = ("default", "finance")
    parameter_names = ("mu", "psi", "rho")
    parameter_labels = (
        r"$\mu$",
        r"$\psi = 2\operatorname{atanh}(\phi)$",
        r"$\rho = \log(\sigma)$",
    )

    qq_colors = {
    "default": (
        "#9ecae1",  # n = 0
        "#4292c6",  # n = 253
        "#08519c",  # n = 1012
    ),
    "finance": (
        "#a1d99b",  # n = 0
        "#41ab5d",  # n = 253
        "#006d2c",  # n = 1012
    ),
    }


    rng = np.random.default_rng(seed=4)
    chains = {}

    def select_transformed_parameters(param_chains):
        param_chains = param_chains.copy()

        for parameter, parameter_transforms in DEFAULT_TRANSFORMS.items():
            for transformed_name, transform_fn in parameter_transforms.items():
                param_chains[transformed_name] = transform_fn(
                    param_chains[parameter]
                )

        return param_chains.loc[:, parameter_names]

    for prior in priors:
        prior_mu, prior_phi, prior_sigma, _, _ = sim.sample_stochvol_prior(
            draws,
            prior=prior,
            fixed_r=0.0,
            fixed_nu=np.inf,
            rng=rng,
        )
        prior_chains = pd.DataFrame(
            {
                "mu": prior_mu,
                "phi": prior_phi,
                "sigma": prior_sigma,
            }
        )
        chains[prior] = {0: select_transformed_parameters(prior_chains)}

        sv_params = sim.sample_stochvol_prior(
            1,
            prior=prior,
            fixed_r=0.0,
            fixed_nu=np.inf,
            rng=rng,
        )

        # Simulate the longest series once.
        y_full = sim.simulate_sv_chunk(
            *sv_params,
            n=n_long,
            rng=rng,
        )

        for n in (n_short, n_long):
            _, param_chains = run_stochvol_mcmc(
                y=y_full[..., :n],
                draws=draws,
                burnin=burnin,
                thinpara=1,
                estimate_ess=False,
                max_cores=-2,
                return_draws=True,
                prior=prior,
            )

            chains[prior][n] = select_transformed_parameters(param_chains)

    # Common plotting probabilities ensure vertically aligned QQ points.
    qq_points = 500
    probabilities = (
        np.arange(1, qq_points + 1) - 0.5
    ) / qq_points
    theoretical_quantiles = stats.norm.ppf(probabilities)

    sample_sizes = (
        (0, r"$n=0$"),
        (n_short, rf"$n={n_short}$"),
        (n_long, rf"$n={n_long}$"),
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(11.5, 13.0),
        sharex=True,
        sharey=True,
    )

    for column, prior in enumerate(priors):
        for row, (parameter, parameter_label) in enumerate(
            zip(parameter_names, parameter_labels)
        ):
            ax = axes[row, column]

            ax.plot(
                theoretical_quantiles,
                theoretical_quantiles,
                color="black",
                linestyle="--",
                linewidth=1.4,
                label="Gaussian reference",
                zorder=1,
            )

            for curve_idx, (n, sample_label) in enumerate(sample_sizes):
                values = chains[prior][n][parameter].to_numpy()

                standardized_values = (
                    values - values.mean()
                ) / values.std(ddof=1)

                empirical_quantiles = np.quantile(
                    standardized_values,
                    probabilities,
                )

                ax.scatter(
                    theoretical_quantiles,
                    empirical_quantiles,
                    color=qq_colors[prior][curve_idx],
                    s=16,
                    alpha=0.85,
                    edgecolors="none",
                    label=sample_label,
                    rasterized=True,
                    zorder=2 + curve_idx,
                )

            ax.set_axisbelow(True)
            ax.grid(
                linestyle=":",
                linewidth=0.8,
            )
            ax.spines[["top", "right"]].set_visible(False)

            if row == 0:
                ax.set_title(f"{prior.capitalize()} prior")
                ax.legend(frameon=False)

            if row == 2:
                ax.set_xlabel(
                    "Theoretical standard-normal quantiles"
                )

            if column == 0:
                ax.set_ylabel(
                    f"{parameter_label}\n"
                    "Empirical quantiles"
                )

    fig.tight_layout()

    fig.savefig(
        "bernstein_von_mises_qq_plots.pdf",
        bbox_inches="tight",
    )

    plt.show()



def main1():

    ## Plot histograms of the minimum effective sample size across MCMC runs.

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
        }
    )
    colors = (
        "#0000ff",
        "#008000",
        "#ff0000",
        "#00bfbf",
        "#bf00bf",
        "#bfbf00",
    )

    
    N = 2000
    draws = 20000
    burnin = 500
    alpha = 0.05

    priors = ("default", "finance")
    ess_transforms = {
        "mu": {
            "mu_centered_sq": centered_square_transform,
        },
        "phi": {
            "psi": psi_transform,
            "psi_centered_sq": psi_sq_transform,
        },
        "sigma": {
            "rho": log_positive_transform,
            "rho_centered_sq": log_positive_sq_transform,
        },
    }
    ess_columns = [
        "mu_ESS",
        "psi_ESS",
        "rho_ESS",
        "mu_centered_sq_ESS",
        "psi_centered_sq_ESS",
        "rho_centered_sq_ESS",
    ]
    param_labels = np.array(
        [
            r"$\mu$",
            r"$\psi$",
            r"$\rho$",
            r"$(\mu-\bar{\mu})^2$",
            r"$(\psi-\bar{\psi})^2$",
            r"$(\rho-\bar{\rho})^2$",
        ]
    )

    rng = np.random.default_rng(seed=2)
    results = {}

    for prior in priors:
        sv_params = sim.sample_stochvol_prior(
            N,
            prior=prior,
            fixed_r=0.0,
            fixed_nu=np.inf,
            rng=rng,
        )

        sv_chunk = sim.simulate_sv_chunk(
            *sv_params,
            n=253,
            rng=rng,
        )

        summary = run_stochvol_mcmc(
            y=sv_chunk,
            draws=draws,
            burnin=burnin,
            thinpara=1,
            alpha=alpha,
            estimate_ess=True,
            transforms=ess_transforms,
            max_cores=-2,
            return_draws=False,
            prior=prior,
        )

        ess_values = summary[ess_columns].to_numpy()

        worst_idx = np.argmin(ess_values, axis=1)
        min_ess = ess_values[np.arange(N), worst_idx]

        worst_param_ratio = (
            pd.Series(param_labels[worst_idx])
            .value_counts(normalize=True)
            .reindex(param_labels, fill_value=0)
        )

        results[prior] = {
            "min_ess": min_ess,
            "lower_bound": np.quantile(min_ess, alpha),
            "param_ratio": worst_param_ratio,
        }

        print(f"\n{prior.capitalize()} prior")
        print(f"Mean minimum ESS: {min_ess.mean():.2f}")
        print(f"{100 * alpha:.0f}% ESS quantile: {results[prior]['lower_bound']:.2f}")
        print("Ratio of runs with lowest ESS:")
        print(worst_param_ratio)


    # Common bin edges make the histograms directly comparable
    all_min_ess = np.concatenate(
        [results[prior]["min_ess"] for prior in priors]
    )
    bins = np.histogram_bin_edges(all_min_ess, bins="fd")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.2),
        sharex=True,
        sharey=True,
    )

    for i, (ax, prior) in enumerate(zip(axes, priors)):
        min_ess = results[prior]["min_ess"]
        lower_bound = results[prior]["lower_bound"]

        ax.set_axisbelow(True)
        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.8,
        )

        ax.hist(
            min_ess,
            bins=bins,
            color=colors[i],
            edgecolor="white",
            linewidth=0.8,
        )

        ax.axvline(
            lower_bound,
            color=colors[2],
            linestyle="--",
            linewidth=2,
            label=rf"{100 * alpha:.0f}% quantile = {lower_bound:.1f}",
        )

        ax.set_title(f"{prior.capitalize()} prior")
        ax.set_xlabel("Minimum effective sample size")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)

    axes[0].set_ylabel("Number of MCMC runs")

    fig.tight_layout()
    fig.savefig("minimum_ess_histograms.pdf", bbox_inches="tight")
    plt.show()




if __name__ == "__main__":
    main0()
