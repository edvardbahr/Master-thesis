import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sim_5_param_data as sim


HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "stochvol_MCMC.R"
PARAMETER_NAMES = ("mu", "phi", "sigma")
DEFAULT_TRANSFORMED_PARAMETER_NAMES = ("mu", "psi", "log_s")


def identity_transform(x):
    return x


def psi_transform(phi, eps=1e-6):
    phi = np.clip(np.asarray(phi, dtype=np.float64), -1.0 + eps, 1.0 - eps)
    return 2.0 * np.arctanh(phi)


def log_positive_transform(x, eps=1e-12):
    x = np.clip(np.asarray(x, dtype=np.float64), eps, None)
    return np.log(x)


DEFAULT_TRANSFORMS = (
    identity_transform,
    psi_transform,
    log_positive_transform,
)


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


def resolve_n_workers(max_cores, n_rows):
    if max_cores is None:
        max_cores = 1

    if int(max_cores) != max_cores or max_cores == 0:
        raise ValueError(
            "max_cores must be a non-zero integer. Use negative values as "
            "CPU offsets, e.g. -2 means all available cores except 2."
        )

    max_cores = int(max_cores)
    available_cpus = os.cpu_count() or 1

    if max_cores < 0:
        max_cores = available_cpus + max_cores
        if max_cores < 1:
            raise ValueError(
                "max_cores leaves no worker processes available. "
                f"With {available_cpus} CPU core(s), use max_cores >= "
                f"{1 - available_cpus}."
            )

    return min(max_cores, available_cpus, n_rows)


def make_row_chunks(n_rows, n_workers):
    chunk_size = max(1, int(np.ceil(n_rows / n_workers)))
    chunks = []
    start = 0

    while start < n_rows:
        stop = min(start + chunk_size, n_rows)
        chunks.append((start, stop))
        start = stop

    return chunks


def normalize_transforms(transform, transformed_parameter_names):
    if transform is None:
        return None, None

    if isinstance(transform, dict):
        transforms = tuple(transform[name] for name in PARAMETER_NAMES)
    else:
        transforms = tuple(transform)

    if len(transforms) != len(PARAMETER_NAMES):
        raise ValueError(
            f"transform must contain {len(PARAMETER_NAMES)} functions, one for each "
            f"parameter in {PARAMETER_NAMES}."
        )

    transformed_parameter_names = tuple(transformed_parameter_names)
    if len(transformed_parameter_names) != len(PARAMETER_NAMES):
        raise ValueError(
            f"transformed_parameter_names must contain {len(PARAMETER_NAMES)} names."
        )

    for transform_fn in transforms:
        if not callable(transform_fn):
            raise TypeError("Each transform entry must be callable.")

    return transforms, transformed_parameter_names


def summarize_values(values, alpha):
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
        }

    return {
        "mean": float(np.mean(values)),
        "var": float(np.var(values, ddof=1)) if values.size > 1 else np.nan,
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "median": float(np.median(values)),
        "ci_lower": float(np.quantile(values, alpha / 2.0)),
        "ci_upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
    }


def add_summary_columns(row, prefix, values, alpha):
    summary = summarize_values(values, alpha)

    for statistic_name, statistic_value in summary.items():
        row[f"{prefix}_{statistic_name}"] = statistic_value


def summarize_parameter_draws(
    parameter_draws,
    alpha=0.05,
    transform=DEFAULT_TRANSFORMS,
    transformed_parameter_names=DEFAULT_TRANSFORMED_PARAMETER_NAMES,
):
    transforms, transformed_parameter_names = normalize_transforms(
        transform,
        transformed_parameter_names,
    )
    required_columns = {"series_index", "draw_index", *PARAMETER_NAMES}
    missing = sorted(required_columns.difference(parameter_draws.columns))

    if missing:
        raise ValueError(
            "Draw frame is missing required column(s): " + ", ".join(missing)
        )

    rows = []
    grouped = parameter_draws.groupby("series_index", sort=True)

    for series_index, group in grouped:
        row = {
            "index": int(series_index),
            "alpha": float(alpha),
            "credible_level": 1.0 - float(alpha),
            "n_draws": int(len(group)),
        }

        for parameter in PARAMETER_NAMES:
            values = group[parameter].to_numpy(dtype=np.float64)
            add_summary_columns(row, parameter, values, alpha)

        if transforms is not None:
            for parameter, transformed_name, transform_fn in zip(
                PARAMETER_NAMES,
                transformed_parameter_names,
                transforms,
            ):
                values = group[parameter].to_numpy(dtype=np.float64)
                transformed_values = transform_fn(values)
                add_summary_columns(
                    row,
                    f"transformed_{transformed_name}",
                    transformed_values,
                    alpha,
                )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


def prepare_chunk_workspace(tmpdir, chunk_id, chunk_start, y_chunk):
    chunk_dir = tmpdir / f"chunk_{chunk_id:04d}"
    input_dir = chunk_dir / "input"
    output_dir = chunk_dir / "output"
    series_dir = chunk_dir / "series"

    input_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    series_dir.mkdir(parents=True, exist_ok=False)

    for local_index in range(y_chunk.shape[0]):
        global_index = chunk_start + local_index + 1
        series_workspace = series_dir / f"series_{global_index:06d}"
        series_input_dir = series_workspace / "input"
        series_output_dir = series_workspace / "output"

        series_input_dir.mkdir(parents=True, exist_ok=False)
        series_output_dir.mkdir(parents=True, exist_ok=False)
        np.savetxt(
            series_input_dir / "y.csv",
            y_chunk[local_index].reshape(1, -1),
            delimiter=",",
        )

    return {
        "chunk_dir": chunk_dir,
        "input_path": input_dir / "y_matrix.csv",
        "draws_path": output_dir / "parameter_draws.csv",
        "series_dir": series_dir,
    }


def run_stochvol_chunk(
    y_chunk,
    chunk_start,
    chunk_id,
    tmpdir,
    rscript,
    prior_constants,
    draws,
    burnin,
    thinpara,
):
    workspace = prepare_chunk_workspace(
        tmpdir=tmpdir,
        chunk_id=chunk_id,
        chunk_start=chunk_start,
        y_chunk=y_chunk,
    )
    input_path = workspace["input_path"]
    draws_path = workspace["draws_path"]
    np.savetxt(input_path, y_chunk, delimiter=",")

    command = [
        rscript,
        str(R_SCRIPT),
        str(input_path),
        str(draws_path),
        str(int(draws)),
        str(int(burnin)),
        str(int(thinpara)),
        str(float(prior_constants.mu_mean)),
        str(float(prior_constants.mu_sd)),
        str(float(prior_constants.phi_a0)),
        str(float(prior_constants.phi_b0)),
        str(float(prior_constants.Bs)),
    ]

    subprocess.run(command, check=True)

    parameter_draws = pd.read_csv(draws_path)
    parameter_draws["series_index"] = (
        parameter_draws["series_index"].astype(int) + int(chunk_start)
    )

    return parameter_draws


def run_and_summarize_chunk(
    y_chunk,
    chunk_start,
    chunk_id,
    tmpdir,
    rscript,
    prior_constants,
    draws,
    burnin,
    thinpara,
    alpha,
    transform,
    transformed_parameter_names,
    return_draws,
):
    parameter_draws = run_stochvol_chunk(
        y_chunk=y_chunk,
        chunk_start=chunk_start,
        chunk_id=chunk_id,
        tmpdir=tmpdir,
        rscript=rscript,
        prior_constants=prior_constants,
        draws=draws,
        burnin=burnin,
        thinpara=thinpara,
    )
    summary = summarize_parameter_draws(
        parameter_draws=parameter_draws,
        alpha=alpha,
        transform=transform,
        transformed_parameter_names=transformed_parameter_names,
    )

    if return_draws:
        return chunk_id, summary, parameter_draws

    return chunk_id, summary, None


def run_stochvol_mcmc(
    y,
    prior="default",
    draws=2000,
    burnin=500,
    thinpara=1,
    alpha=0.05,
    transform=DEFAULT_TRANSFORMS,
    transformed_parameter_names=DEFAULT_TRANSFORMED_PARAMETER_NAMES,
    max_cores=1,
    return_draws=False,
):
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

    transforms, transformed_parameter_names = normalize_transforms(
        transform,
        transformed_parameter_names,
    )

    n_workers = resolve_n_workers(max_cores, y.shape[0])
    chunks = make_row_chunks(y.shape[0], n_workers)
    prior_constants = sim.get_gh_skew_t_prior_constants(prior)
    rscript = find_rscript()

    summary_frames = []
    draw_frames = []

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)

        if n_workers == 1:
            for chunk_id, (start, stop) in enumerate(chunks):
                _, summary, parameter_draws = run_and_summarize_chunk(
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
                    transform=transforms,
                    transformed_parameter_names=transformed_parameter_names,
                    return_draws=return_draws,
                )
                summary_frames.append(summary)
                if return_draws:
                    draw_frames.append(parameter_draws)
        else:
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
                        transform=transforms,
                        transformed_parameter_names=transformed_parameter_names,
                        return_draws=return_draws,
                    )
                    for chunk_id, (start, stop) in enumerate(chunks)
                ]

                for future in as_completed(futures):
                    _, summary, parameter_draws = future.result()
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
    draws,
    output_path,
    true_values=None,
    parameters=PARAMETER_NAMES,
    bins=50,
    transformations=None,
):
    """
    Plot posterior draw histograms with fitted empirical normal overlays.
    """

    if transformations is None:
        transformations = {}

    def identity(x):
        return x

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(parameters), figsize=(5 * len(parameters), 4))

    if len(parameters) == 1:
        axes = [axes]

    for ax, parameter in zip(axes, parameters):
        transform_fn = transformations.get(parameter, identity)
        values = transform_fn(np.asarray(draws[parameter], dtype=float))
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
            transformed_true_value = transform_fn(true_values[parameter])
            ax.axvline(
                transformed_true_value,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"true = {transformed_true_value:.3g}",
            )

        transform_name = getattr(transform_fn, "__name__", "transformed")
        title = parameter if transform_name == "identity" else f"{transform_name}({parameter})"
        ax.set_title(title)
        ax.set_xlabel("Posterior draw")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Posterior draws with empirical normal overlays")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


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


def main():
    mu = np.array([-9.0])
    phi = np.array([0.98])
    s = np.array([0.20])
    r = np.array([0.50])
    nu = np.array([15.0])

    rng = np.random.default_rng(seed=1)
    simulated_data = sim.simulate_sv_chunk(
        mu=mu,
        phi=phi,
        s=s,
        r=r,
        nu=nu,
        n=253 * 2,
        rng=rng,
    )[0]

    summary, draws = run_stochvol_mcmc(
        simulated_data,
        draws=2000,
        burnin=500,
        thinpara=1,
        max_cores=1,
        return_draws=True,
    )

    print(summary[["mu_mean", "phi_mean", "sigma_mean"]])
    print(
        summary[
            [
                "transformed_mu_mean",
                "transformed_psi_mean",
                "transformed_log_s_mean",
            ]
        ]
    )

    true_values = {
        "mu": mu[0],
        "phi": phi[0],
        "sigma": s[0],
    }

    traceplot_path = plot_parameter_trace(
        draws,
        output_path=HERE / "recovered_plots" / "stochvol_traceplot.png",
        true_values=true_values,
    )
    print(f"Saved traceplot to {traceplot_path}")

    hist_path = plot_parameter_histograms_with_normal(
        draws,
        output_path=HERE / "recovered_plots" / "stochvol_hist_normal_overlay_transformed.png",
        true_values=true_values,
        parameters=PARAMETER_NAMES,
        bins=50,
        transformations={
            "phi": psi_transform,
            "sigma": log_positive_transform,
        },
    )
    print(f"Saved histogram plot to {hist_path}")


if __name__ == "__main__":
    main()
