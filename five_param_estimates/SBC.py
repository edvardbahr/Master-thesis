import os
import tempfile
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sim_5_param_data as sim
import torch

from test_TCN import (
    PARAMETER_NAMES,
    TRANSFORMED_TARGET_NAMES,
    load_tcn_model,
    predict_model_transformed_gaussian,
    transform_five_parameters,
)


DEFAULT_CHECKPOINT_PATH = "weights/svghst_posterior_tcn_live_default_n2530_multiscale_topk.pt"
DEFAULT_OUTPUT_DIR = Path("tcn_5_param_sbc")
DEFAULT_N_SIMULATIONS = 5000
DEFAULT_BATCH_SIZE = 1024
DEFAULT_BINS = 50
DEFAULT_PRIOR = "default"
DEFAULT_SEED = 12345
POSTERIOR_VAR_EPS = 1e-12


def standard_normal_cdf(z):
    z = np.asarray(z, dtype=np.float64)
    flat = z.ravel()
    normal = NormalDist()
    cdf = np.fromiter(
        (normal.cdf(float(value)) for value in flat),
        dtype=np.float64,
        count=flat.size,
    )

    return cdf.reshape(z.shape)


def compute_sbc_cdf_values(transformed_theta, posterior_mean, posterior_var):
    transformed_theta = np.asarray(transformed_theta, dtype=np.float64)
    posterior_mean = np.asarray(posterior_mean, dtype=np.float64)
    posterior_var = np.asarray(posterior_var, dtype=np.float64)

    if transformed_theta.shape != posterior_mean.shape:
        raise ValueError(
            "transformed_theta and posterior_mean must have the same shape; "
            f"got {transformed_theta.shape} and {posterior_mean.shape}."
        )

    if transformed_theta.shape != posterior_var.shape:
        raise ValueError(
            "transformed_theta and posterior_var must have the same shape; "
            f"got {transformed_theta.shape} and {posterior_var.shape}."
        )

    posterior_sd = np.sqrt(np.clip(posterior_var, POSTERIOR_VAR_EPS, None))
    z = (transformed_theta - posterior_mean) / posterior_sd

    return standard_normal_cdf(z)


def ks_uniform_statistic(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    n = len(values)

    if n < 1:
        raise ValueError("values must contain at least one element.")

    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("SBC CDF values must lie in [0, 1].")

    empirical_upper = np.arange(1, n + 1, dtype=np.float64) / n
    empirical_lower = np.arange(0, n, dtype=np.float64) / n

    d_plus = np.max(empirical_upper - values)
    d_minus = np.max(values - empirical_lower)

    return float(max(d_plus, d_minus))


def asymptotic_ks_uniform_pvalue(ks_distance, n, n_terms=100):
    if ks_distance <= 0.0:
        return 1.0

    if n < 1:
        raise ValueError("n must be at least one.")

    scaled = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * ks_distance
    terms = [
        ((-1.0) ** (j - 1)) * np.exp(-2.0 * (j * scaled) ** 2)
        for j in range(1, n_terms + 1)
    ]

    return float(np.clip(2.0 * np.sum(terms), 0.0, 1.0))


def uniformity_metrics(cdf_values, bins=DEFAULT_BINS):
    cdf_values = np.asarray(cdf_values, dtype=np.float64)

    if cdf_values.ndim != 2 or cdf_values.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(
            f"cdf_values must have shape (n, {len(PARAMETER_NAMES)})."
        )

    rows = []
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    for index, (parameter, transformed_parameter) in enumerate(
        zip(PARAMETER_NAMES, TRANSFORMED_TARGET_NAMES)
    ):
        values = cdf_values[:, index]
        n = len(values)
        counts, _ = np.histogram(values, bins=bin_edges)
        observed_bin_mass = counts.astype(np.float64) / n
        expected_bin_mass = 1.0 / bins
        bin_deviation = observed_bin_mass - expected_bin_mass
        ks_distance = ks_uniform_statistic(values)

        rows.append({
            "parameter": parameter,
            "transformed_parameter": transformed_parameter,
            "n": n,
            "cdf_mean": float(np.mean(values)),
            "cdf_variance": float(np.var(values, ddof=1)) if n > 1 else np.nan,
            "ks_distance": ks_distance,
            "ks_pvalue_asymptotic": asymptotic_ks_uniform_pvalue(ks_distance, n),
            "histogram_l1_distance": float(np.sum(np.abs(bin_deviation))),
            "histogram_rmse": float(np.sqrt(np.mean(bin_deviation**2))),
            "max_abs_bin_deviation": float(np.max(np.abs(bin_deviation))),
        })

    return pd.DataFrame(rows)


def build_cdf_frame(cdf_values):
    return pd.DataFrame(cdf_values, columns=PARAMETER_NAMES)


def build_prediction_frame(
    theta,
    transformed_theta,
    posterior_mean,
    posterior_var,
    cdf_values,
):
    frame = pd.DataFrame(theta, columns=[f"theta_{name}" for name in PARAMETER_NAMES])

    for index, (parameter, transformed_parameter) in enumerate(
        zip(PARAMETER_NAMES, TRANSFORMED_TARGET_NAMES)
    ):
        frame[f"target_{transformed_parameter}"] = transformed_theta[:, index]
        frame[f"posterior_mean_{transformed_parameter}"] = posterior_mean[:, index]
        frame[f"posterior_sd_{transformed_parameter}"] = np.sqrt(
            np.clip(posterior_var[:, index], POSTERIOR_VAR_EPS, None)
        )
        frame[f"sbc_cdf_{parameter}"] = cdf_values[:, index]

    return frame


def plot_sbc_histograms(cdf_values, metrics, output_dir, bins=DEFAULT_BINS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    plot_paths = []
    metrics_by_parameter = metrics.set_index("parameter")

    for index, parameter in enumerate(PARAMETER_NAMES):
        values = cdf_values[:, index]
        metric_row = metrics_by_parameter.loc[parameter]

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.hist(
            values,
            bins=bin_edges,
            density=True,
            color="tab:blue",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(f"posterior CDF at true {parameter}")
        ax.set_ylabel("density")
        ax.set_title(
            f"SBC histogram for {parameter} "
            f"(KS={metric_row['ks_distance']:.3f}, "
            f"RMSE={metric_row['histogram_rmse']:.3f})"
        )
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        plot_path = output_dir / f"sbc_{parameter}_histogram.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths


def simulate_joint_bayesian_dataset(
    n_simulations=DEFAULT_N_SIMULATIONS,
    sequence_length=None,
    seed=DEFAULT_SEED,
    prior=DEFAULT_PRIOR,
):
    if sequence_length is None:
        raise ValueError("sequence_length must be provided.")

    rng = np.random.default_rng(seed)
    mu, phi, s, r, nu = sim.sample_stochvol_prior(
        n_simulations,
        rng=rng,
        prior=prior,
        return_s2=False,
        dtype=np.float64,
    )
    y = sim.simulate_sv_chunk(
        mu=mu,
        phi=phi,
        s=s,
        r=r,
        nu=nu,
        n=sequence_length,
        rng=rng,
        random_init=True,
    )

    theta = np.column_stack([mu, phi, s, r, nu])
    transformed_theta = transform_five_parameters(
        theta,
        target_names=TRANSFORMED_TARGET_NAMES,
    )

    return y, theta, transformed_theta


def run_sbc(
    tcn_model,
    n_simulations=DEFAULT_N_SIMULATIONS,
    sequence_length=None,
    seed=DEFAULT_SEED,
    prior=DEFAULT_PRIOR,
    batch_size=DEFAULT_BATCH_SIZE,
):
    if sequence_length is None:
        sequence_length = int(tcn_model.checkpoint["sequence_length"])

    y, theta, transformed_theta = simulate_joint_bayesian_dataset(
        n_simulations=n_simulations,
        sequence_length=sequence_length,
        seed=seed,
        prior=prior,
    )

    posterior_mean, posterior_var = predict_model_transformed_gaussian(
        tcn_model,
        y,
        batch_size=batch_size,
    )
    cdf_values = compute_sbc_cdf_values(
        transformed_theta=transformed_theta,
        posterior_mean=posterior_mean,
        posterior_var=posterior_var,
    )

    return theta, transformed_theta, posterior_mean, posterior_var, cdf_values


def print_uniformity_metrics(metrics):
    print("\nSBC uniformity diagnostics:")
    print(
        "Lower KS distance, histogram L1 distance, histogram RMSE, and "
        "max absolute bin deviation mean closer to Uniform(0, 1)."
    )

    display_columns = [
        "parameter",
        "transformed_parameter",
        "cdf_mean",
        "cdf_variance",
        "ks_distance",
        "ks_pvalue_asymptotic",
        "histogram_l1_distance",
        "histogram_rmse",
        "max_abs_bin_deviation",
    ]

    with pd.option_context("display.width", 180):
        print(
            metrics[display_columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.6g}",
            )
        )


def main():
    if DEFAULT_N_SIMULATIONS < 1:
        raise ValueError("DEFAULT_N_SIMULATIONS must be at least one.")

    if DEFAULT_BINS < 2:
        raise ValueError("DEFAULT_BINS must be at least two.")

    if DEFAULT_BATCH_SIZE < 1:
        raise ValueError("DEFAULT_BATCH_SIZE must be at least one.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Loading TCN checkpoint from {DEFAULT_CHECKPOINT_PATH} on {device}.")
    tcn_model = load_tcn_model(DEFAULT_CHECKPOINT_PATH, device)

    sequence_length = int(tcn_model.checkpoint["sequence_length"])
    print(
        f"Running {DEFAULT_N_SIMULATIONS} joint Bayesian simulations "
        f"with sequence length {sequence_length}."
    )

    theta, transformed_theta, posterior_mean, posterior_var, cdf_values = run_sbc(
        tcn_model=tcn_model,
        n_simulations=DEFAULT_N_SIMULATIONS,
        sequence_length=sequence_length,
        seed=DEFAULT_SEED,
        prior=DEFAULT_PRIOR,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    metrics = uniformity_metrics(cdf_values, bins=DEFAULT_BINS)
    cdf_frame = build_cdf_frame(cdf_values)
    prediction_frame = build_prediction_frame(
        theta=theta,
        transformed_theta=transformed_theta,
        posterior_mean=posterior_mean,
        posterior_var=posterior_var,
        cdf_values=cdf_values,
    )

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"n{DEFAULT_N_SIMULATIONS}"
    cdf_path = DEFAULT_OUTPUT_DIR / f"sbc_posterior_cdf_values_{suffix}.csv"
    prediction_path = DEFAULT_OUTPUT_DIR / f"sbc_transformed_predictions_{suffix}.csv"
    metrics_path = DEFAULT_OUTPUT_DIR / f"sbc_uniformity_metrics_{suffix}.csv"

    cdf_frame.to_csv(cdf_path, index_label="simulation")
    prediction_frame.to_csv(prediction_path, index_label="simulation")
    metrics.to_csv(metrics_path, index=False)
    plot_paths = plot_sbc_histograms(
        cdf_values=cdf_values,
        metrics=metrics,
        output_dir=DEFAULT_OUTPUT_DIR,
        bins=DEFAULT_BINS,
    )

    print_uniformity_metrics(metrics)
    print(f"\nSaved SBC CDF values to {cdf_path}")
    print(f"Saved transformed predictions to {prediction_path}")
    print(f"Saved uniformity metrics to {metrics_path}")
    for plot_path in plot_paths:
        print(f"Saved SBC histogram to {plot_path}")


if __name__ == "__main__":
    main()
