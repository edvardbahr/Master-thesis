"""Compare five-parameter TCN posterior scores with MCMC and the prior."""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import digamma, polygamma

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.test_ghst_sv_tcn import load_model, predict
from simulation import sim_5_param_data as sim
from simulation.stochvol_mcmc import (
    log_positive_transform,
    psi_transform,
    run_stochvol_mcmc,
)
from training.sbt_tcn import SVGHST_TARGET_NAMES, theta_to_target_numpy


CHECKPOINT_PATH = PROJECT_DIR / "weights" / "svghst_tcn_default.pt"
SEQUENCE_LENGTH = 253 * 10
PREDICTION_BATCH_SIZE = 128
POSTERIOR_VAR_EPS = 1e-12
DEVICE_NAME: str | None = None

BENCHMARK_SIZE = 5_000
BENCHMARK_SEED = 3
ALPHA = 0.05
MCMC_DRAWS = 20_000
MCMC_BURNIN = 500
MCMC_THINPARA = 1
# The bottleneck. Current implementation of stochvol_mcmc.py
# Is highly memory inefficient, and raising max_cores to a number above
# 3 causes oom errors (on a 32 gb ram system). 
MCMC_MAX_CORES = 3

TRANSFORMED_PARAMETERS = tuple(SVGHST_TARGET_NAMES)
METRIC_PARAMETERS = (
    "mu",
    "psi",
    "rho",
    "logit_r",
    "log_nu_shifted",
)

METRICS_PATH = HERE / "marginal_gaussian_loss_metrics.csv"
SAMPLES_PATH = HERE / "marginal_gaussian_loss_samples.csv"
DIFFERENCES_PATH = HERE / "marginal_gaussian_loss_differences.csv"
MOMENTS_PATH = HERE / "marginal_gaussian_moments.csv"
PRIOR_MOMENTS_PATH = HERE / "transformed_prior_moments.csv"
PLOT_PATH = HERE / "marginal_gaussian_loss_comparison.pdf"

COLORS = {
    "stochvol": "#0000ff",
    "TCN": "#ff0000",
    "prior": "#008000",
}
MARKERS = {"stochvol": "o", "TCN": "^", "prior": "s"}
LOSS_LABELS = {
    "mu": r"$\mu$",
    "psi": r"$\psi$",
    "rho": r"$\rho$",
    "logit_r": r"$\operatorname{logit}(r)$",
    "log_nu_shifted": r"$\log(\nu-\nu_{\min})$",
    "mean_standard": "Mean",
    "mean_ghst": "Mean",
}
METHOD_LABELS = {
    "stochvol": "stochvol",
    "TCN": "TCN",
    "prior": "Prior baseline",
}
MCMC_TRANSFORMS = {
    "phi": {"psi": psi_transform},
    "sigma": {"rho": log_positive_transform},
}


def simulate_benchmark() -> tuple[np.ndarray, np.ndarray]:
    """Simulate prior-predictive series and return their transformed targets."""
    rng = np.random.default_rng(BENCHMARK_SEED)
    theta = np.column_stack(
        sim.sample_stochvol_prior(
            BENCHMARK_SIZE,
            rng=rng,
            prior="default",
            return_s2=False,
            dtype=np.float64,
        )
    )
    y = sim.simulate_sv_chunk(
        mu=theta[:, 0],
        phi=theta[:, 1],
        s=theta[:, 2],
        r=theta[:, 3],
        nu=theta[:, 4],
        n=SEQUENCE_LENGTH,
        rng=rng,
        random_init=True,
    )
    targets = theta_to_target_numpy(
        theta,
        target_names=TRANSFORMED_PARAMETERS,
    ).astype(np.float64)
    return y, targets


def transformed_prior_moments() -> tuple[np.ndarray, np.ndarray]:
    """Return exact moments of the five transformed parameters."""
    prior = sim.get_gh_skew_t_prior_constants("default")
    if prior.r_a0 is not None or prior.r_b0 is not None:
        raise ValueError("The r-moment calculation expects a uniform prior.")

    psi_mean = digamma(prior.phi_a0) - digamma(prior.phi_b0)
    psi_variance = polygamma(1, prior.phi_a0) + polygamma(1, prior.phi_b0)
    rho_mean = 0.5 * (
        np.log(prior.Bs) + digamma(0.5) + np.log(2.0)
    )
    rho_variance = 0.25 * polygamma(1, 0.5)

    def squared_logit(r: float) -> float:
        return float((np.log(r) - np.log1p(-r)) ** 2)

    r_max = prior.r_max
    logit_r_mean = (
        r_max * np.log(r_max)
        + (1.0 - r_max) * np.log1p(-r_max)
    ) / r_max
    logit_r_second_moment = (
        quad(squared_logit, 0.0, r_max, limit=200)[0] / r_max
    )
    logit_r_variance = logit_r_second_moment - logit_r_mean**2

    log_nu_mean = digamma(1.0) - np.log(prior.nu_rate)
    log_nu_variance = polygamma(1, 1.0)

    means = np.array(
        [
            prior.mu_mean,
            psi_mean,
            rho_mean,
            logit_r_mean,
            log_nu_mean,
        ],
        dtype=np.float64,
    )
    variances = np.array(
        [
            prior.mu_sd**2,
            psi_variance,
            rho_variance,
            logit_r_variance,
            log_nu_variance,
        ],
        dtype=np.float64,
    )
    return means, variances


def gaussian_loss(
    targets: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Calculate the elementwise negative log density of a Gaussian."""
    variances = np.clip(variances, POSTERIOR_VAR_EPS, None)
    return 0.5 * (
        np.log(2.0 * np.pi * variances)
        + (targets - means) ** 2 / variances
    )


def mcmc_posterior_moments(
    y: np.ndarray,
    draws: int,
    max_cores: int,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"Running {len(y):,} stochvol chains with {draws:,} draws each.")
    summary = run_stochvol_mcmc(
        y,
        prior="default",
        draws=draws,
        burnin=MCMC_BURNIN,
        thinpara=MCMC_THINPARA,
        alpha=ALPHA,
        transforms=MCMC_TRANSFORMS,
        max_cores=max_cores,
    ).sort_values("index")
    means = summary[["mu_mean", "psi_mean", "rho_mean"]].to_numpy()
    variances = summary[["mu_var", "psi_var", "rho_var"]].to_numpy()
    return means, variances


def loss_components(
    parameter_names: tuple[str, ...],
    losses: np.ndarray,
) -> dict[str, np.ndarray]:
    components = {
        parameter: losses[:, index]
        for index, parameter in enumerate(parameter_names)
    }
    standard_parameters = METRIC_PARAMETERS[:3]
    ghst_parameters = METRIC_PARAMETERS[3:]

    if all(parameter in parameter_names for parameter in standard_parameters):
        indices = [parameter_names.index(name) for name in standard_parameters]
        components["mean_standard"] = np.mean(losses[:, indices], axis=1)
    if all(parameter in parameter_names for parameter in ghst_parameters):
        indices = [parameter_names.index(name) for name in ghst_parameters]
        components["mean_ghst"] = np.mean(losses[:, indices], axis=1)
    if parameter_names == METRIC_PARAMETERS:
        components["mean_all"] = np.mean(losses, axis=1)
    return components


def summarize_method_losses(
    method: str,
    parameter_names: tuple[str, ...],
    losses: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    sample_frames: list[pd.DataFrame] = []

    for parameter, values in loss_components(parameter_names, losses).items():
        sample_sd = np.std(values, ddof=1) if len(values) > 1 else np.nan
        metric_rows.append(
            {
                "method": method,
                "parameter": parameter,
                "benchmark_size": len(values),
                "mean_loss": np.mean(values),
                "sample_sd": sample_sd,
                "standard_error": sample_sd / np.sqrt(len(values)),
            }
        )
        sample_frames.append(
            pd.DataFrame(
                {
                    "method": method,
                    "benchmark_index": np.arange(len(values)),
                    "parameter": parameter,
                    "marginal_loss": values,
                }
            )
        )

    return pd.DataFrame(metric_rows), pd.concat(sample_frames, ignore_index=True)


def calculate_metric_comparison(
    y: np.ndarray,
    targets: np.ndarray,
    tcn_means: np.ndarray,
    tcn_variances: np.ndarray,
    include_mcmc: bool,
    mcmc_draws: int = MCMC_DRAWS,
    mcmc_max_cores: int = MCMC_MAX_CORES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prior_means, prior_variances = transformed_prior_moments()
    estimates = pd.DataFrame({"benchmark_index": np.arange(len(targets))})

    for index, parameter in enumerate(METRIC_PARAMETERS):
        estimates[f"target_{parameter}"] = targets[:, index]
        estimates[f"tcn_mean_{parameter}"] = tcn_means[:, index]
        estimates[f"tcn_var_{parameter}"] = tcn_variances[:, index]
        estimates[f"prior_mean_{parameter}"] = prior_means[index]
        estimates[f"prior_var_{parameter}"] = prior_variances[index]

    method_losses: list[tuple[str, tuple[str, ...], np.ndarray]] = [
        (
            "TCN",
            METRIC_PARAMETERS,
            gaussian_loss(targets, tcn_means, tcn_variances),
        ),
        (
            "prior",
            METRIC_PARAMETERS,
            gaussian_loss(targets, prior_means, prior_variances),
        ),
    ]

    if include_mcmc:
        mcmc_means, mcmc_variances = mcmc_posterior_moments(
            y,
            draws=mcmc_draws,
            max_cores=mcmc_max_cores,
        )
        method_losses.insert(
            0,
            (
                "stochvol",
                METRIC_PARAMETERS[:3],
                gaussian_loss(
                    targets[:, :3],
                    mcmc_means,
                    mcmc_variances,
                ),
            ),
        )
        for index, parameter in enumerate(METRIC_PARAMETERS[:3]):
            estimates[f"stochvol_mean_{parameter}"] = mcmc_means[:, index]
            estimates[f"stochvol_var_{parameter}"] = mcmc_variances[:, index]

    metric_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    for method, parameter_names, losses in method_losses:
        metrics, samples = summarize_method_losses(
            method,
            parameter_names,
            losses,
        )
        metric_frames.append(metrics)
        sample_frames.append(samples)

    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(sample_frames, ignore_index=True),
        estimates,
    )


def apply_plot_style() -> None:
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


def plot_metric_comparison(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    legend_handles: dict[str, object] = {}
    panels = (
        (
            axes[0],
            ("mu", "psi", "rho", "mean_standard"),
            ("stochvol", "TCN", "prior"),
            "Standard SV parameters",
        ),
        (
            axes[1],
            ("logit_r", "log_nu_shifted", "mean_ghst"),
            ("TCN", "prior"),
            "GHST parameters",
        ),
    )

    for axis, parameters, methods, title in panels:
        x = np.arange(len(parameters), dtype=np.float64)
        available_methods = tuple(
            method for method in methods if method in metrics["method"].values
        )
        offsets = dict(
            zip(
                available_methods,
                np.linspace(-0.18, 0.18, len(available_methods)),
            )
        )

        for method in available_methods:
            data = (
                metrics[
                    (metrics["method"] == method)
                    & (metrics["parameter"].isin(parameters))
                ]
                .set_index("parameter")
                .loc[list(parameters)]
            )
            legend_handles[method] = axis.errorbar(
                x + offsets[method],
                data["mean_loss"],
                yerr=data["standard_error"],
                fmt=MARKERS[method],
                color=COLORS[method],
                capsize=3,
                markersize=5,
                linestyle="none",
                label=METHOD_LABELS[method],
            )

        axis.set_xticks(x, [LOSS_LABELS[name] for name in parameters])
        axis.set_title(title)
        axis.set_ylabel("Mean marginal Gaussian loss")
        axis.grid(axis="y", alpha=0.25)

    legend_order = tuple(
        method
        for method in ("stochvol", "TCN", "prior")
        if method in legend_handles
    )
    fig.legend(
        [legend_handles[method] for method in legend_order],
        [METHOD_LABELS[method] for method in legend_order],
        loc="lower center",
        ncol=len(legend_order),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def paired_loss_differences(samples: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired scores; a negative difference favors the method."""
    wide = samples.pivot(
        index=["benchmark_index", "parameter"],
        columns="method",
        values="marginal_loss",
    )
    rows: list[dict[str, object]] = []

    for method, reference in (
        ("TCN", "stochvol"),
        ("TCN", "prior"),
        ("stochvol", "prior"),
    ):
        if method not in wide or reference not in wide:
            continue
        differences = (wide[method] - wide[reference]).dropna()
        for parameter, values in differences.groupby(level="parameter"):
            difference_values = values.to_numpy()
            standard_error = np.std(difference_values, ddof=1) / np.sqrt(
                len(difference_values)
            )
            mean_difference = np.mean(difference_values)
            rows.append(
                {
                    "method": method,
                    "reference": reference,
                    "parameter": parameter,
                    "benchmark_size": len(difference_values),
                    "mean_loss_difference": mean_difference,
                    "standard_error": standard_error,
                    "ci_lower": mean_difference - 1.96 * standard_error,
                    "ci_upper": mean_difference + 1.96 * standard_error,
                }
            )
    return pd.DataFrame(rows)


def save_metric_results(
    y: np.ndarray,
    targets: np.ndarray,
    tcn_means: np.ndarray,
    tcn_variances: np.ndarray,
) -> None:
    metrics, samples, estimates = calculate_metric_comparison(
        y,
        targets,
        tcn_means,
        tcn_variances,
        include_mcmc=True,
    )
    metrics.to_csv(METRICS_PATH, index=False)
    samples.to_csv(SAMPLES_PATH, index=False)
    paired_loss_differences(samples).to_csv(DIFFERENCES_PATH, index=False)
    estimates.to_csv(MOMENTS_PATH, index=False)

    prior_means, prior_variances = transformed_prior_moments()
    pd.DataFrame(
        {
            "parameter": METRIC_PARAMETERS,
            "mean": prior_means,
            "variance": prior_variances,
        }
    ).to_csv(PRIOR_MOMENTS_PATH, index=False)

    plot_metric_comparison(metrics, PLOT_PATH)
    print("\nMarginal Gaussian loss comparison:")
    print(
        metrics.pivot(
            index="parameter",
            columns="method",
            values="mean_loss",
        ).to_string(float_format=lambda value: f"{value:.6g}")
    )


def main() -> None:
    print(f"Simulating {BENCHMARK_SIZE:,} benchmark datasets.")
    y, targets = simulate_benchmark()

    print(f"Loading {CHECKPOINT_PATH.name}.")
    model, checkpoint = load_model(CHECKPOINT_PATH, device=DEVICE_NAME)
    tcn_means, tcn_variances = predict(
        model,
        checkpoint,
        y,
        batch_size=PREDICTION_BATCH_SIZE,
    )
    save_metric_results(y, targets, tcn_means, tcn_variances)


if __name__ == "__main__":
    main()
