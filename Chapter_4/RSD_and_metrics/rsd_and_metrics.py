"""Benchmark standard-SV posterior estimators and report loss uncertainty."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd

from evaluation.test_sv_nn_model import load_model, predict_with_runtimes
from simulation import sim_5_param_data as sim
from simulation.stochvol_mcmc import (
    centered_square_transform,
    log_positive_sq_transform,
    log_positive_transform,
    psi_sq_transform,
    psi_transform,
    run_stochvol_mcmc,
)


ALPHA = 0.05
SEQUENCE_LENGTH = 253
MCMC_DRAWS = 20_000
MCMC_BURNIN = 500
MCMC_THINPARA = 1
MCMC_MAX_CORES = -2
BENCHMARK_SIZE = 5_000
SEED = 3
MCMC_DATA_PERCENTAGES = (100, 95, 90, 85, 80)
ROUNDING_ZERO_TOLERANCE = 0.00005

PRIORS = ("default", "finance")
PARAMETERS = ("mu", "psi", "rho")
METHODS = ("stochvol", "TCN", "Summary NN")

WEIGHTS_DIR = PROJECT_DIR / "weights"
CHECKPOINT_NAMES = {
    ("Summary NN", "default"): "summary_nn_default_arima.pt",
    ("Summary NN", "finance"): "summary_nn_finance_arima.pt",
    ("TCN", "default"): "tcn_default.pt",
    ("TCN", "finance"): "tcn_finance.pt",
}

METRICS_PATH = HERE / "transformed_posterior_benchmark_metrics.csv"
UNCERTAINTY_PATH = HERE / "loss_sampling_uncertainty.csv"
POSTERIOR_MOMENTS_PATH = HERE / "full_data_posterior_moments.csv"
LATEX_TABLES_DIR = HERE / "latex_tables"

MCMC_TRANSFORMS = {
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


def simulate_benchmark(prior: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu, phi, sigma, r, nu = sim.sample_stochvol_prior(
        BENCHMARK_SIZE,
        rng=rng,
        prior=prior,
        fixed_r=0.0,
        fixed_nu=np.inf,
        return_s2=False,
        dtype=np.float64,
    )
    y = sim.simulate_sv_chunk(
        mu=mu,
        phi=phi,
        s=sigma,
        r=r,
        nu=nu,
        n=SEQUENCE_LENGTH,
        rng=rng,
        random_init=True,
    )
    targets = np.column_stack(
        (mu, psi_transform(phi), log_positive_transform(sigma))
    )
    return y, targets


def mcmc_moments(
    summary: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    summary = summary.sort_values("index")
    means = summary[["mu_mean", "psi_mean", "rho_mean"]].to_numpy()
    variances = summary[
        [
            "mu_centered_sq_mean",
            "psi_centered_sq_mean",
            "rho_centered_sq_mean",
        ]
    ].to_numpy()
    runtimes = summary["runtime_seconds"].to_numpy()
    return means, variances, runtimes


def gaussian_loss(
    targets: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    return 0.5 * (np.log(variances) + (targets - means) ** 2 / variances)


def loss_sample_frame(
    method: str,
    prior: str,
    targets: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> pd.DataFrame:
    losses = gaussian_loss(targets, means, variances)
    return pd.DataFrame(
        {
            "method": method,
            "prior": prior,
            "benchmark_index": np.arange(len(targets)),
            **{
                f"loss_{parameter}": losses[:, index]
                for index, parameter in enumerate(PARAMETERS)
            },
            "mean_loss": np.mean(losses, axis=1),
        }
    )


def posterior_moment_frame(
    method: str,
    prior: str,
    targets: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> pd.DataFrame:
    data: dict[str, object] = {
        "method": method,
        "prior": prior,
        "benchmark_index": np.arange(len(targets)),
    }
    for index, parameter in enumerate(PARAMETERS):
        data[f"target_{parameter}"] = targets[:, index]
        data[f"mean_{parameter}"] = means[:, index]
        data[f"variance_{parameter}"] = variances[:, index]
    return pd.DataFrame(data)


def metric_row(
    method: str,
    prior: str,
    data_percentage: int,
    n_observations: int,
    targets: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    runtimes: np.ndarray,
    reference_variances: np.ndarray,
) -> dict[str, object]:
    marginal_loss = np.mean(gaussian_loss(targets, means, variances), axis=0)
    rmse = np.sqrt(np.mean((targets - means) ** 2, axis=0))
    log_variance_ratio = np.mean(
        np.log(variances / reference_variances),
        axis=0,
    )

    row: dict[str, object] = {
        "method": method,
        "prior": prior,
        "data_percentage": data_percentage,
        "n_observations": n_observations,
    }
    for index, parameter in enumerate(PARAMETERS):
        row[f"marginal_loss_{parameter}"] = marginal_loss[index]
        row[f"rmse_{parameter}"] = rmse[index]
        row[f"log_var_ratio_{parameter}"] = log_variance_ratio[index]
    row["mean_runtime_seconds"] = np.mean(runtimes)
    row["sd_runtime_seconds"] = np.std(runtimes, ddof=1)
    return row


def run_mcmc_benchmarks(
    y: np.ndarray,
    prior: str,
) -> dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    results = {}
    for data_percentage in MCMC_DATA_PERCENTAGES:
        n_observations = y.shape[1] * data_percentage // 100
        print(
            f"Running stochvol ({prior}, {data_percentage}% data, "
            f"n={n_observations}) with {MCMC_DRAWS:,} draws per sequence."
        )
        summary = run_stochvol_mcmc(
            y[:, :n_observations],
            prior=prior,
            draws=MCMC_DRAWS,
            burnin=MCMC_BURNIN,
            thinpara=MCMC_THINPARA,
            alpha=ALPHA,
            transforms=MCMC_TRANSFORMS,
            max_cores=MCMC_MAX_CORES,
        )
        results[data_percentage] = (
            n_observations,
            *mcmc_moments(summary),
        )
    return results


def metric_columns() -> list[str]:
    columns = ["method", "prior", "data_percentage", "n_observations"]
    columns += [f"marginal_loss_{name}" for name in PARAMETERS]
    columns += [f"rmse_{name}" for name in PARAMETERS]
    columns += [f"log_var_ratio_{name}" for name in PARAMETERS]
    columns += ["mean_runtime_seconds", "sd_runtime_seconds"]
    return columns


def evaluate_neural_model(
    architecture: str,
    prior: str,
    y: np.ndarray,
    targets: np.ndarray,
    reference_variances: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    checkpoint_path = WEIGHTS_DIR / CHECKPOINT_NAMES[(architecture, prior)]
    model, checkpoint = load_model(checkpoint_path)
    print(f"Timing {architecture} ({prior}) one sequence at a time.")
    means, variances, runtimes = predict_with_runtimes(model, checkpoint, y)

    return (
        metric_row(
            architecture,
            prior,
            100,
            y.shape[1],
            targets,
            means,
            variances,
            runtimes,
            reference_variances,
        ),
        loss_sample_frame(
            architecture,
            prior,
            targets,
            means,
            variances,
        ),
        posterior_moment_frame(
            architecture,
            prior,
            targets,
            means,
            variances,
        ),
    )


def calculate_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    neural_rows = []
    mcmc_rows = []
    loss_frames = []
    moment_frames = []

    for prior_index, prior in enumerate(PRIORS):
        print(f"\nGenerating {BENCHMARK_SIZE} benchmark series for the {prior} prior.")
        y, targets = simulate_benchmark(prior, SEED + prior_index)
        mcmc_results = run_mcmc_benchmarks(y, prior)

        _, mcmc_means, mcmc_variances, _ = mcmc_results[100]
        loss_frames.append(
            loss_sample_frame(
                "stochvol",
                prior,
                targets,
                mcmc_means,
                mcmc_variances,
            )
        )
        moment_frames.append(
            posterior_moment_frame(
                "stochvol",
                prior,
                targets,
                mcmc_means,
                mcmc_variances,
            )
        )

        for architecture in ("TCN", "Summary NN"):
            metric, losses, moments = evaluate_neural_model(
                architecture,
                prior,
                y,
                targets,
                mcmc_variances,
            )
            neural_rows.append(metric)
            loss_frames.append(losses)
            moment_frames.append(moments)

        for data_percentage in MCMC_DATA_PERCENTAGES:
            n_observations, means, variances, runtimes = mcmc_results[
                data_percentage
            ]
            mcmc_rows.append(
                metric_row(
                    "stochvol",
                    prior,
                    data_percentage,
                    n_observations,
                    targets,
                    means,
                    variances,
                    runtimes,
                    mcmc_variances,
                )
            )

    metrics = pd.DataFrame(neural_rows + mcmc_rows)[metric_columns()]
    loss_samples = pd.concat(loss_frames, ignore_index=True)
    posterior_moments = pd.concat(moment_frames, ignore_index=True)
    return metrics, loss_samples, posterior_moments


def loss_component_columns() -> dict[str, str]:
    return {
        **{parameter: f"loss_{parameter}" for parameter in PARAMETERS},
        "mean": "mean_loss",
    }


def uncertainty_row(
    estimate_type: str,
    method: str,
    reference_method: str,
    prior: str,
    component: str,
    values: np.ndarray,
) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    estimate = np.mean(values)
    sample_variance = np.var(values, ddof=1)
    variance_of_mean = sample_variance / len(values)
    sd_of_mean = np.sqrt(variance_of_mean)
    relative_sd = np.nan if estimate == 0.0 else sd_of_mean / abs(estimate)
    return {
        "estimate_type": estimate_type,
        "method": method,
        "reference_method": reference_method,
        "prior": prior,
        "component": component,
        "benchmark_size": len(values),
        "estimate": estimate,
        "sample_sd": np.sqrt(sample_variance),
        "estimated_variance_of_mean": variance_of_mean,
        "estimated_sd_of_mean": sd_of_mean,
        "relative_sd": relative_sd,
    }


def select_loss_samples(
    loss_samples: pd.DataFrame,
    method: str,
    prior: str,
) -> pd.DataFrame:
    selected = loss_samples[
        (loss_samples["method"] == method)
        & (loss_samples["prior"] == prior)
    ].sort_values("benchmark_index")
    expected_indices = np.arange(BENCHMARK_SIZE)
    if not np.array_equal(selected["benchmark_index"].to_numpy(), expected_indices):
        raise ValueError(
            f"Expected benchmark indices 0 through {BENCHMARK_SIZE - 1} for "
            f"{method} ({prior})."
        )
    return selected


def paired_loss_uncertainty_rows(
    method_samples: dict[str, pd.DataFrame],
    method: str,
    reference: str,
    prior: str,
) -> list[dict[str, object]]:
    rows = []
    for component, column in loss_component_columns().items():
        differences = (
            method_samples[method][column].to_numpy(dtype=np.float64)
            - method_samples[reference][column].to_numpy(dtype=np.float64)
        )
        rows.append(
            uncertainty_row(
                "paired_loss_difference",
                method,
                reference,
                prior,
                component,
                differences,
            )
        )
    return rows


def summarize_loss_sampling_uncertainty(
    loss_samples: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    components = loss_component_columns()

    for prior in PRIORS:
        method_samples = {
            method: select_loss_samples(loss_samples, method, prior)
            for method in METHODS
        }
        for method, selected in method_samples.items():
            for component, column in components.items():
                values = selected[column].to_numpy(dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Non-finite {component} loss found for {method} ({prior})."
                    )
                rows.append(
                    uncertainty_row(
                        "raw_loss",
                        method,
                        "",
                        prior,
                        component,
                        values,
                    )
                )

        for method, reference in (
            ("TCN", "stochvol"),
            ("Summary NN", "stochvol"),
            ("Summary NN", "TCN"),
        ):
            rows.extend(
                paired_loss_uncertainty_rows(
                    method_samples,
                    method,
                    reference,
                    prior,
                )
            )

    return pd.DataFrame(rows)


def latex_number(value: float) -> str:
    value = float(value)
    if abs(value) < ROUNDING_ZERO_TOLERANCE:
        value = 0.0
    return rf"\({value:.4f}\)"


def latex_row(values: list[str]) -> str:
    return " & ".join(values) + r" \\"


def latex_metric_row(label: str, values: object) -> str:
    return latex_row([label, *(latex_number(value) for value in values)])


def latex_table_body(header: list[str], row_groups: list[list[str]]) -> str:
    rows = [r"\toprule", latex_row(header), r"\midrule"]
    for group_index, group in enumerate(row_groups):
        if group_index:
            rows.append(r"\midrule")
        rows.extend(group)
    rows.append(r"\bottomrule")
    return "\n".join(rows) + "\n"


def method_label(method: str, prior: str) -> str:
    name = "MCMC" if method == "stochvol" else method
    return f"{name} ({prior})"


def select_metric_row(
    metrics: pd.DataFrame,
    method: str,
    prior: str,
    data_percentage: int = 100,
) -> pd.Series:
    selected = metrics[
        (metrics["method"] == method)
        & (metrics["prior"] == prior)
        & (metrics["data_percentage"] == data_percentage)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {method} ({prior}, {data_percentage}%) metric row, "
            f"found {len(selected)}."
        )
    return selected.iloc[0]


def select_uncertainty_row(
    uncertainty: pd.DataFrame,
    estimate_type: str,
    method: str,
    prior: str,
    component: str,
    reference_method: str | None = None,
) -> pd.Series:
    selected = uncertainty[
        (uncertainty["estimate_type"] == estimate_type)
        & (uncertainty["method"] == method)
        & (uncertainty["prior"] == prior)
        & (uncertainty["component"] == component)
    ]
    if reference_method is not None:
        selected = selected[
            selected["reference_method"].fillna("") == reference_method
        ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {estimate_type} uncertainty row for {method}, "
            f"{prior}, {component}; found {len(selected)}."
        )
    return selected.iloc[0]


def metric_loss_values(row: pd.Series) -> np.ndarray:
    values = row[
        [f"marginal_loss_{parameter}" for parameter in PARAMETERS]
    ].to_numpy(dtype=np.float64)
    return np.append(values, np.mean(values))


def loss_uncertainty_table_body(uncertainty: pd.DataFrame) -> str:
    components = (*PARAMETERS, "mean")
    header = [
        "Method",
        r"\(\operatorname{RSD}_{\mu}\)",
        r"\(\operatorname{RSD}_{\psi}\)",
        r"\(\operatorname{RSD}_{\rho}\)",
        r"\(\operatorname{RSD}_{L_K}\)",
    ]
    row_groups = []

    for prior in PRIORS:
        rows = []
        for method in METHODS:
            values = [
                select_uncertainty_row(
                    uncertainty,
                    "raw_loss",
                    method,
                    prior,
                    component,
                )["relative_sd"]
                for component in components
            ]
            rows.append(latex_metric_row(method_label(method, prior), values))

        for method, reference in (
            ("TCN", "stochvol"),
            ("Summary NN", "stochvol"),
            ("Summary NN", "TCN"),
        ):
            values = [
                select_uncertainty_row(
                    uncertainty,
                    "paired_loss_difference",
                    method,
                    prior,
                    component,
                    reference,
                )["relative_sd"]
                for component in components
            ]
            reference_name = "MCMC" if reference == "stochvol" else reference
            label = f"{method} \\(-\\) {reference_name} ({prior})"
            rows.append(latex_metric_row(label, values))
        row_groups.append(rows)

    return latex_table_body(header, row_groups)


def rmse_table_body(metrics: pd.DataFrame) -> str:
    columns = [f"rmse_{parameter}" for parameter in PARAMETERS]
    header = [
        "Method",
        r"\(\operatorname{RMSE}_{\mu}\)",
        r"\(\operatorname{RMSE}_{\psi}\)",
        r"\(\operatorname{RMSE}_{\rho}\)",
    ]
    row_groups = []
    for prior in PRIORS:
        rows = []
        for method in METHODS:
            row = select_metric_row(metrics, method, prior)
            rows.append(latex_metric_row(method_label(method, prior), row[columns]))
        row_groups.append(rows)
    return latex_table_body(header, row_groups)


def log_var_ratio_table_body(metrics: pd.DataFrame) -> str:
    columns = [f"log_var_ratio_{parameter}" for parameter in PARAMETERS]
    header = [
        "Method",
        r"\(\exp(\overline{r}_{\mathrm{var},\mu})\)",
        r"\(\exp(\overline{r}_{\mathrm{var},\psi})\)",
        r"\(\exp(\overline{r}_{\mathrm{var},\rho})\)",
    ]
    row_groups = []
    for prior in PRIORS:
        rows = []
        for method in ("TCN", "Summary NN"):
            row = select_metric_row(metrics, method, prior)
            rows.append(
                latex_metric_row(
                    method_label(method, prior),
                    np.exp(row[columns].to_numpy(dtype=np.float64)),
                )
            )
        row_groups.append(rows)
    return latex_table_body(header, row_groups)


def loss_difference_table_body(metrics: pd.DataFrame) -> str:
    header = [
        "Method",
        r"\(\widehat{\Delta}_{\mathrm{loss},\mu}\)",
        r"\(\widehat{\Delta}_{\mathrm{loss},\psi}\)",
        r"\(\widehat{\Delta}_{\mathrm{loss},\rho}\)",
        r"\(\overline{\Delta}_{\mathrm{loss}}\)",
    ]
    row_groups = []
    for prior in PRIORS:
        reference = metric_loss_values(
            select_metric_row(metrics, "stochvol", prior)
        )
        rows = []
        for method in ("TCN", "Summary NN"):
            differences = (
                metric_loss_values(select_metric_row(metrics, method, prior))
                - reference
            )
            rows.append(latex_metric_row(method_label(method, prior), differences))
        row_groups.append(rows)
    return latex_table_body(header, row_groups)


def closest_mcmc_row(metrics: pd.DataFrame, prior: str) -> pd.Series:
    tcn_loss = metric_loss_values(select_metric_row(metrics, "TCN", prior))[-1]
    candidates = metrics[
        (metrics["method"] == "stochvol")
        & (metrics["prior"] == prior)
        & (metrics["data_percentage"].isin(MCMC_DATA_PERCENTAGES))
    ]
    if len(candidates) != len(MCMC_DATA_PERCENTAGES):
        raise ValueError(
            f"Expected {len(MCMC_DATA_PERCENTAGES)} MCMC rows for {prior}."
        )
    candidate_losses = candidates[
        [f"marginal_loss_{parameter}" for parameter in PARAMETERS]
    ].mean(axis=1)
    return candidates.loc[(candidate_losses - tcn_loss).abs().idxmin()]


def tcn_mcmc_loss_table_body(metrics: pd.DataFrame) -> str:
    header = [
        "Method",
        r"\(\overline{\ell}_{\mu}\)",
        r"\(\overline{\ell}_{\psi}\)",
        r"\(\overline{\ell}_{\rho}\)",
        r"\(L_K\)",
    ]
    row_groups = []
    for prior in PRIORS:
        mcmc = closest_mcmc_row(metrics, prior)
        tcn = select_metric_row(metrics, "TCN", prior)
        beta = float(mcmc["data_percentage"]) / 100.0
        row_groups.append(
            [
                latex_metric_row(
                    f"MCMC ({prior}, \\(\\beta={beta:.2f}\\))",
                    metric_loss_values(mcmc),
                ),
                latex_metric_row(
                    f"TCN ({prior}, full sequence)",
                    metric_loss_values(tcn),
                ),
            ]
        )
    return latex_table_body(header, row_groups)


def runtime_table_body(metrics: pd.DataFrame) -> str:
    header = ["Method", "Mean runtime (s)", "Runtime SD (s)"]
    row_groups = []
    for prior in PRIORS:
        rows = []
        for method in METHODS:
            row = select_metric_row(metrics, method, prior)
            rows.append(
                latex_metric_row(
                    method_label(method, prior),
                    [
                        row["mean_runtime_seconds"],
                        row["sd_runtime_seconds"],
                    ],
                )
            )
        row_groups.append(rows)
    return latex_table_body(header, row_groups)


def main() -> None:
    metrics, loss_samples, posterior_moments = calculate_metrics()
    uncertainty = summarize_loss_sampling_uncertainty(loss_samples)

    metrics.to_csv(METRICS_PATH, index=False)
    uncertainty.to_csv(UNCERTAINTY_PATH, index=False)
    posterior_moments.to_csv(POSTERIOR_MOMENTS_PATH, index=False)

    tables = {
        "loss_uncertainty_tabular.txt": loss_uncertainty_table_body(uncertainty),
        "rmse_tabular.txt": rmse_table_body(metrics),
        "log_var_ratio_tabular.txt": log_var_ratio_table_body(metrics),
        "loss_difference_tabular.txt": loss_difference_table_body(metrics),
        "tcn_mcmc_loss_comparison_tabular.txt": tcn_mcmc_loss_table_body(metrics),
        "runtime_tabular.txt": runtime_table_body(metrics),
    }
    LATEX_TABLES_DIR.mkdir(exist_ok=True)
    for filename, table in tables.items():
        output_path = LATEX_TABLES_DIR / filename
        output_path.write_text(table, encoding="utf-8")
        print(f"Saved LaTeX tabular body to {output_path}")

    print("\nMetric and runtime comparison:")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(
            metrics.to_string(
                index=False,
                float_format=lambda value: f"{value:.6g}",
            )
        )
    print("\nLoss sampling uncertainty:")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(
            uncertainty.to_string(
                index=False,
                float_format=lambda value: f"{value:.6g}",
            )
        )
    print(f"\nSaved metrics to {METRICS_PATH}")
    print(f"Saved loss uncertainty to {UNCERTAINTY_PATH}")
    print(f"Saved full-data posterior moments to {POSTERIOR_MOMENTS_PATH}")


if __name__ == "__main__":
    main()
