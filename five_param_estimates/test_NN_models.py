"""Compare three-parameter SV posterior estimators with stochvol.

Two intentionally separate tasks are available:

``ci``
    Plot credible intervals for mu, phi, and sigma squared on shared
    parameter sweeps.  The 3-by-2 layout separates the default and finance
    priors while comparing the Summary NN, TCN, and stochvol in each panel.

``metrics``
    On prior-predictive benchmark pairs, compare each neural posterior with
    its matching-prior MCMC reference using the paired marginal Gaussian-loss
    difference, posterior-mean RMSE, and average log variance ratio.  These
    metrics are evaluated on the training scale (mu, psi, rho), where
    psi = 2*atanh(phi) and rho = log(sigma).

The GH skew-t checkpoints are intentionally incompatible with, and excluded
from, both tasks.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Union


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sim_5_param_data as sim
from R_to_py_interface import (
    centered_square_transform,
    log_positive_sq_transform,
    log_positive_transform,
    psi_sq_transform,
    psi_transform,
    run_stochvol_mcmc,
    validate_series_matrix,
)
from train_live_CNN import SVPosteriorTCN
from train_live_summary_nn import SVPosteriorNN


DEFAULT_ALPHA = 0.05
DEFAULT_SEQUENCE_LENGTH = 253
DEFAULT_MCMC_DRAWS = 20_000
DEFAULT_MCMC_BURNIN = 500
DEFAULT_MCMC_THINPARA = 1
DEFAULT_MCMC_MAX_CORES = -2
DEFAULT_BENCHMARK_SIZE = 2_000

PARAMETER_NAMES = ("mu", "phi", "sigma2")
TRANSFORMED_PARAMETER_NAMES = ("mu", "psi", "rho")
PRIOR_NAMES = ("default", "finance")
SUMMARY_TARGET_NAMES = ("mu", "psi", "log_sigma")
TCN_TARGET_NAMES = ("mu", "psi", "log_s")
METHOD_NAMES = ("stochvol", "TCN", "Summary NN")

DEFAULT_BASELINE = {
    "mu": -9.0,
    "phi": 0.95,
    "sigma2": 0.25**2,
}
DEFAULT_SWEEP_BOUNDS = {
    "mu": (-12.0, -6.0),
    "phi": (0.905, 0.995),
    "sigma2": (0.05**2, 0.45**2),
}

DEFAULT_CHECKPOINTS = {
    ("Summary NN", "default"): "sv_posterior_summary_nn_live_default_arima.pt",
    ("Summary NN", "finance"): "sv_posterior_summary_nn_live_finance_arima.pt",
    ("TCN", "default"): "sv_posterior_tcn_live_default_n253_multiscale_topk.pt",
    ("TCN", "finance"): "sv_posterior_tcn_live_finance_n253_multiscale_topk.pt",
}

PLOT_COLORS = (
    "#0000ff",
    "#008000",
    "#ff0000",
    "#00bfbf",
    "#bf00bf",
    "#bfbf00",
)
PRIOR_METHOD_COLORS = {
    "default": {
        "stochvol": PLOT_COLORS[0], #blue
        "TCN": PLOT_COLORS[4], # purple
        "Summary NN": PLOT_COLORS[5], #yellow
        
    },
    "finance": {
        "stochvol": PLOT_COLORS[1], #green
        "TCN": PLOT_COLORS[2], # red
        "Summary NN": PLOT_COLORS[3], # teal
    },
}
METHOD_MARKERS = {
    "stochvol": "o",
    "TCN": "^",
    "Summary NN": "s",
    
    
}
PARAMETER_LABELS = {
    "mu": r"$\mu$",
    "phi": r"$\phi$",
    "sigma2": r"$\sigma^2$",
}


@dataclass(frozen=True)
class LoadedModel:
    architecture: str
    prior: str
    model: nn.Module
    checkpoint: dict
    checkpoint_path: Path
    device: torch.device

    @property
    def label(self) -> str:
        return f"{self.architecture} ({self.prior})"


def resolve_input_path(path: Union[str, Path]) -> Path:
    """Resolve a checkpoint without relying on the process working directory."""
    path = Path(path).expanduser()

    if path.is_absolute():
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    candidates = []
    for root in (HERE, *HERE.parents):
        candidates.extend((root / path, root / "weights" / path))

    # Retain explicit command-line paths relative to the caller as a final
    # convenience, while all bundled defaults resolve from __file__ first.
    candidates.append(Path.cwd() / path)

    seen = set()
    for candidate in candidates:
        candidate_key = os.path.normcase(os.path.abspath(candidate))
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find checkpoint '{path}'. Searched:\n  {searched}")


def resolve_output_dir(path: Union[str, Path]) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = HERE / path
    return path.resolve()


def torch_load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dictionary: {path}")
    return checkpoint


def require_keys(checkpoint: dict, keys, checkpoint_path: Path) -> None:
    missing = [key for key in keys if key not in checkpoint]
    if missing:
        raise KeyError(
            f"{checkpoint_path} is missing required checkpoint key(s): "
            + ", ".join(missing)
        )


def activation_from_checkpoint(checkpoint: dict) -> type[nn.Module]:
    activation_name = checkpoint.get("activation", "ReLU")
    if not isinstance(activation_name, str):
        raise TypeError(f"Checkpoint activation must be a class name, got {activation_name!r}.")

    activation = getattr(nn, activation_name, None)
    if not isinstance(activation, type) or not issubclass(activation, nn.Module):
        raise ValueError(f"Unknown torch.nn activation in checkpoint: {activation_name}")
    return activation


def validate_checkpoint_identity(
    checkpoint: dict,
    checkpoint_path: Path,
    *,
    model_class: str,
    target_names: tuple[str, ...],
    prior: str,
) -> None:
    actual_class = checkpoint.get("model_class")
    if actual_class != model_class:
        raise ValueError(
            f"{checkpoint_path.name} has model_class={actual_class!r}; "
            f"expected {model_class!r}."
        )

    actual_targets = tuple(checkpoint.get("target_names", ()))
    if actual_targets != target_names:
        raise ValueError(
            f"{checkpoint_path.name} has target_names={actual_targets}; expected "
            f"{target_names}. Five-parameter GHST checkpoints are not valid here."
        )

    checkpoint_prior = checkpoint.get("prior")
    if checkpoint_prior is not None and checkpoint_prior != prior:
        raise ValueError(
            f"{checkpoint_path.name} was trained with prior={checkpoint_prior!r}, "
            f"but it was assigned to prior={prior!r}."
        )


def load_summary_model(
    checkpoint_path: Union[str, Path],
    prior: str,
    device: torch.device,
) -> LoadedModel:
    checkpoint_path = resolve_input_path(checkpoint_path)
    checkpoint = torch_load_checkpoint(checkpoint_path, device)

    require_keys(
        checkpoint,
        (
            "model_state_dict",
            "input_dim",
            "hidden_dims_shared_trunk",
            "hidden_dims_head",
            "min_var",
            "layer_norm",
            "z_mean",
            "z_std",
            "feature_names",
            "n_acvf_ratios",
            "n_quantiles",
            "compute_arima_coeff",
            "k",
            "eps",
            "center_y",
            "remove_NaNs",
        ),
        checkpoint_path,
    )
    validate_checkpoint_identity(
        checkpoint,
        checkpoint_path,
        model_class="SVPosteriorNN",
        target_names=SUMMARY_TARGET_NAMES,
        prior=prior,
    )

    model = SVPosteriorNN(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims_shared_trunk=tuple(checkpoint["hidden_dims_shared_trunk"]),
        hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
        activation=activation_from_checkpoint(checkpoint),
        min_var=float(checkpoint["min_var"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
        layer_norm=bool(checkpoint["layer_norm"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return LoadedModel(
        architecture="Summary NN",
        prior=prior,
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def load_tcn_model(
    checkpoint_path: Union[str, Path],
    prior: str,
    device: torch.device,
) -> LoadedModel:
    checkpoint_path = resolve_input_path(checkpoint_path)
    checkpoint = torch_load_checkpoint(checkpoint_path, device)

    require_keys(
        checkpoint,
        (
            "model_state_dict",
            "sequence_length",
            "tcn_channels",
            "kernel_size",
            "dilations",
            "hidden_dims_head",
            "min_var",
            "input_mean",
            "input_std",
            "k",
        ),
        checkpoint_path,
    )
    validate_checkpoint_identity(
        checkpoint,
        checkpoint_path,
        model_class="SVPosteriorTCN",
        target_names=TCN_TARGET_NAMES,
        prior=prior,
    )

    # kernel_sizes and top-k pooling were introduced after the original TCN.
    # Use them when stored, with metadata-compatible fallbacks for older files.
    kernel_size = checkpoint.get("kernel_sizes", checkpoint["kernel_size"])
    model = SVPosteriorTCN(
        tcn_channels=tuple(checkpoint["tcn_channels"]),
        kernel_size=kernel_size,
        dilations=tuple(checkpoint["dilations"]),
        hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
        topk_pool_fraction=checkpoint.get("topk_pool_fraction"),
        activation=activation_from_checkpoint(checkpoint),
        param_names=tuple(checkpoint["target_names"]),
        min_var=float(checkpoint["min_var"]),
        input_mean=float(checkpoint["input_mean"]),
        input_std=float(checkpoint["input_std"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return LoadedModel(
        architecture="TCN",
        prior=prior,
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def load_all_models(checkpoint_paths, device: torch.device) -> dict[tuple[str, str], LoadedModel]:
    models = {}
    for prior in PRIOR_NAMES:
        models[("Summary NN", prior)] = load_summary_model(
            checkpoint_paths[("Summary NN", prior)], prior, device
        )
        models[("TCN", prior)] = load_tcn_model(
            checkpoint_paths[("TCN", prior)], prior, device
        )
    return models


def model_parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def parameter_count_frame(models: dict[tuple[str, str], LoadedModel]) -> pd.DataFrame:
    rows = []
    for architecture in ("Summary NN", "TCN"):
        for prior in PRIOR_NAMES:
            loaded_model = models[(architecture, prior)]
            trainable, total = model_parameter_count(loaded_model.model)
            rows.append(
                {
                    "architecture": architecture,
                    "prior": prior,
                    "trainable_parameters": trainable,
                    "total_parameters": total,
                    "checkpoint": loaded_model.checkpoint_path.name,
                }
            )
    return pd.DataFrame(rows)


def print_parameter_counts(models: dict[tuple[str, str], LoadedModel]) -> pd.DataFrame:
    counts = parameter_count_frame(models)
    print("\nNeural-network parameter counts (buffers are not parameters):")
    print(counts.to_string(index=False))

    for architecture in ("Summary NN", "TCN"):
        subset = counts[counts["architecture"] == architecture]
        trainable_match = subset["trainable_parameters"].nunique() == 1
        total_match = subset["total_parameters"].nunique() == 1
        if trainable_match and total_match:
            row = subset.iloc[0]
            print(
                f"{architecture}: default and finance match at "
                f"{int(row['trainable_parameters']):,} trainable / "
                f"{int(row['total_parameters']):,} total parameters."
            )
        else:
            print(f"WARNING: {architecture} default and finance parameter counts differ.")

    summary_total = int(
        counts.loc[
            (counts["architecture"] == "Summary NN")
            & (counts["prior"] == "default"),
            "total_parameters",
        ].iloc[0]
    )
    tcn_total = int(
        counts.loc[
            (counts["architecture"] == "TCN")
            & (counts["prior"] == "default"),
            "total_parameters",
        ].iloc[0]
    )
    difference = tcn_total - summary_total
    ratio = tcn_total / summary_total
    print(
        f"Architecture comparison: TCN has {difference:,} more parameters "
        f"than Summary NN ({ratio:.2f}x as many)."
    )
    return counts


def checkpoint_sequence_length(loaded_model: LoadedModel) -> int:
    sequence_length = loaded_model.checkpoint.get("sequence_length")
    if sequence_length is None:
        sequence_length = loaded_model.checkpoint.get("dataset_config", {}).get(
            "sequence_length"
        )
    if sequence_length is None:
        raise KeyError(f"{loaded_model.checkpoint_path} does not record sequence_length.")
    return int(sequence_length)


def common_sequence_length(models: dict[tuple[str, str], LoadedModel]) -> int:
    lengths = {
        loaded_model.label: checkpoint_sequence_length(loaded_model)
        for loaded_model in models.values()
    }
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"All checkpoints must use one sequence length; got {lengths}.")
    return unique_lengths.pop()


def make_single_parameter_sweep_datasets(
    baseline=None,
    sweeps=None,
    sweep_bounds=None,
    sweep_size=10,
    n=DEFAULT_SEQUENCE_LENGTH,
    rng=None,
    random_init=True,
):
    """Simulate standard SV by fixing the five-parameter simulator at r=0, nu=inf."""
    baseline = DEFAULT_BASELINE | (baseline or {})
    sweep_bounds = DEFAULT_SWEEP_BOUNDS | (sweep_bounds or {})
    if rng is None:
        rng = np.random.default_rng()

    if sweeps is None:
        sweeps = {
            parameter: np.linspace(
                sweep_bounds[parameter][0],
                sweep_bounds[parameter][1],
                sweep_size,
                dtype=np.float64,
            )
            for parameter in PARAMETER_NAMES
        }
    else:
        sweeps = {
            parameter: np.asarray(sweeps[parameter], dtype=np.float64)
            for parameter in PARAMETER_NAMES
        }

    if any(values.ndim != 1 or len(values) < 1 for values in sweeps.values()):
        raise ValueError("Each parameter sweep must be a non-empty one-dimensional array.")
    if np.any(np.abs(sweeps["phi"]) >= 1.0):
        raise ValueError("All phi sweep values must satisfy |phi| < 1.")
    if np.any(sweeps["sigma2"] <= 0.0):
        raise ValueError("All sigma-squared sweep values must be positive.")

    datasets = {}
    for swept_parameter in PARAMETER_NAMES:
        values = sweeps[swept_parameter]
        n_series = len(values)
        parameters = {
            name: np.full(n_series, baseline[name], dtype=np.float64)
            for name in PARAMETER_NAMES
        }
        parameters[swept_parameter] = values

        datasets[swept_parameter] = sim.simulate_sv_chunk(
            mu=parameters["mu"],
            phi=parameters["phi"],
            s=np.sqrt(parameters["sigma2"]),
            r=np.zeros(n_series, dtype=np.float64),
            nu=np.full(n_series, np.inf, dtype=np.float64),
            n=n,
            rng=rng,
            random_init=random_init,
        )

    return datasets, sweeps, baseline


def prepare_summary_input(y, checkpoint: dict) -> np.ndarray:
    y = validate_series_matrix(y)
    expected_length = int(checkpoint["sequence_length"])
    if y.shape[1] != expected_length:
        raise ValueError(
            f"Summary NN expects sequence_length={expected_length}, got {y.shape[1]}."
        )

    summary_options = {
        "n_acvf_ratios": int(checkpoint["n_acvf_ratios"]),
        "n_quantiles": int(checkpoint["n_quantiles"]),
        "compute_arima_coeff": bool(checkpoint["compute_arima_coeff"]),
    }
    expected_feature_names = sim.summary_stats_sv_feature_names(**summary_options)
    checkpoint_feature_names = list(checkpoint["feature_names"])
    if checkpoint_feature_names != expected_feature_names:
        raise ValueError(
            "Summary checkpoint feature_names do not match sim_5_param_data.py."
        )

    summaries = np.empty(
        (y.shape[0], int(checkpoint["input_dim"])), dtype=np.float32
    )
    for row_index, series in enumerate(y):
        summaries[row_index] = sim.summary_stats_sv(
            series,
            k=float(checkpoint["k"]),
            n_acvf_ratios=summary_options["n_acvf_ratios"],
            n_quantiles=summary_options["n_quantiles"],
            eps=float(checkpoint["eps"]),
            compute_arima_coeff=summary_options["compute_arima_coeff"],
            center_y=bool(checkpoint["center_y"]),
            remove_NaNs=bool(checkpoint["remove_NaNs"]),
        ).astype(np.float32, copy=False)

    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float32).reshape(1, -1)
    z_std = np.asarray(checkpoint["z_std"], dtype=np.float32).reshape(1, -1)
    if z_mean.shape[1] != summaries.shape[1] or z_std.shape[1] != summaries.shape[1]:
        raise ValueError("Summary standardization arrays do not match input_dim.")
    z_std = np.where(z_std < 1e-8, 1.0, z_std)
    return ((summaries - z_mean) / z_std).astype(np.float32, copy=False)


def prepare_tcn_input(y, checkpoint: dict) -> np.ndarray:
    y = validate_series_matrix(y)
    expected_length = int(checkpoint["sequence_length"])
    if y.shape[1] != expected_length:
        raise ValueError(f"TCN expects sequence_length={expected_length}, got {y.shape[1]}.")

    # These checkpoints predate storing center_y, but train_live_CNN uses True.
    if bool(checkpoint.get("center_y", True)):
        y = y - np.mean(y, axis=1, keepdims=True)
    return np.log(y * y + float(checkpoint["k"])).astype(np.float32, copy=False)


def prepare_model_input(loaded_model: LoadedModel, y) -> np.ndarray:
    if loaded_model.architecture == "Summary NN":
        return prepare_summary_input(y, loaded_model.checkpoint)
    if loaded_model.architecture == "TCN":
        return prepare_tcn_input(y, loaded_model.checkpoint)
    raise ValueError(f"Unknown architecture: {loaded_model.architecture}")


@torch.no_grad()
def predict_transformed_gaussian(
    loaded_model: LoadedModel,
    y,
    batch_size=1024,
) -> tuple[np.ndarray, np.ndarray]:
    x = prepare_model_input(loaded_model, y)
    means = []
    variances = []

    for start in range(0, len(x), batch_size):
        x_batch = torch.from_numpy(x[start : start + batch_size]).to(
            loaded_model.device
        )
        mean_batch, variance_batch = loaded_model.model(x_batch)
        means.append(mean_batch.detach().cpu().numpy())
        variances.append(variance_batch.detach().cpu().numpy())

    mean = np.vstack(means).astype(np.float64, copy=False)
    variance = np.vstack(variances).astype(np.float64, copy=False)
    if mean.shape != variance.shape or mean.ndim != 2 or mean.shape[1] != 3:
        raise RuntimeError(
            f"{loaded_model.label} returned mean/variance shapes "
            f"{mean.shape}/{variance.shape}; expected (n, 3)."
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
        raise RuntimeError(f"{loaded_model.label} returned non-finite predictions.")
    return mean, np.clip(variance, 0.0, None)


def transformed_gaussian_to_ci_frame(
    transformed_mean,
    transformed_variance,
    alpha=DEFAULT_ALPHA,
) -> pd.DataFrame:
    mean = np.asarray(transformed_mean, dtype=np.float64)
    variance = np.asarray(transformed_variance, dtype=np.float64)
    if mean.shape != variance.shape or mean.ndim != 2 or mean.shape[1] != 3:
        raise ValueError("Transformed mean and variance must both have shape (n, 3).")

    zcrit = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    sd = np.sqrt(np.clip(variance, 0.0, None))
    lower = mean - zcrit * sd
    upper = mean + zcrit * sd

    return pd.DataFrame(
        {
            "alpha": alpha,
            "credible_level": 1.0 - alpha,
            "mu_median": mean[:, 0],
            "mu_ci_lower": lower[:, 0],
            "mu_ci_upper": upper[:, 0],
            "phi_median": np.tanh(mean[:, 1] / 2.0),
            "phi_ci_lower": np.tanh(lower[:, 1] / 2.0),
            "phi_ci_upper": np.tanh(upper[:, 1] / 2.0),
            # The networks model rho = log(sigma).  Since sigma^2 = exp(2*rho)
            # is strictly increasing, transforming the fixed-alpha Gaussian
            # quantiles gives the exact equal-tailed sigma-squared interval.
            "sigma2_median": np.exp(2.0 * mean[:, 2]),
            "sigma2_ci_lower": np.exp(2.0 * lower[:, 2]),
            "sigma2_ci_upper": np.exp(2.0 * upper[:, 2]),
        }
    )


def sigma_squared_transform(sigma) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=np.float64)
    return sigma**2


CI_MCMC_TRANSFORMS = {
    "sigma": {"sigma2": sigma_squared_transform},
}


def validate_stochvol_sigma_squared_ci(ci_frame: pd.DataFrame) -> pd.DataFrame:
    """Validate draw-level sigma-squared summaries returned by the bridge.

    ``para(svsample(...))[, "sigma"]`` contains sigma, not sigma squared.
    The CI call supplies ``CI_MCMC_TRANSFORMS`` so the bridge squares every
    posterior draw *before* calculating the median and alpha-level quantiles.
    This avoids the small interpolation discrepancy caused by squaring
    already-computed empirical quantiles.
    """
    ci_frame = ci_frame.copy()
    required = ("sigma2_median", "sigma2_ci_lower", "sigma2_ci_upper")
    missing = [column for column in required if column not in ci_frame]
    if missing:
        raise KeyError(
            "stochvol summary is missing draw-level sigma-squared CI column(s): "
            + ", ".join(missing)
        )
    values = ci_frame.loc[:, required].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("stochvol returned an invalid sigma-squared summary.")
    return ci_frame


def predict_model_ci(
    loaded_model: LoadedModel,
    y,
    alpha=DEFAULT_ALPHA,
    batch_size=1024,
) -> pd.DataFrame:
    mean, variance = predict_transformed_gaussian(loaded_model, y, batch_size)
    return transformed_gaussian_to_ci_frame(mean, variance, alpha)


def add_ci_rows(
    rows,
    swept_parameter,
    true_values,
    method,
    prior,
    ci_frame,
) -> None:
    ci_frame = ci_frame.sort_values("index").reset_index(drop=True) if "index" in ci_frame else ci_frame.reset_index(drop=True)
    if len(ci_frame) != len(true_values):
        raise RuntimeError(
            f"{method} ({prior}) returned {len(ci_frame)} intervals for "
            f"{len(true_values)} series."
        )

    for value_index, true_value in enumerate(true_values):
        rows.append(
            {
                "swept_parameter": swept_parameter,
                "value_index": value_index,
                "true_value": float(true_value),
                "method": method,
                "prior": prior,
                "combination": f"{method} ({prior})",
                "median": float(ci_frame.loc[value_index, f"{swept_parameter}_median"]),
                "ci_lower": float(ci_frame.loc[value_index, f"{swept_parameter}_ci_lower"]),
                "ci_upper": float(ci_frame.loc[value_index, f"{swept_parameter}_ci_upper"]),
            }
        )


def run_parameter_sweep_test(
    models: dict[tuple[str, str], LoadedModel],
    baseline=None,
    sweep_bounds=None,
    sweeps=None,
    n=DEFAULT_SEQUENCE_LENGTH,
    sweep_size=10,
    seed=2,
    alpha=DEFAULT_ALPHA,
    mcmc_draws=DEFAULT_MCMC_DRAWS,
    mcmc_burnin=DEFAULT_MCMC_BURNIN,
    mcmc_thinpara=DEFAULT_MCMC_THINPARA,
    mcmc_max_cores=DEFAULT_MCMC_MAX_CORES,
    batch_size=1024,
):
    rng = np.random.default_rng(seed)
    datasets, sweeps, baseline = make_single_parameter_sweep_datasets(
        baseline=baseline,
        sweeps=sweeps,
        sweep_bounds=sweep_bounds,
        sweep_size=sweep_size,
        n=n,
        rng=rng,
    )

    rows = []
    for swept_parameter in PARAMETER_NAMES:
        y = datasets[swept_parameter]
        true_values = sweeps[swept_parameter]
        print(
            f"\n{swept_parameter} sweep: {len(true_values)} shared series, "
            "six method/prior combinations."
        )

        for prior in PRIOR_NAMES:
            print(f"  Running stochvol with {prior} prior ...")
            mcmc_ci = run_stochvol_mcmc(
                y,
                prior=prior,
                draws=mcmc_draws,
                burnin=mcmc_burnin,
                thinpara=mcmc_thinpara,
                alpha=alpha,
                transforms=CI_MCMC_TRANSFORMS,
                max_cores=mcmc_max_cores,
            )
            mcmc_ci = validate_stochvol_sigma_squared_ci(mcmc_ci)
            add_ci_rows(
                rows, swept_parameter, true_values, "stochvol", prior, mcmc_ci
            )

            for architecture in ("Summary NN", "TCN"):
                loaded_model = models[(architecture, prior)]
                print(f"  Predicting with {loaded_model.label} ...")
                model_ci = predict_model_ci(
                    loaded_model, y, alpha=alpha, batch_size=batch_size
                )
                add_ci_rows(
                    rows,
                    swept_parameter,
                    true_values,
                    architecture,
                    prior,
                    model_ci,
                )

    return pd.DataFrame(rows), sweeps, baseline


METRIC_MCMC_TRANSFORMS = {
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


def transform_sv_parameters(theta) -> np.ndarray:
    """Map (mu, phi, sigma) to the networks' (mu, psi, rho) scale."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError("theta must have shape (n, 3) for (mu, phi, sigma).")

    mu = theta[:, 0]
    return np.column_stack(
        [
            mu,
            psi_transform(theta[:, 1]),
            log_positive_transform(theta[:, 2]),
        ]
    )


def simulate_benchmark_dataset(
    prior,
    benchmark_size,
    n,
    seed,
    random_init=True,
):
    """Draw prior-predictive standard-SV benchmark pairs for one prior."""
    rng = np.random.default_rng(seed)
    mu, phi, sigma, r, nu = sim.sample_stochvol_prior(
        benchmark_size,
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
        n=n,
        rng=rng,
        random_init=random_init,
    )
    theta = np.column_stack([mu, phi, sigma])
    targets = transform_sv_parameters(theta)
    return y, theta, targets


def transformed_mcmc_moments(summary):
    """Extract moment-matched (mu, psi, rho) Gaussians from MCMC summaries.

    ``METRIC_MCMC_TRANSFORMS`` first maps the chains to mu, psi, and rho and
    then applies (z - mean(z))**2.  Taking the mean of those centered-square
    draws gives the thesis variance with denominator M directly, without
    retaining all 20,000 raw draws in Python.
    """
    summary = summary.sort_values("index").reset_index(drop=True)
    prefixes = TRANSFORMED_PARAMETER_NAMES
    required = ["index", "n_draws"]
    required.extend(f"{name}_mean" for name in prefixes)
    required.extend(f"{name}_centered_sq_mean" for name in prefixes)
    missing = [column for column in required if column not in summary]
    if missing:
        raise KeyError(
            "stochvol summary is missing transformed moment column(s): "
            + ", ".join(missing)
        )

    n_draws = summary["n_draws"].to_numpy(dtype=np.int64)
    if np.any(n_draws < 2):
        raise ValueError("At least two retained MCMC draws are needed for variance.")

    means = np.column_stack(
        [summary[f"{name}_mean"].to_numpy(dtype=np.float64) for name in prefixes]
    )
    population_variances = np.column_stack(
        [
            summary[f"{name}_centered_sq_mean"].to_numpy(dtype=np.float64)
            for name in prefixes
        ]
    )

    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(population_variances)):
        raise FloatingPointError("MCMC transformed moments contain NaN or Inf values.")
    if np.any(population_variances <= 0.0):
        raise FloatingPointError("MCMC transformed variances must be positive.")

    return (
        summary["index"].to_numpy(dtype=np.int64),
        means,
        population_variances,
        n_draws,
    )


def marginal_gaussian_loss(target, mean, variance) -> np.ndarray:
    """Thesis Gaussian loss, excluding the additive constant that cancels."""
    target = np.asarray(target, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if target.shape != mean.shape or target.shape != variance.shape:
        raise ValueError(
            "target, mean, and variance must have identical (n, 3) shapes."
        )
    if target.ndim != 2 or target.shape[1] != len(TRANSFORMED_PARAMETER_NAMES):
        raise ValueError("Gaussian-loss arrays must have shape (n, 3).")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(mean)):
        raise FloatingPointError("Gaussian-loss targets and means must be finite.")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise FloatingPointError("Gaussian-loss variances must be finite and positive.")

    return 0.5 * (
        np.log(variance)
        + (target - mean) ** 2 / variance
    )


def calculate_paired_metrics(
    targets,
    npe_mean,
    npe_variance,
    mcmc_mean,
    mcmc_variance,
    series_indices,
    retained_draws,
    architecture,
    prior,
):
    """Calculate summary and per-sequence metrics from one paired benchmark."""
    targets = np.asarray(targets, dtype=np.float64)
    npe_mean = np.asarray(npe_mean, dtype=np.float64)
    npe_variance = np.asarray(npe_variance, dtype=np.float64)
    mcmc_mean = np.asarray(mcmc_mean, dtype=np.float64)
    mcmc_variance = np.asarray(mcmc_variance, dtype=np.float64)
    series_indices = np.asarray(series_indices, dtype=np.int64)
    retained_draws = np.asarray(retained_draws, dtype=np.int64)
    benchmark_size = len(targets)
    if len(series_indices) != benchmark_size or len(retained_draws) != benchmark_size:
        raise ValueError("MCMC series indices and draw counts must match target rows.")
    if np.unique(retained_draws).size != 1:
        raise ValueError("All benchmark sequences must have the same retained draw count.")

    npe_loss = marginal_gaussian_loss(targets, npe_mean, npe_variance)
    mcmc_loss = marginal_gaussian_loss(targets, mcmc_mean, mcmc_variance)
    npe_squared_error = (targets - npe_mean) ** 2
    mcmc_squared_error = (targets - mcmc_mean) ** 2
    log_variance_ratio = np.log(npe_variance / mcmc_variance)

    summary_rows = []
    detail_frames = []
    for column, parameter in enumerate(TRANSFORMED_PARAMETER_NAMES):
        loss_difference = npe_loss[:, column] - mcmc_loss[:, column]
        summary_rows.append(
            {
                "architecture": architecture,
                "prior": prior,
                "parameter": parameter,
                "benchmark_size": benchmark_size,
                "mcmc_retained_draws": int(retained_draws[0]),
                "mean_npe_gaussian_loss": float(np.mean(npe_loss[:, column])),
                "mean_mcmc_gaussian_loss": float(np.mean(mcmc_loss[:, column])),
                "gaussian_loss_difference": float(np.mean(loss_difference)),
                "npe_posterior_mean_rmse": float(
                    np.sqrt(np.mean(npe_squared_error[:, column]))
                ),
                "mcmc_posterior_mean_rmse": float(
                    np.sqrt(np.mean(mcmc_squared_error[:, column]))
                ),
                "average_log_variance_ratio": float(
                    np.mean(log_variance_ratio[:, column])
                ),
            }
        )
        detail_frames.append(
            pd.DataFrame(
                {
                    "architecture": architecture,
                    "prior": prior,
                    "series_index": series_indices,
                    "parameter": parameter,
                    "target": targets[:, column],
                    "npe_mean": npe_mean[:, column],
                    "npe_variance": npe_variance[:, column],
                    "mcmc_mean": mcmc_mean[:, column],
                    "mcmc_variance": mcmc_variance[:, column],
                    "npe_gaussian_loss": npe_loss[:, column],
                    "mcmc_gaussian_loss": mcmc_loss[:, column],
                    "gaussian_loss_difference": loss_difference,
                    "npe_squared_error": npe_squared_error[:, column],
                    "mcmc_squared_error": mcmc_squared_error[:, column],
                    "log_variance_ratio": log_variance_ratio[:, column],
                    "mcmc_retained_draws": retained_draws,
                }
            )
        )

    return pd.DataFrame(summary_rows), pd.concat(detail_frames, ignore_index=True)


def run_metric_benchmark(
    models: dict[tuple[str, str], LoadedModel],
    priors=PRIOR_NAMES,
    benchmark_size=DEFAULT_BENCHMARK_SIZE,
    n=DEFAULT_SEQUENCE_LENGTH,
    seed=3,
    alpha=DEFAULT_ALPHA,
    mcmc_draws=DEFAULT_MCMC_DRAWS,
    mcmc_burnin=DEFAULT_MCMC_BURNIN,
    mcmc_thinpara=DEFAULT_MCMC_THINPARA,
    mcmc_max_cores=DEFAULT_MCMC_MAX_CORES,
    batch_size=1024,
):
    """Run the thesis metrics for both neural architectures and selected priors."""
    priors = tuple(priors)
    unknown_priors = set(priors).difference(PRIOR_NAMES)
    if unknown_priors:
        raise ValueError(f"Unknown benchmark prior(s): {sorted(unknown_priors)}.")

    summary_frames = []
    detail_frames = []

    for prior in priors:
        prior_index = PRIOR_NAMES.index(prior)
        child_seed = np.random.SeedSequence([int(seed), prior_index])
        print(
            f"\nGenerating {benchmark_size} benchmark pairs under the {prior} prior."
        )
        y, _theta, targets = simulate_benchmark_dataset(
            prior=prior,
            benchmark_size=benchmark_size,
            n=n,
            seed=child_seed,
        )

        print(
            f"Running stochvol ({prior}) with {mcmc_draws:,} requested draws "
            f"per sequence."
        )
        mcmc_summary = run_stochvol_mcmc(
            y,
            prior=prior,
            draws=mcmc_draws,
            burnin=mcmc_burnin,
            thinpara=mcmc_thinpara,
            alpha=alpha,
            transforms=METRIC_MCMC_TRANSFORMS,
            max_cores=mcmc_max_cores,
            return_draws=False,
        )
        series_indices, mcmc_mean, mcmc_variance, retained_draws = (
            transformed_mcmc_moments(mcmc_summary)
        )
        if len(series_indices) != benchmark_size:
            raise RuntimeError(
                f"stochvol returned {len(series_indices)} series for "
                f"benchmark_size={benchmark_size}."
            )
        expected_indices = np.arange(1, benchmark_size + 1, dtype=np.int64)
        if not np.array_equal(series_indices, expected_indices):
            raise RuntimeError(
                "stochvol series indices do not match the simulated benchmark order."
            )

        for architecture in ("Summary NN", "TCN"):
            loaded_model = models[(architecture, prior)]
            print(f"Predicting benchmark moments with {loaded_model.label}.")
            npe_mean, npe_variance = predict_transformed_gaussian(
                loaded_model,
                y,
                batch_size=batch_size,
            )
            metric_summary, metric_details = calculate_paired_metrics(
                targets=targets,
                npe_mean=npe_mean,
                npe_variance=npe_variance,
                mcmc_mean=mcmc_mean,
                mcmc_variance=mcmc_variance,
                series_indices=series_indices,
                retained_draws=retained_draws,
                architecture=architecture,
                prior=prior,
            )
            summary_frames.append(metric_summary)
            detail_frames.append(metric_details)

    return (
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(detail_frames, ignore_index=True),
    )


def print_metric_table(metrics: pd.DataFrame) -> None:
    columns = [
        "architecture",
        "prior",
        "parameter",
        "gaussian_loss_difference",
        "npe_posterior_mean_rmse",
        "mcmc_posterior_mean_rmse",
        "average_log_variance_ratio",
    ]
    print("\nPaired transformed-scale benchmark metrics:")
    print(
        "  gaussian_loss_difference = mean(NPE loss - MCMC loss); "
        "average_log_variance_ratio = mean(log(v_NPE / v_MCMC))."
    )
    with pd.option_context("display.width", 180):
        print(
            metrics.loc[:, columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.6g}",
            )
        )


def apply_plot_style() -> None:
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


def plot_parameter_sweep_ci(comparison, output_path, alpha=DEFAULT_ALPHA) -> Path:
    apply_plot_style()
    fig, axes = plt.subplots(
        len(PARAMETER_NAMES),
        len(PRIOR_NAMES),
        figsize=(12.5, 13.0),
        sharex="row",
        sharey="row",
    )
    legend_entries = {}

    for row, swept_parameter in enumerate(PARAMETER_NAMES):
        parameter_label = PARAMETER_LABELS[swept_parameter]
        row_y_min = np.inf
        row_y_max = -np.inf

        for column, prior in enumerate(PRIOR_NAMES):
            ax = axes[row, column]
            subset = comparison[
                (comparison["swept_parameter"] == swept_parameter)
                & (comparison["prior"] == prior)
            ]
            true_values = (
                subset[["value_index", "true_value"]]
                .drop_duplicates()
                .sort_values("value_index")["true_value"]
                .to_numpy(dtype=np.float64)
            )
            if len(true_values) == 0:
                raise ValueError(
                    f"Comparison has no rows for {swept_parameter} under {prior}."
                )

            spacing = (
                float(np.min(np.abs(np.diff(true_values))))
                if len(true_values) > 1
                else 1.0
            )
            maximum_offset = 0.18 * spacing
            if swept_parameter == "sigma2":
                maximum_offset = min(
                    maximum_offset,
                    0.4 * float(np.min(true_values)),
                )
            offsets = dict(
                zip(
                    METHOD_NAMES,
                    np.linspace(-maximum_offset, maximum_offset, len(METHOD_NAMES)),
                )
            )
            ax.plot(
                true_values,
                true_values,
                color="0.35",
                linestyle="--",
                linewidth=1.0,
                label="_nolegend_",
                zorder=1,
            )

            y_min = float(np.min(true_values))
            y_max = float(np.max(true_values))
            for method in METHOD_NAMES:
                method_data = (
                    subset[subset["method"] == method]
                    .sort_values("value_index")
                    .reset_index(drop=True)
                )
                if len(method_data) != len(true_values):
                    raise ValueError(
                        f"Missing {method} ({prior}) rows for {swept_parameter}."
                    )

                x = method_data["true_value"].to_numpy() + offsets[method]
                median = method_data["median"].to_numpy()
                lower = method_data["ci_lower"].to_numpy()
                upper = method_data["ci_upper"].to_numpy()
                y_min = min(y_min, float(np.min(lower)))
                y_max = max(y_max, float(np.max(upper)))

                label = f"{method} ({prior})"
                handle = ax.errorbar(
                    x,
                    median,
                    yerr=np.vstack(
                        [
                            np.maximum(median - lower, 0.0),
                            np.maximum(upper - median, 0.0),
                        ]
                    ),
                    fmt=METHOD_MARKERS[method],
                    color=PRIOR_METHOD_COLORS[prior][method],
                    elinewidth=1.4,
                    capsize=3,
                    markersize=4.5,
                    label=label,
                    zorder=2,
                )
                legend_entries[label] = handle

            x_margin = max(0.55 * spacing, 1e-8)
            x_lower = true_values.min() - x_margin
            if swept_parameter == "sigma2":
                x_lower = max(0.0, x_lower)
            ax.set_xlim(x_lower, true_values.max() + x_margin)
            row_y_min = min(row_y_min, y_min)
            row_y_max = max(row_y_max, y_max)
            ax.set_title(
                f"{parameter_label} sweep — {prior.capitalize()} prior"
            )
            ax.set_xlabel(f"True {parameter_label}")
            ax.set_ylabel(f"Posterior {parameter_label}")
            ax.grid(alpha=0.25)

        row_y_margin = max(0.08 * (row_y_max - row_y_min), 1e-8)
        row_y_lower = row_y_min - row_y_margin
        if swept_parameter == "sigma2":
            row_y_lower = max(0.0, row_y_lower)
        axes[row, 0].set_ylim(
            row_y_lower,
            row_y_max + row_y_margin,
        )

    fig.suptitle(
        f"{100.0 * (1.0 - alpha):.0f}% credible intervals for the three-parameter SV model"
    )
    legend_labels = [
        f"{method} ({prior})"
        for method in METHOD_NAMES
        for prior in PRIOR_NAMES
    ]
    fig.legend(
        [legend_entries[label] for label in legend_labels],
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def validate_lightweight_inference(
    models: dict[tuple[str, str], LoadedModel],
    n: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    y = sim.simulate_sv_chunk(
        mu=np.array([DEFAULT_BASELINE["mu"]]),
        phi=np.array([DEFAULT_BASELINE["phi"]]),
        s=np.sqrt(np.array([DEFAULT_BASELINE["sigma2"]])),
        r=np.array([0.0]),
        nu=np.array([np.inf]),
        n=n,
        rng=rng,
        random_init=True,
    )
    for loaded_model in models.values():
        mean, variance = predict_transformed_gaussian(loaded_model, y, batch_size=1)
        if mean.shape != (1, 3) or variance.shape != (1, 3):
            raise RuntimeError(f"Lightweight validation failed for {loaded_model.label}.")
    print("Lightweight load + one-series inference validation passed for all four NNs.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        nargs="?",
        choices=("ci", "metrics", "validate"),
        default="ci",
        help="Run CI sweeps, the intensive metric benchmark, or lightweight validation.",
    )
    parser.add_argument(
        "--summary-default-checkpoint",
        default=DEFAULT_CHECKPOINTS[("Summary NN", "default")],
    )
    parser.add_argument(
        "--summary-finance-checkpoint",
        default=DEFAULT_CHECKPOINTS[("Summary NN", "finance")],
    )
    parser.add_argument(
        "--tcn-default-checkpoint",
        default=DEFAULT_CHECKPOINTS[("TCN", "default")],
    )
    parser.add_argument(
        "--tcn-finance-checkpoint",
        default=DEFAULT_CHECKPOINTS[("TCN", "finance")],
    )
    parser.add_argument("--output-dir", default="nn_model_tests")
    parser.add_argument("--sweep-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--metric-seed", type=int, default=3)
    parser.add_argument(
        "--benchmark-size",
        type=int,
        default=DEFAULT_BENCHMARK_SIZE,
        help="Number K of prior-predictive pairs per selected prior.",
    )
    parser.add_argument(
        "--benchmark-prior",
        choices=("both", *PRIOR_NAMES),
        default="both",
        help="Prior(s) used by the metric task (default: both).",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--mcmc-draws", type=int, default=DEFAULT_MCMC_DRAWS)
    parser.add_argument("--mcmc-burnin", type=int, default=DEFAULT_MCMC_BURNIN)
    parser.add_argument("--mcmc-thinpara", type=int, default=DEFAULT_MCMC_THINPARA)
    parser.add_argument("--mcmc-max-cores", type=int, default=DEFAULT_MCMC_MAX_CORES)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (default: auto-select CUDA, otherwise CPU).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load checkpoints, print counts, and run one NN inference; skip MCMC/sweeps.",
    )
    args = parser.parse_args(argv)

    if args.sweep_size < 1:
        parser.error("--sweep-size must be at least 1")
    if args.benchmark_size < 1:
        parser.error("--benchmark-size must be at least 1")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must lie in (0, 1)")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.mcmc_draws < 2:
        parser.error("--mcmc-draws must be at least 2")
    if args.mcmc_burnin < 0:
        parser.error("--mcmc-burnin must be non-negative")
    if args.mcmc_thinpara < 1:
        parser.error("--mcmc-thinpara must be at least 1")
    if args.mcmc_max_cores == 0:
        parser.error("--mcmc-max-cores must be non-zero")
    return args


def main0(args=None, models=None, counts=None, sequence_length=None) -> None:
    """Run and save the credible-interval sweep only.

    Calling ``main0()`` directly uses the normal command-line defaults.  The
    dispatcher also calls it with already-loaded objects to avoid loading the
    four checkpoints twice.
    """
    if args is None:
        return main(["ci"])

    comparison, sweeps, baseline = run_parameter_sweep_test(
        models=models,
        baseline=DEFAULT_BASELINE,
        sweep_bounds=DEFAULT_SWEEP_BOUNDS,
        n=sequence_length,
        sweep_size=args.sweep_size,
        seed=args.seed,
        alpha=args.alpha,
        mcmc_draws=args.mcmc_draws,
        mcmc_burnin=args.mcmc_burnin,
        mcmc_thinpara=args.mcmc_thinpara,
        mcmc_max_cores=args.mcmc_max_cores,
        batch_size=args.batch_size,
    )

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "three_parameter_sigma2_credible_intervals.csv"
    plot_path = output_dir / "three_parameter_credible_intervals.pdf"
    counts_path = output_dir / "neural_network_parameter_counts.csv"

    comparison.to_csv(comparison_path, index=False)
    counts.to_csv(counts_path, index=False)
    plot_parameter_sweep_ci(comparison, plot_path, args.alpha)

    print("\nBaseline:", baseline)
    print("Sweeps:", {name: values.tolist() for name, values in sweeps.items()})
    print(f"Saved credible-interval data to {comparison_path}")
    print(f"Saved credible-interval plot to {plot_path}")
    print(f"Saved parameter counts to {counts_path}")


def main1(args=None, models=None, counts=None, sequence_length=None) -> None:
    """Run and save the computationally intensive metric benchmark only.

    Calling ``main1()`` directly uses the normal command-line defaults.
    """
    if args is None:
        return main(["metrics"])

    priors = (
        PRIOR_NAMES
        if args.benchmark_prior == "both"
        else (args.benchmark_prior,)
    )
    metrics, per_sequence_metrics = run_metric_benchmark(
        models=models,
        priors=priors,
        benchmark_size=args.benchmark_size,
        n=sequence_length,
        seed=args.metric_seed,
        alpha=args.alpha,
        mcmc_draws=args.mcmc_draws,
        mcmc_burnin=args.mcmc_burnin,
        mcmc_thinpara=args.mcmc_thinpara,
        mcmc_max_cores=args.mcmc_max_cores,
        batch_size=args.batch_size,
    )

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_label = "both_priors" if len(priors) == 2 else f"{priors[0]}_prior"
    metrics_path = output_dir / (
        f"transformed_posterior_benchmark_metrics_{prior_label}.csv"
    )
    details_path = output_dir / (
        f"transformed_posterior_benchmark_per_sequence_{prior_label}.csv"
    )
    counts_path = output_dir / "neural_network_parameter_counts.csv"

    metrics.to_csv(metrics_path, index=False)
    per_sequence_metrics.to_csv(details_path, index=False)
    counts.to_csv(counts_path, index=False)

    print_metric_table(metrics)
    print(f"Saved benchmark metrics to {metrics_path}")
    print(f"Saved per-sequence benchmark details to {details_path}")
    print(f"Saved parameter counts to {counts_path}")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.validate_only:
        args.task = "validate"

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    checkpoint_paths = {
        ("Summary NN", "default"): args.summary_default_checkpoint,
        ("Summary NN", "finance"): args.summary_finance_checkpoint,
        ("TCN", "default"): args.tcn_default_checkpoint,
        ("TCN", "finance"): args.tcn_finance_checkpoint,
    }
    models = load_all_models(checkpoint_paths, device)
    print(f"Loaded four neural checkpoints on {device}.")
    counts = print_parameter_counts(models)
    sequence_length = common_sequence_length(models)

    if args.task == "validate":
        validate_lightweight_inference(models, sequence_length, args.seed)
    elif args.task == "ci":
        main0(args, models, counts, sequence_length)
    elif args.task == "metrics":
        main1(args, models, counts, sequence_length)
    else:
        raise RuntimeError(f"Unhandled task: {args.task}")


if __name__ == "__main__":
    main1()

