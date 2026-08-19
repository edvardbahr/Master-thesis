"""Load standard-SV neural estimators and use them for prediction."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from simulation import sim_5_param_data as sim
from training.sbt_summary_nn import SummaryNN
from training.sbt_tcn import TCN


Checkpoint = dict[str, object]


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Load one Summary NN or TCN checkpoint as an inference-ready model."""
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary at {checkpoint_path}.")

    model_class = checkpoint.get("model_class")
    activation_name = str(checkpoint["activation"])
    activation = getattr(nn, activation_name)

    if model_class == "SummaryNN":
        model: nn.Module = SummaryNN(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims_shared_trunk=tuple(checkpoint["hidden_dims_shared_trunk"]),
            hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
            activation=activation,
            min_var=float(checkpoint["min_var"]),
            layer_norm=bool(checkpoint["layer_norm"]),
        )
    elif model_class == "TCN":
        model = TCN(
            tcn_channels=tuple(checkpoint["tcn_channels"]),
            kernel_size=tuple(checkpoint["kernel_sizes"]),
            dilations=tuple(checkpoint["dilations"]),
            hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
            topk_pool_fraction=checkpoint["topk_pool_fraction"],
            activation=activation,
            param_names=tuple(checkpoint["target_names"]),
            min_var=float(checkpoint["min_var"]),
            input_mean=float(checkpoint["input_mean"]),
            input_std=float(checkpoint["input_std"]),
        )
    else:
        raise ValueError(
            f"Unsupported model_class {model_class!r} in {checkpoint_path}."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    model.checkpoint = checkpoint
    return model


def _checkpoint(model: nn.Module) -> Checkpoint:
    checkpoint = getattr(model, "checkpoint", None)
    if not isinstance(checkpoint, dict):
        raise ValueError("Use load_model() before calling a prediction function.")
    return checkpoint


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def prepare_summary_input(model: SummaryNN, y: np.ndarray) -> np.ndarray:
    checkpoint = _checkpoint(model)
    dataset_config = checkpoint.get("dataset_config", checkpoint)
    if not isinstance(dataset_config, dict):
        raise TypeError("Summary-NN dataset_config must be a dictionary.")

    input_dim = int(checkpoint["input_dim"])
    summaries = np.empty((len(y), input_dim), dtype=np.float32)
    for index, series in enumerate(y):
        summaries[index] = sim.summary_stats_sv(
            series,
            k=float(dataset_config["k"]),
            n_acvf_ratios=int(dataset_config["n_acvf_ratios"]),
            n_quantiles=int(dataset_config["n_quantiles"]),
            eps=float(dataset_config["eps"]),
            compute_arima_coeff=bool(dataset_config["compute_arima_coeff"]),
            center_y=bool(dataset_config.get("center_y", True)),
            remove_NaNs=bool(dataset_config["remove_NaNs"]),
        )

    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float32)
    z_std = np.asarray(checkpoint["z_std"], dtype=np.float32)
    return ((summaries - z_mean) / z_std).astype(np.float32)


def prepare_tcn_input(model: TCN, y: np.ndarray) -> np.ndarray:
    checkpoint = _checkpoint(model)
    centered_y = y - np.mean(y, axis=1, keepdims=True)
    return np.log(centered_y**2 + float(checkpoint["k"])).astype(np.float32)


def prepare_model_input(model: nn.Module, y: np.ndarray) -> np.ndarray:
    if isinstance(model, SummaryNN):
        return prepare_summary_input(model, y)
    if isinstance(model, TCN):
        return prepare_tcn_input(model, y)
    raise TypeError(f"Unsupported model type: {type(model).__name__}.")


@torch.inference_mode()
def predict(model: nn.Module, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict transformed posterior means and variances for a batch of series."""
    model_input = torch.from_numpy(prepare_model_input(model, y)).to(
        _model_device(model)
    )
    mean, variance = model(model_input)
    return (
        mean.cpu().numpy().astype(np.float64),
        variance.cpu().numpy().astype(np.float64),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def predict_with_runtimes(
    model: nn.Module,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict one series at a time and include preprocessing in each runtime."""
    checkpoint = _checkpoint(model)
    device = _model_device(model)
    if isinstance(model, SummaryNN):
        dummy_shape = (1, int(checkpoint["input_dim"]))
    elif isinstance(model, TCN):
        dummy_shape = (1, y.shape[1])
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}.")

    dummy = torch.zeros(dummy_shape, dtype=torch.float32, device=device)
    dummy_mean, _ = model(dummy)
    _synchronize(device)

    output_dim = dummy_mean.shape[1]
    means = np.empty((len(y), output_dim), dtype=np.float64)
    variances = np.empty((len(y), output_dim), dtype=np.float64)
    runtimes = np.empty(len(y), dtype=np.float64)

    for index, series in enumerate(y):
        _synchronize(device)
        started_at = perf_counter()

        model_input = prepare_model_input(model, series[None, :])
        model_input_tensor = torch.from_numpy(model_input).to(device)
        mean, variance = model(model_input_tensor)
        means[index] = mean[0].cpu().numpy()
        variances[index] = variance[0].cpu().numpy()

        _synchronize(device)
        runtimes[index] = perf_counter() - started_at

    return means, variances, runtimes
