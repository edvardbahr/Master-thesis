"""Load the five-parameter SV-GHST TCN and use it for prediction."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.test_nn_model import (
    load_model as load_neural_model,
    predict as predict_batch,
)
from simulation.stochvol_mcmc import validate_series_matrix
from training.sbt_tcn import SVGHST_TARGET_NAMES, TCN


PREDICTION_BATCH_SIZE = 128


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    """Load an inference-ready five-parameter SV-GHST TCN."""
    model, checkpoint = load_neural_model(checkpoint_path, device)

    if not isinstance(model, TCN):
        raise TypeError(
            f"Expected a TCN checkpoint, got {type(model).__name__}."
        )
    if tuple(checkpoint["target_names"]) != SVGHST_TARGET_NAMES:
        raise ValueError(
            "Expected a checkpoint with the five SV-GHST targets "
            f"{SVGHST_TARGET_NAMES}."
        )

    return model, checkpoint


def predict(
    model: nn.Module,
    checkpoint: dict[str, object],
    y: np.ndarray,
    batch_size: int = PREDICTION_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict transformed posterior means and variances in manageable batches."""
    series = validate_series_matrix(y)
    sequence_length = int(checkpoint["sequence_length"])
    if series.shape[1] != sequence_length:
        raise ValueError(
            f"Expected sequences of length {sequence_length}, "
            f"got {series.shape[1]}."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    target_names = tuple(checkpoint["target_names"])
    output_shape = (len(series), len(target_names))
    means = np.empty(output_shape, dtype=np.float64)
    variances = np.empty(output_shape, dtype=np.float64)

    for start in range(0, len(series), batch_size):
        stop = min(start + batch_size, len(series))
        means[start:stop], variances[start:stop] = predict_batch(
            model,
            checkpoint,
            series[start:stop],
        )

    return means, variances
