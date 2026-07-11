import inspect
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


import sim_3_param_data as sim
from train_summary_NN import (
    SVPosteriorNN,
    diagonal_gaussian_nll,
    theta_to_target_numpy,
)

TARGET_NAMES = ("mu", "psi", "log_sigma")
TARGET_TRANSFORMS = {
    "mu": "mu",
    "psi": "2 * atanh(phi)",
    "log_sigma": "log(sigma)",
}
LOSS_REDUCTION = "sum_over_parameters"

CHUNKS_PER_WORKER = 4
KAPPA = 1e-12
SUMMARY_EPS = 1e-12
CENTER_Y = True
REMOVE_NANS = True
EXP_CLIP = 350.0

TRAIN_SEED_STREAM = 101
VALIDATION_SEED_STREAM = 202
FINAL_VALIDATION_SEED_STREAM = 303


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_child_seed(seed, stream, index):
    """Derive reproducible, independent seeds for live data streams."""
    seed_sequence = np.random.SeedSequence([int(seed), int(stream), int(index)])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def default_checkpoint_paths(checkpoint_path):
    base, extension = os.path.splitext(checkpoint_path)
    if extension == "":
        extension = ".pt"
    return f"{base}.latest{extension}", f"{base}.best{extension}"


def torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def state_dict_to_cpu(state_dict):
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def save_checkpoint_atomic(checkpoint, path):
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    temporary_path = f"{path}.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def simulate_live_summary_dataset(
    N,
    sequence_length,
    chunk_size,
    n_workers,
    seed,
    prior,
    n_acvf_ratios,
    compute_arima_coeff,
    out_dtype,
):
    """Simulate summaries and transformed standard-SV targets."""
    summaries, theta, feature_names = sim.simulate_sv_summaries_parallel(
        N=N,
        n=sequence_length,
        chunk_size=chunk_size,
        n_workers=n_workers,
        seed=seed,
        prior=prior,
        random_init=True,
        n_acvf_ratios=n_acvf_ratios,
        compute_arima_coeff=compute_arima_coeff,
        k=KAPPA,
        eps=SUMMARY_EPS,
        arima_method=None,
        center_y=CENTER_Y,
        remove_NaNs=REMOVE_NANS,
        out_dtype=out_dtype,
        exp_clip=EXP_CLIP,
        show_progress=False,
    )

    targets = theta_to_target_numpy(theta).astype(np.float32, copy=False)
    summaries = summaries.astype(np.float32, copy=False)

    if not np.all(np.isfinite(summaries)):
        raise FloatingPointError("Simulated summary statistics contain NaN or Inf values.")
    if not np.all(np.isfinite(targets)):
        raise FloatingPointError("Simulated transformed targets contain NaN or Inf values.")

    return summaries, targets, feature_names


def train_live_summary_nn(
    sequence_length,
    prior="default",
    hidden_dims_shared_trunk=(128, 64),
    hidden_dims_head=(32, 32),
    activation=nn.ReLU,
    checkpoint_path="sv_posterior_summary_nn_live.pt",
    resume_from=None,
    seed=1,
    batch_size=1024,
    n_batches=10,
    val_size=20_000,
    fixed_validation=True,
    lr=5e-4,
    n_epochs=1000,
    patience=100,
    min_delta=1e-5,
    min_var=1e-12,
    dropout=0.0,
    layer_norm=True,
    grad_clip_norm=None,
    deterministic_torch=True,
    n_acvf_ratios=4,
    compute_arima_coeff=True,
    n_workers=-2,
    out_dtype=np.float32,
    verbose=True,
):
    """
    Train ``SVPosteriorNN`` on newly simulated summaries every epoch.

    One epoch generates ``batch_size * n_batches`` training examples, performs
    exactly ``n_batches`` updates, and then evaluates one validation set. With
    ``fixed_validation=True`` the same validation sample is reused throughout;
    otherwise a deterministic fresh sample is generated each epoch. The longer
    default patience is intended for the noisier changing-validation case.

    Summary standardization is estimated once from the first live training set
    and stored in every checkpoint. Resume runs reuse the stored values.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if n_batches < 1:
        raise ValueError("n_batches must be at least 1.")
    if val_size < 1:
        raise ValueError("val_size must be at least 1.")
    if n_epochs < 1:
        raise ValueError("n_epochs must be at least 1.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative.")
    if min_var <= 0:
        raise ValueError("min_var must be positive.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive or None.")
    if not isinstance(n_acvf_ratios, int) or n_acvf_ratios < 1:
        raise ValueError("n_acvf_ratios must be a positive integer.")

    hidden_dims_shared_trunk = tuple(int(value) for value in hidden_dims_shared_trunk)
    hidden_dims_head = tuple(int(value) for value in hidden_dims_head)
    if any(value < 1 for value in hidden_dims_shared_trunk + hidden_dims_head):
        raise ValueError("All hidden dimensions must be positive integers.")

    # Validate the prior before starting expensive simulation work.
    sim.get_stochvol_prior_constants(prior)

    checkpoint_path = os.fspath(checkpoint_path)
    resume_from = None if resume_from is None else os.fspath(resume_from)
    latest_checkpoint_path, best_checkpoint_path = default_checkpoint_paths(checkpoint_path)

    if resume_from is not None and not os.path.isfile(resume_from):
        raise FileNotFoundError(
            "Cannot resume training because the checkpoint does not exist: "
            f"{resume_from}"
        )

    out_dtype = np.dtype(out_dtype).type
    resolved_n_workers = sim.resolve_n_workers(n_workers)
    train_size = batch_size * n_batches
    train_chunk_size = sim.resolve_chunk_size(
        train_size,
        resolved_n_workers,
        CHUNKS_PER_WORKER,
    )
    val_chunk_size = sim.resolve_chunk_size(
        val_size,
        resolved_n_workers,
        CHUNKS_PER_WORKER,
    )
    effective_val_batch_size = min(batch_size, val_size)

    feature_names = sim.summary_stats_sv_feature_names(
        n_acvf_ratios=n_acvf_ratios,
        compute_arima_coeff=compute_arima_coeff,
    )
    input_dim = len(feature_names)
    activation_name = getattr(activation, "__name__", str(activation))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except TypeError:
            torch.use_deterministic_algorithms(True, warn_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SVPosteriorNN(
        input_dim=input_dim,
        hidden_dims_shared_trunk=hidden_dims_shared_trunk,
        hidden_dims_head=hidden_dims_head,
        activation=activation,
        min_var=min_var,
        dropout=dropout,
        layer_norm=layer_norm,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print("Using device:", device)
        print("Prior:", prior)
        print("Sequence length:", sequence_length)
        print("Summary features:", input_dim)
        print("Trainable parameters:", count_parameters(model))
        print("Train samples per validation:", train_size)
        print("Train batches per validation:", n_batches)
        print("Validation size:", val_size)
        print("Fixed validation:", fixed_validation)
        print("Resolved simulation workers:", resolved_n_workers)

    z_mean = None
    z_std = None
    z_mean_tensor = None
    z_std_tensor = None

    def set_standardization(mean, std):
        nonlocal z_mean, z_std, z_mean_tensor, z_std_tensor

        z_mean = np.asarray(mean, dtype=np.float32).reshape(1, input_dim)
        z_std = np.asarray(std, dtype=np.float32).reshape(1, input_dim)
        z_std = np.where(z_std < 1e-8, 1.0, z_std).astype(np.float32, copy=False)
        z_mean_tensor = torch.from_numpy(z_mean).to(device)
        z_std_tensor = torch.from_numpy(z_std).to(device)

    @torch.no_grad()
    def evaluate_array(summaries, targets):
        if z_mean_tensor is None or z_std_tensor is None:
            raise RuntimeError("Summary standardization has not been initialized.")

        model.eval()
        total_losses = None
        total_n = 0

        for start in range(0, len(summaries), effective_val_batch_size):
            stop = min(start + effective_val_batch_size, len(summaries))
            summary_batch = torch.from_numpy(summaries[start:stop]).to(device)
            target_batch = torch.from_numpy(targets[start:stop]).to(device)
            summary_batch = (summary_batch - z_mean_tensor) / z_std_tensor

            mean, var = model(summary_batch)
            losses = diagonal_gaussian_nll(mean, var, target_batch)
            batch_n = summary_batch.shape[0]

            if total_losses is None:
                total_losses = torch.zeros_like(losses)

            total_losses += losses * batch_n
            total_n += batch_n

        return total_losses / total_n

    def has_nonfinite_gradient():
        return any(
            parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )

    train_marginal_loss_history = []
    val_marginal_loss_history = []
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    best_validation_seed = None
    epochs_without_improvement = 0
    start_epoch = 0
    completed_epoch = 0

    def make_checkpoint(
        epoch_completed,
        checkpoint_kind,
        final_val_loss=None,
        final_val_marginal_losses=None,
        final_validation_seed=None,
        include_optimizer=True,
    ):
        if z_mean is None or z_std is None:
            raise RuntimeError("Cannot checkpoint before standardization is initialized.")

        final_marginals = None
        if final_val_marginal_losses is not None:
            final_marginals = np.asarray(final_val_marginal_losses, dtype=np.float32)

        return {
            "checkpoint_kind": checkpoint_kind,
            "epoch": epoch_completed,
            "model_class": "SVPosteriorNN",
            "model_state_dict": state_dict_to_cpu(model.state_dict()),
            "best_model_state_dict": (
                None if best_state is None else state_dict_to_cpu(best_state)
            ),
            "optimizer_state_dict": optimizer.state_dict() if include_optimizer else None,

            "input_dim": input_dim,
            "hidden_dims_shared_trunk": hidden_dims_shared_trunk,
            "hidden_dims_head": hidden_dims_head,
            "activation": activation_name,
            "min_var": min_var,
            "dropout": float(dropout),
            "layer_norm": bool(layer_norm),

            "z_mean": z_mean.astype(np.float32),
            "z_std": z_std.astype(np.float32),
            "standardization_source": "first live training set",

            "feature_names": feature_names,
            "n_acvf_ratios": n_acvf_ratios,
            "compute_arima_coeff": bool(compute_arima_coeff),
            "k": KAPPA,
            "eps": SUMMARY_EPS,
            "center_y": CENTER_Y,
            "remove_NaNs": REMOVE_NANS,
            "dataset_config": {
                "sequence_length": sequence_length,
                "prior": prior,
                "random_init": True,
                "n_acvf_ratios": n_acvf_ratios,
                "compute_arima_coeff": bool(compute_arima_coeff),
                "k": KAPPA,
                "eps": SUMMARY_EPS,
                "center_y": CENTER_Y,
                "remove_NaNs": REMOVE_NANS,
            },

            "target_names": list(TARGET_NAMES),
            "target_transform": TARGET_TRANSFORMS,
            "loss": "mean negative joint Gaussian log score, diagonal covariance",
            "loss_components": "mean marginal Gaussian negative log scores",
            "loss_reduction": LOSS_REDUCTION,

            "best_val_loss": float(best_val_loss),
            "final_val_loss": None if final_val_loss is None else float(final_val_loss),
            "final_val_marginal_losses": final_marginals,
            "best_epoch": best_epoch,
            "best_validation_seed": best_validation_seed,
            "final_validation_seed": final_validation_seed,
            "epochs_without_improvement": epochs_without_improvement,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "train_marginal_loss_history": train_marginal_loss_history,
            "val_marginal_loss_history": val_marginal_loss_history,

            "sequence_length": sequence_length,
            "prior": prior,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "train_size_per_validation": train_size,
            "val_size": val_size,
            "fixed_validation": bool(fixed_validation),
            "effective_val_batch_size": effective_val_batch_size,
            "requested_n_workers": n_workers,
            "resolved_n_workers": resolved_n_workers,
            "chunks_per_worker": CHUNKS_PER_WORKER,
            "train_chunk_size": train_chunk_size,
            "val_chunk_size": val_chunk_size,
            "out_dtype": str(np.dtype(out_dtype)),
            "deterministic_torch": bool(deterministic_torch),
            "lr": lr,
            "n_epochs": n_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "grad_clip_norm": grad_clip_norm,
            "seed": seed,
            "seed_derivation": "SeedSequence([seed, stream, epoch_index])",
            "seed_streams": {
                "train": TRAIN_SEED_STREAM,
                "validation": VALIDATION_SEED_STREAM,
                "final_validation": FINAL_VALIDATION_SEED_STREAM,
            },
            "trainable_parameters": count_parameters(model),
            "latest_checkpoint_path": latest_checkpoint_path,
            "best_checkpoint_path": best_checkpoint_path,
            "resume_from": resume_from,
        }

    if resume_from is not None:
        resume_checkpoint = torch_load_checkpoint(resume_from, map_location=device)

        expected_configuration = {
            "model_class": "SVPosteriorNN",
            "input_dim": input_dim,
            "hidden_dims_shared_trunk": hidden_dims_shared_trunk,
            "hidden_dims_head": hidden_dims_head,
            "activation": activation_name,
            "dropout": float(dropout),
            "layer_norm": bool(layer_norm),
            "sequence_length": sequence_length,
            "prior": prior,
            "n_acvf_ratios": n_acvf_ratios,
            "compute_arima_coeff": bool(compute_arima_coeff),
            "fixed_validation": bool(fixed_validation),
            "seed": seed,
            "loss_reduction": LOSS_REDUCTION,
        }
        mismatches = {
            key: (resume_checkpoint.get(key), expected)
            for key, expected in expected_configuration.items()
            if resume_checkpoint.get(key) != expected
        }
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={actual!r}, requested={expected!r}"
                for key, (actual, expected) in mismatches.items()
            )
            raise ValueError(f"Cannot resume with a different configuration ({details}).")

        if list(resume_checkpoint.get("feature_names", [])) != feature_names:
            raise ValueError("Cannot resume because the summary feature names differ.")

        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = lr

        set_standardization(resume_checkpoint["z_mean"], resume_checkpoint["z_std"])

        train_marginal_loss_history = resume_checkpoint.get(
            "train_marginal_loss_history", []
        )
        val_marginal_loss_history = resume_checkpoint.get(
            "val_marginal_loss_history", []
        )
        train_loss_history = resume_checkpoint.get("train_loss_history", [])
        val_loss_history = resume_checkpoint.get("val_loss_history", [])
        best_val_loss = float(resume_checkpoint.get("best_val_loss", float("inf")))
        best_epoch = resume_checkpoint.get("best_epoch")
        best_validation_seed = resume_checkpoint.get("best_validation_seed")
        epochs_without_improvement = int(
            resume_checkpoint.get("epochs_without_improvement", 0)
        )
        start_epoch = int(resume_checkpoint.get("epoch", len(train_loss_history)))
        completed_epoch = start_epoch

        history_lengths = {
            "train_loss_history": len(train_loss_history),
            "val_loss_history": len(val_loss_history),
            "train_marginal_loss_history": len(train_marginal_loss_history),
            "val_marginal_loss_history": len(val_marginal_loss_history),
        }
        if any(length != start_epoch for length in history_lengths.values()):
            raise ValueError(
                "Checkpoint epoch does not match its loss-history lengths: "
                f"epoch={start_epoch}, lengths={history_lengths}."
            )

        checkpoint_best_state = resume_checkpoint.get("best_model_state_dict")
        if checkpoint_best_state is not None:
            best_state = state_dict_to_cpu(checkpoint_best_state)
        elif np.isfinite(best_val_loss):
            best_state = state_dict_to_cpu(model.state_dict())

        if verbose:
            print(f"Resumed training from {resume_from}")
            print(f"Starting at epoch {start_epoch + 1}")
            print(f"Using learning rate: {lr:g}")

    fixed_val_summaries = None
    fixed_val_targets = None
    fixed_validation_seed = None

    if fixed_validation:
        fixed_validation_seed = make_child_seed(seed, VALIDATION_SEED_STREAM, 0)
        if verbose:
            print("Generating fixed validation set...")

        fixed_val_summaries, fixed_val_targets, generated_feature_names = (
            simulate_live_summary_dataset(
                N=val_size,
                sequence_length=sequence_length,
                chunk_size=val_chunk_size,
                n_workers=resolved_n_workers,
                seed=fixed_validation_seed,
                prior=prior,
                n_acvf_ratios=n_acvf_ratios,
                compute_arima_coeff=compute_arima_coeff,
                out_dtype=out_dtype,
            )
        )
        if generated_feature_names != feature_names:
            raise RuntimeError("Simulator returned unexpected summary feature names.")

    for epoch in range(start_epoch, n_epochs):
        train_seed = make_child_seed(seed, TRAIN_SEED_STREAM, epoch + 1)
        if verbose and epoch == start_epoch:
            print("Generating live training data...")

        train_summaries, train_targets, generated_feature_names = (
            simulate_live_summary_dataset(
                N=train_size,
                sequence_length=sequence_length,
                chunk_size=train_chunk_size,
                n_workers=resolved_n_workers,
                seed=train_seed,
                prior=prior,
                n_acvf_ratios=n_acvf_ratios,
                compute_arima_coeff=compute_arima_coeff,
                out_dtype=out_dtype,
            )
        )
        if generated_feature_names != feature_names:
            raise RuntimeError("Simulator returned unexpected summary feature names.")

        if z_mean is None:
            set_standardization(
                train_summaries.mean(axis=0, keepdims=True),
                train_summaries.std(axis=0, keepdims=True),
            )
            if verbose:
                print("Initialized summary standardization from the first training set.")

        model.train()
        total_train_losses = None
        total_train_n = 0

        for batch_index in range(n_batches):
            start = batch_index * batch_size
            stop = start + batch_size
            summary_batch = torch.from_numpy(train_summaries[start:stop]).to(device)
            target_batch = torch.from_numpy(train_targets[start:stop]).to(device)
            summary_batch = (summary_batch - z_mean_tensor) / z_std_tensor

            optimizer.zero_grad(set_to_none=True)
            mean, var = model(summary_batch)
            marginal_losses = diagonal_gaussian_nll(mean, var, target_batch)
            loss = marginal_losses.sum()
            loss.backward()

            nonfinite_gradient = has_nonfinite_gradient()
            if nonfinite_gradient:
                if verbose:
                    print(
                        "Warning: NaN/Inf gradients detected at "
                        f"epoch {epoch + 1}, batch {batch_index + 1}; "
                        "skipping the optimizer step."
                    )
            else:
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            batch_n = summary_batch.shape[0]
            if total_train_losses is None:
                total_train_losses = torch.zeros_like(marginal_losses)
            total_train_losses += marginal_losses.detach() * batch_n
            total_train_n += batch_n

        del train_summaries
        del train_targets

        train_marginal_losses = total_train_losses / total_train_n

        if fixed_validation:
            val_summaries = fixed_val_summaries
            val_targets = fixed_val_targets
            validation_seed = fixed_validation_seed
        else:
            validation_seed = make_child_seed(seed, VALIDATION_SEED_STREAM, epoch + 1)
            val_summaries, val_targets, generated_feature_names = (
                simulate_live_summary_dataset(
                    N=val_size,
                    sequence_length=sequence_length,
                    chunk_size=val_chunk_size,
                    n_workers=resolved_n_workers,
                    seed=validation_seed,
                    prior=prior,
                    n_acvf_ratios=n_acvf_ratios,
                    compute_arima_coeff=compute_arima_coeff,
                    out_dtype=out_dtype,
                )
            )
            if generated_feature_names != feature_names:
                raise RuntimeError("Simulator returned unexpected summary feature names.")

        val_marginal_losses = evaluate_array(val_summaries, val_targets)
        if not fixed_validation:
            del val_summaries
            del val_targets

        train_marginals_np = train_marginal_losses.cpu().numpy()
        val_marginals_np = val_marginal_losses.cpu().numpy()
        train_loss_value = float(train_marginals_np.sum())
        val_loss_value = float(val_marginals_np.sum())

        train_loss_history.append(train_loss_value)
        val_loss_history.append(val_loss_value)
        train_marginal_loss_history.append(train_marginals_np.tolist())
        val_marginal_loss_history.append(val_marginals_np.tolist())

        improved = val_loss_value < best_val_loss - min_delta
        if improved:
            best_val_loss = val_loss_value
            best_state = state_dict_to_cpu(model.state_dict())
            best_epoch = epoch + 1
            best_validation_seed = validation_seed
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and ((epoch + 1) % 10 == 0 or epoch == start_epoch):
            train_parts = ", ".join(
                f"{name}={value:.4f}"
                for name, value in zip(TARGET_NAMES, train_marginals_np)
            )
            val_parts = ", ".join(
                f"{name}={value:.4f}"
                for name, value in zip(TARGET_NAMES, val_marginals_np)
            )
            print(
                f"Epoch {epoch + 1:4d}: train NLL = {train_loss_value:.4f}, "
                f"val NLL = {val_loss_value:.4f}"
            )
            print(f"             train marginal NLLs: {train_parts}")
            print(f"             val marginal NLLs:   {val_parts}")

        completed_epoch = epoch + 1
        save_checkpoint_atomic(
            make_checkpoint(epoch + 1, checkpoint_kind="latest"),
            latest_checkpoint_path,
        )

        if improved:
            save_checkpoint_atomic(
                make_checkpoint(epoch + 1, checkpoint_kind="best"),
                best_checkpoint_path,
            )
            if verbose:
                print(
                    f"New best model at epoch {epoch + 1}: "
                    f"validation NLL {val_loss_value:.6f}"
                )
                print(f"Best checkpoint saved to {best_checkpoint_path}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        raise RuntimeError("Training completed without a finite best model state.")
    model.eval()

    if fixed_validation:
        final_val_summaries = fixed_val_summaries
        final_val_targets = fixed_val_targets
        final_validation_seed = fixed_validation_seed
    else:
        final_validation_seed = make_child_seed(
            seed,
            FINAL_VALIDATION_SEED_STREAM,
            completed_epoch + 1,
        )
        final_val_summaries, final_val_targets, generated_feature_names = (
            simulate_live_summary_dataset(
                N=val_size,
                sequence_length=sequence_length,
                chunk_size=val_chunk_size,
                n_workers=resolved_n_workers,
                seed=final_validation_seed,
                prior=prior,
                n_acvf_ratios=n_acvf_ratios,
                compute_arima_coeff=compute_arima_coeff,
                out_dtype=out_dtype,
            )
        )
        if generated_feature_names != feature_names:
            raise RuntimeError("Simulator returned unexpected summary feature names.")

    final_val_marginal_losses = evaluate_array(final_val_summaries, final_val_targets)
    final_val_marginals_np = final_val_marginal_losses.cpu().numpy()
    final_val_loss = float(final_val_marginals_np.sum())

    if verbose:
        print()
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation mean joint NLL: {best_val_loss:.6f}")
        print(f"Final validation mean joint NLL: {final_val_loss:.6f}")
        print(
            "Final validation marginal NLLs:",
            {
                name: float(value)
                for name, value in zip(TARGET_NAMES, final_val_marginals_np)
            },
        )

    checkpoint = make_checkpoint(
        epoch_completed=completed_epoch,
        checkpoint_kind="final",
        final_val_loss=final_val_loss,
        final_val_marginal_losses=final_val_marginals_np,
        final_validation_seed=final_validation_seed,
        include_optimizer=False,
    )
    save_checkpoint_atomic(checkpoint, checkpoint_path)

    if verbose:
        print(f"Model saved to {checkpoint_path}")

    return model, checkpoint


def _json_safe(value):
    if isinstance(value, type) and issubclass(value, nn.Module):
        return value.__name__
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _filename_safe(value, max_length=60):
    text = str(_json_safe(value))
    replacements = {
        " ": "",
        "'": "",
        '"': "",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        ":": "-",
        ",": "-",
        "/": "-",
        "\\": "-",
        ".": "p",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    safe_text = "".join(
        character if character.isalnum() or character in "-_=" else "-"
        for character in text
    ).strip("-_")
    return safe_text[:max_length]


def make_unique_search_output_dir(
    base_dir="summary_nn_search",
    run_label="live_summary_nn",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = _filename_safe(run_label)
    directory_name = f"{safe_label}_{timestamp}" if safe_label else timestamp
    output_dir = Path(base_dir) / directory_name

    counter = 2
    while output_dir.exists():
        output_dir = Path(base_dir) / f"{directory_name}_{counter:02d}"
        counter += 1

    return output_dir


def trial_checkpoint_stem(trial_index, trial_parameters):
    parts = [f"trial_{trial_index:03d}"]
    for name, value in trial_parameters.items():
        parts.append(f"{_filename_safe(name, 24)}-{_filename_safe(value, 48)}")
    return "_".join(part for part in parts if part)


def checkpoint_family_exists(checkpoint_path):
    latest_checkpoint_path, best_checkpoint_path = default_checkpoint_paths(
        os.fspath(checkpoint_path)
    )
    return any(
        os.path.exists(path)
        for path in (checkpoint_path, latest_checkpoint_path, best_checkpoint_path)
    )


def unique_checkpoint_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == "":
        checkpoint_path = checkpoint_path.with_suffix(".pt")

    candidate_path = checkpoint_path
    counter = 2
    while checkpoint_family_exists(candidate_path):
        candidate_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_{counter:02d}{checkpoint_path.suffix}"
        )
        counter += 1

    return candidate_path


def run_hyperparameter_search(
    search_space,
    common_config,
    output_dir="summary_nn_search",
    max_trials=None,
    fail_fast=False,
):
    """
    Run a reproducible Cartesian grid search and save progress after each trial.

    ``search_space`` maps argument names to candidate lists. ``common_config``
    contains shared ``train_live_summary_nn`` arguments. Keep one seed and use
    fixed validation when comparing trials; evaluate the eventual winner on a
    separate held-out simulation before reporting its final performance.
    """
    if not search_space:
        raise ValueError("search_space must contain at least one parameter.")

    valid_parameters = set(inspect.signature(train_live_summary_nn).parameters)
    unknown_parameters = (set(search_space) | set(common_config)) - valid_parameters
    if unknown_parameters:
        raise ValueError(
            "Unknown train_live_summary_nn argument(s): "
            f"{sorted(unknown_parameters)}"
        )

    if "checkpoint_path" in search_space or "resume_from" in search_space:
        raise ValueError("checkpoint_path and resume_from cannot be search dimensions.")
    if common_config.get("resume_from") is not None:
        raise ValueError("Hyperparameter-search trials cannot resume a training run.")
    if max_trials is not None and max_trials < 1:
        raise ValueError("max_trials must be positive or None.")

    parameter_names = list(search_space)
    candidate_lists = []
    for name in parameter_names:
        candidates = list(search_space[name])
        if not candidates:
            raise ValueError(f"search_space[{name!r}] must not be empty.")
        candidate_lists.append(candidates)

    combinations = itertools.product(*candidate_lists)
    if max_trials is not None:
        combinations = itertools.islice(combinations, max_trials)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "search_results.json"
    config_path = output_dir / "search_config.json"
    config_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "search_space": search_space,
                    "common_config": common_config,
                    "max_trials": max_trials,
                    "fail_fast": fail_fast,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    results = []

    for trial_index, values in enumerate(combinations, start=1):
        trial_parameters = dict(zip(parameter_names, values))
        trial_config = dict(common_config)
        trial_config.update(trial_parameters)
        checkpoint_stem = trial_checkpoint_stem(trial_index, trial_parameters)
        trial_config["checkpoint_path"] = str(
            unique_checkpoint_path(output_dir / f"{checkpoint_stem}.pt")
        )
        trial_config["resume_from"] = None

        if trial_config.get("verbose", True):
            print()
            print(f"Starting hyperparameter trial {trial_index}")
            print("Parameters:", _json_safe(trial_parameters))

        record = {
            "trial": trial_index,
            "status": "running",
            "parameters": _json_safe(trial_parameters),
            "checkpoint_path": trial_config["checkpoint_path"],
        }
        results.append(record)

        try:
            _, checkpoint = train_live_summary_nn(**trial_config)
            record.update(
                status="completed",
                best_val_loss=float(checkpoint["best_val_loss"]),
                final_val_loss=float(checkpoint["final_val_loss"]),
                best_epoch=checkpoint["best_epoch"],
                trainable_parameters=checkpoint["trainable_parameters"],
            )
        except Exception as error:
            record.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
            if fail_fast:
                results_path.write_text(
                    json.dumps(_json_safe(results), indent=2),
                    encoding="utf-8",
                )
                raise

        results_path.write_text(
            json.dumps(_json_safe(results), indent=2),
            encoding="utf-8",
        )

    completed_results = [
        record for record in results if record["status"] == "completed"
    ]
    completed_results.sort(key=lambda record: record["final_val_loss"])

    if completed_results:
        print()
        print("Best completed trial:", completed_results[0])
    else:
        print("No hyperparameter-search trial completed successfully.")

    return completed_results


def main():
    common_config = {
        "sequence_length": 253,
        "prior": "default",
        "seed": 2,
        "batch_size": 1024,
        "n_batches": 10,
        "val_size": 20_000,
        "fixed_validation": True,
        "n_epochs": 2000,
        "patience": 75,
        "min_delta": 1e-5,
        "layer_norm": True,
        "n_workers": -4,
        "verbose": True,
    }

    search_space = {
        "lr": [3e-4, 5e-4, 8e-4],
        "hidden_dims_shared_trunk": [(128, 64), (256, 128)],
        "hidden_dims_head": [(32, 32), (64, 32)],
        "dropout": [0.0, 0.05],
    }

    output_dir = make_unique_search_output_dir(
        base_dir=Path(__file__).resolve().parent / "summary_nn_search",
        run_label="three_param_default",
    )
    print(f"Saving hyperparameter-search outputs to {output_dir}")

    run_hyperparameter_search(
        search_space=search_space,
        common_config=common_config,
        output_dir=output_dir,
        fail_fast=False,
    )


if __name__ == "__main__":
    main()
