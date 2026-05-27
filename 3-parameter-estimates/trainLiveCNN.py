import copy
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F

import simulateData as sim
from trainCNN import (
    SVPosteriorTCN,
    count_parameters,
    diagonal_gaussian_nll,
    temporal_receptive_field,
    theta_to_target_numpy,
)


K_MOMENT_WARNING_THRESHOLD = 1e-10

TRAIN_SEED_STREAM = 101
VALIDATION_SEED_STREAM = 202
FINAL_VALIDATION_SEED_STREAM = 303


# ============================================================
# Live simulation helpers
# ============================================================

class LiveSVPosteriorTCN(SVPosteriorTCN):
    """
    State-dict compatible TCN using deterministic pooling reductions.

    The base SVPosteriorTCN uses adaptive max pooling. On CUDA, PyTorch can
    mark its backward pass as nondeterministic. These reductions preserve the
    same representation shape without introducing extra parameters.
    """

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError("x must have shape (batch_size, n) or (batch_size, 1, n).")

        if x.shape[1] != 1:
            raise ValueError("x must have exactly one input channel.")

        x = (x - self.input_mean) / self.input_std.clamp_min(1e-8)

        h = self.encoder(x)

        h_avg = h.mean(dim=-1)
        h_max = h.amax(dim=-1)
        h = torch.cat([h_avg, h_max], dim=1)

        means = []
        variances = []

        for name in self.param_names:
            out = self.heads[name](h)

            mean = out[:, 0:1]
            raw_var = out[:, 1:2]
            var = F.softplus(raw_var) + self.min_var

            means.append(mean)
            variances.append(var)

        mean = torch.cat(means, dim=1)
        var = torch.cat(variances, dim=1)

        return mean, var

def make_child_seed(seed, stream, index):
    """
    Derive a deterministic 32-bit seed from a master seed and stream id.

    This keeps training and validation simulations reproducible even when the
    work is split across multiple processes.
    """
    seed_sequence = np.random.SeedSequence([int(seed), int(stream), int(index)])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def default_checkpoint_paths(checkpoint_path):
    base, ext = os.path.splitext(checkpoint_path)

    if ext == "":
        ext = ".pt"

    return f"{base}.latest{ext}", f"{base}.best{ext}"


def torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint_atomic(checkpoint, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def state_dict_to_cpu(state_dict):
    return {
        key: value.detach().cpu()
        for key, value in state_dict.items()
    }


def simulate_live_dataset(
    N,
    sequence_length,
    chunk_size,
    n_workers,
    seed,
    prior,
    random_init,
    k,
    center_y,
    out_dtype,
    exp_clip,
    show_progress,
):
    log_y_squared, theta = sim.simulate_sv_log_y_squared_parallel(
        N=N,
        n=sequence_length,
        chunk_size=chunk_size,
        n_workers=n_workers,
        seed=seed,
        prior=prior,
        random_init=random_init,
        k=k,
        center_y=center_y,
        out_dtype=out_dtype,
        exp_clip=exp_clip,
        show_progress=show_progress,
    )

    target = theta_to_target_numpy(theta).astype(np.float32, copy=False)

    return log_y_squared.astype(np.float32, copy=False), target


# ============================================================
# Training
# ============================================================

def train_live_cnn(
    sequence_length,
    prior="default",
    tcn_channels=(16, 32, 32, 64, 64, 64),
    kernel_size=5,
    dilations=None,
    hidden_dims_head=(64, 64),
    activation=nn.ReLU,
    dropout=0.0,
    use_batch_norm=True,
    checkpoint_path="sv_posterior_tcn_live.pt",
    latest_checkpoint_path=None,
    best_checkpoint_path=None,
    resume_from=None,
    seed=1,
    batch_size=1024,
    n_batches=100,
    val_size=500_000,
    fixed_validation=False,
    lr=1e-4,
    weight_decay=0.0,
    n_epochs=1000,
    patience=50,
    min_delta=1e-4,
    min_var=1e-12,
    standardize_input=True,
    use_amp=None,
    grad_clip_norm=None,
    warn_nonfinite_grad=True,
    deterministic_torch=True,
    n_workers=-2,
    chunks_per_worker=4,
    random_init=True,
    k=1e-12,
    center_y=True,
    out_dtype=np.float32,
    exp_clip=350.0,
    simulation_progress=False,
    verbose=True,
    plot=True,
):
    """
    Train a TCN on SV time series generated live during training.

    One training epoch here means:
        1. generate batch_size * n_batches training samples,
        2. fit exactly n_batches mini-batches,
        3. compute one validation loss on val_size generated samples.

    If fixed_validation=True, the validation set is generated once and reused.
    Otherwise, a new deterministic validation set is generated each epoch.
    """

    # ============================================================
    # Validate configuration
    # ============================================================

    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")

    if batch_size is None or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    if n_batches < 1:
        raise ValueError("n_batches must be at least 1.")

    if val_size < 1:
        raise ValueError("val_size must be at least 1.")

    if n_epochs < 1:
        raise ValueError("n_epochs must be at least 1.")

    if patience < 1:
        raise ValueError("patience must be at least 1.")

    if chunks_per_worker < 1:
        raise ValueError("chunks_per_worker must be at least 1.")

    if k <= 0:
        raise ValueError("k must be positive.")

    # Validate the prior name through simulateData's public API.
    sim.get_stochvol_prior_constants(prior)

    default_latest_path, default_best_path = default_checkpoint_paths(checkpoint_path)

    if latest_checkpoint_path is None:
        latest_checkpoint_path = default_latest_path

    if best_checkpoint_path is None:
        best_checkpoint_path = default_best_path

    if k > K_MOMENT_WARNING_THRESHOLD:
        warnings.warn(
            "Input standardization moments are stored for log(y_t^2), but "
            f"k={k:g} means the model sees log(y_t^2 + k). "
            "For this k, the stored moments may no longer be accurate.",
            RuntimeWarning,
            stacklevel=2,
        )

    out_dtype = np.dtype(out_dtype).type

    resolved_n_workers = sim.resolve_n_workers(n_workers)
    train_size = batch_size * n_batches
    train_chunk_size = sim.resolve_chunk_size(
        train_size,
        resolved_n_workers,
        chunks_per_worker,
    )
    val_chunk_size = sim.resolve_chunk_size(
        val_size,
        resolved_n_workers,
        chunks_per_worker,
    )
    effective_val_batch_size = min(val_size, batch_size)

    if standardize_input:
        moments = sim.log_y_squared_moments(prior=prior)
        input_mean = np.float32(moments["mean"])
        input_std = np.float32(moments["std"])

    else:
        input_mean = np.float32(0.0)
        input_std = np.float32(1.0)

    # ============================================================
    # Reproducibility
    # ============================================================

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False

        try:
            torch.use_deterministic_algorithms(True)
        except TypeError:
            torch.use_deterministic_algorithms(True, warn_only=False)

    # ============================================================
    # Device
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print("Using device:", device)

    if use_amp is None:
        use_amp = device.type == "cuda"
    elif use_amp and device.type != "cuda":
        warnings.warn(
            "Automatic mixed precision (AMP) is only supported on CUDA devices. "
            "Disabling AMP.",
            RuntimeWarning,
            stacklevel=2,
        )

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if verbose and amp_enabled:
        print("Using CUDA automatic mixed precision")

    def has_nonfinite_gradient(model):
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                return True

        return False

    # ============================================================
    # Evaluation helper
    # ============================================================

    @torch.no_grad()
    def evaluate_array(model, x, target):
        """
        Return one mean validation NLL per transformed parameter.
        """
        model.eval()

        total_losses = None
        total_n = 0

        for start in range(0, len(x), effective_val_batch_size):
            stop = min(start + effective_val_batch_size, len(x))

            x_batch = torch.from_numpy(x[start:stop]).to(device)
            target_batch = torch.from_numpy(target[start:stop]).to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                mean, var = model(x_batch)

            losses = diagonal_gaussian_nll(
                mean.float(),
                var.float(),
                target_batch.float(),
            )

            batch_n = x_batch.shape[0]

            if total_losses is None:
                total_losses = torch.zeros_like(losses)

            total_losses += losses * batch_n
            total_n += batch_n

        return total_losses / total_n

    # ============================================================
    # Initialize model
    # ============================================================

    model = LiveSVPosteriorTCN(
        sequence_length=sequence_length,
        tcn_channels=tcn_channels,
        kernel_size=kernel_size,
        dilations=dilations,
        hidden_dims_head=hidden_dims_head,
        activation=activation,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        min_var=min_var,
        input_mean=input_mean,
        input_std=input_std,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if verbose:
        print("Prior:", prior)
        print("Sequence length:", sequence_length)
        print("Input mean:", float(input_mean))
        print("Input std:", float(input_std))
        print("Temporal receptive field:", temporal_receptive_field(kernel_size, model.dilations))
        print("Trainable parameters:", count_parameters(model))
        print("Train samples per validation:", train_size)
        print("Train batch size:", batch_size)
        print("Train batches per validation:", n_batches)
        print("Validation size:", val_size)
        print("Validation batch size:", effective_val_batch_size)
        print("Fixed validation:", fixed_validation)
        print("Requested simulation workers:", n_workers)
        print("Resolved simulation workers:", resolved_n_workers)
        print("Train chunk size:", train_chunk_size)
        print("Validation chunk size:", val_chunk_size)

    # ============================================================
    # Optional fixed validation set
    # ============================================================

    fixed_val_x = None
    fixed_val_target = None
    fixed_validation_seed = None

    if fixed_validation:
        fixed_validation_seed = make_child_seed(
            seed,
            VALIDATION_SEED_STREAM,
            0,
        )

        if verbose:
            print("Generating fixed validation set...")

        fixed_val_x, fixed_val_target = simulate_live_dataset(
            N=val_size,
            sequence_length=sequence_length,
            chunk_size=val_chunk_size,
            n_workers=resolved_n_workers,
            seed=fixed_validation_seed,
            prior=prior,
            random_init=random_init,
            k=k,
            center_y=center_y,
            out_dtype=out_dtype,
            exp_clip=exp_clip,
            show_progress=simulation_progress,
        )

    # ============================================================
    # Training loop with early stopping
    # ============================================================

    target_names = ["mu", "psi", "log_sigma"]

    train_marginal_loss_history = []
    val_marginal_loss_history = []
    train_loss_history = []
    val_loss_history = []
    train_seed_history = []
    validation_seed_history = []

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    best_validation_seed = None
    epochs_without_improvement = 0
    start_epoch = 0

    def make_training_checkpoint(epoch_completed, checkpoint_kind):
        best_model_state_cpu = None

        if best_state is not None:
            best_model_state_cpu = state_dict_to_cpu(best_state)

        return {
            "checkpoint_kind": checkpoint_kind,
            "epoch": epoch_completed,
            "model_state_dict": state_dict_to_cpu(model.state_dict()),
            "best_model_state_dict": best_model_state_cpu,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),

            "model_class": "SVPosteriorTCN",
            "sequence_length": sequence_length,
            "tcn_channels": tuple(tcn_channels),
            "kernel_size": kernel_size,
            "dilations": tuple(model.dilations),
            "temporal_receptive_field": temporal_receptive_field(kernel_size, model.dilations),
            "hidden_dims_head": tuple(hidden_dims_head),
            "activation": getattr(activation, "__name__", str(activation)),
            "dropout": dropout,
            "use_batch_norm": use_batch_norm,
            "min_var": min_var,

            "input_mean": np.float32(input_mean),
            "input_std": np.float32(input_std),
            "standardize_input": standardize_input,
            "standardization_source": "LOG_Y_SQUARED_INPUT_MOMENTS",

            "target_names": target_names,
            "target_transform": {
                "mu": "mu",
                "psi": "2 * atanh(phi)",
                "log_sigma": "log(sigma)",
            },

            "best_val_loss": float(best_val_loss),
            "best_epoch": best_epoch,
            "best_validation_seed": best_validation_seed,
            "epochs_without_improvement": epochs_without_improvement,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "train_marginal_loss_history": train_marginal_loss_history,
            "val_marginal_loss_history": val_marginal_loss_history,
            "train_seed_history": train_seed_history,
            "validation_seed_history": validation_seed_history,

            "live_training": True,
            "prior": prior,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "train_size_per_validation": train_size,
            "val_size": val_size,
            "fixed_validation": fixed_validation,
            "effective_val_batch_size": effective_val_batch_size,
            "requested_n_workers": n_workers,
            "resolved_n_workers": resolved_n_workers,
            "chunks_per_worker": chunks_per_worker,
            "train_chunk_size": train_chunk_size,
            "val_chunk_size": val_chunk_size,
            "random_init": random_init,
            "k": k,
            "center_y": center_y,
            "out_dtype": str(np.dtype(out_dtype)),
            "exp_clip": exp_clip,
            "simulation_progress": simulation_progress,

            "use_amp": use_amp,
            "deterministic_torch": deterministic_torch,
            "lr": lr,
            "weight_decay": weight_decay,
            "n_epochs": n_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "grad_clip_norm": grad_clip_norm,
            "warn_nonfinite_grad": warn_nonfinite_grad,
            "seed": seed,
            "trainable_parameters": count_parameters(model),
        }

    if resume_from is not None:
        resume_checkpoint = torch_load_checkpoint(resume_from, map_location=device)

        model.load_state_dict(resume_checkpoint["model_state_dict"])

        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if "scaler_state_dict" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])

        train_marginal_loss_history = resume_checkpoint.get("train_marginal_loss_history", [])
        val_marginal_loss_history = resume_checkpoint.get("val_marginal_loss_history", [])
        train_loss_history = resume_checkpoint.get("train_loss_history", [])
        val_loss_history = resume_checkpoint.get("val_loss_history", [])
        train_seed_history = resume_checkpoint.get("train_seed_history", [])
        validation_seed_history = resume_checkpoint.get("validation_seed_history", [])

        best_val_loss = float(resume_checkpoint.get("best_val_loss", float("inf")))
        best_epoch = resume_checkpoint.get("best_epoch")
        best_validation_seed = resume_checkpoint.get("best_validation_seed")
        epochs_without_improvement = int(
            resume_checkpoint.get("epochs_without_improvement", 0)
        )
        start_epoch = int(resume_checkpoint.get("epoch", len(train_loss_history)))

        if "best_model_state_dict" in resume_checkpoint:
            best_state = resume_checkpoint["best_model_state_dict"]
        elif np.isfinite(best_val_loss):
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"Resumed training from {resume_from}")
            print(f"Starting at epoch {start_epoch + 1}")
            print(f"Using learning rate: {lr:g}")

    for epoch in range(start_epoch, n_epochs):
        train_seed = make_child_seed(
            seed,
            TRAIN_SEED_STREAM,
            epoch + 1,
        )
        train_seed_history.append(train_seed)

        if verbose and epoch == 0:
            print("Generating live training data...")

        train_x, train_target = simulate_live_dataset(
            N=train_size,
            sequence_length=sequence_length,
            chunk_size=train_chunk_size,
            n_workers=resolved_n_workers,
            seed=train_seed,
            prior=prior,
            random_init=random_init,
            k=k,
            center_y=center_y,
            out_dtype=out_dtype,
            exp_clip=exp_clip,
            show_progress=simulation_progress,
        )

        model.train()

        total_train_losses = None
        total_train_n = 0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            stop = start + batch_size

            x_batch = torch.from_numpy(train_x[start:stop]).to(device)
            target_batch = torch.from_numpy(train_target[start:stop]).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                mean, var = model(x_batch)

            train_marginal_losses = diagonal_gaussian_nll(
                mean.float(),
                var.float(),
                target_batch.float(),
            )
            train_loss = train_marginal_losses.sum()

            scaler.scale(train_loss).backward()

            if grad_clip_norm is not None or warn_nonfinite_grad:
                scaler.unscale_(optimizer)

            nonfinite_grad = warn_nonfinite_grad and has_nonfinite_gradient(model)

            if nonfinite_grad and verbose:
                if amp_enabled:
                    print(
                        "Warning: NaN/Inf gradients detected "
                        f"at epoch {epoch + 1}, batch {batch_idx + 1}; "
                        "GradScaler will skip the optimizer step and lower the scale."
                    )
                else:
                    print(
                        "Warning: NaN/Inf gradients detected "
                        f"at epoch {epoch + 1}, batch {batch_idx + 1}; "
                        "skipping the optimizer step."
                    )

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            # GradScaler handles inf/NaN gradients itself when AMP is enabled,
            # so only manually skip the step when AMP is disabled.
            if amp_enabled or not nonfinite_grad:
                scaler.step(optimizer)

            scaler.update()

            batch_n = x_batch.shape[0]

            if total_train_losses is None:
                total_train_losses = torch.zeros_like(train_marginal_losses)

            # Avoid extending the graph while accumulating training metrics.
            total_train_losses += train_marginal_losses.detach() * batch_n
            total_train_n += batch_n

        del train_x
        del train_target

        train_marginal_losses_value = total_train_losses / total_train_n

        if fixed_validation:
            val_x = fixed_val_x
            val_target = fixed_val_target
            validation_seed = fixed_validation_seed
        else:
            validation_seed = make_child_seed(
                seed,
                VALIDATION_SEED_STREAM,
                epoch + 1,
            )

            val_x, val_target = simulate_live_dataset(
                N=val_size,
                sequence_length=sequence_length,
                chunk_size=val_chunk_size,
                n_workers=resolved_n_workers,
                seed=validation_seed,
                prior=prior,
                random_init=random_init,
                k=k,
                center_y=center_y,
                out_dtype=out_dtype,
                exp_clip=exp_clip,
                show_progress=simulation_progress,
            )

        validation_seed_history.append(validation_seed)

        val_marginal_losses_value = evaluate_array(model, val_x, val_target)

        if not fixed_validation:
            del val_x
            del val_target

        train_marginal_losses_np = train_marginal_losses_value.cpu().numpy()
        val_marginal_losses_np = val_marginal_losses_value.cpu().numpy()

        train_loss_value = float(train_marginal_losses_np.sum())
        val_loss_value = float(val_marginal_losses_np.sum())

        train_loss_history.append(train_loss_value)
        val_loss_history.append(val_loss_value)
        train_marginal_loss_history.append(train_marginal_losses_np.tolist())
        val_marginal_loss_history.append(val_marginal_losses_np.tolist())

        improved = val_loss_value < best_val_loss - min_delta

        if improved:
            best_val_loss = val_loss_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_validation_seed = validation_seed
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            train_parts = ", ".join(
                f"{name}={loss:.4f}"
                for name, loss in zip(target_names, train_marginal_losses_np)
            )
            val_parts = ", ".join(
                f"{name}={loss:.4f}"
                for name, loss in zip(target_names, val_marginal_losses_np)
            )

            print(
                f"Epoch {epoch + 1:4d}: "
                f"train NLL = {train_loss_value:.4f}, "
                f"val NLL = {val_loss_value:.4f}"
            )
            print(
                f"             "
                f"train marginal NLLs: {train_parts}"
            )
            print(
                f"             "
                f"val marginal NLLs:   {val_parts}"
            )

        latest_checkpoint = make_training_checkpoint(
            epoch_completed=epoch + 1,
            checkpoint_kind="latest",
        )
        save_checkpoint_atomic(latest_checkpoint, latest_checkpoint_path)

        if improved:
            best_checkpoint = make_training_checkpoint(
                epoch_completed=epoch + 1,
                checkpoint_kind="best",
            )
            save_checkpoint_atomic(best_checkpoint, best_checkpoint_path)
            if verbose:
                print(f"New best model found at epoch {epoch + 1} with validation NLL {val_loss_value:.6f}")
                print(f"val marginal NLLs: {', '.join(f'{name}={loss:.4f}' for name, loss in zip(target_names, val_marginal_losses_np))}")
                print(f"Best checkpoint saved to {best_checkpoint_path}")

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Latest checkpoint saved to {latest_checkpoint_path}")


        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    # ============================================================
    # Restore best model
    # ============================================================

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("Warning: best_state is None, using current model.")

    model.eval()

    # Use the fixed validation set when available. With changing validation,
    # report a fresh deterministic final validation score for the restored model.
    if fixed_validation:
        final_val_x = fixed_val_x
        final_val_target = fixed_val_target
        final_validation_seed = fixed_validation_seed
    else:
        final_validation_seed = make_child_seed(
            seed,
            FINAL_VALIDATION_SEED_STREAM,
            len(val_loss_history) + 1,
        )

        final_val_x, final_val_target = simulate_live_dataset(
            N=val_size,
            sequence_length=sequence_length,
            chunk_size=val_chunk_size,
            n_workers=resolved_n_workers,
            seed=final_validation_seed,
            prior=prior,
            random_init=random_init,
            k=k,
            center_y=center_y,
            out_dtype=out_dtype,
            exp_clip=exp_clip,
            show_progress=simulation_progress,
        )

    final_val_marginal_losses = evaluate_array(model, final_val_x, final_val_target)
    final_val_marginal_losses_np = final_val_marginal_losses.detach().cpu().numpy()
    final_val_loss = float(final_val_marginal_losses_np.sum())

    if not fixed_validation:
        del final_val_x
        del final_val_target

    if verbose:
        print()
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation mean negative joint log score: {best_val_loss:.6f}")
        print(f"Best validation seed: {best_validation_seed}")
        print(f"Final validation mean negative joint log score: {final_val_loss:.6f}")
        print(f"Final validation seed: {final_validation_seed}")
        print(
            "Final validation marginal NLLs:",
            {
                name: float(loss)
                for name, loss in zip(target_names, final_val_marginal_losses_np)
            },
        )

    # ============================================================
    # Save checkpoint
    # ============================================================

    activation_name = getattr(activation, "__name__", str(activation))

    model_state_cpu = state_dict_to_cpu(model.state_dict())

    checkpoint = {
        "checkpoint_kind": "final",
        "epoch": len(train_loss_history),
        "model_state_dict": model_state_cpu,
        "best_model_state_dict": state_dict_to_cpu(best_state) if best_state is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),

        "model_class": "SVPosteriorTCN",
        "sequence_length": sequence_length,
        "tcn_channels": tuple(tcn_channels),
        "kernel_size": kernel_size,
        "dilations": tuple(model.dilations),
        "temporal_receptive_field": temporal_receptive_field(kernel_size, model.dilations),
        "hidden_dims_head": tuple(hidden_dims_head),
        "activation": activation_name,
        "dropout": dropout,
        "use_batch_norm": use_batch_norm,
        "min_var": min_var,

        "input_mean": np.float32(input_mean),
        "input_std": np.float32(input_std),
        "standardize_input": standardize_input,
        "standardization_source": "LOG_Y_SQUARED_INPUT_MOMENTS",

        "target_names": target_names,
        "target_transform": {
            "mu": "mu",
            "psi": "2 * atanh(phi)",
            "log_sigma": "log(sigma)",
        },

        "loss": "mean negative joint Gaussian log score, diagonal covariance",
        "loss_components": "mean marginal Gaussian negative log scores for mu, psi, and log_sigma",
        "best_val_loss": float(best_val_loss),
        "final_val_loss": float(final_val_loss),
        "final_val_marginal_losses": final_val_marginal_losses_np.astype(np.float32),
        "best_epoch": best_epoch,
        "best_validation_seed": best_validation_seed,
        "final_validation_seed": final_validation_seed,

        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "train_marginal_loss_history": train_marginal_loss_history,
        "val_marginal_loss_history": val_marginal_loss_history,
        "train_seed_history": train_seed_history,
        "validation_seed_history": validation_seed_history,

        "live_training": True,
        "prior": prior,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "train_size_per_validation": train_size,
        "val_size": val_size,
        "fixed_validation": fixed_validation,
        "effective_val_batch_size": effective_val_batch_size,
        "requested_n_workers": n_workers,
        "resolved_n_workers": resolved_n_workers,
        "chunks_per_worker": chunks_per_worker,
        "train_chunk_size": train_chunk_size,
        "val_chunk_size": val_chunk_size,
        "random_init": random_init,
        "k": k,
        "center_y": center_y,
        "out_dtype": str(np.dtype(out_dtype)),
        "exp_clip": exp_clip,
        "simulation_progress": simulation_progress,

        "use_amp": use_amp,
        "deterministic_torch": deterministic_torch,
        "lr": lr,
        "weight_decay": weight_decay,
        "n_epochs": n_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "grad_clip_norm": grad_clip_norm,
        "warn_nonfinite_grad": warn_nonfinite_grad,
        "seed": seed,
        "trainable_parameters": count_parameters(model),
        "latest_checkpoint_path": latest_checkpoint_path,
        "best_checkpoint_path": best_checkpoint_path,
        "resume_from": resume_from,
    }

    save_checkpoint_atomic(checkpoint, checkpoint_path)

    if verbose:
        print(f"Model saved to {checkpoint_path}")

    # ============================================================
    # Plot loss curves
    # ============================================================

    if plot:
        plt.plot(train_loss_history, label="train")
        plt.plot(val_loss_history, label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Mean negative joint log score")
        plt.legend()
        plt.show()

        train_marginal_loss_array = np.array(train_marginal_loss_history)
        val_marginal_loss_array = np.array(val_marginal_loss_history)

        for i, name in enumerate(target_names):
            plt.plot(train_marginal_loss_array[:, i], label=f"train {name}")
            plt.plot(val_marginal_loss_array[:, i], label=f"validation {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Mean marginal negative log score")
        plt.legend()
        plt.show()

    return model, checkpoint


def main():
    train_live_cnn(
        sequence_length=253,
        prior="default",
        tcn_channels=(16, 32, 32, 64, 64),
        kernel_size=5,
        dilations=None, # Use default exponentially increasing dilations
        hidden_dims_head=(32, 32),
        activation=nn.ReLU,
        dropout=0.0,
        use_batch_norm=True,
        checkpoint_path="sv_posterior_tcn_live.pt",
        latest_checkpoint_path="sv_posterior_tcn_live.latest.pt",
        best_checkpoint_path="sv_posterior_tcn_live.best.pt",
        resume_from="sv_posterior_tcn_live.latest.pt",  # Set to "sv_posterior_tcn_live.latest.pt" to continue. Leave as None to start fresh.
        seed=1,
        batch_size=1024 * 8,
        n_batches=100,    # Number of batches done before each validation
        val_size=500_000, # Larger validation to reduce noise between each epoch's validation scores 
        fixed_validation=False,
        lr=0.5e-3,
        weight_decay=0.0,
        n_epochs=2000,
        patience=75, # A bit higher patience since live training is noisier than fixed datasets
        min_delta=1e-5,
        min_var=1e-12, # Minimum variance to ensure numerical stability in the loss and gradients
        standardize_input=True,
        use_amp=True,  # Use automatic mixed precision to save on vram
        grad_clip_norm=5.0,
        warn_nonfinite_grad=True,
        deterministic_torch=True,
        n_workers=-2, # Uses all but 2 cpu cores for data simulation
        chunks_per_worker=4,
        random_init=True,
        k=1e-12,
        center_y=True,
        out_dtype=np.float32,
        exp_clip=350.0,
        simulation_progress=False,
        verbose=True,
        plot=True,
    )


if __name__ == "__main__":
    main()
