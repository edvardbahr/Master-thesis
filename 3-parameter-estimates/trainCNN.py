import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Shared helpers
# ============================================================

def make_mlp(
    input_dim,
    hidden_dims,
    output_dim=None,
    activation=nn.ReLU,
    dropout=0.0,
    layer_norm=False,
):
    layers = []
    d_prev = input_dim

    for d_hidden in hidden_dims:
        layers.append(nn.Linear(d_prev, d_hidden))

        if layer_norm:
            layers.append(nn.LayerNorm(d_hidden))

        layers.append(activation())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        d_prev = d_hidden

    if output_dim is not None:
        layers.append(nn.Linear(d_prev, output_dim))
        d_prev = output_dim

    if len(layers) == 0:
        return nn.Identity(), input_dim

    return nn.Sequential(*layers), d_prev


def theta_to_target_numpy(theta, eps=1e-6):
    """
    Converts original SV parameters to transformed training targets.

    theta has shape (n_samples, 3), with columns:
        theta[:, 0] = mu
        theta[:, 1] = phi
        theta[:, 2] = sigma

    Returns target with columns:
        mu
        psi = 2 * atanh(phi)
        log_sigma = log(sigma)
    """
    mu = theta[:, 0]
    phi = theta[:, 1]
    sigma = theta[:, 2]

    phi = np.clip(phi, -1.0 + eps, 1.0 - eps)
    sigma = np.clip(sigma, eps, None)

    psi = 2 * np.arctanh(phi)
    log_sigma = np.log(sigma)

    target = np.column_stack([mu, psi, log_sigma])

    return target.astype(np.float32)


def diagonal_gaussian_nll(mean, var, target):
    """
    Computes marginal negative log scores under a diagonal Gaussian.

    mean:
        shape (batch_size, param_dim)

    var:
        shape (batch_size, param_dim)

    target:
        shape (batch_size, param_dim)

    Returns:
        losses:
            shape (param_dim,), one mean NLL per transformed parameter.
    """
    elementwise_nll = F.gaussian_nll_loss(
        input=mean,
        target=target,
        var=var,
        full=True,
        reduction="none",
    )

    losses = elementwise_nll.mean(dim=0)

    return losses


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def temporal_receptive_field(kernel_size, dilations, convs_per_block=2):
    return 1 + convs_per_block * (kernel_size - 1) * sum(dilations)


def estimate_indexed_mean_std(x, indices, chunk_size=100_000, eps=1e-8):
    """
    Estimate one global mean/std over all time points in the indexed rows.

    This avoids materializing a second standardized copy of the full time-series
    matrix, which matters more here than in the summary-statistic trainer.
    """
    total = 0.0
    total_sq = 0.0
    total_n = 0

    for start in range(0, len(indices), chunk_size):
        idx = indices[start:start + chunk_size]
        chunk = x[idx].astype(np.float64, copy=False)

        total += chunk.sum(dtype=np.float64)
        total_sq += np.square(chunk, dtype=np.float64).sum(dtype=np.float64)
        total_n += chunk.size

    mean = total / total_n
    var = max(total_sq / total_n - mean * mean, eps)
    std = np.sqrt(var)

    return np.float32(mean), np.float32(std)


class SVTimeSeriesDataset(Dataset):
    """
    Dataset for precomputed log(y_t^2 + k) time series.

    x:
        NumPy array of shape (N, n).

    target:
        NumPy array of shape (N, 3), containing transformed parameters.

    indices:
        Row indices defining the split.
    """

    def __init__(self, x, target, indices):
        self.x = x
        self.target = target
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row = int(self.indices[item])

        return (
            torch.from_numpy(self.x[row]).float(),
            torch.from_numpy(self.target[row]).float(),
        )


# ============================================================
# TCN model definition
# ============================================================

class TemporalResidualBlock(nn.Module):
    """
    Non-causal TCN-style residual block.

    Causality is unnecessary here because the posterior is conditioned on the
    full observed series. Symmetric padding lets each location use neighboring
    observations on both sides while preserving sequence length.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        dilation=1,
        activation=nn.ReLU,
        dropout=0.0,
        use_batch_norm=True,
    ):
        super().__init__()

        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1.")

        if kernel_size % 2 == 0:
            raise ValueError("Use an odd kernel_size so symmetric padding preserves length.")

        padding = dilation * (kernel_size - 1) // 2

        layers = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))

        layers.append(activation())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.activation = activation()

    def forward(self, x):
        return self.activation(self.net(x) + self.residual(x))


class SVPosteriorTCN(nn.Module):
    """
    Temporal convolutional network for amortized inference in the standard SV model.

    Input:
        x: log(y_t^2 + k), shape (batch_size, sequence_length)
           or (batch_size, 1, sequence_length)

    Output:
        mean: shape (batch_size, 3)
              columns: [mu_mean, psi_mean, log_sigma_mean]

        var: shape (batch_size, 3)
             columns: [mu_var, psi_var, log_sigma_var]

    where:
        psi = 2 * atanh(phi)
        log_sigma = log(sigma)
    """

    param_names = ("mu", "psi", "log_sigma")

    def __init__(
        self,
        sequence_length,
        tcn_channels=(16, 32, 32, 64, 64, 64),
        kernel_size=5,
        dilations=None,
        hidden_dims_head=(64, 64),
        activation=nn.ReLU,
        dropout=0.0,
        use_batch_norm=True,
        min_var=1e-12,
        input_mean=0.0,
        input_std=1.0,
    ):
        super().__init__()

        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")

        if len(tcn_channels) == 0:
            raise ValueError("tcn_channels must contain at least one channel size.")

        if dilations is None:
            dilations = tuple(2 ** i for i in range(len(tcn_channels)))

        if len(dilations) != len(tcn_channels):
            raise ValueError("dilations must have the same length as tcn_channels.")

        self.sequence_length = sequence_length
        self.tcn_channels = tuple(tcn_channels)
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.hidden_dims_head = tuple(hidden_dims_head)
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.min_var = min_var

        self.register_buffer(
            "input_mean",
            torch.tensor(float(input_mean), dtype=torch.float32).view(1, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor(float(input_std), dtype=torch.float32).view(1, 1, 1),
        )

        blocks = []
        in_channels = 1

        for out_channels, dilation in zip(tcn_channels, dilations):
            blocks.append(
                TemporalResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                )
            )
            in_channels = out_channels

        self.encoder = nn.Sequential(*blocks)

        representation_dim = 2 * tcn_channels[-1]

        self.heads = nn.ModuleDict()

        for name in self.param_names:
            head, _ = make_mlp(
                input_dim=representation_dim,
                hidden_dims=hidden_dims_head,
                output_dim=2,
                activation=activation,
                dropout=dropout,
                layer_norm=False,
            )
            self.heads[name] = head

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError("x must have shape (batch_size, n) or (batch_size, 1, n).")

        if x.shape[1] != 1:
            raise ValueError("x must have exactly one input channel.")

        x = (x - self.input_mean) / self.input_std.clamp_min(1e-8)

        h = self.encoder(x)

        h_avg = F.adaptive_avg_pool1d(h, output_size=1).squeeze(-1)
        h_max = F.adaptive_max_pool1d(h, output_size=1).squeeze(-1)
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


# ============================================================
# Training
# ============================================================

def train_cnn(
    data_path,
    tcn_channels=(16, 32, 32, 64, 64, 64),
    kernel_size=5,
    dilations=None,
    hidden_dims_head=(64, 64),
    activation=nn.ReLU,
    dropout=0.0,
    use_batch_norm=True,
    checkpoint_path="sv_posterior_tcn.pt",
    seed=1,
    val_fraction=0.2,
    batch_size=1024,
    val_batch_size=None,
    lr=1e-4,
    weight_decay=0.0,
    n_epochs=1000,
    patience=50,
    min_delta=1e-4,
    min_var=1e-12,
    standardize_input=True,
    normalization_chunk_size=100_000,
    use_amp=None,
    grad_clip_norm=None,
    verbose=True,
    plot=True,
    num_workers=0,
):
    """
    Train a TCN to predict transformed SV parameters from log(y_t^2 + k) series.

    Parameters
    ----------
    data_path:
        Path to .npz file containing a time-series matrix and parameters.
        Path to .npz file containing arrays with keys "log_y_squared" and
        "params". "log_y_squared" must have shape (N, n), where each row is
        log(y_t^2 + k). "params" must have shape (N, 3), with columns
        [mu, phi, sigma].

    tcn_channels:
        Output channel count for each residual temporal block.

    kernel_size:
        Odd temporal convolution kernel size.

    dilations:
        Dilation for each residual temporal block. If None, powers of two are
        used: 1, 2, 4, ...

    hidden_dims_head:
        Hidden dimensions for each posterior head.

    activation:
        PyTorch activation class, e.g. nn.ReLU, nn.GELU.

    dropout:
        Dropout used inside TCN blocks and posterior heads.

    use_batch_norm:
        Whether to use BatchNorm1d inside TCN blocks.

    checkpoint_path:
        Path where the trained checkpoint is saved.

    seed:
        Random seed used for train/validation split and PyTorch initialization.

    val_fraction:
        Fraction of data used for validation.

    batch_size:
        Mini-batch size for training.
        If batch_size=None, full-batch training is used.

    val_batch_size:
        Mini-batch size for validation. If None, validation uses batch_size.
        Use a smaller value if validation runs out of GPU memory.

    lr:
        Learning rate for Adam.

    weight_decay:
        Adam weight decay.

    n_epochs:
        Maximum number of epochs.

    patience:
        Number of epochs without validation improvement before early stopping.

    min_delta:
        Minimum validation improvement required to reset patience.

    min_var:
        Lower bound added inside model variance output.

    standardize_input:
        If True, standardize log(y_t^2 + k) using one global mean/std estimated
        from the training split only. The normalization is done inside the model.

    normalization_chunk_size:
        Number of indexed rows used at a time when computing training mean/std.

    use_amp:
        Whether to use CUDA automatic mixed precision. If None, AMP is enabled
        automatically on CUDA and disabled on CPU.

    grad_clip_norm:
        If not None, clip gradient norm to this value.

    verbose:
        Whether to print progress.

    plot:
        Whether to plot training and validation loss curves.

    num_workers:
        Number of subprocesses for DataLoader. Keep at 0 unless you need more.
    """

    # ============================================================
    # Reproducibility
    # ============================================================

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ============================================================
    # Device
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print("Using device:", device)

    pin_memory = device.type == "cuda"

    if use_amp is None:
        use_amp = device.type == "cuda"

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if verbose and amp_enabled:
        print("Using CUDA automatic mixed precision")

    # ============================================================
    # Evaluation helper
    # ============================================================

    @torch.no_grad()
    def evaluate(model, loader, device):
        """
        Return one mean validation NLL per transformed parameter.
        """
        model.eval()

        total_losses = None
        total_n = 0

        for x_batch, target_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

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
    # Load data
    # ============================================================

    data = np.load(data_path, allow_pickle=True)

    X = data["log_y_squared"].astype(np.float32, copy=False)
    theta = data["params"].astype(np.float32, copy=False)

    if X.ndim != 2:
        raise ValueError("Time-series data must have shape (N, n).")

    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError("Parameter data must have shape (N, 3).")

    if len(X) != len(theta):
        raise ValueError("Time-series data and parameter data must have the same number of rows.")

    if verbose:
        print("X shape:", X.shape)
        print("theta shape:", theta.shape)

    target = theta_to_target_numpy(theta).astype(np.float32, copy=False)

    if verbose:
        print("target shape:", target.shape)

    # ============================================================
    # Train-validation split
    # ============================================================

    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1.")

    N, sequence_length = X.shape

    if N == 0:
        raise ValueError("Dataset is empty.")

    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(N)

    n_val = int(val_fraction * N)

    if n_val == 0:
        raise ValueError("Validation set is empty. Increase val_fraction or dataset size.")

    if n_val == N:
        raise ValueError("Training set is empty. Decrease val_fraction.")

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    # ============================================================
    # Standardize input using training set only
    # ============================================================

    if standardize_input:
        input_mean, input_std = estimate_indexed_mean_std(
            X,
            train_idx,
            chunk_size=normalization_chunk_size,
        )
    else:
        input_mean = np.float32(0.0)
        input_std = np.float32(1.0)

    if verbose:
        print("Input mean:", float(input_mean))
        print("Input std:", float(input_std))

    # ============================================================
    # Create PyTorch datasets
    # ============================================================

    train_dataset = SVTimeSeriesDataset(X, target, train_idx)
    val_dataset = SVTimeSeriesDataset(X, target, val_idx)

    # ============================================================
    # Decide batch sizes
    # ============================================================

    if batch_size is None:
        effective_batch_size = len(train_dataset)
        train_shuffle = False

        if verbose:
            print("Training mode: full-batch")
    else:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None.")

        effective_batch_size = batch_size
        train_shuffle = True

        if verbose:
            print(f"Training mode: mini-batch, batch_size={batch_size}")

    if val_batch_size is None:
        effective_val_batch_size = effective_batch_size
    else:
        if val_batch_size <= 0:
            raise ValueError("val_batch_size must be a positive integer or None.")

        effective_val_batch_size = val_batch_size

    # ============================================================
    # Initialize model
    # ============================================================

    model = SVPosteriorTCN(
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

    # ============================================================
    # Create DataLoaders
    # ============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=train_shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if verbose:
        print("Train size:", len(train_dataset))
        print("Validation size:", len(val_dataset))
        print("Sequence length:", sequence_length)
        print("Temporal receptive field:", temporal_receptive_field(kernel_size, model.dilations))
        print("Train batch size:", effective_batch_size)
        print("Validation batch size:", effective_val_batch_size)
        print("Number of train batches per epoch:", len(train_loader))
        print("Number of validation batches:", len(val_loader))
        print("Trainable parameters:", count_parameters(model))

    # ============================================================
    # Training loop with early stopping
    # ============================================================

    target_names = ["mu", "psi", "log_sigma"]

    train_marginal_loss_history = []
    val_marginal_loss_history = []
    train_loss_history = []
    val_loss_history = []

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        model.train()

        total_train_losses = None
        total_train_n = 0

        for x_batch, target_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

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

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            batch_n = x_batch.shape[0]

            if total_train_losses is None:
                total_train_losses = torch.zeros_like(train_marginal_losses)

            total_train_losses += train_marginal_losses.detach() * batch_n
            total_train_n += batch_n

        train_marginal_losses_value = total_train_losses / total_train_n
        val_marginal_losses_value = evaluate(model, val_loader, device)

        train_marginal_losses_np = train_marginal_losses_value.cpu().numpy()
        val_marginal_losses_np = val_marginal_losses_value.cpu().numpy()

        train_loss_value = float(train_marginal_losses_np.sum())
        val_loss_value = float(val_marginal_losses_np.sum())

        train_loss_history.append(train_loss_value)
        val_loss_history.append(val_loss_value)
        train_marginal_loss_history.append(train_marginal_losses_np.tolist())
        val_marginal_loss_history.append(val_marginal_losses_np.tolist())

        if val_loss_value < best_val_loss - min_delta:
            best_val_loss = val_loss_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
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

    final_val_marginal_losses = evaluate(model, val_loader, device)
    final_val_marginal_losses_np = final_val_marginal_losses.detach().cpu().numpy()
    final_val_loss = float(final_val_marginal_losses_np.sum())

    if verbose:
        print()
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation mean negative joint log score: {best_val_loss:.6f}")
        print(f"Final validation mean negative joint log score: {final_val_loss:.6f}")
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

    model_state_cpu = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": model_state_cpu,

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

        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "train_marginal_loss_history": train_marginal_loss_history,
        "val_marginal_loss_history": val_marginal_loss_history,

        "data_path": data_path,
        "val_fraction": val_fraction,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "val_batch_size": val_batch_size,
        "effective_val_batch_size": effective_val_batch_size,
        "use_amp": use_amp,
        "lr": lr,
        "weight_decay": weight_decay,
        "n_epochs": n_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "grad_clip_norm": grad_clip_norm,
        "seed": seed,
        "trainable_parameters": count_parameters(model),
    }

    torch.save(checkpoint, checkpoint_path)

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
    train_cnn(
        data_path="sv_log_y_squared_default_1M.npz",
        tcn_channels=(16, 32, 32, 64, 64, 64),
        kernel_size=5,
        dilations=None,
        hidden_dims_head=(64, 64),
        activation=nn.ReLU,
        dropout=0.05,
        use_batch_norm=True,
        checkpoint_path="sv_posterior_tcn_1M.pt",
        seed=1,
        val_fraction=0.2,
        batch_size=1024*4,
        val_batch_size=1024*16,
        lr=0.5e-3,
        weight_decay=0.0,
        n_epochs=2000,
        patience=50,
        min_delta=1e-5,
        min_var=1e-12,
        standardize_input=True,
        use_amp=True,
        grad_clip_norm=5.0,
        verbose=True,
        plot=True,
        num_workers=0,
    )


if __name__ == "__main__":
    main()
