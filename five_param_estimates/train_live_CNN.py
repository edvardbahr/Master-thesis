import os
import warnings

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F

import sim_5_param_data as sim


SVGHST_TARGET_NAMES = ("mu", "psi", "log_s", "logit_r", "log_nu")
TARGET_TRANSFORMS = {
    "mu": "mu",
    "psi": "2 * atanh(phi)",
    "log_s": "log(s)",
    "logit_r": "log(r/(1-r))",
    "log_nu": "log(nu)",
}
LOSS_REDUCTION = "mean_over_active_parameters"
CHUNKS_PER_WORKER = 4
KAPPA = 1e-12
CENTER_Y = True
EXP_CLIP = 350.0


def make_mlp(
    input_dim,
    hidden_dims,
    output_dim=None,
    activation=nn.ReLU,
):
    layers = []
    d_prev = input_dim

    for d_hidden in hidden_dims:
        layers.append(nn.Linear(d_prev, d_hidden))
        layers.append(activation())

        d_prev = d_hidden

    if output_dim is not None:
        layers.append(nn.Linear(d_prev, output_dim))
        d_prev = output_dim

    if len(layers) == 0:
        return nn.Identity(), input_dim

    return nn.Sequential(*layers), d_prev


def resolve_per_block_values(value, n_blocks, name):
    if isinstance(value, int):
        values = (int(value),) * n_blocks
    else:
        try:
            values = tuple(int(item) for item in value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an integer or a sequence with one value per TCN block."
            ) from exc

    if len(values) != n_blocks:
        raise ValueError(f"{name} must have the same length as tcn_channels.")

    return values



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
        use_batch_norm=False,
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
        mean: shape (batch_size, n_parameters)
        var: shape (batch_size, n_parameters)

    The output columns follow ``param_names``.
    """

    supported_param_names = SVGHST_TARGET_NAMES

    def __init__(
        self,
        tcn_channels=(16, 32, 32, 64, 64),
        kernel_size=5,
        dilations=None,
        hidden_dims_head=(32, 32),
        topk_pool_fraction=None,
        activation=nn.ReLU,
        use_batch_norm=False,
        param_names=SVGHST_TARGET_NAMES,
        min_var=1e-12,
        input_mean=0.0,
        input_std=1.0,
    ):
        super().__init__()

        if len(tcn_channels) == 0:
            raise ValueError("tcn_channels must contain at least one channel size.")

        param_names = tuple(param_names)
        if not param_names:
            raise ValueError("param_names must contain at least one parameter name.")
        if len(set(param_names)) != len(param_names):
            raise ValueError("param_names must not contain duplicates.")

        unknown_param_names = set(param_names) - set(self.supported_param_names)
        if unknown_param_names:
            raise ValueError(
                "Unknown parameter names: "
                f"{sorted(unknown_param_names)}. "
                f"Valid names are: {self.supported_param_names}."
            )

        if dilations is None:
            dilations = tuple(2 ** i for i in range(len(tcn_channels)))
        else:
            dilations = tuple(int(dilation) for dilation in dilations)

        if len(dilations) != len(tcn_channels):
            raise ValueError("dilations must have the same length as tcn_channels.")
        if any(dilation < 1 for dilation in dilations):
            raise ValueError("dilations must be positive integers.")

        kernel_sizes = resolve_per_block_values(
            kernel_size,
            len(tcn_channels),
            "kernel_size",
        )
        for block_kernel_size in kernel_sizes:
            if block_kernel_size < 1:
                raise ValueError("kernel_size values must be at least 1.")
            if block_kernel_size % 2 == 0:
                raise ValueError("Use odd kernel_size values so padding preserves length.")

        if topk_pool_fraction is not None:
            topk_pool_fraction = float(topk_pool_fraction)
            if not 0.0 < topk_pool_fraction <= 1.0:
                raise ValueError("topk_pool_fraction must be in (0, 1] or None.")

        self.tcn_channels = tuple(tcn_channels)
        self.kernel_size = kernel_size
        self.kernel_sizes = kernel_sizes
        self.dilations = tuple(dilations)
        self.hidden_dims_head = tuple(hidden_dims_head)
        self.topk_pool_fraction = topk_pool_fraction
        self.use_batch_norm = use_batch_norm
        self.param_names = param_names
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

        for out_channels, dilation, block_kernel_size in zip(
            tcn_channels,
            dilations,
            kernel_sizes,
        ):
            blocks.append(
                TemporalResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=block_kernel_size,
                    dilation=dilation,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
            )
            in_channels = out_channels

        self.encoder = nn.Sequential(*blocks)

        n_pooling_summaries = 3 if topk_pool_fraction is not None else 2
        representation_dim = n_pooling_summaries * tcn_channels[-1]

        self.heads = nn.ModuleDict()

        for name in self.param_names:
            head, _ = make_mlp(
                input_dim=representation_dim,
                hidden_dims=hidden_dims_head,
                output_dim=2,
                activation=activation,
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

        # Direct reductions avoid nondeterministic adaptive-max-pool backward
        # implementations while preserving the same representation shape.
        h_avg = h.mean(dim=-1)
        h_max = h.amax(dim=-1)
        pooled = [h_avg, h_max]

        if self.topk_pool_fraction is not None:
            topk_count = max(1, int(h.shape[-1] * self.topk_pool_fraction))
            h_topk_mean = h.topk(
                topk_count,
                dim=-1,
                sorted=False,
            ).values.mean(dim=-1)
            pooled.append(h_topk_mean)

        h = torch.cat(pooled, dim=1)

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


def theta_to_target_numpy(theta, target_names=SVGHST_TARGET_NAMES, eps=1e-6):
    """
    Converts original SV parameters to transformed training targets.

    theta has shape (n_samples, 5), with columns:
        theta[:, 0] = mu
        theta[:, 1] = phi
        theta[:, 2] = s
        theta[:, 3] = r
        theta[:, 4] = nu

    Returns only the transformed columns listed in ``target_names``, in the
    same order.
    """
    target_names = tuple(target_names)
    unknown_target_names = set(target_names) - set(SVGHST_TARGET_NAMES)
    if unknown_target_names:
        raise ValueError(
            f"Unknown target names: {sorted(unknown_target_names)}."
        )

    mu = theta[:, 0]
    phi = theta[:, 1]
    s = theta[:, 2]
    r = theta[:, 3]
    nu = theta[:, 4]
    nu_min = sim.get_gh_skew_t_prior_constants(prior="default").nu_min


    phi = np.clip(phi, -1.0 + eps, 1.0 - eps)
    s= np.clip(s, eps, None)
    r = np.clip(r, eps, 1.0 - eps)
    nu_adjusted = np.clip(nu-nu_min, eps, None)

    psi = 2 * np.arctanh(phi)
    log_s = np.log(s)
    logit_r = np.log(r / (1 - r))
    log_nu = np.log(nu_adjusted)


    transformed = {
        "mu": mu,
        "psi": psi,
        "log_s": log_s,
        "logit_r": logit_r,
        "log_nu": log_nu,
    }
    target = np.column_stack([transformed[name] for name in target_names])

    return target.astype(np.float32)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def temporal_receptive_field(kernel_size, dilations, convs_per_block=2):
    dilations = tuple(dilations)
    kernel_sizes = resolve_per_block_values(
        kernel_size,
        len(dilations),
        "kernel_size",
    )

    return 1 + convs_per_block * sum(
        (block_kernel_size - 1) * dilation
        for block_kernel_size, dilation in zip(kernel_sizes, dilations)
    )

def diagonal_gaussian_nll(mean, var, target, min_var):
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
        eps=min_var,
        reduction="none",
    )

    losses = elementwise_nll.mean(dim=0)

    return losses


TRAIN_SEED_STREAM = 101
VALIDATION_SEED_STREAM = 202
FINAL_VALIDATION_SEED_STREAM = 303


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
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def simulate_live_dataset(
    N,
    sequence_length,
    chunk_size,
    n_workers,
    seed,
    prior,
    fixed_nu,
    fixed_r,
    target_names,
    out_dtype,
):
    log_y_squared, theta = sim.simulate_sv_log_y_squared_parallel(
        N=N,
        n=sequence_length,
        chunk_size=chunk_size,
        n_workers=n_workers,
        seed=seed,
        prior=prior,
        fixed_nu=fixed_nu,
        fixed_r=fixed_r,
        random_init=True,
        k=KAPPA,
        center_y=CENTER_Y,
        out_dtype=out_dtype,
        exp_clip=EXP_CLIP,
        show_progress=False,
    )

    target = theta_to_target_numpy(
        theta,
        target_names=target_names,
    ).astype(np.float32, copy=False)

    return log_y_squared.astype(np.float32, copy=False), target


# ============================================================
# Training
# ============================================================

def train_live_cnn(
    sequence_length,
    prior="default",
    fixed_nu=None,
    fixed_r=None,
    tcn_channels=(16, 32, 32, 64, 64, 64),
    kernel_size=5,
    dilations=None,
    hidden_dims_head=(64, 64),
    topk_pool_fraction=None,
    activation=nn.ReLU,
    use_batch_norm=False,
    checkpoint_path="sv_posterior_tcn_live.pt",
    resume_from=None,
    seed=1,
    batch_size=1024,
    n_batches=100,
    val_size=500_000,
    fixed_validation=False,
    lr=1e-4,
    n_epochs=1000,
    patience=50,
    min_delta=1e-4,
    min_var=1e-12,
    use_amp=None,
    grad_clip_norm=None,
    deterministic_torch=True,
    n_workers=-2,
    out_dtype=np.float32,
    verbose=True,
):
    """
    Train a TCN on SV time series generated live during training.

    One training epoch here means:
        1. generate batch_size * n_batches training samples,
        2. fit exactly n_batches mini-batches,
        3. compute one validation loss on val_size generated samples.

    If fixed_validation=True, the validation set is generated once and reused.
    Otherwise, a new deterministic validation set is generated each epoch.

    If fixed_nu or fixed_r is not None, simulations condition on that value
    and the corresponding log_nu or logit_r target and model head are omitted.
    Set fixed_nu=np.inf and fix r to obtain the three-parameter standard SV
    model setup.
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

    if min_var <= 0:
        raise ValueError("min_var must be positive.")

    if topk_pool_fraction is not None:
        topk_pool_fraction = float(topk_pool_fraction)
        if not 0.0 < topk_pool_fraction <= 1.0:
            raise ValueError("topk_pool_fraction must be in (0, 1] or None.")

    if fixed_r is not None and (
        not np.isfinite(fixed_r) or not 0.0 <= fixed_r < 1.0
    ):
        raise ValueError("fixed_r must satisfy 0 <= fixed_r < 1.")

    if fixed_nu is not None and (
        not (np.isfinite(fixed_nu) or np.isposinf(fixed_nu))
        or fixed_nu <= 4.0
    ):
        raise ValueError("fixed_nu must be greater than 4 or np.inf.")

    fixed_r = None if fixed_r is None else float(fixed_r)
    fixed_nu = None if fixed_nu is None else float(fixed_nu)

    target_names = tuple(
        name
        for name in SVGHST_TARGET_NAMES
        if not (
            (name == "logit_r" and fixed_r is not None)
            or (name == "log_nu" and fixed_nu is not None)
        )
    )

    # Validate the prior name through the 3-parameter simulator API.
    sim.get_gh_skew_t_prior_constants(prior)


    latest_checkpoint_path, best_checkpoint_path = default_checkpoint_paths(
        checkpoint_path
    )

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
    effective_val_batch_size = min(val_size, batch_size)

    moments = sim.log_y_squared_moments(prior=prior)
    input_mean = np.float32(moments["mean"])
    input_std = np.float32(moments["std"])

    # ============================================================
    # Reproducibility
    # ============================================================

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
        use_amp = False

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
                min_var=min_var,
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

    model = SVPosteriorTCN(
        tcn_channels=tcn_channels,
        kernel_size=kernel_size,
        dilations=dilations,
        hidden_dims_head=hidden_dims_head,
        topk_pool_fraction=topk_pool_fraction,
        activation=activation,
        use_batch_norm=use_batch_norm,
        param_names=target_names,
        min_var=min_var,
        input_mean=input_mean,
        input_std=input_std,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    if verbose:
        print("Prior:", prior)
        print("Fixed r:", fixed_r)
        print("Fixed nu:", fixed_nu)
        print("Estimated parameters:", target_names)
        print("Sequence length:", sequence_length)
        print("Input mean:", float(input_mean))
        print("Input std:", float(input_std))
        print("Kernel sizes:", model.kernel_sizes)
        print("Dilations:", model.dilations)
        print(
            "Temporal receptive field:",
            temporal_receptive_field(model.kernel_sizes, model.dilations),
        )
        print("Top-k pooling fraction:", model.topk_pool_fraction)
        if model.topk_pool_fraction is not None:
            print(
                "Top-k activations per channel:",
                max(1, int(sequence_length * model.topk_pool_fraction)),
            )
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
            fixed_nu=fixed_nu,
            fixed_r=fixed_r,
            target_names=target_names,
            out_dtype=out_dtype,
        )

    # ============================================================
    # Training loop with early stopping
    # ============================================================

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

    activation_name = getattr(activation, "__name__", str(activation))

    def make_checkpoint(
        epoch_completed,
        checkpoint_kind,
        final_val_loss=None,
        final_val_marginal_losses=None,
        final_validation_seed=None,
    ):
        best_model_state_cpu = None

        if best_state is not None:
            best_model_state_cpu = state_dict_to_cpu(best_state)

        if final_val_marginal_losses is not None:
            final_val_marginal_losses = np.asarray(
                final_val_marginal_losses,
                dtype=np.float32,
            )

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
            "kernel_sizes": tuple(model.kernel_sizes),
            "dilations": tuple(model.dilations),
            "temporal_receptive_field": temporal_receptive_field(
                model.kernel_sizes,
                model.dilations,
            ),
            "hidden_dims_head": tuple(hidden_dims_head),
            "topk_pool_fraction": model.topk_pool_fraction,
            "pooling_modes": (
                ("mean", "max", "topk_mean")
                if model.topk_pool_fraction is not None
                else ("mean", "max")
            ),
            "activation": activation_name,
            "use_batch_norm": use_batch_norm,
            "min_var": min_var,

            "input_mean": np.float32(input_mean),
            "input_std": np.float32(input_std),
            "standardize_input": True,
            "standardization_source": "sim_5_param_data.log_y_squared_moments",

            "target_names": target_names,
            "target_transform": {
                name: TARGET_TRANSFORMS[name]
                for name in target_names
            },
            "loss": "mean marginal Gaussian NLL across active parameters",
            "loss_components": "mean marginal Gaussian negative log scores",
            "loss_reduction": LOSS_REDUCTION,

            "best_val_loss": float(best_val_loss),
            "final_val_loss": (
                None if final_val_loss is None else float(final_val_loss)
            ),
            "final_val_marginal_losses": final_val_marginal_losses,
            "best_epoch": best_epoch,
            "best_validation_seed": best_validation_seed,
            "final_validation_seed": final_validation_seed,
            "epochs_without_improvement": epochs_without_improvement,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "train_marginal_loss_history": train_marginal_loss_history,
            "val_marginal_loss_history": val_marginal_loss_history,

            "prior": prior,
            "fixed_r": fixed_r,
            "fixed_nu": fixed_nu,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "train_size_per_validation": train_size,
            "val_size": val_size,
            "fixed_validation": fixed_validation,
            "effective_val_batch_size": effective_val_batch_size,
            "requested_n_workers": n_workers,
            "resolved_n_workers": resolved_n_workers,
            "chunks_per_worker": CHUNKS_PER_WORKER,
            "train_chunk_size": train_chunk_size,
            "val_chunk_size": val_chunk_size,
            "random_init": True,
            "k": KAPPA,
            "out_dtype": str(np.dtype(out_dtype)),

            "use_amp": use_amp,
            "deterministic_torch": deterministic_torch,
            "lr": lr,
            "n_epochs": n_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "grad_clip_norm": grad_clip_norm,
            "warn_nonfinite_grad": True,
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

        checkpoint_loss_reduction = resume_checkpoint.get("loss_reduction")
        if checkpoint_loss_reduction != LOSS_REDUCTION:
            raise ValueError(
                "Cannot resume a checkpoint created with a different loss "
                "reduction: "
                f"checkpoint={checkpoint_loss_reduction}, "
                f"requested={LOSS_REDUCTION}."
            )

        checkpoint_target_names = tuple(
            resume_checkpoint.get("target_names", ())
        )
        if checkpoint_target_names != target_names:
            raise ValueError(
                "Cannot resume with different estimated parameters: "
                f"checkpoint={checkpoint_target_names}, requested={target_names}."
            )

        checkpoint_fixed_nu = resume_checkpoint.get("fixed_nu")
        same_fixed_nu = (
            checkpoint_fixed_nu is None and fixed_nu is None
        ) or (
            checkpoint_fixed_nu is not None
            and fixed_nu is not None
            and float(checkpoint_fixed_nu) == fixed_nu
        )
        if not same_fixed_nu:
            raise ValueError(
                "Cannot resume with a different fixed_nu value: "
                f"checkpoint={checkpoint_fixed_nu}, requested={fixed_nu}."
            )

        checkpoint_fixed_r = resume_checkpoint.get("fixed_r")
        same_fixed_r = (
            checkpoint_fixed_r is None and fixed_r is None
        ) or (
            checkpoint_fixed_r is not None
            and fixed_r is not None
            and float(checkpoint_fixed_r) == fixed_r
        )
        if not same_fixed_r:
            raise ValueError(
                "Cannot resume with a different fixed_r value: "
                f"checkpoint={checkpoint_fixed_r}, requested={fixed_r}."
            )

        checkpoint_topk_pool_fraction = resume_checkpoint.get("topk_pool_fraction")
        if checkpoint_topk_pool_fraction != model.topk_pool_fraction:
            raise ValueError(
                "Cannot resume with a different top-k pooling fraction: "
                f"checkpoint={checkpoint_topk_pool_fraction}, "
                f"requested={model.topk_pool_fraction}."
            )

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
        inconsistent_histories = {
            name: length
            for name, length in history_lengths.items()
            if length != start_epoch
        }
        if inconsistent_histories:
            raise ValueError(
                "Checkpoint epoch does not match its loss-history lengths: "
                f"epoch={start_epoch}, lengths={history_lengths}."
            )

        if resume_checkpoint.get("best_model_state_dict") is not None:
            best_state = state_dict_to_cpu(
                resume_checkpoint["best_model_state_dict"]
            )
        elif np.isfinite(best_val_loss):
            best_state = state_dict_to_cpu(model.state_dict())

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

        if verbose and epoch == 0:
            print("Generating live training data...")

        train_x, train_target = simulate_live_dataset(
            N=train_size,
            sequence_length=sequence_length,
            chunk_size=train_chunk_size,
            n_workers=resolved_n_workers,
            seed=train_seed,
            prior=prior,
            fixed_nu=fixed_nu,
            fixed_r=fixed_r,
            target_names=target_names,
            out_dtype=out_dtype,
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
                min_var=min_var,
            )
            # Averaging over active parameters keeps the gradient scale, and
            # therefore the learning-rate interpretation, independent of how
            # many parameter heads are enabled.
            train_loss = train_marginal_losses.mean()

            scaler.scale(train_loss).backward()

            scaler.unscale_(optimizer)

            nonfinite_grad = has_nonfinite_gradient(model)

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
                fixed_nu=fixed_nu,
                fixed_r=fixed_r,
                target_names=target_names,
                out_dtype=out_dtype,
            )

        val_marginal_losses_value = evaluate_array(model, val_x, val_target)

        if not fixed_validation:
            del val_x
            del val_target

        train_marginal_losses_np = train_marginal_losses_value.cpu().numpy()
        val_marginal_losses_np = val_marginal_losses_value.cpu().numpy()

        train_loss_value = float(train_marginal_losses_np.mean())
        val_loss_value = float(val_marginal_losses_np.mean())

        train_loss_history.append(train_loss_value)
        val_loss_history.append(val_loss_value)
        train_marginal_loss_history.append(train_marginal_losses_np.tolist())
        val_marginal_loss_history.append(val_marginal_losses_np.tolist())

        improved = val_loss_value < best_val_loss - min_delta

        if improved:
            best_val_loss = val_loss_value
            best_state = state_dict_to_cpu(model.state_dict())
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

        completed_epoch = epoch + 1

        latest_checkpoint = make_checkpoint(
            epoch_completed=epoch + 1,
            checkpoint_kind="latest",
        )
        save_checkpoint_atomic(latest_checkpoint, latest_checkpoint_path)

        if improved:
            best_checkpoint = make_checkpoint(
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
            completed_epoch + 1,
        )

        final_val_x, final_val_target = simulate_live_dataset(
            N=val_size,
            sequence_length=sequence_length,
            chunk_size=val_chunk_size,
            n_workers=resolved_n_workers,
            seed=final_validation_seed,
            prior=prior,
            fixed_nu=fixed_nu,
            fixed_r=fixed_r,
            target_names=target_names,
            out_dtype=out_dtype,
        )

    final_val_marginal_losses = evaluate_array(model, final_val_x, final_val_target)
    final_val_marginal_losses_np = final_val_marginal_losses.detach().cpu().numpy()
    final_val_loss = float(final_val_marginal_losses_np.mean())

    if not fixed_validation:
        del final_val_x
        del final_val_target

    if verbose:
        print()
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation mean marginal NLL: {best_val_loss:.6f}")
        print(f"Best validation seed: {best_validation_seed}")
        print(f"Final validation mean marginal NLL: {final_val_loss:.6f}")
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

    checkpoint = make_checkpoint(
        epoch_completed=completed_epoch,
        checkpoint_kind="final",
        final_val_loss=final_val_loss,
        final_val_marginal_losses=final_val_marginal_losses_np,
        final_validation_seed=final_validation_seed,
    )

    save_checkpoint_atomic(checkpoint, checkpoint_path)

    if verbose:
        print(f"Model saved to {checkpoint_path}")

    return model, checkpoint


def main():
    train_live_cnn(
        sequence_length=253 * 10,
        prior="default",
        fixed_r=None,
        fixed_nu=None,  # Set to 12 as this was our EM estimate using 2000-2020 5 min RV of S&P500. Set to np.inf for Gaussian innovations
        tcn_channels=(16, 32, 32, 64, 64, 64),
        kernel_size=(9, 9, 7, 5, 5, 5),
        dilations=(1, 2, 4, 16, 64, 256),
        hidden_dims_head=(32, 32),
        topk_pool_fraction=0.05,
        activation=nn.ReLU,
        use_batch_norm=False,
        checkpoint_path="svghst_posterior_tcn_live_default_n2530_multiscale_topk.pt",
        resume_from=None,  # Set to the n2530_multiscale_topk latest checkpoint to continue.
        seed=2,
        batch_size=1024 * 4,
        n_batches=100,    # Number of batches done before each validation
        val_size=200_000, # Similar validation memory footprint to the 253 * 2 run
        fixed_validation=False,
        lr=0.5e-3,
        n_epochs=2000,
        patience=75, # A bit higher patience since live training is noisier than fixed datasets
        min_delta=1e-5,
        min_var=1e-12, # Minimum variance to ensure numerical stability in the loss and gradients
        use_amp=True,  # Use automatic mixed precision to save on vram
        grad_clip_norm=5.0,
        deterministic_torch=True,
        n_workers= -2, # Uses all but 2 cpu cores for data simulation
        out_dtype=np.float32,
        verbose=True,
    )


if __name__ == "__main__":
    main()
