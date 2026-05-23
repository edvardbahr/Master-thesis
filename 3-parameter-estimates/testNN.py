import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
import pandas as pd
from R_to_py_interface import run_stochvol_mcmc, validate_series_matrix
import simulateData as sim
import torch
import torch.nn as nn
from trainLiveCNN import LiveSVPosteriorTCN
from trainSummaryNN import SVPosteriorNN


HERE = Path(__file__).resolve().parent

DEFAULT_SUMMARY_CHECKPOINT_PATH = "sv_posterior_nn_1M_ARIMA_finance.pt"
DEFAULT_OUTPUT_DIR = "nn_vs_mcmc_comparison"
DEFAULT_ALPHA = 0.05
DEFAULT_K = 1e-12

PARAMETER_NAMES = ("mu", "phi", "sigma")
TRANSFORMED_TARGET_NAMES = ["mu", "psi", "log_sigma"]


@dataclass
class LoadedSVModel:
    model: nn.Module
    checkpoint: dict
    checkpoint_path: Path
    model_kind: str
    label: str
    prefix: str
    device: torch.device


# ============================================================
# Path and checkpoint helpers
# ============================================================

def resolve_existing_path(path):
    """
    Resolve a path that may be absolute or relative. If relative, search in cwd,
    script directory, and parent directories. Return the first existing path found.
    """
    path = Path(path).expanduser()

    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Could not find {path}.")

    search_roots = [Path.cwd(), HERE, *HERE.parents]
    seen = set()

    for root in search_roots:
        candidate = (root / path).resolve()

        if candidate in seen:
            continue

        seen.add(candidate)

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {path}. Looked relative to cwd, script dir, and parent dirs."
    )


def resolve_output_dir(path):
    """
    Resolve an output directory path.
    If it doesn't exist, create it. Return the resolved path.
    """
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    path.mkdir(parents=True, exist_ok=True)

    return path


def torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def activation_from_checkpoint(checkpoint):
    """
    Get the activation function class from the checkpoint. Default to ReLU if not specified.
    """
    activation_name = checkpoint.get("activation", -1)
    if activation_name == -1:
        print("Warning: checkpoint does not specify activation. Defaulting to ReLU.")
        activation_name = "ReLU"
    activation = getattr(nn, activation_name, None)

    if activation is None:
        raise ValueError(f"Unknown activation in checkpoint: {activation_name}")

    return activation


def state_dict_from_checkpoint(checkpoint, prefer_best=True):
    if prefer_best:
        best_state = checkpoint.get("best_model_state_dict")

        if best_state is not None:
            return best_state

    return checkpoint["model_state_dict"]


def validate_target_names(checkpoint):
    target_names = list(checkpoint.get("target_names", TRANSFORMED_TARGET_NAMES))

    if target_names != TRANSFORMED_TARGET_NAMES:
        raise ValueError(
            "This comparison expects target_names "
            f"{TRANSFORMED_TARGET_NAMES}, got {target_names}."
        )


def infer_model_kind(checkpoint, model_kind):
    if model_kind != "auto":
        return model_kind

    if checkpoint.get("model_class") == "SVPosteriorTCN":
        return "cnn"

    if "tcn_channels" in checkpoint or "sequence_length" in checkpoint:
        return "cnn"

    if "input_dim" in checkpoint and "hidden_dims_shared_trunk" in checkpoint:
        return "summary"

    raise ValueError("Could not infer model kind from checkpoint. Pass model_kind explicitly.")


def sanitize_prefix(label):
    prefix = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()

    if prefix == "":
        prefix = "model"

    if prefix[0].isdigit():
        prefix = f"model_{prefix}"

    return prefix


def make_unique_prefix(prefix, used_prefixes):
    if prefix not in used_prefixes:
        used_prefixes.add(prefix)
        return prefix

    index = 2

    while f"{prefix}_{index}" in used_prefixes:
        index += 1

    unique_prefix = f"{prefix}_{index}"
    used_prefixes.add(unique_prefix)

    return unique_prefix


# ============================================================
# Model loading
# ============================================================

def build_summary_nn(checkpoint, state_dict, device):
    activation = activation_from_checkpoint(checkpoint)
    layer_norm_candidates = []

    if "layer_norm" in checkpoint:
        layer_norm_candidates.append(bool(checkpoint["layer_norm"]))

    layer_norm_candidates.extend([False, True])
    layer_norm_candidates = list(dict.fromkeys(layer_norm_candidates))

    errors = []

    for layer_norm in layer_norm_candidates:
        model = SVPosteriorNN(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims_shared_trunk=tuple(checkpoint["hidden_dims_shared_trunk"]),
            hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
            activation=activation,
            min_var=float(checkpoint.get("min_var", 1e-12)),
            dropout=float(checkpoint.get("dropout", 0.0)),
            layer_norm=layer_norm,
        ).to(device)

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as error:
            errors.append(f"layer_norm={layer_norm}: {error}")
            continue

        model.eval()
        checkpoint["resolved_layer_norm"] = layer_norm

        return model

    raise RuntimeError(
        "Could not load summary NN state_dict with any known layer_norm setting.\n"
        + "\n".join(errors)
    )


def build_cnn(checkpoint, state_dict, device):
    activation = activation_from_checkpoint(checkpoint)

    model = LiveSVPosteriorTCN(
        sequence_length=int(checkpoint["sequence_length"]),
        tcn_channels=tuple(checkpoint["tcn_channels"]),
        kernel_size=int(checkpoint["kernel_size"]),
        dilations=tuple(checkpoint["dilations"]),
        hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
        activation=activation,
        dropout=float(checkpoint.get("dropout", 0.0)),
        use_batch_norm=bool(checkpoint.get("use_batch_norm", True)),
        min_var=float(checkpoint.get("min_var", 1e-12)),
        input_mean=float(checkpoint.get("input_mean", 0.0)),
        input_std=float(checkpoint.get("input_std", 1.0)),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_sv_model(
    checkpoint_path,
    model_kind="auto",
    device=None,
    label=None,
    prefix=None,
    prefer_best=True,
    used_prefixes=None,
):
    checkpoint_path = resolve_existing_path(checkpoint_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch_load_checkpoint(checkpoint_path, map_location=device)
    validate_target_names(checkpoint)

    model_kind = infer_model_kind(checkpoint, model_kind)
    state_dict = state_dict_from_checkpoint(checkpoint, prefer_best=prefer_best)

    if model_kind == "summary":
        model = build_summary_nn(checkpoint, state_dict, device)
    elif model_kind == "cnn":
        model = build_cnn(checkpoint, state_dict, device)
    else:
        raise ValueError("model_kind must be 'auto', 'summary', or 'cnn'.")

    if label is None:
        label = checkpoint_path.stem

    if prefix is None:
        prefix = sanitize_prefix(label)
    else:
        prefix = sanitize_prefix(prefix)

    if used_prefixes is not None:
        prefix = make_unique_prefix(prefix, used_prefixes)

    return LoadedSVModel(
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        model_kind=model_kind,
        label=label,
        prefix=prefix,
        device=device,
    )


# ============================================================
# Model inputs
# ============================================================

def infer_summary_config(checkpoint):
    input_dim = int(checkpoint["input_dim"])

    if "n_acvf_ratios" in checkpoint and "compute_arima_coeff" in checkpoint:
        n_acvf_ratios = int(checkpoint["n_acvf_ratios"])
        compute_arima_coeff = bool(checkpoint["compute_arima_coeff"])
        feature_names = sim.summary_stats_sv_feature_names(
            n_acvf_ratios=n_acvf_ratios,
            compute_arima_coeff=compute_arima_coeff,
        )

        if len(feature_names) == input_dim:
            return n_acvf_ratios, compute_arima_coeff

    for n_acvf_ratios, compute_arima_coeff in [
        (4, True),
        (4, False),
        (8, True),
        (8, False),
    ]:
        feature_names = sim.summary_stats_sv_feature_names(
            n_acvf_ratios=n_acvf_ratios,
            compute_arima_coeff=compute_arima_coeff,
        )

        if len(feature_names) == input_dim:
            return n_acvf_ratios, compute_arima_coeff

    raise ValueError(
        f"Could not infer summary-statistic setup for input_dim={input_dim}."
    )


def compute_summary_matrix(y, checkpoint):
    y = validate_series_matrix(y)

    n_acvf_ratios, compute_arima_coeff = infer_summary_config(checkpoint)
    input_dim = int(checkpoint["input_dim"])
    summaries = np.empty((y.shape[0], input_dim), dtype=np.float32)

    k = float(checkpoint.get("k", DEFAULT_K))
    eps = float(checkpoint.get("eps", 1e-12))
    center_y = bool(checkpoint.get("center_y", True))

    for i in range(y.shape[0]):
        summaries[i] = sim.summary_stats_sv(
            y[i],
            k=k,
            n_acvf_ratios=n_acvf_ratios,
            eps=eps,
            compute_arima_coeff=compute_arima_coeff,
            center_y=center_y,
        ).astype(np.float32, copy=False)

    return summaries


def prepare_summary_input(y, checkpoint):
    summaries = compute_summary_matrix(y, checkpoint)

    z_mean = checkpoint.get("z_mean")
    z_std = checkpoint.get("z_std")

    if z_mean is None:
        z_mean = np.zeros((1, summaries.shape[1]), dtype=np.float32)

    if z_std is None:
        z_std = np.ones((1, summaries.shape[1]), dtype=np.float32)

    z_mean = np.asarray(z_mean, dtype=np.float32)
    z_std = np.asarray(z_std, dtype=np.float32)
    z_std = np.where(z_std < 1e-8, 1.0, z_std)

    return ((summaries - z_mean) / z_std).astype(np.float32, copy=False)


def prepare_cnn_input(y, checkpoint):
    y = validate_series_matrix(y)

    expected_length = int(checkpoint["sequence_length"])

    if y.shape[1] != expected_length:
        raise ValueError(
            f"CNN checkpoint expects sequence_length={expected_length}, "
            f"but y has length {y.shape[1]}."
        )

    k = float(checkpoint.get("k", DEFAULT_K))
    center_y = bool(checkpoint.get("center_y", True))

    if center_y:
        y = y - np.mean(y, axis=1, keepdims=True)

    return np.log(y * y + k).astype(np.float32, copy=False)


def prepare_model_input(loaded_model, y):
    if loaded_model.model_kind == "summary":
        return prepare_summary_input(y, loaded_model.checkpoint)

    if loaded_model.model_kind == "cnn":
        return prepare_cnn_input(y, loaded_model.checkpoint)

    raise ValueError(f"Unknown model_kind: {loaded_model.model_kind}")


# ============================================================
# Model prediction
# ============================================================

@torch.no_grad()
def predict_transformed_gaussian(model, x, device, batch_size=4096):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    model.eval()

    means = []
    variances = []

    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        x_batch = torch.from_numpy(x[start:stop]).float().to(device)
        mean_batch, var_batch = model(x_batch)

        means.append(mean_batch.detach().cpu().float().numpy())
        variances.append(var_batch.detach().cpu().float().numpy())

    return np.vstack(means), np.vstack(variances)


def transformed_gaussian_to_parameter_frame(
    transformed_mean,
    transformed_var,
    alpha=DEFAULT_ALPHA,
    prefix="model",
):
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    transformed_mean = np.asarray(transformed_mean, dtype=np.float64)
    transformed_var = np.asarray(transformed_var, dtype=np.float64)

    if transformed_mean.shape != transformed_var.shape:
        raise ValueError("transformed_mean and transformed_var must have the same shape.")

    if transformed_mean.ndim != 2 or transformed_mean.shape[1] != 3:
        raise ValueError("transformed_mean and transformed_var must have shape (m, 3).")

    if np.any(transformed_var < 0.0):
        raise ValueError("transformed_var must be non-negative.")

    zcrit = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    transformed_sd = np.sqrt(transformed_var)

    mu_mean = transformed_mean[:, 0]
    mu_sd = transformed_sd[:, 0]
    mu_lower = mu_mean - zcrit * mu_sd
    mu_upper = mu_mean + zcrit * mu_sd

    psi_mean = transformed_mean[:, 1]
    psi_sd = transformed_sd[:, 1]
    phi_median = np.tanh(psi_mean / 2.0)
    phi_lower = np.tanh((psi_mean - zcrit * psi_sd) / 2.0)
    phi_upper = np.tanh((psi_mean + zcrit * psi_sd) / 2.0)

    log_sigma_mean = transformed_mean[:, 2]
    log_sigma_sd = transformed_sd[:, 2]
    sigma_median = np.exp(log_sigma_mean)
    sigma_lower = np.exp(log_sigma_mean - zcrit * log_sigma_sd)
    sigma_upper = np.exp(log_sigma_mean + zcrit * log_sigma_sd)

    return pd.DataFrame({
        f"{prefix}_alpha": alpha,
        f"{prefix}_credible_level": 1.0 - alpha,
        f"{prefix}_mu_median": mu_mean,
        f"{prefix}_mu_ci_lower": mu_lower,
        f"{prefix}_mu_ci_upper": mu_upper,
        f"{prefix}_mu_mean_transformed": mu_mean,
        f"{prefix}_mu_sd_transformed": mu_sd,
        f"{prefix}_phi_median": phi_median,
        f"{prefix}_phi_ci_lower": phi_lower,
        f"{prefix}_phi_ci_upper": phi_upper,
        f"{prefix}_psi_mean": psi_mean,
        f"{prefix}_psi_sd": psi_sd,
        f"{prefix}_sigma_median": sigma_median,
        f"{prefix}_sigma_ci_lower": sigma_lower,
        f"{prefix}_sigma_ci_upper": sigma_upper,
        f"{prefix}_log_sigma_mean": log_sigma_mean,
        f"{prefix}_log_sigma_sd": log_sigma_sd,
    })


def predict_sv_parameters(loaded_model, y, alpha=DEFAULT_ALPHA, batch_size=4096):
    x = prepare_model_input(loaded_model, y)
    transformed_mean, transformed_var = predict_transformed_gaussian(
        model=loaded_model.model,
        x=x,
        device=loaded_model.device,
        batch_size=batch_size,
    )

    return transformed_gaussian_to_parameter_frame(
        transformed_mean=transformed_mean,
        transformed_var=transformed_var,
        alpha=alpha,
        prefix=loaded_model.prefix,
    )


# ============================================================
# Simulation grid
# ============================================================

def make_finance_parameter_grid(baseline=None, sweeps=None, n_replicates=1):
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least 1.")

    if baseline is None:
        baseline = {
            "mu": -9.0,
            "phi": 0.98,
            "sigma": 0.20,
        }

    if sweeps is None:
        sweeps = {
            "mu": np.linspace(-11.0, -7.0, 9),
            "phi": np.array([0.90, 0.93, 0.95, 0.97, 0.98, 0.985, 0.99]),
            "sigma": np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]),
        }

    rows = []

    for parameter_name, values in sweeps.items():
        if parameter_name not in PARAMETER_NAMES:
            raise ValueError(f"Unknown sweep parameter: {parameter_name}")

        for value_index, value in enumerate(values):
            for replicate in range(n_replicates):
                theta = dict(baseline)
                theta[parameter_name] = float(value)

                rows.append({
                    "series_index": len(rows) + 1,
                    "sweep": parameter_name,
                    "sweep_value_index": value_index,
                    "replicate": replicate,
                    "sweep_value": float(value),
                    "mu_true": theta["mu"],
                    "phi_true": theta["phi"],
                    "sigma_true": theta["sigma"],
                })

    return pd.DataFrame(rows)


def simulate_grid(grid, n_obs=253, seed=12345, random_init=True, exp_clip=350.0):
    if n_obs < 1:
        raise ValueError("n_obs must be at least 1.")

    rng = np.random.default_rng(seed)

    return sim.simulate_sv_chunk(
        mu=grid["mu_true"].to_numpy(dtype=np.float64),
        phi=grid["phi_true"].to_numpy(dtype=np.float64),
        sigma=grid["sigma_true"].to_numpy(dtype=np.float64),
        n=n_obs,
        rng=rng,
        random_init=random_init,
        exp_clip=exp_clip,
    )


# ============================================================
# Plotting
# ============================================================

def format_tick(value, parameter):
    if parameter == "mu":
        return f"{value:.1f}"
    if parameter == "phi":
        return f"{value:.3f}"
    return f"{value:.2f}"


def plot_parameter_ci(comparison, parameter, output_dir, model_specs, show=False):
    import matplotlib.pyplot as plt

    subset = comparison[comparison["sweep"] == parameter].copy()
    subset = subset.sort_values(["sweep_value_index", "replicate"])

    x = np.arange(len(subset))
    methods = [("mcmc", "stochvol MCMC")]
    methods.extend((model.prefix, model.label) for model in model_specs)

    offsets = np.linspace(-0.28, 0.28, len(methods))

    if len(methods) == 1:
        offsets = np.array([0.0])

    fig, ax = plt.subplots(figsize=(11, 6))

    for x_offset, (prefix, label) in zip(offsets, methods):
        center = subset[f"{prefix}_{parameter}_median"].to_numpy()
        lower = subset[f"{prefix}_{parameter}_ci_lower"].to_numpy()
        upper = subset[f"{prefix}_{parameter}_ci_upper"].to_numpy()

        ax.errorbar(
            x + x_offset,
            center,
            yerr=[
                center - lower,
                upper - center,
            ],
            fmt="o",
            capsize=5,
            label=label,
        )

    ax.scatter(
        x,
        subset[f"{parameter}_true"].to_numpy(),
        marker="x",
        s=80,
        color="black",
        label=f"true {parameter}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_tick(value, parameter) for value in subset["sweep_value"]],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel(f"True {parameter}")
    ax.set_ylabel(f"Posterior median and CI for {parameter}")
    ax.set_title(f"{parameter}: amortized model(s) vs stochvol MCMC")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f"{parameter}_ci_comparison.png"
    fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    plt.close(fig)

    return output_path


# ============================================================
# Experiment
# ============================================================

def load_requested_models(
    summary_checkpoint_paths=None,
    cnn_checkpoint_paths=None,
    checkpoint_paths=None,
    device=None,
    prefer_best=True,
):
    used_prefixes = set()
    loaded_models = []

    for checkpoint_path in summary_checkpoint_paths or []:
        loaded_models.append(
            load_sv_model(
                checkpoint_path=checkpoint_path,
                model_kind="summary",
                device=device,
                prefer_best=prefer_best,
                used_prefixes=used_prefixes,
            )
        )

    for checkpoint_path in cnn_checkpoint_paths or []:
        loaded_models.append(
            load_sv_model(
                checkpoint_path=checkpoint_path,
                model_kind="cnn",
                device=device,
                prefer_best=prefer_best,
                used_prefixes=used_prefixes,
            )
        )

    for checkpoint_path in checkpoint_paths or []:
        loaded_models.append(
            load_sv_model(
                checkpoint_path=checkpoint_path,
                model_kind="auto",
                device=device,
                prefer_best=prefer_best,
                used_prefixes=used_prefixes,
            )
        )

    if len(loaded_models) == 0:
        loaded_models.append(
            load_sv_model(
                checkpoint_path=DEFAULT_SUMMARY_CHECKPOINT_PATH,
                model_kind="summary",
                device=device,
                prefer_best=prefer_best,
                used_prefixes=used_prefixes,
            )
        )

    return loaded_models


def run_comparison_experiment(
    summary_checkpoint_paths=None,
    cnn_checkpoint_paths=None,
    checkpoint_paths=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    mcmc_prior="finance",
    draws=2000,
    burnin=500,
    thinpara=1,
    alpha=DEFAULT_ALPHA,
    n_obs=253,
    seed=12345,
    n_replicates=1,
    prediction_batch_size=4096,
    device=None,
    prefer_best=True,
    show_plots=False,
    sweeps=None,
):
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    loaded_models = load_requested_models(
        summary_checkpoint_paths=summary_checkpoint_paths,
        cnn_checkpoint_paths=cnn_checkpoint_paths,
        checkpoint_paths=checkpoint_paths,
        device=device,
        prefer_best=prefer_best,
    )

    grid = make_finance_parameter_grid(
        sweeps=sweeps,
        n_replicates=n_replicates,
    )
    y = simulate_grid(grid, n_obs=n_obs, seed=seed)

    print("Loaded model checkpoints:")

    for loaded_model in loaded_models:
        print(
            f"  {loaded_model.label} "
            f"({loaded_model.model_kind}, device={loaded_model.device}) "
            f"from {loaded_model.checkpoint_path}"
        )

    print(f"Simulated {len(grid)} series with n_obs={n_obs}.")
    print(
        f"Running stochvol MCMC with prior='{mcmc_prior}', "
        f"draws={draws}, burnin={burnin}, thinpara={thinpara}, alpha={alpha}."
    )

    model_frames = []

    for loaded_model in loaded_models:
        model_frames.append(
            predict_sv_parameters(
                loaded_model=loaded_model,
                y=y,
                alpha=alpha,
                batch_size=prediction_batch_size,
            )
        )

    mcmc_df = run_stochvol_mcmc(
        y,
        prior=mcmc_prior,
        draws=draws,
        burnin=burnin,
        thinpara=thinpara,
        alpha=alpha,
    )
    mcmc_df = mcmc_df.sort_values("index").reset_index(drop=True)
    mcmc_df = mcmc_df.add_prefix("mcmc_")

    output_dir = resolve_output_dir(output_dir)

    comparison = pd.concat(
        [
            grid.reset_index(drop=True),
            mcmc_df.reset_index(drop=True),
            *[frame.reset_index(drop=True) for frame in model_frames],
        ],
        axis=1,
    )

    model_label = "_".join(model.prefix for model in loaded_models)
    comparison_path = output_dir / f"{model_label}_vs_stochvol_mcmc_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    plot_paths = []

    for parameter in PARAMETER_NAMES:
        plot_paths.append(
            plot_parameter_ci(
                comparison=comparison,
                parameter=parameter,
                output_dir=output_dir,
                model_specs=loaded_models,
                show=show_plots,
            )
        )

    print(f"Saved comparison table to {comparison_path}")

    for plot_path in plot_paths:
        print(f"Saved plot to {plot_path}")

    return comparison, plot_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare summary NN and CNN SV posterior models against stochvol MCMC."
    )
    parser.add_argument(
        "--summary-checkpoint",
        action="append",
        default=None,
        help="Summary-NN checkpoint path. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--cnn-checkpoint",
        action="append",
        default=None,
        help="CNN/TCN checkpoint path. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint path with model type inferred automatically. Can be supplied multiple times.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mcmc-prior", default="finance", choices=["default", "finance"])
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--thinpara", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--n-obs", type=int, default=253)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-replicates", type=int, default=1)
    parser.add_argument("--prediction-batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-final-weights", action="store_true")
    parser.add_argument("--show-plots", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    run_comparison_experiment(
        summary_checkpoint_paths=args.summary_checkpoint,
        cnn_checkpoint_paths=args.cnn_checkpoint,
        checkpoint_paths=args.checkpoint,
        output_dir=args.output_dir,
        mcmc_prior=args.mcmc_prior,
        draws=args.draws,
        burnin=args.burnin,
        thinpara=args.thinpara,
        alpha=args.alpha,
        n_obs=args.n_obs,
        seed=args.seed,
        n_replicates=args.n_replicates,
        prediction_batch_size=args.prediction_batch_size,
        device=args.device,
        prefer_best=not args.use_final_weights,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()
