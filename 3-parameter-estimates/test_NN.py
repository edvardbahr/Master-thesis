import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from R_to_py_interface import run_stochvol_mcmc, validate_series_matrix
import simulate_data as sim
import torch
import torch.nn as nn
from train_live_CNN import LiveSVPosteriorTCN
from train_summary_NN import SVPosteriorNN


HERE = Path(__file__).resolve().parent

DEFAULT_ALPHA = 0.05

PARAMETER_NAMES = ("mu", "phi", "sigma")
TRANSFORMED_TARGET_NAMES = ["mu", "psi", "log_sigma"]
METHOD_NAMES = ("MCMC", "NN", "TCN")

DEFAULT_BASELINE = {
    "mu": -9.0,
    "phi": 0.98,
    "sigma": 0.20,
}
DEFAULT_SWEEP_DELTAS = {
    "mu": 2.0,
    "phi": 0.015,
    "sigma": 0.10,
}


@dataclass
class LoadedModel:
    label: str
    kind: str
    model: nn.Module
    checkpoint: dict
    device: torch.device


def resolve_path(path):
    path = Path(path)

    if path.is_absolute() and path.exists():
        return path

    search_roots = [Path.cwd(), HERE, *HERE.parents]
    for root in search_roots:
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find path: {path}")


def torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def require_keys(checkpoint, keys, checkpoint_path):
    missing = [key for key in keys if key not in checkpoint]
    if missing:
        raise KeyError(
            f"{checkpoint_path} is missing required checkpoint key(s): "
            + ", ".join(missing)
        )


def activation_from_checkpoint(checkpoint):
    activation_name = checkpoint["activation"]
    activation = getattr(nn, activation_name, None)

    if activation is None:
        raise ValueError(f"Unknown activation in checkpoint: {activation_name}")

    return activation


def validate_target_names(checkpoint, checkpoint_path):
    target_names = list(checkpoint["target_names"])

    if target_names != TRANSFORMED_TARGET_NAMES:
        raise ValueError(
            f"{checkpoint_path} has target_names={target_names}; expected "
            f"{TRANSFORMED_TARGET_NAMES}."
        )


def load_summary_model(checkpoint_path, device):
    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = torch_load_checkpoint(checkpoint_path, device)

    require_keys(
        checkpoint,
        [
            "model_class",
            "model_state_dict",
            "input_dim",
            "hidden_dims_shared_trunk",
            "hidden_dims_head",
            "activation",
            "min_var",
            "dropout",
            "layer_norm",
            "z_mean",
            "z_std",
            "target_names",
            "feature_names",
            "n_acvf_ratios",
            "compute_arima_coeff",
            "k",
            "eps",
            "center_y",
            "remove_NaNs",
        ],
        checkpoint_path,
    )

    if checkpoint["model_class"] != "SVPosteriorNN":
        raise ValueError(
            f"{checkpoint_path} has model_class={checkpoint['model_class']}; "
            "expected SVPosteriorNN."
        )

    validate_target_names(checkpoint, checkpoint_path)

    if not bool(checkpoint["center_y"]):
        raise ValueError(f"{checkpoint_path} must have center_y=True.")

    model = SVPosteriorNN(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims_shared_trunk=tuple(checkpoint["hidden_dims_shared_trunk"]),
        hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
        activation=activation_from_checkpoint(checkpoint),
        min_var=float(checkpoint["min_var"]),
        dropout=float(checkpoint["dropout"]),
        layer_norm=bool(checkpoint["layer_norm"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LoadedModel(
        label="NN",
        kind="summary",
        model=model,
        checkpoint=checkpoint,
        device=device,
    )


def load_tcn_model(checkpoint_path, device):
    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = torch_load_checkpoint(checkpoint_path, device)

    require_keys(
        checkpoint,
        [
            "model_class",
            "model_state_dict",
            "sequence_length",
            "tcn_channels",
            "kernel_size",
            "dilations",
            "hidden_dims_head",
            "activation",
            "dropout",
            "use_batch_norm",
            "min_var",
            "input_mean",
            "input_std",
            "target_names",
            "k",
            "center_y",
        ],
        checkpoint_path,
    )

    if checkpoint["model_class"] != "SVPosteriorTCN":
        raise ValueError(
            f"{checkpoint_path} has model_class={checkpoint['model_class']}; "
            "expected SVPosteriorTCN."
        )

    validate_target_names(checkpoint, checkpoint_path)

    if not bool(checkpoint["center_y"]):
        raise ValueError(f"{checkpoint_path} must have center_y=True.")

    model = LiveSVPosteriorTCN(
        sequence_length=int(checkpoint["sequence_length"]),
        tcn_channels=tuple(checkpoint["tcn_channels"]),
        kernel_size=int(checkpoint["kernel_size"]),
        dilations=tuple(checkpoint["dilations"]),
        hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
        activation=activation_from_checkpoint(checkpoint),
        dropout=float(checkpoint["dropout"]),
        use_batch_norm=bool(checkpoint["use_batch_norm"]),
        min_var=float(checkpoint["min_var"]),
        input_mean=float(checkpoint["input_mean"]),
        input_std=float(checkpoint["input_std"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LoadedModel(
        label="TCN",
        kind="tcn",
        model=model,
        checkpoint=checkpoint,
        device=device,
    )


def make_single_parameter_sweep_datasets(
    baseline=None,
    sweeps=None,
    sweep_deltas=None,
    sweep_size=9,
    n=253,
    rng=None,
    random_init=True,
):
    baseline = DEFAULT_BASELINE | (baseline or {})
    sweep_deltas = DEFAULT_SWEEP_DELTAS | (sweep_deltas or {})

    if rng is None:
        rng = np.random.default_rng()

    if sweeps is None:
        sweeps = {
            parameter: np.linspace(
                baseline[parameter] - sweep_deltas[parameter],
                baseline[parameter] + sweep_deltas[parameter],
                sweep_size,
            )
            for parameter in PARAMETER_NAMES
        }

    datasets = {}
    for swept_parameter in PARAMETER_NAMES:
        m = len(sweeps[swept_parameter])
        params = {
            parameter: np.full(m, baseline[parameter])
            for parameter in PARAMETER_NAMES
        }
        params[swept_parameter] = np.asarray(sweeps[swept_parameter])

        datasets[swept_parameter] = sim.simulate_sv_chunk(
            mu=params["mu"],
            phi=params["phi"],
            sigma=params["sigma"],
            n=n,
            rng=rng,
            random_init=random_init,
        )

    return datasets, sweeps, baseline


def prepare_summary_input(y, checkpoint):
    y = validate_series_matrix(y)

    expected_feature_names = sim.summary_stats_sv_feature_names(
        n_acvf_ratios=int(checkpoint["n_acvf_ratios"]),
        compute_arima_coeff=bool(checkpoint["compute_arima_coeff"]),
    )

    if list(checkpoint["feature_names"]) != expected_feature_names:
        raise ValueError("Checkpoint feature_names do not match simulateData.py.")

    summaries = np.empty((y.shape[0], int(checkpoint["input_dim"])), dtype=np.float32)

    for i in range(y.shape[0]):
        summaries[i] = sim.summary_stats_sv(
            y[i],
            k=float(checkpoint["k"]),
            n_acvf_ratios=int(checkpoint["n_acvf_ratios"]),
            eps=float(checkpoint["eps"]),
            compute_arima_coeff=bool(checkpoint["compute_arima_coeff"]),
            center_y=bool(checkpoint["center_y"]),
            remove_NaNs=bool(checkpoint["remove_NaNs"]),
        ).astype(np.float32, copy=False)

    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float32)
    z_std = np.asarray(checkpoint["z_std"], dtype=np.float32)
    z_std = np.where(z_std < 1e-8, 1.0, z_std)

    return ((summaries - z_mean) / z_std).astype(np.float32, copy=False)


def prepare_tcn_input(y, checkpoint):
    y = validate_series_matrix(y)
    expected_length = int(checkpoint["sequence_length"])

    if y.shape[1] != expected_length:
        raise ValueError(
            f"TCN checkpoint expects sequence_length={expected_length}, "
            f"but y has length {y.shape[1]}."
        )

    if bool(checkpoint["center_y"]):
        y = y - np.mean(y, axis=1, keepdims=True)

    return np.log(y * y + float(checkpoint["k"])).astype(np.float32, copy=False)


def prepare_model_input(loaded_model, y):
    if loaded_model.kind == "summary":
        return prepare_summary_input(y, loaded_model.checkpoint)

    if loaded_model.kind == "tcn":
        return prepare_tcn_input(y, loaded_model.checkpoint)

    raise ValueError(f"Unknown model kind: {loaded_model.kind}")


@torch.no_grad()
def predict_transformed_gaussian(model, x, device, batch_size=4096):
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


def transformed_gaussian_to_ci_frame(transformed_mean, transformed_var, alpha):
    transformed_mean = np.asarray(transformed_mean, dtype=np.float64)
    transformed_var = np.asarray(transformed_var, dtype=np.float64)

    zcrit = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    transformed_sd = np.sqrt(transformed_var)

    mu_mean = transformed_mean[:, 0]
    mu_sd = transformed_sd[:, 0]

    psi_mean = transformed_mean[:, 1]
    psi_sd = transformed_sd[:, 1]

    log_sigma_mean = transformed_mean[:, 2]
    log_sigma_sd = transformed_sd[:, 2]

    return pd.DataFrame({
        "alpha": alpha,
        "credible_level": 1.0 - alpha,
        "mu_median": mu_mean,
        "mu_ci_lower": mu_mean - zcrit * mu_sd,
        "mu_ci_upper": mu_mean + zcrit * mu_sd,
        "phi_median": np.tanh(psi_mean / 2.0),
        "phi_ci_lower": np.tanh((psi_mean - zcrit * psi_sd) / 2.0),
        "phi_ci_upper": np.tanh((psi_mean + zcrit * psi_sd) / 2.0),
        "sigma_median": np.exp(log_sigma_mean),
        "sigma_ci_lower": np.exp(log_sigma_mean - zcrit * log_sigma_sd),
        "sigma_ci_upper": np.exp(log_sigma_mean + zcrit * log_sigma_sd),
    })


def predict_model_ci(loaded_model, y, alpha, batch_size):
    x = prepare_model_input(loaded_model, y)
    mean, var = predict_transformed_gaussian(
        model=loaded_model.model,
        x=x,
        device=loaded_model.device,
        batch_size=batch_size,
    )

    return transformed_gaussian_to_ci_frame(mean, var, alpha)


def add_ci_rows(rows, swept_parameter, true_values, method, ci_frame):
    ci_frame = ci_frame.reset_index(drop=True)

    for value_index, true_value in enumerate(true_values):
        rows.append({
            "swept_parameter": swept_parameter,
            "value_index": value_index,
            "true_value": float(true_value),
            "method": method,
            "median": float(ci_frame.loc[value_index, f"{swept_parameter}_median"]),
            "ci_lower": float(ci_frame.loc[value_index, f"{swept_parameter}_ci_lower"]),
            "ci_upper": float(ci_frame.loc[value_index, f"{swept_parameter}_ci_upper"]),
        })


def run_parameter_sweep_test(
    summary_model,
    tcn_model,
    baseline=None,
    sweep_deltas=None,
    sweeps=None,
    n=253,
    sweep_size=9,
    seed=12345,
    alpha=DEFAULT_ALPHA,
    mcmc_prior="finance",
    mcmc_draws=2000,
    mcmc_burnin=500,
    mcmc_thinpara=1,
    batch_size=4096,
):
    rng = np.random.default_rng(seed)
    datasets, sweeps, baseline = make_single_parameter_sweep_datasets(
        baseline=baseline,
        sweeps=sweeps,
        sweep_deltas=sweep_deltas,
        n=n,
        sweep_size=sweep_size,
        rng=rng,
    )

    rows = []

    for swept_parameter in PARAMETER_NAMES:
        y = datasets[swept_parameter]
        true_values = sweeps[swept_parameter]

        print(f"Running sweep for {swept_parameter} with {len(true_values)} values.")

        mcmc_ci = run_stochvol_mcmc(
            y,
            prior=mcmc_prior,
            draws=mcmc_draws,
            burnin=mcmc_burnin,
            thinpara=mcmc_thinpara,
            alpha=alpha,
        ).sort_values("index").reset_index(drop=True)

        nn_ci = predict_model_ci(summary_model, y, alpha, batch_size)
        tcn_ci = predict_model_ci(tcn_model, y, alpha, batch_size)

        add_ci_rows(rows, swept_parameter, true_values, "MCMC", mcmc_ci)
        add_ci_rows(rows, swept_parameter, true_values, "NN", nn_ci)
        add_ci_rows(rows, swept_parameter, true_values, "TCN", tcn_ci)

    return pd.DataFrame(rows), sweeps, baseline


def plot_parameter_sweep_ci(comparison, output_path, alpha):
    colors = {
        "MCMC": "tab:blue",
        "NN": "tab:orange",
        "TCN": "tab:green",
    }
    markers = {
        "MCMC": "o",
        "NN": "s",
        "TCN": "^",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, swept_parameter in zip(axes, PARAMETER_NAMES):
        subset = comparison[comparison["swept_parameter"] == swept_parameter]
        true_values = (
            subset[["value_index", "true_value"]]
            .drop_duplicates()
            .sort_values("value_index")["true_value"]
            .to_numpy()
        )

        if len(true_values) > 1:
            spacing = float(np.min(np.diff(true_values)))
        else:
            spacing = 1.0

        offsets = {
            "MCMC": -0.18 * spacing,
            "NN": 0.0,
            "TCN": 0.18 * spacing,
        }

        ax.plot(true_values, true_values, color="0.45", linewidth=1.0, linestyle="--")

        y_min = np.inf
        y_max = -np.inf

        for method in METHOD_NAMES:
            method_data = (
                subset[subset["method"] == method]
                .sort_values("value_index")
                .reset_index(drop=True)
            )

            x = method_data["true_value"].to_numpy() + offsets[method]
            y = method_data["median"].to_numpy()
            lower = method_data["ci_lower"].to_numpy()
            upper = method_data["ci_upper"].to_numpy()

            y_min = min(y_min, float(np.min(lower)), float(np.min(true_values)))
            y_max = max(y_max, float(np.max(upper)), float(np.max(true_values)))

            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lower, upper - y]),
                fmt=markers[method],
                color=colors[method],
                elinewidth=1.5,
                capsize=3,
                markersize=4,
                label=method,
            )

        x_margin = max(abs(spacing) * 0.55, 1e-8)
        y_margin = max((y_max - y_min) * 0.08, 1e-8)

        ax.set_xlim(float(np.min(true_values)) - x_margin, float(np.max(true_values)) + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_title(f"{swept_parameter} sweep")
        ax.set_xlabel(f"true {swept_parameter}")
        ax.set_ylabel(f"posterior {swept_parameter}")
        ax.grid(alpha=0.25)

    axes[0].legend(loc="best")
    fig.suptitle(f"{100 * (1.0 - alpha):.0f}% credible intervals by swept parameter using default prior")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def main():
    summary_checkpoint_path = "sv_posterior_nn_1M_ARIMA.pt"
    tcn_checkpoint_path = "sv_posterior_tcn_live.best.pt"
    output_dir = Path("nn_parameter_sweep_test")

    baseline = {
        "mu": -9.0,
        "phi": 0.95,
        "sigma": 0.25,
    }
    sweep_deltas = {
        "mu": 3.0,
        "phi": 0.045,
        "sigma": 0.20,
    }
    sweeps = None

    n = 253
    sweep_size = 10
    seed = 2
    alpha = 0.05

    mcmc_prior = "default"
    mcmc_draws = 2000*10
    mcmc_burnin = 500*2
    mcmc_thinpara = 1

    batch_size = 4096
    device_name = None

    device = torch.device(
        device_name if device_name is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    summary_model = load_summary_model(summary_checkpoint_path, device)
    tcn_model = load_tcn_model(tcn_checkpoint_path, device)

    comparison, sweeps, baseline = run_parameter_sweep_test(
        summary_model=summary_model,
        tcn_model=tcn_model,
        baseline=baseline,
        sweep_deltas=sweep_deltas,
        sweeps=sweeps,
        n=n,
        sweep_size=sweep_size,
        seed=seed,
        alpha=alpha,
        mcmc_prior=mcmc_prior,
        mcmc_draws=mcmc_draws,
        mcmc_burnin=mcmc_burnin,
        mcmc_thinpara=mcmc_thinpara,
        batch_size=batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "parameter_sweep_ci_default_comparison.csv"
    plot_path = output_dir / "parameter_sweep_ci_default_comparison.png"

    comparison.to_csv(csv_path, index=False)
    plot_parameter_sweep_ci(comparison, plot_path, alpha)

    print("Baseline:", baseline)
    print("Sweeps:", {key: values.tolist() for key, values in sweeps.items()})
    print(f"Saved comparison CSV to {csv_path}")
    print(f"Saved CI plot to {plot_path}")


if __name__ == "__main__":
    main()
