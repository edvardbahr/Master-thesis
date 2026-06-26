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
import sim_5_param_data as sim
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_live_CNN import SVGHST_TARGET_NAMES, SVPosteriorTCN


HERE = Path(__file__).resolve().parent

DEFAULT_ALPHA = 0.05
DEFAULT_SEQUENCE_LENGTH = 253 * 20
DEFAULT_PRIOR_DRAWS = 1000
DEFAULT_MCMC_DRAWS = 2000
DEFAULT_MCMC_BURNIN = 500
DEFAULT_MCMC_THINPARA = 1
DEFAULT_MCMC_MAX_CORES = -2
GAUSSIAN_NLL_EPS = 1e-6

PARAMETER_NAMES = ("mu", "phi", "s", "r", "nu")
MCMC_COMPARISON_PARAMETERS = ("mu", "phi", "s")
TRANSFORMED_TARGET_NAMES = tuple(SVGHST_TARGET_NAMES)
MCMC_TRANSFORMED_TARGET_NAMES = ("mu", "psi", "log_s")
METHOD_NAMES = ("MCMC", "TCN")

DEFAULT_BASELINE = {
    "mu": -9.0,
    "phi": 0.95,
    "s": 0.25,
    "r": 0.50,
    "nu": 15.0,
}
DEFAULT_SWEEP_DELTAS = {
    "mu": 3.0,
    "phi": 0.045,
    "s": 0.20,
    "r": 0.30,
    "nu": 7.0,
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
        candidates = [root / path, root / "weights" / path]
        for candidate in candidates:
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
    target_names = tuple(checkpoint["target_names"])

    if target_names != TRANSFORMED_TARGET_NAMES:
        raise ValueError(
            f"{checkpoint_path} has target_names={target_names}; expected "
            f"{TRANSFORMED_TARGET_NAMES}. This analysis needs all five parameters."
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

    model = SVPosteriorTCN(
        tcn_channels=tuple(checkpoint["tcn_channels"]),
        kernel_size=int(checkpoint["kernel_size"]),
        dilations=tuple(checkpoint["dilations"]),
        hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
        activation=activation_from_checkpoint(checkpoint),
        use_batch_norm=bool(checkpoint["use_batch_norm"]),
        param_names=tuple(checkpoint["target_names"]),
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


def validate_original_parameters(theta):
    theta = np.asarray(theta, dtype=np.float64)

    if theta.ndim != 2 or theta.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(
            f"theta must have shape (n, {len(PARAMETER_NAMES)}) with columns "
            f"{PARAMETER_NAMES}."
        )

    return theta


def transform_five_parameters(theta, target_names=TRANSFORMED_TARGET_NAMES, eps=1e-6):
    theta = validate_original_parameters(theta)
    target_names = tuple(target_names)

    unknown_target_names = set(target_names).difference(TRANSFORMED_TARGET_NAMES)
    if unknown_target_names:
        raise ValueError(f"Unknown target name(s): {sorted(unknown_target_names)}.")

    nu_min = sim.get_gh_skew_t_prior_constants("default").nu_min

    mu = theta[:, 0]
    phi = np.clip(theta[:, 1], -1.0 + eps, 1.0 - eps)
    s = np.clip(theta[:, 2], eps, None)
    r = np.clip(theta[:, 3], eps, 1.0 - eps)
    nu_minus_min = np.clip(theta[:, 4] - nu_min, eps, None)

    transformed = {
        "mu": mu,
        "psi": 2.0 * np.arctanh(phi),
        "log_s": np.log(s),
        "logit_r": np.log(r / (1.0 - r)),
        "log_nu": np.log(nu_minus_min),
    }

    return np.column_stack([transformed[name] for name in target_names])


def simulate_prior_sv_dataset(
    n_prior_draws=DEFAULT_PRIOR_DRAWS,
    n=DEFAULT_SEQUENCE_LENGTH,
    seed=12345,
    random_init=True,
):
    rng = np.random.default_rng(seed)
    mu, phi, s, r, nu = sim.sample_stochvol_prior(
        n_prior_draws,
        rng=rng,
        prior="default",
        return_s2=False,
        dtype=np.float64,
    )
    y = sim.simulate_sv_chunk(
        mu=mu,
        phi=phi,
        s=s,
        r=r,
        nu=nu,
        n=n,
        rng=rng,
        random_init=random_init,
    )

    theta = np.column_stack([mu, phi, s, r, nu])
    transformed_targets = transform_five_parameters(theta)

    return y, theta, transformed_targets


def make_single_parameter_sweep_datasets(
    baseline=None,
    sweeps=None,
    sweep_deltas=None,
    sweep_size=9,
    n=DEFAULT_SEQUENCE_LENGTH,
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
        params[swept_parameter] = np.asarray(sweeps[swept_parameter], dtype=np.float64)

        datasets[swept_parameter] = sim.simulate_sv_chunk(
            mu=params["mu"],
            phi=params["phi"],
            s=params["s"],
            r=params["r"],
            nu=params["nu"],
            n=n,
            rng=rng,
            random_init=random_init,
        )

    return datasets, sweeps, baseline


def prepare_tcn_input(y, checkpoint):
    y = validate_series_matrix(y)
    expected_length = int(checkpoint["sequence_length"])

    if y.shape[1] != expected_length and False:
        raise ValueError(
            f"TCN checkpoint expects sequence_length={expected_length}, "
            f"but y has length {y.shape[1]}."
        )

    if bool(checkpoint["center_y"]):
        y = y - np.mean(y, axis=1, keepdims=True)

    return np.log(y * y + float(checkpoint["k"])).astype(np.float32, copy=False)


def prepare_model_input(loaded_model, y):
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


def predict_model_transformed_gaussian(loaded_model, y, batch_size):
    x = prepare_model_input(loaded_model, y)
    return predict_transformed_gaussian(
        model=loaded_model.model,
        x=x,
        device=loaded_model.device,
        batch_size=batch_size,
    )


def inverse_logit(x):
    x = np.asarray(x, dtype=np.float64)
    positive = x >= 0.0
    out = np.empty_like(x, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def transformed_gaussian_to_ci_frame(transformed_mean, transformed_var, alpha):
    transformed_mean = np.asarray(transformed_mean, dtype=np.float64)
    transformed_var = np.asarray(transformed_var, dtype=np.float64)

    if transformed_mean.shape[1] != len(TRANSFORMED_TARGET_NAMES):
        raise ValueError(
            f"Expected {len(TRANSFORMED_TARGET_NAMES)} transformed columns, "
            f"got {transformed_mean.shape[1]}."
        )

    zcrit = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    transformed_sd = np.sqrt(transformed_var)
    nu_min = sim.get_gh_skew_t_prior_constants("default").nu_min

    mu_mean = transformed_mean[:, 0]
    mu_sd = transformed_sd[:, 0]

    psi_mean = transformed_mean[:, 1]
    psi_sd = transformed_sd[:, 1]

    log_s_mean = transformed_mean[:, 2]
    log_s_sd = transformed_sd[:, 2]

    logit_r_mean = transformed_mean[:, 3]
    logit_r_sd = transformed_sd[:, 3]

    log_nu_mean = transformed_mean[:, 4]
    log_nu_sd = transformed_sd[:, 4]

    return pd.DataFrame({
        "alpha": alpha,
        "credible_level": 1.0 - alpha,
        "mu_median": mu_mean,
        "mu_ci_lower": mu_mean - zcrit * mu_sd,
        "mu_ci_upper": mu_mean + zcrit * mu_sd,
        "phi_median": np.tanh(psi_mean / 2.0),
        "phi_ci_lower": np.tanh((psi_mean - zcrit * psi_sd) / 2.0),
        "phi_ci_upper": np.tanh((psi_mean + zcrit * psi_sd) / 2.0),
        "s_median": np.exp(log_s_mean),
        "s_ci_lower": np.exp(log_s_mean - zcrit * log_s_sd),
        "s_ci_upper": np.exp(log_s_mean + zcrit * log_s_sd),
        "r_median": inverse_logit(logit_r_mean),
        "r_ci_lower": inverse_logit(logit_r_mean - zcrit * logit_r_sd),
        "r_ci_upper": inverse_logit(logit_r_mean + zcrit * logit_r_sd),
        "nu_median": nu_min + np.exp(log_nu_mean),
        "nu_ci_lower": nu_min + np.exp(log_nu_mean - zcrit * log_nu_sd),
        "nu_ci_upper": nu_min + np.exp(log_nu_mean + zcrit * log_nu_sd),
    })


def predict_model_ci(loaded_model, y, alpha, batch_size):
    mean, var = predict_model_transformed_gaussian(loaded_model, y, batch_size)
    return transformed_gaussian_to_ci_frame(mean, var, alpha)


def marginal_gaussian_metrics(
    target,
    mean,
    var,
    method,
    parameter_names,
    eps=GAUSSIAN_NLL_EPS,
):
    target = np.asarray(target, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    parameter_names = tuple(parameter_names)

    if target.shape != mean.shape or target.shape != var.shape:
        raise ValueError(
            f"target, mean, and var must have equal shapes; got "
            f"{target.shape}, {mean.shape}, and {var.shape}."
        )

    if target.ndim != 2 or target.shape[1] != len(parameter_names):
        raise ValueError(
            f"target must have shape (n, {len(parameter_names)})."
        )

    var = np.clip(var, eps, None)

    target_t = torch.as_tensor(target, dtype=torch.float64)
    mean_t = torch.as_tensor(mean, dtype=torch.float64)
    var_t = torch.as_tensor(var, dtype=torch.float64)

    nll = F.gaussian_nll_loss(
        input=mean_t,
        target=target_t,
        var=var_t,
        full=True,
        reduction="none",
        eps=eps,
    ).mean(dim=0)

    mse = torch.mean((mean_t - target_t) ** 2, dim=0)
    mean_variance = torch.mean(var_t, dim=0)

    rows = []
    for index, parameter in enumerate(parameter_names):
        rows.append({
            "method": method,
            "parameter": parameter,
            "mse": float(mse[index].item()),
            "mean_variance": float(mean_variance[index].item()),
            "negative_log_score": float(nll[index].item()),
        })

    return pd.DataFrame(rows)


def blank_metric_rows(method, parameter_names):
    return pd.DataFrame([
        {
            "method": method,
            "parameter": parameter,
            "mse": np.nan,
            "mean_variance": np.nan,
            "negative_log_score": np.nan,
        }
        for parameter in parameter_names
    ])


def transformed_mcmc_summary_arrays(summary):
    summary = summary.sort_values("index").reset_index(drop=True)

    mean = np.column_stack([
        summary[f"transformed_{parameter}_mean"].to_numpy(dtype=np.float64)
        for parameter in MCMC_TRANSFORMED_TARGET_NAMES
    ])
    var = np.column_stack([
        summary[f"transformed_{parameter}_var"].to_numpy(dtype=np.float64)
        for parameter in MCMC_TRANSFORMED_TARGET_NAMES
    ])

    return mean, var


def build_transformed_estimate_frame(theta, targets, estimates_by_method):
    frame = pd.DataFrame(theta, columns=PARAMETER_NAMES)

    for index, parameter in enumerate(TRANSFORMED_TARGET_NAMES):
        frame[f"target_{parameter}"] = targets[:, index]

    for method, estimate in estimates_by_method.items():
        method_key = method.lower()
        mean = estimate["mean"]
        var = estimate["var"]
        parameter_names = tuple(estimate["parameter_names"])

        for index, parameter in enumerate(parameter_names):
            frame[f"{method_key}_{parameter}_mean"] = mean[:, index]
            frame[f"{method_key}_{parameter}_var"] = var[:, index]

    return frame


def print_transformed_metric_table(metrics):
    table = metrics.pivot(
        index="parameter",
        columns="method",
        values=["mse", "mean_variance", "negative_log_score"],
    )
    table = table.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

    print("\nTransformed-scale marginal Gaussian metrics:")
    with pd.option_context("display.width", 180):
        print(
            table.to_string(
                float_format=lambda value: f"{value:.6g}",
                na_rep="",
            )
        )


def add_ci_rows(rows, swept_parameter, true_values, method, ci_frame, ci_parameter=None):
    ci_frame = ci_frame.reset_index(drop=True)
    ci_parameter = swept_parameter if ci_parameter is None else ci_parameter

    for value_index, true_value in enumerate(true_values):
        rows.append({
            "swept_parameter": swept_parameter,
            "value_index": value_index,
            "true_value": float(true_value),
            "method": method,
            "median": float(ci_frame.loc[value_index, f"{ci_parameter}_median"]),
            "ci_lower": float(ci_frame.loc[value_index, f"{ci_parameter}_ci_lower"]),
            "ci_upper": float(ci_frame.loc[value_index, f"{ci_parameter}_ci_upper"]),
        })


def run_parameter_sweep_test(
    tcn_model,
    baseline=None,
    sweep_deltas=None,
    sweeps=None,
    n=DEFAULT_SEQUENCE_LENGTH,
    sweep_size=9,
    seed=12345,
    alpha=DEFAULT_ALPHA,
    mcmc_draws=DEFAULT_MCMC_DRAWS,
    mcmc_burnin=DEFAULT_MCMC_BURNIN,
    mcmc_thinpara=DEFAULT_MCMC_THINPARA,
    mcmc_max_cores=DEFAULT_MCMC_MAX_CORES,
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

        if swept_parameter in MCMC_COMPARISON_PARAMETERS:
            mcmc_ci = run_stochvol_mcmc(
                y,
                prior="default",
                draws=mcmc_draws,
                burnin=mcmc_burnin,
                thinpara=mcmc_thinpara,
                alpha=alpha,
                max_cores=mcmc_max_cores,
            ).sort_values("index").reset_index(drop=True)
            mcmc_ci_parameter = "sigma" if swept_parameter == "s" else swept_parameter
            add_ci_rows(
                rows,
                swept_parameter,
                true_values,
                "MCMC",
                mcmc_ci,
                ci_parameter=mcmc_ci_parameter,
            )

        tcn_ci = predict_model_ci(tcn_model, y, alpha, batch_size)
        add_ci_rows(rows, swept_parameter, true_values, "TCN", tcn_ci)

    return pd.DataFrame(rows), sweeps, baseline


def run_prior_draw_metric_test(
    tcn_model,
    n_prior_draws=DEFAULT_PRIOR_DRAWS,
    n=DEFAULT_SEQUENCE_LENGTH,
    seed=12345,
    mcmc_draws=DEFAULT_MCMC_DRAWS,
    mcmc_burnin=DEFAULT_MCMC_BURNIN,
    mcmc_thinpara=DEFAULT_MCMC_THINPARA,
    mcmc_max_cores=DEFAULT_MCMC_MAX_CORES,
    alpha=DEFAULT_ALPHA,
    batch_size=4096,
):
    y, theta, targets = simulate_prior_sv_dataset(
        n_prior_draws=n_prior_draws,
        n=n,
        seed=seed,
    )

    print(
        "Running transformed-scale metrics for "
        f"{n_prior_draws} prior draws, sequence length {n}."
    )
    print(
        f"Running stochvol MCMC with draws={mcmc_draws}, "
        f"burnin={mcmc_burnin}, thinpara={mcmc_thinpara}."
    )

    mcmc_summary = run_stochvol_mcmc(
        y,
        prior="default",
        draws=mcmc_draws,
        burnin=mcmc_burnin,
        thinpara=mcmc_thinpara,
        alpha=alpha,
        max_cores=mcmc_max_cores,
    )
    mcmc_mean, mcmc_var = transformed_mcmc_summary_arrays(mcmc_summary)

    print("Predicting transformed posterior Gaussian with TCN.")
    tcn_mean, tcn_var = predict_model_transformed_gaussian(
        tcn_model,
        y,
        batch_size=batch_size,
    )

    mcmc_target_indices = [
        TRANSFORMED_TARGET_NAMES.index(parameter)
        for parameter in MCMC_TRANSFORMED_TARGET_NAMES
    ]
    metric_frames = [
        marginal_gaussian_metrics(
            target=targets[:, mcmc_target_indices],
            mean=mcmc_mean,
            var=mcmc_var,
            method="MCMC",
            parameter_names=MCMC_TRANSFORMED_TARGET_NAMES,
        ),
        blank_metric_rows(
            method="MCMC",
            parameter_names=tuple(
                parameter
                for parameter in TRANSFORMED_TARGET_NAMES
                if parameter not in MCMC_TRANSFORMED_TARGET_NAMES
            ),
        ),
        marginal_gaussian_metrics(
            target=targets,
            mean=tcn_mean,
            var=tcn_var,
            method="TCN",
            parameter_names=TRANSFORMED_TARGET_NAMES,
        ),
    ]

    estimates_by_method = {
        "MCMC": {
            "mean": mcmc_mean,
            "var": mcmc_var,
            "parameter_names": MCMC_TRANSFORMED_TARGET_NAMES,
        },
        "TCN": {
            "mean": tcn_mean,
            "var": tcn_var,
            "parameter_names": TRANSFORMED_TARGET_NAMES,
        },
    }

    metrics = pd.concat(metric_frames, ignore_index=True)
    estimates = build_transformed_estimate_frame(theta, targets, estimates_by_method)

    return metrics, estimates


def plot_parameter_sweep_ci(comparison, output_path, alpha):
    colors = {
        "MCMC": "tab:blue",
        "TCN": "tab:green",
    }
    markers = {
        "MCMC": "o",
        "TCN": "^",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    axes = axes.ravel()

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

        methods = [
            method for method in METHOD_NAMES
            if method in set(subset["method"])
        ]
        if len(methods) > 1:
            offsets = {
                method: offset * spacing
                for method, offset in zip(methods, np.linspace(-0.16, 0.16, len(methods)))
            }
        else:
            offsets = {method: 0.0 for method in methods}

        ax.plot(true_values, true_values, color="0.45", linewidth=1.0, linestyle="--")

        y_min = float(np.min(true_values))
        y_max = float(np.max(true_values))

        for method in methods:
            method_data = (
                subset[subset["method"] == method]
                .sort_values("value_index")
                .reset_index(drop=True)
            )

            x = method_data["true_value"].to_numpy() + offsets[method]
            y = method_data["median"].to_numpy()
            lower = method_data["ci_lower"].to_numpy()
            upper = method_data["ci_upper"].to_numpy()

            y_min = min(y_min, float(np.min(lower)))
            y_max = max(y_max, float(np.max(upper)))

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

    axes[len(PARAMETER_NAMES)].axis("off")
    axes[0].legend(loc="best")
    fig.suptitle(f"{100 * (1.0 - alpha):.0f}% credible intervals by swept parameter")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def main():
    tcn_checkpoint_path = "weights/svghst_posterior_tcn_live_default.pt"
    output_dir = Path("tcn_5_param_test")

    baseline = DEFAULT_BASELINE.copy()
    sweep_deltas = DEFAULT_SWEEP_DELTAS.copy()
    sweeps = None

    n = DEFAULT_SEQUENCE_LENGTH
    sweep_size = 10
    seed = 2
    metric_seed = 3
    alpha = DEFAULT_ALPHA

    mcmc_draws = DEFAULT_MCMC_DRAWS
    mcmc_burnin = DEFAULT_MCMC_BURNIN
    mcmc_thinpara = DEFAULT_MCMC_THINPARA
    mcmc_max_cores = DEFAULT_MCMC_MAX_CORES
    n_prior_draws = DEFAULT_PRIOR_DRAWS

    batch_size = 4096
    device_name = None

    device = torch.device(
        device_name if device_name is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tcn_model = load_tcn_model(tcn_checkpoint_path, device)

    output_dir.mkdir(parents=True, exist_ok=True)

    comparison, sweeps, baseline = run_parameter_sweep_test(
        tcn_model=tcn_model,
        baseline=baseline,
        sweep_deltas=sweep_deltas,
        sweeps=sweeps,
        n=n,
        sweep_size=sweep_size,
        seed=seed,
        alpha=alpha,
        mcmc_draws=mcmc_draws,
        mcmc_burnin=mcmc_burnin,
        mcmc_thinpara=mcmc_thinpara,
        mcmc_max_cores=mcmc_max_cores,
        batch_size=batch_size,
    )

    metrics, transformed_estimates = run_prior_draw_metric_test(
        tcn_model=tcn_model,
        n_prior_draws=n_prior_draws,
        n=n,
        seed=metric_seed,
        mcmc_draws=mcmc_draws,
        mcmc_burnin=mcmc_burnin,
        mcmc_thinpara=mcmc_thinpara,
        mcmc_max_cores=mcmc_max_cores,
        alpha=alpha,
        batch_size=batch_size,
    )

    csv_path = output_dir / "parameter_sweep_ci_default_comparison.csv"
    plot_path = output_dir / "parameter_sweep_ci_default_comparison.png"
    metrics_path = output_dir / "prior_draw_transformed_metrics.csv"
    estimates_path = output_dir / "prior_draw_transformed_estimates.csv"

    comparison.to_csv(csv_path, index=False)
    plot_parameter_sweep_ci(comparison, plot_path, alpha)
    metrics.to_csv(metrics_path, index=False)
    transformed_estimates.to_csv(estimates_path, index=False)

    print("Baseline:", baseline)
    print("Sweeps:", {key: values.tolist() for key, values in sweeps.items()})
    print_transformed_metric_table(metrics)
    print(f"Saved comparison CSV to {csv_path}")
    print(f"Saved CI plot to {plot_path}")
    print(f"Saved transformed metrics CSV to {metrics_path}")
    print(f"Saved transformed estimates CSV to {estimates_path}")


if __name__ == "__main__":
    main()
