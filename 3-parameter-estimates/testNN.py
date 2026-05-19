import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simulateData as sim
import torch
import torch.nn as nn
from trainSummaryNN import SVPosteriorNN


# Identify the path to the R script that performs stochvol MCMC estimation.
# The R script should be located in the same directory as this Python script.
HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "stochvolMCMC.R"

DEFAULT_CHECKPOINT_PATH = "sv_posterior_nn_1M_ARIMA_finance.pt"
DEFAULT_OUTPUT_DIR = "nn_vs_mcmc_comparison"
NORMAL_95_Z = 1.959963984540054


def find_rscript():
    rscript = shutil.which("Rscript")

    if rscript is not None:
        return rscript

    program_files = Path("C:/Program Files")
    candidates = sorted(
        program_files.glob("R/R-*/bin/Rscript.exe"),
        reverse=True,
    )

    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(
        "Could not find Rscript. Add R's bin folder to PATH, or install R."
    )


def resolve_existing_path(path):
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


def estimate_sv_with_r(y, draws=2000, burnin=500, thinpara=1, prior="default"):
    """
    Run stochvol MCMC in R for one or more simulated SV series.

    Parameters
    ----------
    y:
        One series with shape (n,) or many series with shape (m, n).

    prior:
        Prior constants to use in stochvol. Must be either "default" or
        "finance", matching simulateData.sample_stochvol_prior().

    Returns
    -------
    result:
        Structured NumPy array with one row per input series.
        The columns include posterior means, standard deviations, medians,
        and 95% posterior interval endpoints for mu, phi, and sigma.
    """
    y = np.asarray(y, dtype=float)

    if y.ndim == 1:
        y = y.reshape(1, -1)
    elif y.ndim != 2:
        raise ValueError("y must have shape (n,) or (m, n).")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")

    sim.get_stochvol_prior_constants(prior)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "y.csv"
        output_path = tmpdir / "sv_result.csv"

        np.savetxt(input_path, y, delimiter=",")

        subprocess.run(
            [
                find_rscript(),
                str(R_SCRIPT),
                str(input_path),
                str(output_path),
                str(draws),
                str(burnin),
                str(thinpara),
                str(prior),
            ],
            check=True,
        )

        result = np.genfromtxt(
            output_path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )

    return np.atleast_1d(result)


def structured_array_to_frame(records):
    records = np.atleast_1d(records)

    return pd.DataFrame({
        name: records[name]
        for name in records.dtype.names
    })


def activation_from_checkpoint(checkpoint):
    activation_name = checkpoint.get("activation", "ReLU")
    activation = getattr(nn, activation_name, None)

    if activation is None:
        raise ValueError(f"Unknown activation in checkpoint: {activation_name}")

    return activation


def load_summary_nn(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=None):
    checkpoint_path = resolve_existing_path(checkpoint_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    target_names = list(checkpoint.get("target_names", []))

    if target_names and target_names != ["mu", "psi", "log_sigma"]:
        raise ValueError(
            "This comparison expects target_names ['mu', 'psi', 'log_sigma'], "
            f"got {target_names}."
        )

    model = SVPosteriorNN(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims_shared_trunk=tuple(checkpoint["hidden_dims_shared_trunk"]),
        hidden_dims_head=tuple(checkpoint["hidden_dims_head"]),
        activation=activation_from_checkpoint(checkpoint),
        min_var=float(checkpoint.get("min_var", 1e-12)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, checkpoint_path, device


def infer_summary_config(checkpoint):
    input_dim = int(checkpoint["input_dim"])

    # Current trained summary models use n_acvf_ratios=4. Try nearby known
    # configurations so the loader fails loudly if a future checkpoint differs.
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
    y = np.asarray(y, dtype=np.float64)

    if y.ndim == 1:
        y = y.reshape(1, -1)

    n_acvf_ratios, compute_arima_coeff = infer_summary_config(checkpoint)
    input_dim = int(checkpoint["input_dim"])
    summaries = np.empty((y.shape[0], input_dim), dtype=np.float32)

    for i in range(y.shape[0]):
        summaries[i] = sim.summary_stats_sv(
            y[i],
            n_acvf_ratios=n_acvf_ratios,
            compute_arima_coeff=compute_arima_coeff,
        ).astype(np.float32, copy=False)

    return summaries


def predict_with_summary_nn(model, checkpoint, y, device, zcrit=NORMAL_95_Z):
    summaries = compute_summary_matrix(y, checkpoint)

    z_mean = checkpoint.get("z_mean")
    z_std = checkpoint.get("z_std")

    if z_mean is None:
        z_mean = np.zeros((1, summaries.shape[1]), dtype=np.float32)

    if z_std is None:
        z_std = np.ones((1, summaries.shape[1]), dtype=np.float32)

    z_std = np.where(z_std < 1e-8, 1.0, z_std)
    summaries_scaled = (summaries - z_mean) / z_std

    with torch.no_grad():
        z_tensor = torch.from_numpy(summaries_scaled).float().to(device)
        mean_t, var_t = model(z_tensor)

    transformed_mean = mean_t.detach().cpu().numpy()
    transformed_var = var_t.detach().cpu().numpy()
    transformed_sd = np.sqrt(transformed_var)

    mu_mean = transformed_mean[:, 0]
    mu_sd = transformed_sd[:, 0]
    mu_lower = mu_mean - zcrit * mu_sd
    mu_upper = mu_mean + zcrit * mu_sd

    psi_mean = transformed_mean[:, 1]
    psi_sd = transformed_sd[:, 1]
    phi_lower = np.tanh((psi_mean - zcrit * psi_sd) / 2.0)
    phi_upper = np.tanh((psi_mean + zcrit * psi_sd) / 2.0)
    phi_median = np.tanh(psi_mean / 2.0)

    log_sigma_mean = transformed_mean[:, 2]
    log_sigma_sd = transformed_sd[:, 2]
    sigma_lower = np.exp(log_sigma_mean - zcrit * log_sigma_sd)
    sigma_upper = np.exp(log_sigma_mean + zcrit * log_sigma_sd)
    sigma_median = np.exp(log_sigma_mean)

    return pd.DataFrame({
        "nn_mu_median": mu_mean,
        "nn_mu_ci_lower": mu_lower,
        "nn_mu_ci_upper": mu_upper,
        "nn_mu_sd_transformed": mu_sd,
        "nn_phi_median": phi_median,
        "nn_phi_ci_lower": phi_lower,
        "nn_phi_ci_upper": phi_upper,
        "nn_psi_mean": psi_mean,
        "nn_psi_sd": psi_sd,
        "nn_sigma_median": sigma_median,
        "nn_sigma_ci_lower": sigma_lower,
        "nn_sigma_ci_upper": sigma_upper,
        "nn_log_sigma_mean": log_sigma_mean,
        "nn_log_sigma_sd": log_sigma_sd,
    })


def make_financial_sweep_grid(baseline=None, sweeps=None, n_replicates=1):
    """
    Build a small set of finance-like SV scenarios.

    The baseline roughly corresponds to one-year daily equity/FX-style returns:
    exp(mu / 2) is about 1.1% daily volatility, phi is highly persistent, and
    sigma is moderate volatility-of-volatility.
    """
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

    for sweep_name, values in sweeps.items():
        if sweep_name not in baseline:
            raise ValueError(f"Unknown sweep parameter: {sweep_name}")

        for value_index, value in enumerate(values):
            for replicate in range(n_replicates):
                theta = dict(baseline)
                theta[sweep_name] = float(value)

                rows.append({
                    "series_index": len(rows) + 1,
                    "sweep": sweep_name,
                    "sweep_value_index": value_index,
                    "replicate": replicate,
                    "sweep_value": float(value),
                    "mu_true": theta["mu"],
                    "phi_true": theta["phi"],
                    "sigma_true": theta["sigma"],
                })

    return pd.DataFrame(rows)


def simulate_grid(grid, n_obs=253, seed=12345):
    rng = np.random.default_rng(seed)

    return sim.simulate_sv_chunk(
        mu=grid["mu_true"].to_numpy(dtype=np.float64),
        phi=grid["phi_true"].to_numpy(dtype=np.float64),
        sigma=grid["sigma_true"].to_numpy(dtype=np.float64),
        n=n_obs,
        rng=rng,
        random_init=True,
    )


def format_tick(value, parameter):
    if parameter == "mu":
        return f"{value:.1f}"
    if parameter == "phi":
        return f"{value:.3f}"
    return f"{value:.2f}"


def plot_parameter_ci(comparison, parameter, output_dir, model_label, show=False):
    subset = comparison[comparison["sweep"] == parameter].copy()
    subset = subset.sort_values(["sweep_value_index", "replicate"])

    x = np.arange(len(subset))
    offset = 0.14

    fig, ax = plt.subplots(figsize=(11, 6))

    for x_offset, prefix, label in [
        (-offset, "mcmc", "stochvol MCMC"),
        (offset, "nn", model_label),
    ]:
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
            capsize=6,
            label=label,
        )

    true_values = subset[f"{parameter}_true"].to_numpy()

    ax.scatter(
        x,
        true_values,
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
    ax.set_ylabel(f"Posterior median and 95% CI for {parameter}")
    ax.set_title(f"{parameter}: summary NN vs stochvol MCMC")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f"{parameter}_ci_comparison_finance.png"
    fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    plt.close(fig)

    return output_path


def run_comparison_experiment(
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    mcmc_prior="default",
    draws=2000,
    burnin=500,
    thinpara=1,
    n_obs=253,
    seed=12345,
    n_replicates=1,
    show_plots=False,
    sweeps=None,
):
    model, checkpoint, checkpoint_path, device = load_summary_nn(checkpoint_path)
    model_label = checkpoint_path.stem

    grid = make_financial_sweep_grid(
        sweeps=sweeps,
        n_replicates=n_replicates,
    )
    y = simulate_grid(grid, n_obs=n_obs, seed=seed)

    print(f"Loaded NN checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Simulated {len(grid)} series with n_obs={n_obs}.")
    print(
        f"Running stochvol MCMC with prior='{mcmc_prior}', "
        f"draws={draws}, burnin={burnin}, thinpara={thinpara}."
    )

    nn_df = predict_with_summary_nn(
        model=model,
        checkpoint=checkpoint,
        y=y,
        device=device,
    )

    mcmc_records = estimate_sv_with_r(
        y,
        draws=draws,
        burnin=burnin,
        thinpara=thinpara,
        prior=mcmc_prior,
    )
    mcmc_df = structured_array_to_frame(mcmc_records)
    mcmc_df = mcmc_df.sort_values("index").reset_index(drop=True)
    mcmc_df = mcmc_df.add_prefix("mcmc_")

    if len(mcmc_df) != len(grid):
        raise RuntimeError(
            f"MCMC returned {len(mcmc_df)} rows for {len(grid)} simulated series."
        )

    output_dir = Path(output_dir).expanduser()

    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.concat(
        [
            grid.reset_index(drop=True),
            nn_df.reset_index(drop=True),
            mcmc_df.reset_index(drop=True),
        ],
        axis=1,
    )

    comparison_path = output_dir / f"{model_label}_vs_stochvol_mcmc_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    plot_paths = []

    for parameter in ["mu", "phi", "sigma"]:
        plot_paths.append(
            plot_parameter_ci(
                comparison=comparison,
                parameter=parameter,
                output_dir=output_dir,
                model_label=model_label,
                show=show_plots,
            )
        )

    print(f"Saved comparison table to {comparison_path}")

    for plot_path in plot_paths:
        print(f"Saved plot to {plot_path}")

    return comparison, plot_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare a summary NN SV posterior model against stochvol MCMC."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mcmc-prior", default="default", choices=["default", "finance"])
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--thinpara", type=int, default=1)
    parser.add_argument("--n-obs", type=int, default=253)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-replicates", type=int, default=1)
    parser.add_argument("--show-plots", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    run_comparison_experiment(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        mcmc_prior=args.mcmc_prior,
        draws=args.draws,
        burnin=args.burnin,
        thinpara=args.thinpara,
        n_obs=args.n_obs,
        seed=args.seed,
        n_replicates=args.n_replicates,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()
