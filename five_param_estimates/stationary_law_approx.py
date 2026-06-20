import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sim_5_param_data as sim


# Grid configuration. Rows vary phi and columns vary nu.
PHI_VALUES = np.array([0.0, 0.5, 0.8, 0.95, 0.99], dtype=np.float64)
NU_VALUES = np.array([8.1, 10.0, 15.0, 30.0, 100.0], dtype=np.float64)

# One 5x5 QQ figure is created for each value of K.
K_VALUES = (0, 5, 10, 20, 50)

SV_MU = -9.0
S = 1.0
R = 0.0

N_APPROX_DRAWS = 10_000
N_REFERENCE_DRAWS = 50_000

# For each phi, the reference K is large enough that |phi|**K is no larger
# than this tolerance. This avoids unnecessary work for low-persistence rows.
REFERENCE_RESIDUAL_TOLERANCE = 1e-8

SEED = 20260620
SHOW_PLOTS = False
OUTPUT_DIR = Path(__file__).resolve().parent / "stationary_law_approx_results"


def reference_burn_in_steps(phi, residual_tolerance):
    """
    Choose K so the coefficient on the initial approximation is negligible.

    After K transitions, the contribution from the initial state is phi**K.
    At phi=0, one transition gives the exact stationary law.
    """

    abs_phi = abs(float(phi))

    if abs_phi >= 1.0:
        raise ValueError("phi must satisfy abs(phi) < 1.")

    if not (0.0 < residual_tolerance < 1.0):
        raise ValueError("residual_tolerance must be between 0 and 1.")

    if abs_phi == 0.0:
        return 1

    return max(1, int(np.ceil(np.log(residual_tolerance) / np.log(abs_phi))))


def sample_hybrid_stationary_states(
    n_draws,
    phi,
    nu_values,
    steps_to_save,
    rng,
    mu=SV_MU,
    s=S,
    r=R,
):
    """
    Draw hybrid stationary approximations at selected warm-up steps.

    The initial state is Gaussian with the exact stationary mean and variance.
    Each warm-up transition replaces part of the Gaussian approximation with
    a centered GHST innovation. The returned arrays have shape
    (len(nu_values), n_draws).
    """

    if n_draws < 1:
        raise ValueError("n_draws must be at least 1.")

    phi = float(phi)
    mu = float(mu)
    s = float(s)
    r = float(r)
    nu_values = np.asarray(nu_values, dtype=np.float64)
    steps_to_save = tuple(sorted(set(int(step) for step in steps_to_save)))

    if abs(phi) >= 1.0:
        raise ValueError("phi must satisfy abs(phi) < 1.")

    if s <= 0.0:
        raise ValueError("s must be positive.")

    if not (0.0 <= r < 1.0):
        raise ValueError("r must satisfy 0 <= r < 1.")

    if nu_values.ndim != 1 or nu_values.size == 0:
        raise ValueError("nu_values must be a non-empty one-dimensional array.")

    if np.any(nu_values <= 4.0):
        raise ValueError("All nu values must be greater than 4.")

    if not steps_to_save or steps_to_save[0] < 0:
        raise ValueError("steps_to_save must contain non-negative integers.")

    shape = (nu_values.size, n_draws)
    stationary_sd = s / np.sqrt(1.0 - phi**2)
    h = mu + stationary_sd * rng.standard_normal(shape)

    s_values = np.full(shape, s, dtype=np.float64)
    r_values = np.full(shape, r, dtype=np.float64)
    nu_grid = np.broadcast_to(nu_values[:, np.newaxis], shape)

    states = {}

    if 0 in steps_to_save:
        states[0] = h.copy()

    steps_set = set(steps_to_save)

    for step in range(1, steps_to_save[-1] + 1):
        innovations = sim.sample_centered_gh_skew_t_innovations(
            s=s_values,
            r=r_values,
            nu=nu_grid,
            rng=rng,
            dtype=np.float64,
        )
        h = mu + phi * (h - mu) + innovations

        if step in steps_set:
            states[step] = h.copy()

    return states


def qq_probabilities(n_draws):
    """Return the plotting probabilities associated with sorted observations."""

    return (np.arange(1, n_draws + 1, dtype=np.float64) - 0.5) / n_draws


def empirical_quantiles(samples, probabilities):
    """Compute linearly interpolated empirical quantiles row by row."""

    samples = np.asarray(samples, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if samples.ndim != 2:
        raise ValueError("samples must be a two-dimensional array.")

    sorted_samples = np.sort(samples, axis=1)
    positions = probabilities * (samples.shape[1] - 1)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    upper_weight = positions - lower

    return (
        sorted_samples[:, lower] * (1.0 - upper_weight)
        + sorted_samples[:, upper] * upper_weight
    )


def generate_qq_quantiles(
    phi_values=PHI_VALUES,
    nu_values=NU_VALUES,
    k_values=K_VALUES,
    n_approx_draws=N_APPROX_DRAWS,
    n_reference_draws=N_REFERENCE_DRAWS,
    reference_tolerance=REFERENCE_RESIDUAL_TOLERANCE,
    seed=SEED,
):
    """
    Generate reference and hybrid quantiles for every grid cell and K.

    Returns
    -------
    reference_quantiles:
        Array with shape (n_phi, n_nu, n_approx_draws).

    hybrid_quantiles:
        Dictionary mapping K to an array with the same shape.

    reference_steps:
        Array containing the reference K used for each phi value.
    """

    phi_values = np.asarray(phi_values, dtype=np.float64)
    nu_values = np.asarray(nu_values, dtype=np.float64)
    k_values = tuple(sorted(set(int(k) for k in k_values)))

    if phi_values.ndim != 1 or phi_values.size == 0:
        raise ValueError("phi_values must be a non-empty one-dimensional array.")

    if nu_values.ndim != 1 or nu_values.size == 0:
        raise ValueError("nu_values must be a non-empty one-dimensional array.")

    if not k_values or k_values[0] < 0:
        raise ValueError("k_values must contain non-negative integers.")

    if n_reference_draws <= n_approx_draws:
        raise ValueError("n_reference_draws must be larger than n_approx_draws.")

    probabilities = qq_probabilities(n_approx_draws)
    quantile_shape = (phi_values.size, nu_values.size, n_approx_draws)
    reference_quantiles = np.empty(quantile_shape, dtype=np.float64)
    hybrid_quantiles = {
        k: np.empty(quantile_shape, dtype=np.float64)
        for k in k_values
    }
    reference_steps = np.empty(phi_values.size, dtype=int)

    seed_sequences = np.random.SeedSequence(seed).spawn(2 * phi_values.size)

    for phi_index, phi in enumerate(phi_values):
        reference_k = reference_burn_in_steps(phi, reference_tolerance)
        reference_steps[phi_index] = reference_k
        print(
            f"phi={phi:g}: generating reference with K={reference_k} "
            f"and {n_reference_draws:,} draws"
        )

        reference_rng = np.random.default_rng(seed_sequences[2 * phi_index])
        reference_state = sample_hybrid_stationary_states(
            n_draws=n_reference_draws,
            phi=phi,
            nu_values=nu_values,
            steps_to_save=(reference_k,),
            rng=reference_rng,
        )[reference_k]
        reference_quantiles[phi_index] = empirical_quantiles(
            reference_state,
            probabilities,
        )

        approx_rng = np.random.default_rng(seed_sequences[2 * phi_index + 1])
        approx_states = sample_hybrid_stationary_states(
            n_draws=n_approx_draws,
            phi=phi,
            nu_values=nu_values,
            steps_to_save=k_values,
            rng=approx_rng,
        )

        for k in k_values:
            hybrid_quantiles[k][phi_index] = np.sort(approx_states[k], axis=1)

    return reference_quantiles, hybrid_quantiles, reference_steps


def qq_error_metrics(reference_quantiles, hybrid_quantiles, stationary_sd):
    """Compute raw and stationary-SD-normalized QQ errors."""

    differences = hybrid_quantiles - reference_quantiles
    mean_absolute_error = float(np.mean(np.abs(differences)))
    rmse = float(np.sqrt(np.mean(differences**2)))
    maximum_absolute_error = float(np.max(np.abs(differences)))

    return {
        "mean_absolute_error": mean_absolute_error,
        "rmse": rmse,
        "maximum_absolute_error": maximum_absolute_error,
        "normalized_mean_absolute_error": mean_absolute_error / stationary_sd,
        "normalized_rmse": rmse / stationary_sd,
        "normalized_maximum_absolute_error": maximum_absolute_error / stationary_sd,
    }


def build_summary_rows(
    reference_quantiles,
    hybrid_quantiles,
    reference_steps,
    phi_values=PHI_VALUES,
    nu_values=NU_VALUES,
):
    """Build one convergence-summary row for each K, phi, and nu."""

    rows = []

    for k, k_quantiles in hybrid_quantiles.items():
        for phi_index, phi in enumerate(phi_values):
            stationary_sd = S / np.sqrt(1.0 - phi**2)

            for nu_index, nu in enumerate(nu_values):
                metrics = qq_error_metrics(
                    reference_quantiles[phi_index, nu_index],
                    k_quantiles[phi_index, nu_index],
                    stationary_sd,
                )
                rows.append(
                    {
                        "K": k,
                        "reference_K": int(reference_steps[phi_index]),
                        "phi": float(phi),
                        "nu": float(nu),
                        "mu": SV_MU,
                        "s": S,
                        "r": R,
                        **metrics,
                    }
                )

    return rows


def plot_qq_grid(
    k,
    reference_quantiles,
    hybrid_quantiles,
    reference_steps,
    output_path,
    phi_values=PHI_VALUES,
    nu_values=NU_VALUES,
    show=False,
):
    """Create and save one phi-by-nu grid of stationary-law QQ plots."""

    n_rows = len(phi_values)
    n_columns = len(nu_values)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.6 * n_columns, 3.4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for phi_index, phi in enumerate(phi_values):
        stationary_sd = S / np.sqrt(1.0 - phi**2)

        for nu_index, nu in enumerate(nu_values):
            ax = axes[phi_index, nu_index]
            theoretical = reference_quantiles[phi_index, nu_index]
            approximate = hybrid_quantiles[phi_index, nu_index]
            metrics = qq_error_metrics(theoretical, approximate, stationary_sd)

            line_min = float(min(np.min(theoretical), np.min(approximate)))
            line_max = float(max(np.max(theoretical), np.max(approximate)))

            ax.scatter(
                theoretical,
                approximate,
                s=5,
                alpha=0.45,
                color="#2364aa",
                edgecolors="none",
                rasterized=True,
            )
            ax.plot(
                [line_min, line_max],
                [line_min, line_max],
                color="#b22234",
                linewidth=1.2,
            )
            ax.text(
                0.04,
                0.94,
                f"W1/sd={metrics['normalized_mean_absolute_error']:.3g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )
            ax.grid(color="#e1e5ea", linewidth=0.6)

            if phi_index == 0:
                ax.set_title(f"nu = {nu:g}")

            if nu_index == 0:
                ax.set_ylabel(
                    f"phi = {phi:g} (reference K={reference_steps[phi_index]})\n"
                    f"Hybrid K={k} quantiles"
                )

            if phi_index == n_rows - 1:
                ax.set_xlabel("Reference quantiles")

    fig.suptitle(
        f"Hybrid stationary-law approximation: K={k}\n"
        f"mu={SV_MU:g}, s={S:g}, r={R:g}",
        fontsize=15,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)

    if show:
        plt.show()

    plt.close(fig)


def write_summary_csv(rows, output_path):
    """Write convergence metrics to a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_quantiles, hybrid_quantiles, reference_steps = generate_qq_quantiles()
    summary_rows = build_summary_rows(
        reference_quantiles,
        hybrid_quantiles,
        reference_steps,
    )

    summary_path = OUTPUT_DIR / "stationary_law_qq_errors.csv"
    write_summary_csv(summary_rows, summary_path)

    for k in K_VALUES:
        figure_path = OUTPUT_DIR / f"stationary_law_qq_K_{k:04d}.png"
        plot_qq_grid(
            k=k,
            reference_quantiles=reference_quantiles,
            hybrid_quantiles=hybrid_quantiles[k],
            reference_steps=reference_steps,
            output_path=figure_path,
            show=SHOW_PLOTS,
        )
        print(f"Saved QQ grid: {figure_path}")

    print(f"Saved convergence summary: {summary_path}")


if __name__ == "__main__":
    main()
