from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from scipy import optimize, special

SCRIPT_DIR = Path(__file__).resolve().parent
GHST_SAMPLE_PATH = SCRIPT_DIR / "ghst_eta_hat_samples.csv"


def load_ghst_samples(path: str | Path = GHST_SAMPLE_PATH) -> np.ndarray:
    sample_path = Path(path)
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Could not find residual samples at {sample_path}. "
            "Run RV_AR_1_estm.R first to create ghst_eta_hat_samples.csv."
        )
    return np.genfromtxt(sample_path, delimiter=",", skip_header=1)


ghst_samples = np.genfromtxt("ghst_eta_hat_samples.csv", delimiter=",", skip_header=1)


@dataclass(frozen=True)
class GHSTParameters:
    """Aas-Haff GH skew Student t parameters."""

    mu: float
    delta: float
    beta: float
    nu: float


@dataclass(frozen=True)
class CenteredGHSTParameters:
    """Centered simulator parameters used in five_param_estimates/sim_5_param_data.py."""

    mean: float
    s: float
    r: float
    nu: float
    beta_sign: float


@dataclass(frozen=True)
class GHSTEMResult:
    params: GHSTParameters
    centered_params: CenteredGHSTParameters
    loglikelihood: float
    converged: bool
    n_iter: int
    history: np.ndarray


def _clean_sample(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("Need at least 10 finite observations to fit GHST EM.")
    return x


def _log_kv(order: float | np.ndarray, z: np.ndarray) -> np.ndarray:
    """log(K_order(z)), using exponentially scaled K for better stability."""

    z = np.asarray(z, dtype=np.float64)
    z = np.maximum(z, np.finfo(np.float64).tiny)
    value = special.kve(order, z)
    return np.log(np.maximum(value, np.finfo(np.float64).tiny)) - z


def _d_log_kv_d_order(order: float, z: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Numerical derivative of log(K_order(z)) with respect to order."""

    return (_log_kv(order + h, z) - _log_kv(order - h, z)) / (2.0 * h)


def ghst_logpdf(x: np.ndarray, params: GHSTParameters) -> np.ndarray:
    """
    Log-density of the GH skew Student t distribution from Aas and Haff.

    The distribution is represented as
        X = mu + beta * W + sqrt(W) * Y,
        W ~ InvGamma(nu / 2, delta^2 / 2), Y ~ N(0, 1).
    """

    x = np.asarray(x, dtype=np.float64)
    mu, delta, beta, nu = params.mu, params.delta, params.beta, params.nu

    if delta <= 0.0 or nu <= 0.0:
        return np.full_like(x, -np.inf, dtype=np.float64)

    y = x - mu
    q = np.sqrt(delta * delta + y * y)
    a = 0.5 * nu
    m = 0.5 * (nu + 1.0)

    if abs(beta) < 1e-10:
        return (
            special.gammaln(m)
            - special.gammaln(a)
            - 0.5 * np.log(np.pi)
            - np.log(delta)
            - m * np.log1p((y * y) / (delta * delta))
        )

    abs_beta = abs(beta)
    z = abs_beta * q

    return (
        (1.0 - 0.5 * nu) * np.log(2.0)
        - 0.5 * np.log(2.0 * np.pi)
        + nu * np.log(delta)
        - special.gammaln(a)
        + m * np.log(abs_beta)
        - m * np.log(q)
        + _log_kv(m, z)
        + beta * y
    )


def ghst_loglikelihood(x: np.ndarray, params: GHSTParameters) -> float:
    logpdf = ghst_logpdf(x, params)
    if not np.all(np.isfinite(logpdf)):
        return -np.inf
    return float(np.sum(logpdf))


def sample_ghst(
    params: GHSTParameters,
    size: int | tuple[int, ...],
    rng: np.random.Generator,
    dtype=np.float64,
) -> np.ndarray:
    """
    Sample from the fitted Aas-Haff GH skew Student t distribution.

    This uses the same normal-variance-mean mixture convention as
    five_param_estimates/sim_5_param_data.py, but keeps the signed beta from
    the EM fit instead of using only the positive-skew centered convention.
    """

    mu, delta, beta, nu = params.mu, params.delta, params.beta, params.nu
    gamma_draw = rng.gamma(shape=0.5 * nu, scale=1.0, size=size)
    w = 0.5 * delta * delta / gamma_draw
    z = rng.standard_normal(size=size)
    samples = mu + beta * w + np.sqrt(w) * z

    return np.asarray(samples, dtype=dtype)


def _initial_params(x: np.ndarray, r0: float = 0.08) -> GHSTParameters:
    """
    Conservative moment-style initializer in the centered parameterization.

    The exact moment initializer in the paper is algebraically brittle. This
    initializer uses sample variance, skewness direction, and excess kurtosis to
    start EM in a plausible finite-variance region.
    """

    mean = float(np.mean(x))
    centered = x - mean
    var = float(np.mean(centered * centered))
    sd = float(np.sqrt(max(var, 1e-12)))
    skew = float(np.mean(centered**3) / max(var, 1e-12) ** 1.5)
    kurt_excess = float(np.mean(centered**4) / max(var, 1e-12) ** 2 - 3.0)

    # Symmetric Student t has excess kurtosis 6 / (nu - 4). Use this only as a
    # stable starting point; EM updates nu afterward.
    if kurt_excess > 0.05:
        nu = 4.0 + 6.0 / kurt_excess
    else:
        nu = 30.0
    nu = float(np.clip(nu, 4.5, 80.0))

    r = float(np.clip(r0 + min(abs(skew), 1.0) * 0.10, 0.02, 0.35))
    beta_sign = 1.0 if skew >= 0.0 else -1.0

    skew_scale = np.sqrt(0.5 * r * (nu - 4.0))
    delta = sd * np.sqrt((nu - 2.0) * (1.0 - r))
    beta = beta_sign * skew_scale / (sd * (1.0 - r))
    mu = mean - beta * delta * delta / (nu - 2.0)

    return GHSTParameters(mu=float(mu), delta=float(delta), beta=float(beta), nu=float(nu))


def _e_step(x: np.ndarray, params: GHSTParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute E[W | X], E[1/W | X], and E[log W | X].

    W is the inverse-gamma mixing variable in the GH skew Student t
    normal-variance-mean mixture.
    """

    mu, delta, beta, nu = params.mu, params.delta, params.beta, params.nu
    abs_beta = max(abs(beta), 1e-10)
    q = np.sqrt(delta * delta + (x - mu) ** 2)
    z = np.maximum(abs_beta * q, 1e-12)

    lambda_cond = -0.5 * (nu + 1.0)
    log_k_den = _log_kv(lambda_cond, z)

    xi = (q / abs_beta) * np.exp(_log_kv(lambda_cond + 1.0, z) - log_k_den)
    rho = (abs_beta / q) * np.exp(_log_kv(lambda_cond - 1.0, z) - log_k_den)
    chi = np.log(q / abs_beta) + _d_log_kv_d_order(lambda_cond, z)

    xi = np.maximum(xi, np.finfo(np.float64).tiny)
    rho = np.maximum(rho, np.finfo(np.float64).tiny)

    if not (np.all(np.isfinite(xi)) and np.all(np.isfinite(rho)) and np.all(np.isfinite(chi))):
        raise FloatingPointError("Non-finite values encountered in the GHST E-step.")

    return xi, rho, chi


def _solve_nu(
    rho: np.ndarray,
    chi: np.ndarray,
    *,
    min_nu: float = 4.000001,
    max_nu: float = 500.0,
) -> float:
    n = rho.size
    constant = np.log(n / np.sum(rho)) - float(np.mean(chi))

    def score(nu: float) -> float:
        return np.log(0.5 * nu) - special.digamma(0.5 * nu) + constant

    lower = min_nu
    upper = max(lower * 1.05, 8.0)
    f_lower = score(lower)
    f_upper = score(upper)

    while np.sign(f_lower) == np.sign(f_upper) and upper < max_nu:
        upper *= 1.8
        f_upper = score(upper)

    if np.sign(f_lower) != np.sign(f_upper):
        return float(optimize.brentq(score, lower, upper, xtol=1e-10, rtol=1e-10))

    # If the likelihood equation has no root in the finite-variance region, use
    # the closest boundary/minimum. This keeps the EM iteration numerically alive
    # and makes the fallback visible through the returned nu.
    objective = lambda v: score(float(v[0])) ** 2
    result = optimize.minimize(
        objective,
        x0=np.array([min(max(20.0, lower), max_nu)]),
        bounds=[(lower, max_nu)],
        method="L-BFGS-B",
    )
    return float(result.x[0])


def _m_step(x: np.ndarray, xi: np.ndarray, rho: np.ndarray, chi: np.ndarray) -> GHSTParameters:
    n = x.size
    x_bar = float(np.mean(x))
    xi_bar = float(np.mean(xi))
    sum_rho = float(np.sum(rho))
    sum_x_rho = float(np.sum(x * rho))

    denominator = n - xi_bar * sum_rho
    if abs(denominator) < 1e-12:
        raise FloatingPointError("Degenerate beta update in GHST M-step.")

    beta = (sum_x_rho - x_bar * sum_rho) / denominator
    mu = x_bar - beta * xi_bar
    nu = _solve_nu(rho, chi)
    delta = np.sqrt(n * nu / sum_rho)

    if delta <= 0.0 or nu <= 4.0 or not np.isfinite(beta + mu + delta + nu):
        raise FloatingPointError("Invalid parameter update in GHST M-step.")

    return GHSTParameters(mu=float(mu), delta=float(delta), beta=float(beta), nu=float(nu))


def aas_haff_to_centered(params: GHSTParameters) -> CenteredGHSTParameters:
    """Convert Aas-Haff parameters to mean, sd, skew-variance fraction, and nu."""

    mu, delta, beta, nu = params.mu, params.delta, params.beta, params.nu
    if nu <= 4.0:
        raise ValueError("nu must be greater than 4 to compute finite variance.")

    mean = mu + beta * delta * delta / (nu - 2.0)
    normal_var = delta * delta / (nu - 2.0)
    skew_var = 2.0 * beta * beta * delta**4 / ((nu - 2.0) ** 2 * (nu - 4.0))
    total_var = normal_var + skew_var
    s = np.sqrt(total_var)
    r = skew_var / total_var
    beta_sign = 1.0 if beta >= 0.0 else -1.0

    return CenteredGHSTParameters(
        mean=float(mean),
        s=float(s),
        r=float(r),
        nu=float(nu),
        beta_sign=float(beta_sign),
    )


def centered_to_aas_haff(
    s: float,
    r: float,
    nu: float,
    *,
    mean: float = 0.0,
    beta_sign: float = 1.0,
) -> GHSTParameters:
    """Convert centered simulator-style parameters to Aas-Haff parameters."""

    if s <= 0.0:
        raise ValueError("s must be positive.")
    if not (0.0 <= r < 1.0):
        raise ValueError("r must satisfy 0 <= r < 1.")
    if nu <= 4.0:
        raise ValueError("nu must be greater than 4.")

    sign = 1.0 if beta_sign >= 0.0 else -1.0
    skew_scale = np.sqrt(0.5 * r * (nu - 4.0))
    delta = s * np.sqrt((nu - 2.0) * (1.0 - r))
    beta = sign * skew_scale / (s * (1.0 - r))
    mu = mean - beta * delta * delta / (nu - 2.0)

    return GHSTParameters(mu=float(mu), delta=float(delta), beta=float(beta), nu=float(nu))


def fit_ghst_em(
    x: np.ndarray,
    *,
    initial_params: GHSTParameters | None = None,
    max_iter: int = 500,
    tol: float = 1e-7,
    verbose: bool = True,
) -> GHSTEMResult:
    x = _clean_sample(x)
    params = initial_params if initial_params is not None else _initial_params(x)
    previous_loglik = ghst_loglikelihood(x, params)

    history: list[tuple[float, float, float, float, float]] = [
        (0.0, previous_loglik, params.mu, params.delta, params.beta, params.nu)
    ]

    converged = False
    for iteration in range(1, max_iter + 1):
        xi, rho, chi = _e_step(x, params)
        new_params = _m_step(x, xi, rho, chi)
        loglik = ghst_loglikelihood(x, new_params)

        if loglik + 1e-7 < previous_loglik and verbose:
            print(
                "Warning: log-likelihood decreased by "
                f"{previous_loglik - loglik:.6g} at iteration {iteration}."
            )

        rel_change = abs(loglik - previous_loglik) / (1.0 + abs(previous_loglik))
        param_change = max(
            abs(new_params.mu - params.mu) / (1.0 + abs(params.mu)),
            abs(new_params.delta - params.delta) / (1.0 + abs(params.delta)),
            abs(new_params.beta - params.beta) / (1.0 + abs(params.beta)),
            abs(new_params.nu - params.nu) / (1.0 + abs(params.nu)),
        )

        history.append(
            (
                float(iteration),
                loglik,
                new_params.mu,
                new_params.delta,
                new_params.beta,
                new_params.nu,
            )
        )

        params = new_params
        previous_loglik = loglik

        if verbose and (iteration <= 5 or iteration % 25 == 0):
            print(
                f"iter={iteration:4d} loglik={loglik: .6f} "
                f"mu={params.mu: .6g} delta={params.delta: .6g} "
                f"beta={params.beta: .6g} nu={params.nu: .6g}"
            )

        if max(rel_change, param_change) < tol:
            converged = True
            break

    return GHSTEMResult(
        params=params,
        centered_params=aas_haff_to_centered(params),
        loglikelihood=previous_loglik,
        converged=converged,
        n_iter=iteration,
        history=np.asarray(history, dtype=np.float64),
    )


def print_fit_summary(result: GHSTEMResult) -> None:
    p = result.params
    c = result.centered_params

    print("\nGH skew Student t EM fit")
    print(f"converged:      {result.converged}")
    print(f"iterations:     {result.n_iter}")
    print(f"loglikelihood:  {result.loglikelihood:.6f}")

    print("\nAas-Haff parameters")
    print(f"mu:             {p.mu:.10g}")
    print(f"delta:          {p.delta:.10g}")
    print(f"beta:           {p.beta:.10g}")
    print(f"nu:             {p.nu:.10g}")

    print("\nCentered parameters for simulation")
    print(f"mean:           {c.mean:.10g}")
    print(f"s:              {c.s:.10g}")
    print(f"r:              {c.r:.10g}")
    print(f"nu:             {c.nu:.10g}")
    print(f"beta_sign:      {c.beta_sign:.0f}")

    if c.beta_sign < 0:
        print(
            "\nNote: beta_sign is negative. The current five-parameter simulator "
            "uses r >= 0 as a positive-skew parameter, so negative-skew fitting "
            "would need a signed-r extension or mirrored samples."
        )


def plot_histogram_vs_pdf(
    x: np.ndarray,
    result: GHSTEMResult,
    *,
    output_path: str | Path | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the empirical residual histogram against the fitted GHST pdf."""

    x = _clean_sample(x)
    owns_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
    else:
        fig = ax.figure

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    span = max(x_max - x_min, float(np.std(x)), 1e-8)
    grid = np.linspace(x_min - 0.06 * span, x_max + 0.06 * span, 1200)
    pdf = np.exp(ghst_logpdf(grid, result.params))

    ax.hist(
        x,
        bins="fd",
        density=True,
        color="#d8dee9",
        edgecolor="white",
        linewidth=0.8,
        label="Residual samples",
    )
    ax.plot(grid, pdf, color="#b22234", linewidth=2.2, label="Fitted GHST pdf")
    ax.axvline(float(np.mean(x)), color="#2f3b52", linestyle="--", linewidth=1.1, label="Sample mean")
    ax.set_title("Residual Histogram vs Fitted GHST PDF")
    ax.set_xlabel("Residual innovation")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#edf0f5", linewidth=0.8)

    if owns_figure:
        fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")

    return fig, ax


def plot_ghst_qq(
    x: np.ndarray,
    result: GHSTEMResult,
    *,
    n_sim: int = 500_000,
    seed: int = 20260610,
    output_path: str | Path | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Make a QQ plot using Monte Carlo quantiles from the fitted GHST model.

    The theoretical quantiles are empirical quantiles from a large simulated
    sample because this GHST parameterization does not have a simple inverse CDF.
    """

    x = _clean_sample(x)
    n_sim = int(n_sim)
    if n_sim < 10_000:
        raise ValueError("n_sim should be at least 10,000 for a stable GHST QQ plot.")

    owns_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 6.2))
    else:
        fig = ax.figure

    observed_quantiles = np.sort(x)
    probs = (np.arange(1, observed_quantiles.size + 1) - 0.5) / observed_quantiles.size
    rng = np.random.default_rng(seed)
    simulated = sample_ghst(result.params, size=n_sim, rng=rng)
    theoretical_quantiles = np.quantile(simulated, probs)

    ax.scatter(
        theoretical_quantiles,
        observed_quantiles,
        s=20,
        color="#234f7e",
        alpha=0.68,
        linewidths=0,
    )

    line_min = float(min(np.min(theoretical_quantiles), np.min(observed_quantiles)))
    line_max = float(max(np.max(theoretical_quantiles), np.max(observed_quantiles)))
    ax.plot([line_min, line_max], [line_min, line_max], color="#b22234", linewidth=1.8)
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("GHST QQ Plot")
    ax.set_xlabel("Fitted GHST quantiles")
    ax.set_ylabel("Observed residual quantiles")
    ax.grid(color="#edf0f5", linewidth=0.8)

    if owns_figure:
        fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")

    return fig, ax


def create_ghst_diagnostic_plots(
    x: np.ndarray,
    result: GHSTEMResult,
    *,
    output_dir: str | Path = SCRIPT_DIR / "plots",
    qq_sim_size: int = 500_000,
    seed: int = 20260610,
    show: bool = True,
) -> tuple[Path, Path]:
    """Create and save the two main fitted-GHST diagnostics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_path = output_dir / "ghst_histogram_vs_pdf.png"
    qq_path = output_dir / "ghst_qq_plot.png"

    hist_fig, _ = plot_histogram_vs_pdf(x, result, output_path=hist_path)
    qq_fig, _ = plot_ghst_qq(x, result, n_sim=qq_sim_size, seed=seed, output_path=qq_path)

    print("\nSaved GHST diagnostic plots")
    print(f"histogram vs pdf: {hist_path}")
    print(f"GHST QQ plot:     {qq_path}")

    if show:
        plt.show()
    else:
        plt.close(hist_fig)
        plt.close(qq_fig)

    return hist_path, qq_path


if __name__ == "__main__":
    result = fit_ghst_em(ghst_samples, max_iter=1000, tol=1e-7, verbose=True)
    print_fit_summary(result)
    create_ghst_diagnostic_plots(ghst_samples, result)
