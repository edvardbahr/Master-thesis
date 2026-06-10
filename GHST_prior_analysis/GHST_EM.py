from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, special


def _find_samples_csv() -> Path:
    """Find the eta_hat sample file from common project working directories."""

    here = Path(__file__).resolve()
    candidates = (
        Path.cwd() / "ghst_eta_hat_samples.csv",
        here.parent / "ghst_eta_hat_samples.csv",
        here.parents[2] / "ghst_eta_hat_samples.csv",
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find ghst_eta_hat_samples.csv in the cwd, beside this file, "
        "or at the workspace root."
    )


ghst_samples = np.genfromtxt(_find_samples_csv(), delimiter=",", skip_header=1)


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


if __name__ == "__main__":
    result = fit_ghst_em(ghst_samples, max_iter=1000, tol=1e-7, verbose=True)
    print_fit_summary(result)

    