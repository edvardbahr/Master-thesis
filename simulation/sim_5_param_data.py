from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acovf


_STATIONARY_INIT_BURN_IN_STEPS = 20


def _simulate_and_summarize_chunk(job):
    """Sample, simulate, and summarize one parallel-work chunk."""

    (
        _chunk_id,
        chunk_start,
        n_chunk,
        n,
        seed_seq,
        prior,
        fixed_r,
        fixed_nu,
        random_init,
        n_acvf_ratios,
        n_quantiles,
        compute_arima_coeff,
        k,
        eps,
        arima_method,
        center_y,
        remove_NaNs,
        out_dtype,
        exp_clip,
        p,
    ) = job

    rng = np.random.default_rng(seed_seq)

    mu, phi, s, r, nu = sample_stochvol_prior(
        n_chunk,
        rng=rng,
        prior=prior,
        fixed_r=fixed_r,
        fixed_nu=fixed_nu,
        return_s2=False,
        dtype=np.float64,
    )

    y_chunk = simulate_sv_chunk(
        mu=mu,
        phi=phi,
        s=s,
        r=r,
        nu=nu,
        n=n,
        rng=rng,
        random_init=random_init,
        dtype=np.float64,
        exp_clip=exp_clip,
    )

    summary_chunk = np.empty((n_chunk, p), dtype=out_dtype)

    # summary_stats_sv operates on one observed series at a time.
    for i in range(n_chunk):
        summary_chunk[i, :] = summary_stats_sv(
            y_chunk[i, :],
            k=k,
            n_acvf_ratios=n_acvf_ratios,
            n_quantiles=n_quantiles,
            eps=eps,
            compute_arima_coeff=compute_arima_coeff,
            arima_method=arima_method,
            center_y=center_y,
            remove_NaNs=remove_NaNs,
        ).astype(out_dtype, copy=False)

    # Keep the native five-parameter API.  Three-parameter consumers use the
    # leading (mu, phi, s) columns and ignore fixed r and nu.
    theta_chunk = np.column_stack([mu, phi, s, r, nu]).astype(
        out_dtype,
        copy=False,
    )

    return chunk_start, n_chunk, summary_chunk, theta_chunk


def summary_stats_sv(
    y,
    k=1e-12,
    n_acvf_ratios=4,
    n_quantiles=5,
    eps=1e-12,
    compute_arima_coeff=True,
    arima_method=None,
    center_y=True,
    remove_NaNs=True,
):
    """
    Compute the legacy standard-SV summary vector for one observed series.

    The feature order intentionally matches ``sim_3_param_data`` and existing
    summary-network checkpoints:

    1. mean and 5%-to-95% quantiles of ``log(y**2 + k)``;
    2. transformed consecutive autocovariance ratios;
    3. optionally, transformed ARMA(1, 1) AR/MA coefficients and log
       innovation standard deviation;
    4. log standard deviation, log MAD, and a plug-in log-s estimate.

    Parameters ``k`` and ``eps`` must be positive; ``eps`` must also be below
    one because it is used to clip values before ``atanh``.
    """

    def clip_unit(z):
        return np.clip(z, -1.0 + eps, 1.0 - eps)

    def safe_log(z):
        return np.log(np.maximum(z, eps))

    def psi_phi(z):
        return 2.0 * np.arctanh(clip_unit(z))

    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")

    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("k must be finite and positive.")

    if not np.isfinite(eps) or not 0.0 < eps < 1.0:
        raise ValueError("eps must be finite and satisfy 0 < eps < 1.")

    if not isinstance(n_acvf_ratios, int):
        raise TypeError("n_acvf_ratios must be an integer.")

    if n_acvf_ratios < 1:
        raise ValueError("n_acvf_ratios must be at least 1.")

    quantile_probs = summary_quantile_probabilities(n_quantiles)

    if len(y) <= n_acvf_ratios:
        raise ValueError("y is too short for the requested number of ACVF ratios.")

    if center_y:
        y = y - np.mean(y)

    x = np.log(y**2 + k)

    mean_x = np.mean(x)
    q_x = np.quantile(x, quantile_probs)

    gamma = acovf(
        x,
        adjusted=False,
        demean=True,
        fft=False,
        nlag=n_acvf_ratios,
        missing="raise",
    )

    numerator = gamma[1:n_acvf_ratios + 1]
    denominator = gamma[0:n_acvf_ratios]
    raw_ratios = np.divide(
        numerator,
        denominator,
        out=np.zeros(n_acvf_ratios, dtype=float),
        where=np.abs(denominator) > eps,
    )
    acvf_ratio_features = psi_phi(raw_ratios)

    var_x = np.var(x, ddof=1)
    mad_x = np.median(np.abs(x - np.median(x)))
    log_sd_x = 0.5 * safe_log(var_x)
    log_mad_x = safe_log(mad_x)

    if n_acvf_ratios >= 2:
        phi_proxy = clip_unit(np.median(raw_ratios[1:]))
    else:
        phi_proxy = clip_unit(raw_ratios[0])

    if compute_arima_coeff:
        sigma2_start = max(var_x * (1.0 - phi_proxy**2), eps)
        start_params = np.array(
            [
                mean_x,
                phi_proxy,
                0.0,
                sigma2_start,
            ]
        )

        try:
            model = ARIMA(
                x,
                order=(1, 0, 1),
                trend="c",
                enforce_stationarity=True,
                enforce_invertibility=True,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)

                if arima_method is None:
                    fit = model.fit(
                        start_params=start_params,
                        method="statespace",
                        method_kwargs={"maxiter": 50, "disp": 0},
                    )
                else:
                    fit = model.fit(
                        start_params=start_params,
                        method=arima_method,
                        method_kwargs={"maxiter": 50, "disp": 0},
                    )

            params = dict(zip(fit.param_names, fit.params))
            alpha_arma = clip_unit(params.get("ar.L1", 0.0))
            beta_arma = clip_unit(params.get("ma.L1", 0.0))
            arma_sigma2 = params.get("sigma2", np.var(fit.resid, ddof=1))
            arma_innov_sd = np.sqrt(max(arma_sigma2, eps))
            arma_features = np.array(
                [
                    psi_phi(alpha_arma),
                    psi_phi(beta_arma),
                    safe_log(arma_innov_sd),
                ],
                dtype=float,
            )
            phi_proxy = alpha_arma

        except Exception:
            # Retain the legacy deterministic fallback so the feature layout
            # stays usable if an individual ARMA fit does not converge.
            alpha_arma = phi_proxy
            beta_arma = 0.0
            arma_innov_sd = np.std(x, ddof=1)
            arma_features = np.array(
                [
                    psi_phi(alpha_arma),
                    psi_phi(beta_arma),
                    safe_log(arma_innov_sd),
                ],
                dtype=float,
            )
            phi_proxy = alpha_arma

    log_eps2_var = np.pi**2 / 2.0
    latent_var_est = max(var_x - log_eps2_var, eps)
    one_minus_r2 = max(1.0 - phi_proxy**2, eps)
    log_s_plugin = 0.5 * (
        np.log(latent_var_est)
        + np.log(one_minus_r2)
    )

    spread_features = np.array(
        [
            log_sd_x,
            log_mad_x,
            log_s_plugin,
        ],
        dtype=float,
    )

    if compute_arima_coeff:
        p = 1 + n_quantiles + n_acvf_ratios + 3 + 3
    else:
        p = 1 + n_quantiles + n_acvf_ratios + 3

    out = np.empty(p, dtype=float)
    i = 0

    out[i] = mean_x
    i += 1
    out[i:i + n_quantiles] = q_x
    i += n_quantiles
    out[i:i + n_acvf_ratios] = acvf_ratio_features
    i += n_acvf_ratios

    if compute_arima_coeff:
        out[i:i + 3] = arma_features
        i += 3

    out[i:i + 3] = spread_features

    if remove_NaNs:
        out[~np.isfinite(out)] = 0.0

    return out


def summary_quantile_probabilities(n_quantiles):
    """Return the evenly spaced probabilities used by summary_stats_sv."""

    if not isinstance(n_quantiles, int):
        raise TypeError("n_quantiles must be an integer.")

    if n_quantiles < 1:
        raise ValueError("n_quantiles must be at least 1.")

    return np.linspace(0.05, 0.95, n_quantiles)


def quantile_feature_name(probability):
    """Format one summary-quantile probability as its checkpoint feature name."""

    percentage = 100.0 * float(probability)
    if np.isclose(percentage, round(percentage)):
        return f"q{int(round(percentage)):02d}_x"

    label = f"{percentage:05.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"q{label}_x"


def summary_stats_sv_feature_names(
    n_acvf_ratios=4,
    n_quantiles=5,
    compute_arima_coeff=True,
):
    """Return names in exactly the order emitted by summary_stats_sv."""

    if not isinstance(n_acvf_ratios, int):
        raise TypeError("n_acvf_ratios must be an integer.")

    if n_acvf_ratios < 1:
        raise ValueError("n_acvf_ratios must be at least 1.")

    names = ["mean_x"]
    names.extend(
        quantile_feature_name(probability)
        for probability in summary_quantile_probabilities(n_quantiles)
    )
    names.extend(
        f"psi_gamma{j}_over_gamma{j-1}"
        for j in range(1, n_acvf_ratios + 1)
    )

    if compute_arima_coeff:
        names.extend(
            [
                "psi_alpha_arma",
                "psi_beta_arma",
                "log_arma_innov_sd",
            ]
        )

    # Keep the legacy name `log_sigma_plugin`; the five-parameter simulator's
    # scale `s` is the standard-SV sigma when r=0 and nu=np.inf.
    names.extend(
        [
            "log_sd_x",
            "log_mad_x",
            "log_sigma_plugin",
        ]
    )

    return names


@dataclass(frozen=True)
class GHSkewTPriorConstants:
    mu_mean: float
    mu_sd: float
    phi_a0: float
    phi_b0: float
    Bs: float
    r_a0: float | None
    r_b0: float | None
    r_max: float
    nu_min: float
    nu_rate: float


_GH_SKEW_T_PRIORS = {
     #TODO: Adjust the hyper parameters for nu and r to better match financial time series.
     "finance": GHSkewTPriorConstants(
         mu_mean=-9.0,
         mu_sd=1.0,
         phi_a0=20.0,
         phi_b0=1.5,
         Bs=1.0,

         # These are just the default parameter values for r and nu
         r_a0=1.0,
         r_b0=9.0,
         r_max=0.8,
         nu_min=8.0,
         nu_rate=0.1,
    ),
    "default": GHSkewTPriorConstants(
        mu_mean=0.0,
        mu_sd=10.0,
        phi_a0=5.0,
        phi_b0=1.5,
        Bs=1.0,
        r_a0=None,  # When r_a0, r_b0 is None, r is sampled
        r_b0=None,  # from a uniform distribution on [0, r_max).
        r_max=0.999999,
        nu_min=6.0,  # Under this condition skew exists ish
        nu_rate=0.1, # Picked so that there is a 10% chance of observing nu > 30 (approx Gaussian)
    ),
}


def get_gh_skew_t_prior_constants(prior="default"):
    """
    Return prior constants for the five-parameter SV model.

    The SV-level priors match the three-parameter model:

        mu ~ N(mu_mean, mu_sd^2)
        (phi + 1) / 2 ~ Beta(phi_a0, phi_b0)
        s^2 ~ Bs * ChiSq(df = 1)

    The GH skew-t innovation is parameterized by (s, nu, r), where r controls
    the positive-skew variance fraction and nu controls tail thickness:

        r / r_max ~ Beta(r_a0, r_b0), or
        r ~ Uniform(0, r_max) if r_a0 or r_b0 are None

    The tail parameter follows a shifted exponential distribution:

        nu = nu_min + X, X ~ Exponential(rate = nu_rate).
    """

    if prior not in _GH_SKEW_T_PRIORS:
        valid = ", ".join(_GH_SKEW_T_PRIORS)
        raise ValueError(f"Unknown prior '{prior}'. Valid choices are: {valid}.")

    return _GH_SKEW_T_PRIORS[prior]


def validate_gh_skew_t_prior_constants(hyper):
    """
    Validate prior constants whose constraints are needed by the sampler.
    """

    if not np.isfinite(hyper.r_max) or not (0.0 < hyper.r_max < 1.0):
        raise ValueError("r_max must satisfy 0 < r_max < 1.")

    has_r_a0 = hyper.r_a0 is not None
    has_r_b0 = hyper.r_b0 is not None

    if has_r_a0 != has_r_b0:
        raise ValueError("r_a0 and r_b0 must either both be specified or both be None.")

    if has_r_a0 and (
        not np.isfinite(hyper.r_a0)
        or not np.isfinite(hyper.r_b0)
        or hyper.r_a0 <= 0.0
        or hyper.r_b0 <= 0.0
    ):
        raise ValueError("r_a0 and r_b0 must be positive when specified.")

    if not np.isfinite(hyper.nu_min) or hyper.nu_min <= 4.0:
        raise ValueError("nu_min must be greater than 4.")

    if not np.isfinite(hyper.nu_rate) or hyper.nu_rate <= 0.0:
        raise ValueError("nu_rate must be positive.")


def sample_stochvol_prior(
    n,
    rng,
    prior="default",
    fixed_r=None,
    fixed_nu=None,
    return_s2=False,
    dtype=np.float64,
):
    """
    Sample from the prior for the five-parameter SV model.

    Parameterization:

        mu ~ N(mu_mean, mu_sd^2)
        (phi + 1) / 2 ~ Beta(phi_a0, phi_b0)
        s^2 ~ Bs * ChiSq(df = 1)
        r = fixed_r if fixed_r is not None, otherwise
        r / r_max ~ Beta(r_a0, r_b0), or
        r ~ Uniform(0, r_max) if r_a0 and r_b0 are both None
        nu = fixed_nu if fixed_nu is not None (np.inf gives Gaussian
        innovations), otherwise
        nu = nu_min + X, X ~ Exponential(rate = nu_rate)

    Returns
    -------
    If return_s2=False:
        mu, phi, s, r, nu

    If return_s2=True:
        mu, phi, s, r, nu, s2
    """

    if n < 1:
        raise ValueError("n must be at least 1.")

    hyper = get_gh_skew_t_prior_constants(prior)
    validate_gh_skew_t_prior_constants(hyper)

    if fixed_r is not None and (
        not np.isfinite(fixed_r) or not 0.0 <= fixed_r < 1.0
    ):
        raise ValueError("fixed_r must satisfy 0 <= fixed_r < 1.")

    if fixed_nu is not None and (
        not (np.isfinite(fixed_nu) or np.isposinf(fixed_nu))
        or fixed_nu <= 4.0
    ):
        raise ValueError("fixed_nu must be greater than 4 or np.inf.")

    mu = rng.normal(
        loc=hyper.mu_mean,
        scale=hyper.mu_sd,
        size=n,
    ).astype(dtype, copy=False)

    phi = (
        2.0 * rng.beta(
            a=hyper.phi_a0,
            b=hyper.phi_b0,
            size=n,
        )
        - 1.0
    ).astype(dtype, copy=False)

    s2 = (
        hyper.Bs * rng.chisquare(df=1.0, size=n)
    ).astype(dtype, copy=False)
    s = np.sqrt(s2).astype(dtype, copy=False)

    if fixed_r is None:
        if hyper.r_a0 is not None and hyper.r_b0 is not None:
            r = (
                hyper.r_max * rng.beta(
                    a=hyper.r_a0,
                    b=hyper.r_b0,
                    size=n,
                )
            ).astype(dtype, copy=False)
        else:
            r = (
                hyper.r_max * rng.uniform(low=0.0, high=1.0, size=n)
            ).astype(dtype, copy=False)
    else:
        r = np.full((n,), fixed_r, dtype=dtype)

    if fixed_nu is None:
        nu = (
            hyper.nu_min + rng.exponential(scale=1.0 / hyper.nu_rate, size=n)
        ).astype(dtype, copy=False)
    else:
        nu = np.full((n,), fixed_nu, dtype=dtype)

    if return_s2:
        return mu, phi, s, r, nu, s2

    return mu, phi, s, r, nu


def gh_skew_t_params_from_s_r_nu(s, r, nu):
    """
    Convert the interpretable centered parameterization (s, r, nu) to
    GH skew-t parameters (mu_GH, delta, beta).
    """

    s = np.asarray(s, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    s, r, nu = np.broadcast_arrays(s, r, nu)

    if np.any(s <= 0.0):
        raise ValueError("All s values must be positive.")

    if np.any((r < 0.0) | (r >= 1.0)):
        raise ValueError("All r values must satisfy 0 <= r < 1.")

    if np.any(nu <= 4.0):
        raise ValueError("All nu values must be greater than 4.")

    skew_scale = np.sqrt(0.5 * r * (nu - 4.0))
    delta = s * np.sqrt((nu - 2.0) * (1.0 - r))
    beta = skew_scale / (s * (1.0 - r))
    mu_gh = -s * skew_scale

    return mu_gh, delta, beta


def sample_centered_gh_skew_t_innovations(s, r, nu, rng, dtype=np.float64):
    """
    Sample centered GH skew-t innovations with mean 0 and standard deviation s.

    Entries with nu=np.inf are sampled directly from N(0, s^2).
    """

    s = np.asarray(s, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    s, r, nu = np.broadcast_arrays(s, r, nu)

    if np.any(s <= 0.0):
        raise ValueError("All s values must be positive.")

    if np.any((r < 0.0) | (r >= 1.0)):
        raise ValueError("All r values must satisfy 0 <= r < 1.")

    if np.any(~(np.isfinite(nu) | np.isposinf(nu))) or np.any(nu <= 4.0):
        raise ValueError("All nu values must be greater than 4 or np.inf.")

    gaussian = np.isposinf(nu)
    if np.all(gaussian):
        return np.asarray(
            s * rng.standard_normal(size=s.shape),
            dtype=dtype,
        )

    if not np.any(gaussian):
        mu_gh, delta, beta = gh_skew_t_params_from_s_r_nu(s, r, nu)
        gamma_draw = rng.gamma(shape=0.5 * nu, scale=1.0)
        # 1/gamma(shape = 0.5 * nu, rate = k) -> inv-gamma(shape = 0.5 * nu, scale = k)
        # 0.5 * delta^2 * inv-gamma(shape = 0.5 * nu, scale = 1.0) -> inv-gamma(shape = 0.5 * nu, scale = 0.5 * delta^2)
        w = 0.5 * delta * delta / gamma_draw
        z = rng.standard_normal(size=np.shape(w))
        innovations = mu_gh + beta * w + np.sqrt(w) * z

        return np.asarray(innovations, dtype=dtype)

    innovations = np.empty(s.shape, dtype=np.float64)
    innovations[gaussian] = (
        s[gaussian] * rng.standard_normal(size=np.count_nonzero(gaussian))
    )

    finite = ~gaussian
    mu_gh, delta, beta = gh_skew_t_params_from_s_r_nu(
        s[finite],
        r[finite],
        nu[finite],
    )
    gamma_draw = rng.gamma(shape=0.5 * nu[finite], scale=1.0)
    # 1/gamma(shape = 0.5 * nu, rate = k) -> inv-gamma(shape = 0.5 * nu, scale = k)
    # 0.5 * delta^2 * inv-gamma(shape = 0.5 * nu, scale = 1.0) -> inv-gamma(shape = 0.5 * nu, scale = 0.5 * delta^2)
    w = 0.5 * delta * delta / gamma_draw
    z = rng.standard_normal(size=np.shape(w))
    innovations[finite] = mu_gh + beta * w + np.sqrt(w) * z

    return np.asarray(innovations, dtype=dtype)


def simulate_sv_chunk(
    mu,
    phi,
    s,
    r,
    nu,
    n,
    rng,
    random_init=True,
    dtype=np.float64,
    exp_clip=350.0,
):
    """
    Simulate a chunk of stochastic-volatility series with centered GH skew-t
    innovations in the log-volatility process.

    Model:
        h_t = mu + phi * (h_{t-1} - mu) + eta_t
        eta_t ~ centered GH skew-t(s, r, nu)
        y_t = exp(h_t / 2) * eps_t
        eps_t ~ N(0, 1)

    Parameters
    ----------
    mu, phi, s, r, nu:
        Arrays of shape (m,), where m is the chunk size.

    n:
        Length of each time series.

    rng:
        np.random.Generator.

    random_init:
        If True, initialize from a Gaussian with the stationary mean and
        variance, then apply 20 GHST transitions before generating y_0.
        For Gaussian innovations (all nu=np.inf), no burn-in transitions are
        needed because this initialization is exactly stationary.

    dtype:
        Floating point type for the returned y array.

    exp_clip:
        Clips h_t / 2 before exponentiating to avoid overflow.

    Returns
    -------
    y:
        Array of shape (m, n).
    """

    mu = np.asarray(mu, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)

    if not (mu.shape == phi.shape == s.shape == r.shape == nu.shape):
        raise ValueError("mu, phi, s, r, and nu must have the same shape.")

    if mu.ndim != 1:
        raise ValueError("mu, phi, s, r, and nu must be one-dimensional arrays.")

    if n < 1:
        raise ValueError("n must be at least 1.")

    if np.any(np.abs(phi) >= 1.0):
        raise ValueError("All phi values must satisfy abs(phi) < 1.")

    if np.any(s <= 0.0):
        raise ValueError("All s values must be positive.")

    if np.any((r < 0.0) | (r >= 1.0)):
        raise ValueError("All r values must satisfy 0 <= r < 1.")

    if np.any(~(np.isfinite(nu) | np.isposinf(nu))) or np.any(nu <= 4.0):
        raise ValueError("All nu values must be greater than 4 or np.inf.")

    m = len(mu)
    y = np.empty((m, n), dtype=dtype)

    if random_init:
        stationary_sd = s / np.sqrt(1.0 - phi**2)
        h_prev = mu + stationary_sd * rng.standard_normal(m)

        stationary_init_burn_in_steps = (
            0 if np.all(np.isposinf(nu)) else _STATIONARY_INIT_BURN_IN_STEPS
        )

        for _ in range(stationary_init_burn_in_steps):
            h_prev = (
                mu
                + phi * (h_prev - mu)
                + sample_centered_gh_skew_t_innovations(
                    s,
                    r,
                    nu,
                    rng,
                    dtype=np.float64,
                )
            )
    else:
        h_prev = mu.copy()

    y[:, 0] = (
        np.exp(np.clip(0.5 * h_prev, -exp_clip, exp_clip))
        * rng.standard_normal(m)
    )

    for t in range(1, n):
        h_prev = (
            mu
            + phi * (h_prev - mu)
            + sample_centered_gh_skew_t_innovations(s, r, nu, rng, dtype=np.float64)
        )

        y[:, t] = (
            np.exp(np.clip(0.5 * h_prev, -exp_clip, exp_clip))
            * rng.standard_normal(m)
        )

    return y


def _simulate_log_y_squared_chunk(job):
    """
    Worker function for parallel log(y^2 + k) simulation.
    """

    (
        _chunk_id,
        chunk_start,
        n_chunk,
        n,
        seed_seq,
        prior,
        fixed_r,
        fixed_nu,
        random_init,
        k,
        center_y,
        out_dtype,
        exp_clip,
    ) = job

    rng = np.random.default_rng(seed_seq)

    mu, phi, s, r, nu = sample_stochvol_prior(
        n_chunk,
        rng=rng,
        prior=prior,
        fixed_r=fixed_r,
        fixed_nu=fixed_nu,
        return_s2=False,
        dtype=np.float64,
    )

    y_chunk = simulate_sv_chunk(
        mu=mu,
        phi=phi,
        s=s,
        r=r,
        nu=nu,
        n=n,
        rng=rng,
        random_init=random_init,
        dtype=np.float64,
        exp_clip=exp_clip,
    )

    if center_y:
        y_chunk = y_chunk - np.mean(y_chunk, axis=1, keepdims=True)

    log_y_squared_chunk = np.log(y_chunk * y_chunk + k).astype(out_dtype, copy=False)
    theta_chunk = np.column_stack([mu, phi, s, r, nu]).astype(out_dtype, copy=False)

    return chunk_start, n_chunk, log_y_squared_chunk, theta_chunk


def resolve_n_workers(n_workers):
    """
    Resolve the number of worker processes.

    Negative values work like offsets from the total CPU count:
        -1 uses all cores except one,
        -2 uses all cores except two,
        etc.
    """

    n_cpus = os.cpu_count() or 1

    if n_workers is None:
        raise ValueError(
            "n_workers cannot be None. Use a positive worker count or a negative CPU offset."
        )

    if n_workers < 0:
        resolved = n_cpus + n_workers

        if resolved < 1:
            raise ValueError(
                "n_workers leaves no worker processes available. "
                f"With {n_cpus} CPU core(s), use n_workers >= {1 - n_cpus}."
            )

        return resolved

    if n_workers == 0:
        raise ValueError("n_workers must not be 0.")

    if n_workers > n_cpus:
        raise ValueError(
            f"n_workers={n_workers} exceeds the available CPU count ({n_cpus})."
        )

    return n_workers


def resolve_chunk_size(N, n_workers, chunks_per_worker):
    """
    Compute a chunk size from the number of simulations, workers, and chunks
    per worker.
    """

    if N < 1:
        raise ValueError("N must be at least 1.")

    if n_workers < 1:
        raise ValueError("n_workers must be at least 1.")

    if chunks_per_worker < 1:
        raise ValueError("chunks_per_worker must be at least 1.")

    return max(1, int(np.ceil(N / (n_workers * chunks_per_worker))))


def simulate_sv_log_y_squared_parallel(
    N,
    n,
    chunk_size,
    fixed_r=None,
    fixed_nu=None,
    n_workers=-1,
    seed=1,
    prior="default",
    random_init=True,
    k=1e-12,
    center_y=True,
    out_dtype=np.float32,
    exp_clip=350.0,
    show_progress=True,
):
    """
    Generate log(y^2 + k) series and true five-parameter SV values in parallel.

    If fixed_r or fixed_nu is provided, that value is used for every simulated
    series instead of sampling the corresponding parameter from its prior.
    Setting fixed_nu=np.inf gives Gaussian log-volatility innovations.

    Returns
    -------
    log_y_squared:
        Matrix of shape (N, n), where row i is log(y_i^2 + k).

    theta:
        Parameter matrix of shape (N, 5), columns are mu, phi, s, r, nu.
    """

    if N < 1:
        raise ValueError("N must be at least 1.")

    if n < 1:
        raise ValueError("n must be at least 1.")

    n_workers = resolve_n_workers(n_workers)

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    if k <= 0:
        raise ValueError("k must be positive.")

    log_y_squared = np.empty((N, n), dtype=out_dtype)
    theta = np.empty((N, 5), dtype=out_dtype)

    n_chunks = np.ceil(N / chunk_size).astype(int)

    master_ss = np.random.SeedSequence(seed)
    child_seeds = master_ss.spawn(n_chunks)

    chunk_jobs = []

    for chunk_id in range(n_chunks):
        chunk_start = chunk_id * chunk_size
        chunk_stop = min(chunk_start + chunk_size, N)
        n_chunk = chunk_stop - chunk_start

        chunk_jobs.append(
            (
                chunk_id,
                chunk_start,
                n_chunk,
                n,
                child_seeds[chunk_id],
                prior,
                fixed_r,
                fixed_nu,
                random_init,
                k,
                center_y,
                out_dtype,
                exp_clip,
            )
        )

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_simulate_log_y_squared_chunk, job) for job in chunk_jobs]

        completed_iter = as_completed(futures)

        if show_progress:
            try:
                from tqdm.auto import tqdm
                completed_iter = tqdm(completed_iter, total=n_chunks)
            except ImportError:
                pass

        for future in completed_iter:
            chunk_start, n_chunk, log_y_squared_chunk, theta_chunk = future.result()

            chunk_stop = chunk_start + n_chunk
            log_y_squared[chunk_start:chunk_stop, :] = log_y_squared_chunk
            theta[chunk_start:chunk_stop, :] = theta_chunk

    return log_y_squared, theta


def simulate_sv_summaries_parallel(
    N,
    n,
    chunk_size,
    n_workers=-1,
    seed=1,
    prior="default",
    fixed_r=None,
    fixed_nu=None,
    random_init=True,
    n_acvf_ratios=4,
    n_quantiles=5,
    compute_arima_coeff=True,
    k=1e-12,
    eps=1e-12,
    arima_method=None,
    center_y=True,
    remove_NaNs=True,
    out_dtype=np.float32,
    exp_clip=350.0,
    show_progress=True,
):
    """
    Generate summary statistics and five-parameter SV values in parallel.

    The summary feature definition and names match the three-parameter
    simulator, so existing summary-network checkpoints can consume the
    result.  To simulate the standard three-parameter stochastic-volatility
    model, pass ``fixed_r=0.0`` and ``fixed_nu=np.inf``.  In that case ``s``
    is exactly the Gaussian log-volatility innovation standard deviation.

    Returns
    -------
    Z:
        Summary matrix of shape ``(N, p)``.

    theta:
        Parameter matrix of shape ``(N, 5)`` with columns
        ``(mu, phi, s, r, nu)``.  Three-parameter target transforms may use
        the first three columns.

    feature_names:
        Names of the ``p`` summary features, in checkpoint-compatible order.
    """

    if N < 1:
        raise ValueError("N must be at least 1.")

    if n < 1:
        raise ValueError("n must be at least 1.")

    if chunk_size is None:
        raise ValueError("chunk_size must be explicitly specified.")

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("k must be finite and positive.")

    if not np.isfinite(eps) or not 0.0 < eps < 1.0:
        raise ValueError("eps must be finite and satisfy 0 < eps < 1.")

    if fixed_r is not None and (
        not np.isfinite(fixed_r) or not 0.0 <= fixed_r < 1.0
    ):
        raise ValueError("fixed_r must satisfy 0 <= fixed_r < 1.")

    if fixed_nu is not None and (
        not (np.isfinite(fixed_nu) or np.isposinf(fixed_nu))
        or fixed_nu <= 4.0
    ):
        raise ValueError("fixed_nu must be greater than 4 or np.inf.")

    # Validate the named prior before starting subprocesses so input errors
    # fail immediately in the caller.
    validate_gh_skew_t_prior_constants(get_gh_skew_t_prior_constants(prior))
    n_workers = resolve_n_workers(n_workers)

    feature_names = summary_stats_sv_feature_names(
        n_acvf_ratios=n_acvf_ratios,
        n_quantiles=n_quantiles,
        compute_arima_coeff=compute_arima_coeff,
    )
    p = len(feature_names)

    Z = np.empty((N, p), dtype=out_dtype)
    theta = np.empty((N, 5), dtype=out_dtype)

    n_chunks = int(np.ceil(N / chunk_size))
    master_ss = np.random.SeedSequence(seed)
    child_seeds = master_ss.spawn(n_chunks)

    jobs = []

    for chunk_id in range(n_chunks):
        start_idx = chunk_id * chunk_size
        stop_idx = min(start_idx + chunk_size, N)
        n_chunk = stop_idx - start_idx
        jobs.append(
            (
                chunk_id,
                start_idx,
                n_chunk,
                n,
                child_seeds[chunk_id],
                prior,
                fixed_r,
                fixed_nu,
                random_init,
                n_acvf_ratios,
                n_quantiles,
                compute_arima_coeff,
                k,
                eps,
                arima_method,
                center_y,
                remove_NaNs,
                out_dtype,
                exp_clip,
                p,
            )
        )

    # Spawn gives independent, reproducible child processes.  Callers must use
    # the usual `if __name__ == "__main__":` guard on spawn-based platforms.
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(_simulate_and_summarize_chunk, job)
            for job in jobs
        ]
        completed_iter = as_completed(futures)

        if show_progress:
            try:
                from tqdm.auto import tqdm
                completed_iter = tqdm(completed_iter, total=n_chunks)
            except ImportError:
                pass

        for future in completed_iter:
            start_idx, n_chunk, Z_chunk, theta_chunk = future.result()
            stop_idx = start_idx + n_chunk
            Z[start_idx:stop_idx, :] = Z_chunk
            theta[start_idx:stop_idx, :] = theta_chunk

    return Z, theta, feature_names


def log_y_squared_moments(prior="default"):
    """
    Computes the prior-predictive mean and variance of

        log(y_t^2) = h_t + log(epsilon_t^2)

    where

        E(h_t | mu, phi, s, r, nu) = mu;
        var(h_t | mu, phi, s, r, nu) =  s^2 / (1 - phi^2);
        epsilon_t ~ N(0, 1);

    and mu, phi, s are drawn from the stochvol-style prior.
    """

    EULER_GAMMA = 0.5772156649015329

    hyper = get_gh_skew_t_prior_constants(prior)

    a = hyper.phi_a0
    b = hyper.phi_b0
    Bs = hyper.Bs

    if a <= 1 or b <= 1:
        raise ValueError(
            "The analytic variance requires phi_a0 > 1 and phi_b0 > 1, "
            "otherwise E[1 / (1 - phi^2)] is infinite."
        )

    # Moments of log(epsilon_t^2), where epsilon_t ~ N(0, 1)
    mean_log_eps2 = -EULER_GAMMA - np.log(2.0)
    var_log_eps2 = np.pi**2 / 2.0

    # E[mu]
    mean_mu = hyper.mu_mean

    # Var(mu)
    var_mu = hyper.mu_sd**2

    # E[sigma^2], since sigma^2 = Bs * chi^2_1
    mean_sigma2 = Bs

    # If phi = 2U - 1, U ~ Beta(a, b), then
    #
    # E[1 / (1 - phi^2)]
    # =
    # (a + b - 1) / 4 * (1 / (a - 1) + 1 / (b - 1))
    mean_inv_one_minus_phi2 = (
        (a + b - 1.0) / 4.0
        * (1.0 / (a - 1.0) + 1.0 / (b - 1.0))
    )

    # E[sigma^2 / (1 - phi^2)]
    mean_stationary_h_var = mean_sigma2 * mean_inv_one_minus_phi2

    # Law of total expectation
    mean_log_y2 = mean_mu + mean_log_eps2

    # Law of total variance
    var_log_y2 = (
        var_mu
        + mean_stationary_h_var
        + var_log_eps2
    )

    return {"mean": mean_log_y2, "var": var_log_y2, "std": np.sqrt(var_log_y2)}




def main():
    N = 100000
    n = 5
    n_workers = resolve_n_workers(-2)
    chunk_size = resolve_chunk_size(N, n_workers, chunks_per_worker=4)
    seed = 1

    log_y_squared, theta = simulate_sv_log_y_squared_parallel(
        N=N,
        n=n,
        fixed_nu=10,
        chunk_size=chunk_size,
        n_workers=n_workers,
        seed=seed,
        prior="default",
        random_init=True,
        k=1e-12,
        center_y=True,
        out_dtype=np.float32,
        exp_clip=350.0,
        show_progress=True,
    )

    import matplotlib.pyplot as plt
    plt.hist(theta[:,3], density=True)
    plt.show()

    print(np.mean(log_y_squared[:,4]))
    print(np.var(log_y_squared[:,4]))

    print(log_y_squared_moments())



if __name__ == "__main__":
    main()
