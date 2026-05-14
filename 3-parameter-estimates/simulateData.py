import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acovf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from dataclasses import dataclass



def _simulate_and_summarize_chunk(job):
    """
    Worker function.

    It samples parameters, simulates a chunk of SV series, computes summaries,
    and returns compact arrays.
    """
    (
        _chunk_id,
        chunk_start,
        n_chunk,
        n,
        seed_seq,
        prior,
        random_init,
        n_acvf_ratios,
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

    # 1. Sample parameters
    mu, phi, sigma = sample_stochvol_prior(
        n_chunk,
        rng=rng,
        prior=prior,
        return_sigma2=False,
        dtype=np.float64,
    )

    # 2. Simulate data
    y_chunk = simulate_sv_chunk(
        mu=mu,
        phi=phi,
        sigma=sigma,
        n=n,
        rng=rng,
        random_init=random_init,
        dtype=np.float64,
        exp_clip=exp_clip,
    )

    # 3. Compute summaries row by row
    summary_chunk = np.empty((n_chunk, p), dtype=out_dtype)
    
    for i in range(n_chunk):  # We use a Python for loop as summary_stats_sv() is not vectorized
        summary_chunk[i, :] = summary_stats_sv(
            y_chunk[i, :],
            k=k,
            n_acvf_ratios=n_acvf_ratios,
            eps=eps,
            compute_arima_coeff=compute_arima_coeff,
            arima_method=arima_method,
            center_y=center_y,
            remove_NaNs=remove_NaNs,
        ).astype(out_dtype, copy=False)

    theta_chunk = np.column_stack([mu, phi, sigma]).astype(out_dtype, copy=False)

    return chunk_start, n_chunk, summary_chunk, theta_chunk


def _simulate_log_y_squared_chunk(job):
    """
    Worker function.

    It samples parameters, simulates a chunk of SV series, computes log(y^2+k),
    and returns compact arrays.
    """
    (
        _chunk_id,
        chunk_start,
        n_chunk,
        n,
        seed_seq,
        prior,
        random_init,
        k,
        center_y,
        out_dtype,
        exp_clip,
    ) = job

    rng = np.random.default_rng(seed_seq)

    # 1. Sample parameters
    mu, phi, sigma = sample_stochvol_prior(
        n_chunk,
        rng=rng,
        prior=prior,
        return_sigma2=False,
        dtype=np.float64,
    )

    # 2. Simulate data
    y_chunk = simulate_sv_chunk(
        mu=mu,
        phi=phi,
        sigma=sigma,
        n=n,
        rng=rng,
        random_init=random_init,
        dtype=np.float64,
        exp_clip=exp_clip,
    )

    if center_y:
        y_chunk = y_chunk - np.mean(y_chunk, axis=1, keepdims=True)

    log_y_squared_chunk = np.log(y_chunk * y_chunk + k).astype(out_dtype, copy=False)

    theta_chunk = np.column_stack([mu, phi, sigma]).astype(out_dtype, copy=False)

    return chunk_start, n_chunk, log_y_squared_chunk, theta_chunk




def summary_stats_sv(
    y,
    k=1e-12,
    n_acvf_ratios=4,
    eps=1e-12,
    compute_arima_coeff=True,
    arima_method=None,
    center_y=True,
    remove_NaNs=True,
):
    """
    Summary statistic for one observed SV series y_{1:n}.

    If compute_arima_coeff=True, feature order is:
        1. mean(log(y^2 + k))
        2. q05
        3. q25
        4. q50
        5. q75
        6. q95
        7. transformed ACVF ratios:

              gamma(1) / gamma(0),
              gamma(2) / gamma(1),
              ...,
              gamma(n_acvf_ratios) / gamma(n_acvf_ratios - 1)

        8. transformed ARMA(1,1) AR coefficient
        9. transformed ARMA(1,1) MA coefficient
        10. log ARMA innovation SD
        11. 0.5 * log(var(log(y^2 + k)))
        12. log MAD(log(y^2 + k))
        13. plug-in log sigma estimate

    If compute_arima_coeff=False, the ARMA-related features are omitted.
    The feature order is then:
        1. mean(log(y^2 + k))
        2. q05
        3. q25
        4. q50
        5. q75
        6. q95
        7. transformed ACVF ratios
        8. 0.5 * log(var(log(y^2 + k)))
        9. log MAD(log(y^2 + k))
        10. plug-in log sigma estimate
    
    NOTE: Remember to update summary_stats_sv_feature_names() if you change the features generated here.
    """

    def clip_unit(z):
        return np.clip(z, -1.0 + eps, 1.0 - eps)

    def safe_log(z):
        return np.log(np.maximum(z, eps))

    def psi_phi(z):
        return 2.0 * np.arctanh(clip_unit(z))

    # Convert and validate input
    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")

    if not isinstance(n_acvf_ratios, int):
        raise TypeError("n_acvf_ratios must be an integer.")

    if n_acvf_ratios < 1:
        raise ValueError("n_acvf_ratios must be at least 1.")

    if len(y) <= n_acvf_ratios:
        raise ValueError("y is too short for the requested number of ACVF ratios.")

    # Transform data
    if center_y:
        y = y - np.mean(y)

    x = np.log(y**2 + k)

    # Location features
    mean_x = np.mean(x)
    q_x = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])

    # ACVF ratio features
    gamma = acovf(
        x,
        adjusted=False,
        demean=True,
        fft=False,
        nlag=n_acvf_ratios,
        missing="raise",
    )

    num = gamma[1:n_acvf_ratios + 1]
    den = gamma[0:n_acvf_ratios]

    raw_ratios = np.divide(
        num,
        den,
        out=np.zeros(n_acvf_ratios, dtype=float),
        where=np.abs(den) > eps,
    )

    # Apply transformation to raw ACVF ratios to map them from (-1, 1) to the real line
    acvf_ratio_features = psi_phi(raw_ratios)

    # Spread features
    var_x = np.var(x, ddof=1)
    mad_x = np.median(np.abs(x - np.median(x)))

    # Apply tranformation to variance and MAD to map them from (0, inf) to the real line
    log_sd_x = 0.5 * safe_log(var_x)
    log_mad_x = safe_log(mad_x)

    
    # Default persistence proxy used for log_sigma_plugin and ARMA.
    if n_acvf_ratios >= 2:
        phi_proxy = clip_unit(np.median(raw_ratios[1:]))
    else:
        phi_proxy = clip_unit(raw_ratios[0])

    if compute_arima_coeff:
        
        sigma2_start = max(var_x * (1.0 - phi_proxy**2), eps)

        start_params = np.array([
            mean_x,           # const
            phi_proxy,        # ar.L1
            0.0,              # ma.L1
            sigma2_start,     # sigma2
        ])

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

            alpha_arma = params.get("ar.L1", 0.0)
            beta_arma = params.get("ma.L1", 0.0)
            arma_sigma2 = params.get("sigma2", np.var(fit.resid, ddof=1))

            alpha_arma = clip_unit(alpha_arma)
            beta_arma = clip_unit(beta_arma)
            arma_innov_sd = np.sqrt(max(arma_sigma2, eps))

            psi_alpha_arma = psi_phi(alpha_arma)
            psi_beta_arma = psi_phi(beta_arma)
            log_arma_innov_sd = safe_log(arma_innov_sd)

            arma_features = np.array(
                [
                    psi_alpha_arma,
                    psi_beta_arma,
                    log_arma_innov_sd,
                ],
                dtype=float,
            )

            # If the ARMA fit succeeds, use the AR coefficient
            # as the persistence proxy in the plug-in sigma estimate.
            phi_proxy = alpha_arma

        except Exception:
            # If ARMA fitting fails, include explicit fallback values,
            # since compute_arima_coeff=True requested ARMA features.
            print("ARMA(1,1) fitting failed. Using fallback values for ARMA features.")
            alpha_arma = phi_proxy
            beta_arma = 0.0
            arma_innov_sd = np.std(x, ddof=1)

            psi_alpha_arma = psi_phi(alpha_arma)
            psi_beta_arma = psi_phi(beta_arma)
            log_arma_innov_sd = safe_log(arma_innov_sd)

            arma_features = np.array(
                [
                    psi_alpha_arma,
                    psi_beta_arma,
                    log_arma_innov_sd,
                ],
                dtype=float,
            )

            phi_proxy = alpha_arma

    # Plug-in log sigma estimate
    #
    # var(log(y_t^2 + k)) approx var(h_t) + var(log(epsilon_t^2))
    # var(log(epsilon_t^2)) = pi^2 / 2
    # var(h_t) = sigma^2 / (1 - phi^2)
    #
    # log(sigma) approx 0.5 * [log(var(x) - pi^2/2) + log(1 - phi^2)]
    log_eps2_var = np.pi**2 / 2.0

    latent_var_est = max(var_x - log_eps2_var, eps)
    one_minus_r2 = max(1.0 - phi_proxy**2, eps)

    log_sigma_plugin = 0.5 * (
        np.log(latent_var_est)
        + np.log(one_minus_r2)
    )

    spread_features = np.array(
        [
            log_sd_x,
            log_mad_x,
            log_sigma_plugin,
        ],
        dtype=float,
    )

    # Preallocate output
    if compute_arima_coeff:
        p = 6 + n_acvf_ratios + 3 + 3
    else:
        p = 6 + n_acvf_ratios + 3

    out = np.empty(p, dtype=float)

    i = 0

    # Location features
    out[i:i + 6] = [
        mean_x,
        q_x[0],
        q_x[1],
        q_x[2],
        q_x[3],
        q_x[4],
    ]
    i += 6

    # Persistence features
    out[i:i + n_acvf_ratios] = acvf_ratio_features
    i += n_acvf_ratios

    # Additional persistience features derived from ARMA(1,1) fit
    if compute_arima_coeff:
        out[i:i + 3] = arma_features
        i += 3
    
    # Volatility of volatility features
    out[i:i + 3] = spread_features

    if remove_NaNs:
        out[~np.isfinite(out)] = 0.0

    return out



def summary_stats_sv_feature_names(n_acvf_ratios=4, compute_arima_coeff=True):
    """
    Creates a list of the feature names generated in summary_stats_sv.
    The number of feature names is equal to the dimension of the output of summary_stats_sv.
    """
    names = [
        "mean_x",
        "q05_x",
        "q25_x",
        "q50_x",
        "q75_x",
        "q95_x",
    ]

    for j in range(1, n_acvf_ratios + 1):
        names.append(f"psi_gamma{j}_over_gamma{j-1}")

    if compute_arima_coeff:
        names.extend([
            "psi_alpha_arma",
            "psi_beta_arma",
            "log_arma_innov_sd",
        ])

    names.extend([
        "log_sd_x",
        "log_mad_x",
        "log_sigma_plugin",
    ])

    return names


def simulate_sv_chunk(
    mu,
    phi,
    sigma,
    n,
    rng,
    random_init=True,
    dtype=np.float64,
    exp_clip=350.0,
):
    """
    Simulate a chunk of standard log-normal SV series.

    Model:
        h_t = mu + phi * (h_{t-1} - mu) + sigma * eta_t
        y_t = exp(h_t / 2) * eps_t

    Parameters
    ----------
    mu, phi, sigma:
        Arrays of shape (m,), where m is the chunk size.

    n:
        Length of each time series.

    rng:
        np.random.Generator.

    random_init:
        If True, initialize h_0 from the stationary distribution.
        If False, initialize h_0 = mu.

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
    sigma = np.asarray(sigma, dtype=np.float64)

    if not (mu.shape == phi.shape == sigma.shape):
        raise ValueError("mu, phi, and sigma must have the same shape.")

    if mu.ndim != 1:
        raise ValueError("mu, phi, and sigma must be one-dimensional arrays.")

    if n < 1:
        raise ValueError("n must be at least 1.")

    if np.any(np.abs(phi) >= 1.0):
        raise ValueError("All phi values must satisfy abs(phi) < 1.")

    if np.any(sigma <= 0.0):
        raise ValueError("All sigma values must be positive.")

    m = len(mu)

    y = np.empty((m, n), dtype=dtype)

    if random_init:
        stationary_sd = sigma / np.sqrt(1.0 - phi**2)
        h_prev = mu + stationary_sd * rng.standard_normal(m)
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
            + sigma * rng.standard_normal(m)
        )

        y[:, t] = (
            np.exp(np.clip(0.5 * h_prev, -exp_clip, exp_clip))
            * rng.standard_normal(m)
        )

    return y


@dataclass(frozen=True)
class StochvolPriorConstants:
    mu_mean: float
    mu_sd: float
    phi_a0: float
    phi_b0: float
    Bsigma: float


_STOCHVOL_PRIORS = {
    "finance": StochvolPriorConstants(
        mu_mean=-9.0,
        mu_sd=1.0,
        phi_a0=20.0,
        phi_b0=1.5,
        Bsigma=1.0,
    ),
    "default": StochvolPriorConstants(
        mu_mean=0.0,
        mu_sd=10.0,
        phi_a0=5.0,
        phi_b0=1.5,
        Bsigma=1.0,
    ),
}


def get_stochvol_prior_constants(prior="default"):
    """
    Return prior constants matching the R function get_stochvol_prior_constants().
    """

    if prior not in _STOCHVOL_PRIORS:
        valid = ", ".join(_STOCHVOL_PRIORS)
        raise ValueError(f"Unknown prior '{prior}'. Valid choices are: {valid}.")

    return _STOCHVOL_PRIORS[prior]


def sample_stochvol_prior(
    n,
    rng,
    prior="default",
    return_sigma2=False,
    dtype=np.float64,
):
    """
    Sample from the stochvol-style prior for (mu, phi, sigma).

    Matches the R code:

        mu     ~ N(mu_mean, mu_sd^2)
        (phi + 1) / 2 ~ Beta(phi_a0, phi_b0)
        sigma^2 ~ Bsigma * ChiSq(df = 1)

    Parameters
    ----------
    n:
        Number of parameter draws.

    rng:
        np.random.Generator object.

    prior:
        Either "finance" or "default".

    return_sigma2:
        If True, also return sigma2.

    dtype:
        Floating point dtype for output arrays.

    Returns
    -------
    If return_sigma2=False:
        mu, phi, sigma

    If return_sigma2=True:
        mu, phi, sigma, sigma2
    """

    if n < 1:
        raise ValueError("n must be at least 1.")

    hyper = get_stochvol_prior_constants(prior)

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

    sigma2 = (
        hyper.Bsigma * rng.chisquare(df=1.0, size=n)
    ).astype(dtype, copy=False)

    sigma = np.sqrt(sigma2).astype(dtype, copy=False)

    if return_sigma2:
        return mu, phi, sigma, sigma2

    return mu, phi, sigma


def resolve_n_workers(n_workers):
    """
    Resolve the number of worker processes.

    Negative values work like offsets from the total CPU count:
        -1 uses all cores except one,
        -2 uses all cores except two,
        etc.

    To use all available cores, pass the explicit positive worker count.
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
    Generate log(y^2 + k) series and true SV parameters in parallel.

    Returns
    -------
    log_y_squared:
        Matrix of shape (N, n), where row i is log(y_i^2 + k).

    theta:
        Parameter matrix of shape (N, 3), columns are mu, phi, sigma
    """

    if N < 1:
        raise ValueError("N must be at least 1.")

    if n < 1:
        raise ValueError("n must be at least 1.")

    n_workers = resolve_n_workers(n_workers)

    if chunk_size is None:
        raise ValueError("chunk_size must be explicitly specified.")

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    if k <= 0:
        raise ValueError("k must be positive.")

    log_y_squared = np.empty((N, n), dtype=out_dtype)
    theta = np.empty((N, 3), dtype=out_dtype)

    n_chunks = np.ceil(N / chunk_size).astype(int)

    # Independent, reproducible RNG streams for each chunk.
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
                random_init,
                k,
                center_y,
                out_dtype,
                exp_clip,
            )
        )

    # "spawn" is safer and reproducible, but requires the usual
    # if __name__ == "__main__" guard in scripts.
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
    random_init=True,
    n_acvf_ratios=4,
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
    Generate a dataset of summaries and true SV parameters in parallel.

    Returns
    -------
    Z:
        Summary matrix of shape (N, p)

    theta:
        Parameter matrix of shape (N, 3), columns are mu, phi, sigma

    feature_names:
        Names of summary-statistic features
    """

    if N < 1:
        raise ValueError("N must be at least 1.")

    if n < 1:
        raise ValueError("n must be at least 1.")

    n_workers = resolve_n_workers(n_workers)

    if chunk_size is None:
        raise ValueError("chunk_size must be explicitly specified.")

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    feature_names = summary_stats_sv_feature_names(
        n_acvf_ratios=n_acvf_ratios,
        compute_arima_coeff=compute_arima_coeff,
    )
    p = len(feature_names)

    Z = np.empty((N, p), dtype=out_dtype)
    theta = np.empty((N, 3), dtype=out_dtype)

    n_chunks = np.ceil(N / chunk_size).astype(int)

    # Independent, reproducible RNG streams for each chunk.
    master_ss = np.random.SeedSequence(seed)
    child_seeds = master_ss.spawn(n_chunks)

    jobs = []

    for chunk_id in range(n_chunks):
        start_idx = chunk_id * chunk_size
        stop_idx = min(start_idx + chunk_size, N)
        m_chunk = stop_idx - start_idx

        jobs.append(
            (
                chunk_id,
                start_idx,
                m_chunk,
                n,
                child_seeds[chunk_id],
                prior,
                random_init,
                n_acvf_ratios,
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

    # "spawn" is safer and reproducible, but requires the usual
    # if __name__ == "__main__" guard in scripts.
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_simulate_and_summarize_chunk, job) for job in jobs]

        completed_iter = as_completed(futures)

        if show_progress:
            try:
                from tqdm.auto import tqdm
                completed_iter = tqdm(completed_iter, total=n_chunks)
            except ImportError:
                pass

        for future in completed_iter:
            start_idx, m_chunk, Z_chunk, theta_chunk = future.result()

            Z[start_idx:start_idx + m_chunk, :] = Z_chunk
            theta[start_idx:start_idx + m_chunk, :] = theta_chunk

    return Z, theta, feature_names


def main1():
    """
    Generate a large summary-statistic dataset for the MLP trainer.

    The saved .npz file is compatible with trainSummaryNN.py and contains:
        summaries:
            Array of shape (N, p), where each row contains summary statistics.

        params:
            Array of shape (N, 3), with columns [mu, phi, sigma].

        feature_names:
            Names of the summary-statistic columns.

        param_names:
            Names of the parameter columns.

        config:
            JSON metadata describing the simulation settings.
    """

    N = 1_000_000
    n = 253

    prior = "finance"
    chunks_per_worker = 8
    compute_arima_coeff = False
    n_workers = -2 # -2 means "use all available cores minus two"
    seed = 1
    k = 1e-12
    eps = 1e-12
    center_y = True
    random_init = True
    n_acvf_ratios = 4
    out_dtype = np.float32
    exp_clip = 350.0

    resolved_n_workers = resolve_n_workers(n_workers)
    chunk_size = resolve_chunk_size(N, resolved_n_workers, chunks_per_worker)

    arima_label = "ARIMA" if compute_arima_coeff else "no_ARIMA"
    file_name = f"sv_summaries_{prior}_1M_n{n}_{arima_label}.npz"

    Z, theta, feature_names = simulate_sv_summaries_parallel(
        N=N,
        n=n,
        chunk_size=chunk_size,
        seed=seed,
        prior=prior,
        random_init=random_init,
        n_acvf_ratios=n_acvf_ratios,
        compute_arima_coeff=compute_arima_coeff,
        k=k,
        eps=eps,
        center_y=center_y,
        out_dtype=out_dtype,
        exp_clip=exp_clip,
        show_progress=True,
        n_workers=resolved_n_workers,
    )
    
    np.savez(
        file_name,
        summaries=Z,
        params=theta,
        feature_names=np.array(feature_names),
        param_names=np.array(["mu", "phi", "sigma"]),
        config=json.dumps({  #Store metadata
            "dataset_type": "summary_statistics",
            "N": N,
            "n": n,
            "chunk_size": chunk_size,
            "chunks_per_worker": chunks_per_worker,
            "requested_n_workers": n_workers,
            "resolved_n_workers": resolved_n_workers,
            "prior": prior,
            "random_init": random_init,
            "n_acvf_ratios": n_acvf_ratios,
            "compute_arima_coeff": compute_arima_coeff,
            "k": k,
            "eps": eps,
            "center_y": center_y,
            "out_dtype": str(np.dtype(out_dtype)),
            "exp_clip": exp_clip,
            "seed": seed,
        }),
    )

    print("Done.")
    print("Saved to:", file_name)
    print("Z shape:", Z.shape)
    print("theta shape:", theta.shape)

    df = pd.DataFrame(Z, columns=feature_names)
    print(df.describe())


def main2():
    """
    Generate a large log(y_t^2 + k) dataset for the TCN/CNN trainer.

    This is the dataset format expected by trainCNN.py. The saved .npz file
    contains:
        log_y_squared:
            Array of shape (N, n), where row i is log(y_i,t^2 + k).

        params:
            Array of shape (N, 3), with columns [mu, phi, sigma].

        param_names:
            Names of the parameter columns.

        config:
            JSON metadata describing the simulation settings.

    The primary intended dataset is N=1,000,000, n=253, default prior.
    """

    N = 1_000_000
    n = 253

    prior = "default"
    chunks_per_worker = 8
    n_workers = -2 # -2 means "use all available cores minus two"
    seed = 1
    k = 1e-12
    center_y = True
    random_init = True
    out_dtype = np.float32
    exp_clip = 350.0

    resolved_n_workers = resolve_n_workers(n_workers)
    chunk_size = resolve_chunk_size(N, resolved_n_workers, chunks_per_worker)

    file_name = f"sv_log_y_squared_{prior}_1M_n{n}.npz"

    log_y_squared, theta = simulate_sv_log_y_squared_parallel(
        N=N,
        n=n,
        chunk_size=chunk_size,
        n_workers=resolved_n_workers,
        seed=seed,
        prior=prior,
        random_init=random_init,
        k=k,
        center_y=center_y,
        out_dtype=out_dtype,
        exp_clip=exp_clip,
        show_progress=True,
    )
    
    np.savez(
        file_name,
        log_y_squared=log_y_squared,
        params=theta,
        param_names=np.array(["mu", "phi", "sigma"]),
        config=json.dumps({  #Store metadata
            "dataset_type": "log_y_squared",
            "N": N,
            "n": n,
            "chunk_size": chunk_size,
            "chunks_per_worker": chunks_per_worker,
            "requested_n_workers": n_workers,
            "resolved_n_workers": resolved_n_workers,
            "prior": prior,
            "random_init": random_init,
            "k": k,
            "center_y": center_y,
            "out_dtype": str(np.dtype(out_dtype)),
            "exp_clip": exp_clip,
            "seed": seed,
        }),
    )

    print("Done.")
    print("Saved to:", file_name)
    print("log_y_squared shape:", log_y_squared.shape)
    print("theta shape:", theta.shape)
    print("log_y_squared mean:", float(np.mean(log_y_squared)))
    print("log_y_squared std:", float(np.std(log_y_squared)))
    print("log_y_squared min:", float(np.min(log_y_squared)))
    print("log_y_squared max:", float(np.max(log_y_squared)))


if __name__ == "__main__":

    main2()
