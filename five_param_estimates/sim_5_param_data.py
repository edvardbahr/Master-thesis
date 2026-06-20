import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np


_STATIONARY_INIT_BURN_IN_STEPS = 20


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
    # TODO: Finalize finance prior before making it selectable. This can be done through "adaptive learning"
    # by conditioning on the default prior first and then use the resulting posterior as a finance prior.
    # "finance": GHSkewTPriorConstants(
    #     mu_mean=-9.0,
    #     mu_sd=1.0,
    #     phi_a0=20.0,
    #     phi_b0=1.5,
    #     Bs=1.0,
    #     r_a0=1.0,
    #     r_b0=9.0,
    #     r_max=0.8,
    #     nu_min=8.0,
    #     nu_rate=0.1,
    # ),
    "default": GHSkewTPriorConstants(
        mu_mean=0.0,
        mu_sd=10.0,
        phi_a0=5.0,
        phi_b0=1.5,
        Bs=1.0,
        r_a0=None,  # When r_a0, r_b0 is None, r is sampled
        r_b0=None,  # from a uniform distribution on [0, r_max).
        r_max=0.999999,
        nu_min=8.0,  # Under this condition the kurtosis exists
        nu_rate=0.1, # Picked so that there should be at least 5% chance of observing dof > 30 (approx Gaussian)
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
        r / r_max ~ Beta(r_a0, r_b0), or
        r ~ Uniform(0, r_max) if r_a0 and r_b0 are both None
        nu = fixed_nu if fixed_nu is not None, otherwise
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

    if fixed_nu is not None and (not np.isfinite(fixed_nu) or fixed_nu <= 4.0):
        raise ValueError("fixed_nu must be greater than 4.")

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
    """

    s = np.asarray(s, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    s, r, nu = np.broadcast_arrays(s, r, nu)
    mu_gh, delta, beta = gh_skew_t_params_from_s_r_nu(s, r, nu)
    gamma_draw = rng.gamma(shape=0.5 * nu, scale=1.0)
    # 1/gamma(shape = 0.5 * nu, rate = k) -> inv-gamma(shape = 0.5 * nu, scale = k)
    # 0.5 * delta^2 * inv-gamma(shape = 0.5 * nu, scale = 1.0) -> inv-gamma(shape = 0.5 * nu, scale = 0.5 * delta^2)
    w = 0.5 * delta * delta / gamma_draw    
    z = rng.standard_normal(size=np.shape(w))
    innovations = mu_gh + beta * w + np.sqrt(w) * z

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

    gh_skew_t_params_from_s_r_nu(s, r, nu)

    m = len(mu)
    y = np.empty((m, n), dtype=dtype)

    if random_init:
        stationary_sd = s / np.sqrt(1.0 - phi**2)
        h_prev = mu + stationary_sd * rng.standard_normal(m)

        for _ in range(_STATIONARY_INIT_BURN_IN_STEPS):
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
    #plt.hist(theta[:,3], density=True)
    #plt.show()

    print(np.mean(log_y_squared[:,4]))
    print(np.var(log_y_squared[:,4]))

    print(log_y_squared_moments())



if __name__ == "__main__":
    main()
