from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GHSkewTPriorConstants:
    mu_mean: float
    mu_sd: float
    phi_a0: float
    phi_b0: float
    Bs: float
    r_a0: float
    r_b0: float
    r_max: float
    nu_min: float
    nu_rate: float


_GH_SKEW_T_PRIORS = {
    "finance": GHSkewTPriorConstants(
        mu_mean=-9.0,
        mu_sd=1.0,
        phi_a0=20.0,
        phi_b0=1.5,
        Bs=1.0,
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
        r_a0=1.0,
        r_b0=9.0,
        r_max=0.8,
        nu_min=8.0,
        nu_rate=0.1,
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

        r / r_max ~ Beta(r_a0, r_b0)

    The tail parameter follows a shifted exponential distribution:

        nu = nu_min + X, X ~ Exponential(rate = nu_rate).
    """

    if prior not in _GH_SKEW_T_PRIORS:
        valid = ", ".join(_GH_SKEW_T_PRIORS)
        raise ValueError(f"Unknown prior '{prior}'. Valid choices are: {valid}.")

    return _GH_SKEW_T_PRIORS[prior]


# Backwards-compatible alias with the same naming style as sim_3_param_data.py.
get_stochvol_prior_constants = get_gh_skew_t_prior_constants


def sample_stochvol_prior(
    n,
    rng,
    prior="default",
    return_s2=False,
    dtype=np.float64,
):
    """
    Sample from the prior for the five-parameter SV model.

    Parameterization:

        mu ~ N(mu_mean, mu_sd^2)
        (phi + 1) / 2 ~ Beta(phi_a0, phi_b0)
        s^2 ~ Bs * ChiSq(df = 1)
        r / r_max ~ Beta(r_a0, r_b0)
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

    r = (
        hyper.r_max * rng.beta(
            a=hyper.r_a0,
            b=hyper.r_b0,
            size=n,
        )
    ).astype(dtype, copy=False)

    nu = (
        hyper.nu_min + rng.exponential(scale=1.0 / hyper.nu_rate, size=n)
    ).astype(dtype, copy=False)

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
        If True, initialize h_0 using the stationary mean and variance. This
        matches the 3-parameter simulator's initialization style.

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




def main():
    pass


if __name__ == "__main__":
    main()