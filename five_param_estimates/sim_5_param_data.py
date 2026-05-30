from dataclasses import dataclass


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
    nu_max: float
    nu_log_mean: float
    nu_log_sd: float


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
        nu_max=150.0,
        nu_log_mean=2.5,
        nu_log_sd=0.75,
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
        nu_max=150.0,
        nu_log_mean=2.5,
        nu_log_sd=0.75,
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

    The nu constants are intentionally open-ended for the next step. A natural
    starting point is a shifted, truncated lognormal:

        nu = nu_min + exp(z), z ~ N(nu_log_mean, nu_log_sd^2),
        truncated to nu <= nu_max.
    """

    if prior not in _GH_SKEW_T_PRIORS:
        valid = ", ".join(_GH_SKEW_T_PRIORS)
        raise ValueError(f"Unknown prior '{prior}'. Valid choices are: {valid}.")

    return _GH_SKEW_T_PRIORS[prior]


# Backwards-compatible alias with the same naming style as sim_3_param_data.py.
get_stochvol_prior_constants = get_gh_skew_t_prior_constants
