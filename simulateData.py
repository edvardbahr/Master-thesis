import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acovf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def summary_stats_sv(
    y,
    k=1e-8,
    n_acvf_ratios=4,
    eps=1e-8,
    compute_arima_coeff=True,
    arima_method=None,
    center_y=True,
    clean_output=True,
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

    
    # Default persistence proxy used for log_sigma_plugin.
    # This is always available because n_acvf_ratios >= 1.
    phi_proxy = clip_unit(raw_ratios[0])

    if compute_arima_coeff:
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
                    fit = model.fit()
                else:
                    fit = model.fit(method=arima_method)

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
            alpha_arma = clip_unit(raw_ratios[0])
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

    if clean_output:
        out[~np.isfinite(out)] = 0.0

    return out




