import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acovf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def summary_stats_sv(
    y,
    k=1e-8,
    ratio_lags=(1, 2, 3, 4),
    eps=1e-8,
    compute_arima_coeff=True,
    arima_method=None,
    arima_fallback="acf_proxy",
    center_y=True,
    clean_output=True,
):
    """
    Summary statistic for one observed SV series y_{1:n}.

    Feature order:
        1. mean(log(y^2 + k))
        2. q05
        3. q25
        4. q50
        5. q75
        6. q95
        7. transformed ACVF ratios for given ratio_lags
        8. transformed ARMA/auxiliary AR coefficient
        9. transformed ARMA/auxiliary MA coefficient
        10. log auxiliary innovation SD
        11. 0.5 * log(var(log(y^2 + k)))
        12. log MAD(log(y^2 + k))
        13. plug-in log sigma estimate

    If compute_arima_coeff=False, the output dimension is unchanged.
    The ARMA features are replaced by cheap fallback features.
    """

    def clip_unit(z):
        # Clip to (-1 + eps, 1 - eps)
        # e.g. -1 + eps/2 is clipped to -1 + eps, and 1 - eps/2 is clipped to 1 - eps.
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

    ratio_lags = np.asarray(ratio_lags, dtype=int)

    if ratio_lags.ndim != 1:
        raise ValueError("ratio_lags must be one-dimensional.")

    if np.any(ratio_lags < 0):
        raise ValueError("ratio_lags must contain nonnegative integers.")

    max_lag = int(np.max(ratio_lags)) + 1 if len(ratio_lags) > 0 else 1

    if len(y) <= max_lag:
        raise ValueError("y is too short for the requested ratio_lags.")

    # Transform data
    if center_y:
        y = y - np.mean(y)

    x = np.log(y**2 + k)

    # Location features
    mean_x = np.mean(x)
    q_x = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])

    # ACVF features
    # adjusted=False gives the biased denominator n.
    # demean=True matches your previous x_centered logic.
    gamma = acovf(
        x,
        adjusted=False,
        demean=True,
        fft=False,
        nlag=max_lag,
        missing="raise",
    )

    if len(ratio_lags) > 0:
        num = gamma[ratio_lags + 1]
        den = gamma[ratio_lags]

        raw_ratios = np.divide(
            num,
            den,
            out=np.zeros_like(num, dtype=float),
            where=np.abs(den) > eps,
        )

        acvf_ratio_features = psi_phi(raw_ratios)
    else:
        raw_ratios = np.empty(0)
        acvf_ratio_features = np.empty(0)

    # Cheap defaults used if ARMA is skipped or fails
    if arima_fallback == "acf_proxy":
        alpha_arma = raw_ratios[0] if len(raw_ratios) > 0 else 0.0
        beta_arma = 0.0
        arma_innov_sd = np.std(x, ddof=1)
    elif arima_fallback == "zero":
        alpha_arma = 0.0
        beta_arma = 0.0
        arma_innov_sd = np.std(x, ddof=1)
    else:
        raise ValueError("arima_fallback must be either 'acf_proxy' or 'zero'.")

    # ARMA(1,1) auxiliary fit
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

            alpha_arma = params.get("ar.L1", alpha_arma)
            beta_arma = params.get("ma.L1", beta_arma)
            arma_sigma2 = params.get("sigma2", arma_innov_sd**2)

            arma_innov_sd = np.sqrt(max(arma_sigma2, eps))

        except Exception:
            # Keep fallback values.
            pass

    alpha_arma = clip_unit(alpha_arma)
    beta_arma = clip_unit(beta_arma)

    psi_alpha_arma = psi_phi(alpha_arma)
    psi_beta_arma = psi_phi(beta_arma)
    log_arma_innov_sd = safe_log(arma_innov_sd)

    # Spread features
    var_x = np.var(x, ddof=1)
    mad_x = np.median(np.abs(x - np.median(x)))

    half_log_var_x = 0.5 * safe_log(var_x)
    log_mad_x = safe_log(mad_x)

    # Plug-in log sigma estimate
    #
    # var(log(y_t^2 + k)) approx var(h_t) + var(log(eps_t^2))
    # var(log(eps_t^2)) = pi^2 / 2
    # var(h_t) = sigma^2 / (1 - phi^2)
    #
    # log(sigma) approx 0.5 * [log(var(x) - pi^2/2) + log(1 - phi^2)]
    log_eps2_var = np.pi**2 / 2.0

    latent_var_est = max(var_x - log_eps2_var, eps)
    one_minus_r2 = max(1.0 - alpha_arma**2, eps)

    log_sigma_plugin = 0.5 * (
        np.log(latent_var_est)
        + np.log(one_minus_r2)
    )

    # Preallocate output instead of repeatedly appending/concatenating
    p = 6 + len(acvf_ratio_features) + 6
    out = np.empty(p, dtype=float)

    i = 0

    out[i:i + 6] = [
        mean_x,
        q_x[0],
        q_x[1],
        q_x[2],
        q_x[3],
        q_x[4],
    ]
    i += 6

    out[i:i + len(acvf_ratio_features)] = acvf_ratio_features
    i += len(acvf_ratio_features)

    out[i:i + 6] = [
        psi_alpha_arma,
        psi_beta_arma,
        log_arma_innov_sd,
        half_log_var_x,
        log_mad_x,
        log_sigma_plugin,
    ]

    if clean_output:
        out[~np.isfinite(out)] = 0.0

    return out