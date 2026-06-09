library(data.table)
library(rumidas)
library(zoo)

data("rv5", package = "rumidas")

rumidas_rv <- data.table(
  date = as.Date(zoo::index(rv5)),
  RV = as.numeric(rv5)
)

setorder(rumidas_rv, date)
rumidas_rv <- rumidas_rv[is.finite(RV) & RV > 0]
rumidas_rv[, log_RV := log(RV)]

# Model:
#   log_RV_t = mu + phi * (log_RV_{t-1} - mu) + u_t
#
# Equivalently:
#   log_RV_t = alpha + phi * log_RV_{t-1} + u_t,
#   alpha = mu * (1 - phi).
#
# We estimate alpha and phi by conditional least squares / OLS. This only uses
# the conditional-mean restriction E[u_t | log_RV_{t-1}] = 0, so u_t can be
# non-Gaussian, skewed, and heavy-tailed. The estimated residuals below are then
# treated as empirical samples from the GHST innovation distribution.
ar_data <- rumidas_rv[, .(date, RV, log_RV)]
ar_data[, log_RV_lag := shift(log_RV)]
ar_data <- ar_data[is.finite(log_RV) & is.finite(log_RV_lag)]

ar1_fit <- lm(log_RV ~ log_RV_lag, data = ar_data)

alpha_hat <- unname(coef(ar1_fit)[["(Intercept)"]])
phi_hat <- unname(coef(ar1_fit)[["log_RV_lag"]])

if (abs(1 - phi_hat) < sqrt(.Machine$double.eps)) {
  stop("phi_hat is too close to 1; mu_hat = alpha_hat / (1 - phi_hat) is unstable.")
}

mu_hat <- alpha_hat / (1 - phi_hat)

ar_data[, fitted_log_RV := mu_hat + phi_hat * (log_RV_lag - mu_hat)]
ar_data[, u_hat := log_RV - fitted_log_RV]

ghst_samples <- ar_data$u_hat

n_u <- length(ghst_samples)
u_centered <- ghst_samples - mean(ghst_samples)
u_var_mle <- mean(u_centered^2)

skewness_hat <- mean(u_centered^3) / u_var_mle^(3 / 2)
excess_kurtosis_hat <- mean(u_centered^4) / u_var_mle^2 - 3

# Approximate normal-reference moment tests:
#   H0 skewness = 0 against H1 skewness != 0
#   H0 excess kurtosis = 0 against H1 excess kurtosis > 0
#
# These are asymptotic checks. They are useful diagnostics, but the p-values
# should not be treated as exact because u_hat is estimated from a time series.
skewness_z <- skewness_hat / sqrt(6 / n_u)
skewness_p_two_sided <- 2 * pnorm(abs(skewness_z), lower.tail = FALSE)

kurtosis_z <- excess_kurtosis_hat / sqrt(24 / n_u)
kurtosis_p_two_sided <- 2 * pnorm(abs(kurtosis_z), lower.tail = FALSE)
leptokurtosis_p_one_sided <- pnorm(kurtosis_z, lower.tail = FALSE)

jarque_bera_stat <- n_u / 6 * (
  skewness_hat^2 + (excess_kurtosis_hat^2 / 4)
)
jarque_bera_p <- pchisq(jarque_bera_stat, df = 2, lower.tail = FALSE)

moment_tests <- data.table(
  test = c(
    "Skewness: H0 skewness = 0",
    "Leptokurtosis: H0 excess kurtosis = 0, H1 > 0",
    "Jarque-Bera: H0 Gaussian skewness and kurtosis"
  ),
  estimate = c(skewness_hat, excess_kurtosis_hat, jarque_bera_stat),
  statistic = c(skewness_z, kurtosis_z, jarque_bera_stat),
  p_value = c(
    skewness_p_two_sided,
    leptokurtosis_p_one_sided,
    jarque_bera_p
  )
)

tail_probs <- c(0.10, 0.05, 0.025, 0.01, 0.005, 0.001)
u_median <- median(ghst_samples)
left_quantiles <- as.numeric(quantile(ghst_samples, probs = tail_probs, na.rm = TRUE))
right_quantiles <- as.numeric(quantile(ghst_samples, probs = 1 - tail_probs, na.rm = TRUE))

tail_quantile_balance <- data.table(
  tail_prob = tail_probs,
  left_quantile = left_quantiles,
  right_quantile = right_quantiles,
  left_distance_from_median = u_median - left_quantiles,
  right_distance_from_median = right_quantiles - u_median
)
tail_quantile_balance[, right_left_distance_ratio :=
  right_distance_from_median / left_distance_from_median
]

tail_thresholds_sd <- c(1.0, 1.5, 2.0, 2.5, 3.0)
tail_sign_tests <- rbindlist(lapply(tail_thresholds_sd, function(k) {
  threshold <- k * sd(ghst_samples)
  n_left <- sum(ghst_samples < -threshold)
  n_right <- sum(ghst_samples > threshold)
  n_tail <- n_left + n_right
  expected_each_gaussian_tail <- n_u * pnorm(-k)

  data.table(
    threshold_sd = k,
    threshold = threshold,
    n_left = n_left,
    n_right = n_right,
    n_tail = n_tail,
    right_tail_share = if (n_tail > 0) n_right / n_tail else NA_real_,
    heavier_empirical_tail = if (n_right > n_left) {
      "right"
    } else if (n_left > n_right) {
      "left"
    } else {
      "balanced"
    },
    left_vs_gaussian = n_left / expected_each_gaussian_tail,
    right_vs_gaussian = n_right / expected_each_gaussian_tail,
    sign_test_p_two_sided = if (n_tail > 0) {
      binom.test(n_right, n_tail, p = 0.5)$p.value
    } else {
      NA_real_
    }
  )
}))

ar1_estimates <- data.table(
  start_date = min(ar_data$date),
  end_date = max(ar_data$date),
  n = nrow(ar_data),
  alpha_hat = alpha_hat,
  mu_hat = mu_hat,
  phi_hat = phi_hat,
  u_mean = mean(ghst_samples),
  u_sd = sd(ghst_samples),
  skewness_hat = skewness_hat,
  excess_kurtosis_hat = excess_kurtosis_hat
)

cat("\nAR(1) estimates for log realized variance\n")
print(ar1_estimates)

cat("\nNormal-reference tests for skewness and heavy tails\n")
print(moment_tests)

cat(sprintf(
  "\nTwo-sided excess-kurtosis p-value: %.4g\n",
  kurtosis_p_two_sided
))

cat("\nTail quantile balance around the residual median\n")
print(tail_quantile_balance)

cat("\nTail sign tests conditional on large absolute residuals\n")
print(tail_sign_tests)

cat("\nQuantiles of u_hat residual samples\n")
print(quantile(
  ghst_samples,
  probs = c(0.001, 0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995, 0.999),
  na.rm = TRUE
))

cat("\nLargest absolute residuals\n")
print(head(ar_data[
  order(-abs(u_hat)),
  .(date, log_RV, fitted_log_RV, u_hat)
], 10))

old_par <- par(no.readonly = TRUE)
on.exit(par(old_par), add = TRUE)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(
  ar_data$log_RV_lag,
  ar_data$log_RV,
  pch = 16,
  col = rgb(0, 0, 0, 0.25),
  xlab = expression(log(RV)[t - 1]),
  ylab = expression(log(RV)[t]),
  main = "AR(1) Fit for log(RV)"
)
abline(a = alpha_hat, b = phi_hat, col = "firebrick", lwd = 2)

hist(
  ghst_samples,
  breaks = "FD",
  probability = TRUE,
  col = "grey85",
  border = "white",
  xlab = expression(hat(u)[t]),
  main = "Residual Samples for GHST Fit"
)
lines(density(ghst_samples, na.rm = TRUE), col = "firebrick", lwd = 2)
abline(v = 0, col = "black")

qqnorm(
  ghst_samples,
  pch = 16,
  col = rgb(0, 0, 0, 0.25),
  main = "Normal QQ Plot of Residuals",
  xlab = "Theoretical normal quantiles",
  ylab = expression(hat(u)[t])
)
qqline(ghst_samples, col = "firebrick", lwd = 2)
