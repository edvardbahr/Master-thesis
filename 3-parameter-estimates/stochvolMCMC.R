args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 10) {
  stop(
    paste(
      "Usage:",
      "Rscript stochvolMCMC.R input_csv output_csv draws burnin thinpara",
      "mu_mean mu_sd phi_a0 phi_b0 Bsigma [alpha] [draws_output_csv]"
    )
  )
}

input_path <- args[[1]]
output_path <- args[[2]]
draws <- as.integer(args[[3]])
burnin <- as.integer(args[[4]])
thinpara <- as.integer(args[[5]])

prior_mu_mean <- as.numeric(args[[6]])
prior_mu_sd <- as.numeric(args[[7]])
prior_phi_a0 <- as.numeric(args[[8]])
prior_phi_b0 <- as.numeric(args[[9]])
prior_Bsigma <- as.numeric(args[[10]])
alpha <- if (length(args) >= 11) as.numeric(args[[11]]) else 0.05
draws_output_path <- if (length(args) >= 12 && nzchar(args[[12]])) {
  args[[12]]
} else {
  NA_character_
}
save_parameter_draws <- !is.na(draws_output_path)

if (is.na(draws) || draws < 1L) {
  stop("draws must be a positive integer.")
}

if (is.na(burnin) || burnin < 0L) {
  stop("burnin must be a non-negative integer.")
}

if (is.na(thinpara) || thinpara < 1L) {
  stop("thinpara must be a positive integer.")
}

if (
  any(is.na(c(
    prior_mu_mean,
    prior_mu_sd,
    prior_phi_a0,
    prior_phi_b0,
    prior_Bsigma,
    alpha
  )))
) {
  stop("Prior constants and alpha must be numeric.")
}

if (prior_mu_sd <= 0.0) {
  stop("mu_sd must be positive.")
}

if (prior_phi_a0 <= 0.0 || prior_phi_b0 <= 0.0) {
  stop("phi_a0 and phi_b0 must be positive.")
}

if (prior_Bsigma <= 0.0) {
  stop("Bsigma must be positive.")
}

if (alpha <= 0.0 || alpha >= 1.0) {
  stop("alpha must be between 0 and 1.")
}

suppressPackageStartupMessages(library(stochvol))

y_matrix <- as.matrix(
  read.csv(
    input_path,
    header = FALSE,
    check.names = FALSE
  )
)

if (!is.numeric(y_matrix)) {
  stop("Input CSV must contain only numeric values.")
}

lower_probability <- alpha / 2.0
upper_probability <- 1.0 - alpha / 2.0
credible_level <- 1.0 - alpha

summarize_draws <- function(draws_matrix, parameter_name) {
  values <- draws_matrix[, parameter_name]

  c(
    mean = mean(values),
    sd = sd(values),
    median = median(values),
    ci_lower = unname(quantile(values, lower_probability)),
    ci_upper = unname(quantile(values, upper_probability))
  )
}

sample_series <- function(series_index, y) {
  fit <- svsample(
    as.numeric(y),
    draws = draws,
    burnin = burnin,
    priormu = c(prior_mu_mean, prior_mu_sd), # Mu prior: Normal with mean and sd
    priorphi = c(prior_phi_a0, prior_phi_b0),# Phi prior: Beta with shape parameters a0 and b0
    priorsigma = prior_Bsigma,               # 
    thinpara = thinpara,
    quiet = TRUE
  )

  draws_matrix <- as.matrix(para(fit))
  mu_summary <- summarize_draws(draws_matrix, "mu")
  phi_summary <- summarize_draws(draws_matrix, "phi")
  sigma_summary <- summarize_draws(draws_matrix, "sigma")

  summary <- data.frame(
    index = series_index,
    alpha = alpha,
    credible_level = credible_level,
    mu_mean = mu_summary[["mean"]],
    phi_mean = phi_summary[["mean"]],
    sigma_mean = sigma_summary[["mean"]],
    mu_sd = mu_summary[["sd"]],
    phi_sd = phi_summary[["sd"]],
    sigma_sd = sigma_summary[["sd"]],
    mu_median = mu_summary[["median"]],
    phi_median = phi_summary[["median"]],
    sigma_median = sigma_summary[["median"]],
    mu_ci_lower = mu_summary[["ci_lower"]],
    mu_ci_upper = mu_summary[["ci_upper"]],
    phi_ci_lower = phi_summary[["ci_lower"]],
    phi_ci_upper = phi_summary[["ci_upper"]],
    sigma_ci_lower = sigma_summary[["ci_lower"]],
    sigma_ci_upper = sigma_summary[["ci_upper"]]
  )

  parameter_draws <- NULL
  if (save_parameter_draws) {
    parameter_draws <- data.frame(
      index = series_index,
      draw_index = seq_len(nrow(draws_matrix)),
      mu = draws_matrix[, "mu"],
      phi = draws_matrix[, "phi"],
      sigma = draws_matrix[, "sigma"]
    )
  }

  list(summary = summary, parameter_draws = parameter_draws)
}

fits <- lapply(
  seq_len(nrow(y_matrix)),
  function(i) sample_series(i, y_matrix[i, ])
)

results <- do.call(
  rbind,
  lapply(fits, function(fit) fit$summary)
)

write.csv(results, output_path, row.names = FALSE)

if (!is.na(draws_output_path)) {
  all_parameter_draws <- do.call(
    rbind,
    lapply(fits, function(fit) fit$parameter_draws)
  )

  write.csv(all_parameter_draws, draws_output_path, row.names = FALSE)
}
