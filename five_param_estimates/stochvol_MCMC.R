args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 10) {
  stop(
    paste(
      "Usage:",
      "Rscript stochvol_MCMC.R input_csv draws_output_csv draws burnin thinpara",
      "mu_mean mu_sd phi_a0 phi_b0 Bs"
    )
  )
}

input_path <- args[[1]]
draws_output_path <- args[[2]]
draws <- as.integer(args[[3]])
burnin <- as.integer(args[[4]])
thinpara <- as.integer(args[[5]])

prior_mu_mean <- as.numeric(args[[6]])
prior_mu_sd <- as.numeric(args[[7]])
prior_phi_a0 <- as.numeric(args[[8]])
prior_phi_b0 <- as.numeric(args[[9]])
prior_Bs <- as.numeric(args[[10]])

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
    prior_Bs
  )))
) {
  stop("Prior constants must be numeric.")
}

if (prior_mu_sd <= 0.0) {
  stop("mu_sd must be positive.")
}

if (prior_phi_a0 <= 0.0 || prior_phi_b0 <= 0.0) {
  stop("phi_a0 and phi_b0 must be positive.")
}

if (prior_Bs <= 0.0) {
  stop("Bs must be positive.")
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

sample_series <- function(series_index) {
  fit <- svsample(
    as.numeric(y_matrix[series_index, ]),
    draws = draws,
    burnin = burnin,
    priormu = c(prior_mu_mean, prior_mu_sd),
    priorphi = c(prior_phi_a0, prior_phi_b0),
    priorsigma = prior_Bs,
    thinpara = thinpara,
    quiet = TRUE
  )

  draws_matrix <- as.matrix(para(fit))

  data.frame(
    series_index = series_index,
    draw_index = seq_len(nrow(draws_matrix)),
    mu = draws_matrix[, "mu"],
    phi = draws_matrix[, "phi"],
    sigma = draws_matrix[, "sigma"]
  )
}

parameter_draws <- do.call(
  rbind,
  lapply(seq_len(nrow(y_matrix)), sample_series)
)

dir.create(dirname(draws_output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(parameter_draws, draws_output_path, row.names = FALSE)
