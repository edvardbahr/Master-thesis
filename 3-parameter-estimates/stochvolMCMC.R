args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript stochvolMCMC.R input_csv output_csv [draws] [burnin] [thinpara] [prior]")
}

input_path <- args[[1]]
output_path <- args[[2]]
draws <- if (length(args) >= 3) as.integer(args[[3]]) else 2000L
burnin <- if (length(args) >= 4) as.integer(args[[4]]) else 500L
thinpara <- if (length(args) >= 5) as.integer(args[[5]]) else 1L
prior <- if (length(args) >= 6) args[[6]] else "default"

get_stochvol_prior_constants <- function(prior) {
  priors <- list(
    finance = list(
      mu_mean = -9.0,
      mu_sd = 1.0,
      phi_a0 = 20.0,
      phi_b0 = 1.5,
      Bsigma = 1.0
    ),
    default = list(
      mu_mean = 0.0,
      mu_sd = 10.0,
      phi_a0 = 5.0,
      phi_b0 = 1.5,
      Bsigma = 1.0
    )
  )

  if (!prior %in% names(priors)) {
    stop(
      sprintf(
        "Unknown prior '%s'. Valid choices are: %s.",
        prior,
        paste(names(priors), collapse = ", ")
      )
    )
  }

  priors[[prior]]
}

prior_constants <- get_stochvol_prior_constants(prior)

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

summarize_series <- function(series_index, y) {
  fit <- svsample(
    as.numeric(y),
    draws = draws,
    burnin = burnin,
    priormu = c(prior_constants$mu_mean, prior_constants$mu_sd^2),
    priorphi = c(prior_constants$phi_a0, prior_constants$phi_b0),
    priorsigma = prior_constants$Bsigma,
    thinpara = thinpara,
    quiet = TRUE
  )

  draws_matrix <- as.matrix(para(fit))

  data.frame(
    index = series_index,
    mu_mean = mean(draws_matrix[, "mu"]),
    phi_mean = mean(draws_matrix[, "phi"]),
    sigma_mean = mean(draws_matrix[, "sigma"]),
    mu_sd = sd(draws_matrix[, "mu"]),
    phi_sd = sd(draws_matrix[, "phi"]),
    sigma_sd = sd(draws_matrix[, "sigma"]),
    mu_median = median(draws_matrix[, "mu"]),
    phi_median = median(draws_matrix[, "phi"]),
    sigma_median = median(draws_matrix[, "sigma"])
  )
}

results <- do.call(
  rbind,
  lapply(seq_len(nrow(y_matrix)), function(i) summarize_series(i, y_matrix[i, ]))
)

write.csv(results, output_path, row.names = FALSE)
