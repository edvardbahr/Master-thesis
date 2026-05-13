args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript stochvolMCMC.R input_csv output_csv [draws] [burnin] [thinpara]")
}

input_path <- args[[1]]
output_path <- args[[2]]
draws <- if (length(args) >= 3) as.integer(args[[3]]) else 2000L
burnin <- if (length(args) >= 4) as.integer(args[[4]]) else 500L
thinpara <- if (length(args) >= 5) as.integer(args[[5]]) else 1L

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
