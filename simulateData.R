# ============================================================
# Summary statistics for the simulated data
# ============================================================

summary_stats <- function(y, k = 1e-12, acf_lags = 8L) {
  stopifnot(length(y) > acf_lags)

  y <- y - mean(y)
  x <- log(y^2 + k)

  ac <- as.numeric(acf(x, lag.max = acf_lags, plot = FALSE)$acf)[-1]

  qs <- as.numeric(
    quantile(x, probs = c(0.05, 0.5, 0.95), names = FALSE)
  )

  out <- c(
    mean = mean(x),
    var = var(x),
    ac,
    q05 = qs[1],
    q50 = qs[2],
    q95 = qs[3]
  )

  names(out)[3:(2 + acf_lags)] <- paste0("acf", seq_len(acf_lags))

  out
}



# ============================================================
# Return preset prior constants for the stochvol parameters
# ============================================================

get_stochvol_prior_constants <- function(prior = c("finance", "default")) {
  prior <- match.arg(prior)

  switch(
    prior,
    finance = list(
      mu_mean = -9,   # suitable for daily raw log returns
      mu_sd   = 1,
      phi_a0  = 20,
      phi_b0  = 1.5,
      Bsigma  = 1
    ),
    default = list(
      mu_mean = 0,
      mu_sd   = 10,
      phi_a0  = 5,
      phi_b0  = 1.5,
      Bsigma  = 1
    )
  )
}

# ============================================================
# Sample from the stochvol prior for (mu, phi, sigma)
# ============================================================

rprior_stochvol <- function(n,
                            prior = c("finance", "default"),
                            return_sigma2 = FALSE) {
  stopifnot(n >= 1)

  prior <- match.arg(prior)

  hyper <- get_stochvol_prior_constants(prior)

  mu <- rnorm(n, mean = hyper$mu_mean, sd = hyper$mu_sd)
  phi <- 2 * rbeta(n, shape1 = hyper$phi_a0, shape2 = hyper$phi_b0) - 1
  sigma2 <- hyper$Bsigma * rchisq(n, df = 1)
  sigma <- sqrt(sigma2)

  out <- data.frame(
    mu = mu,
    phi = phi,
    sigma = sigma
  )

  if (return_sigma2) {
    out$sigma2 <- sigma2
  }

  out
}





simulate_data_parallel <- function(m, n,
                                   prior = c("default", "finance"),
                                   n_cores = max(1L, parallel::detectCores() - 1L),
                                   chunk_size = NULL,
                                   seed = NULL) {
  prior <- match.arg(prior)

  stopifnot(m >= 1, n >= 1, n_cores >= 1)

  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Sample parameters on the main process
  params <- rprior_stochvol(m, prior = prior, return_sigma2 = FALSE)

  # Use a non-constant probe vector to determine output length
  p <- length(summary_stats(seq_len(n)))

  simulated_data <- matrix(NA_real_, nrow = m, ncol = p)

  # Row-specific seeds make results reproducible regardless of chunking/cores
  row_seeds <- sample.int(.Machine$integer.max, size = m)

  if (is.null(chunk_size)) {
    # Several chunks per core usually gives decent load balancing
    chunk_size <- max(1L, ceiling(m / (4L * n_cores)))
  }

  idx_chunks <- split(seq_len(m), ceiling(seq_len(m) / chunk_size))

  cl <- parallel::makeCluster(n_cores)

  on.exit({
    parallel::stopCluster(cl)
  }, add = TRUE)

  parallel::clusterEvalQ(cl, {
    library(stochvol)
    NULL
  })

  parallel::clusterExport(
    cl,
    varlist = c("params", "n", "p", "summary_stats", "row_seeds"),
    envir = environment()
  )

  simulate_chunk <- function(idx) {
    out <- matrix(NA_real_, nrow = length(idx), ncol = p)

    for (j in seq_along(idx)) {
      i <- idx[j]

      set.seed(row_seeds[i])

      sim <- svsim(
        len = n,
        mu = params$mu[i],
        phi = params$phi[i],
        sigma = params$sigma[i]
      )

      out[j, ] <- summary_stats(sim$y)
    }

    out
  }

  chunk_results <- parallel::parLapply(cl, idx_chunks, simulate_chunk)

  simulated_data <- do.call(rbind, chunk_results)

  list(
    params = params,
    simulated_data = simulated_data
  )
}



# ============================================================
# Parameter moments
# ============================================================


get_phi_prior_moments <- function(prior = c("default", "finance"),
                                  transform = c("none", "shifted_logit", "fisher")) {
  prior <- match.arg(prior)
  transform <- match.arg(transform)

  pc <- get_stochvol_prior_constants(prior)
  a_0 <- pc$phi_a0
  b_0 <- pc$phi_b0

  if (transform == "none") {
    mean <- (a_0 - b_0) / (a_0 + b_0)
    variance <- 4 * a_0 * b_0 / ((a_0 + b_0)^2 * (a_0 + b_0 + 1))
  } else if (transform == "shifted_logit") {
    mean <- digamma(a_0) - digamma(b_0)
    variance <- trigamma(a_0) + trigamma(b_0)
  } else if (transform == "fisher") {
    mean <- 0.5 * (digamma(a_0) - digamma(b_0))
    variance <- 0.25 * (trigamma(a_0) + trigamma(b_0))
  }

  list(mean = mean, variance = variance, sd = sqrt(variance))
}



# ============================================================
# Generate and store the simulated data
# ============================================================


res <- simulate_data_parallel(
  m = 100000,
  n = 253,
  prior = "default",
  chunk_size = 500,
  seed = 123
)

library(reticulate)

np <- import("numpy")

np$savez_compressed(
  "training_data.npz",
  z = res$simulated_data,
  theta = as.matrix(res$params)
)