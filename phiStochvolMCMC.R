library(stochvol)



# -----------------------------
# Function: posterior summary for psi = 2 * atanh(phi)
# -----------------------------

psi_posterior_summary <- function(y,
                                  alpha = 0.05,
                                  draws = 20000,
                                  burnin = 5000,
                                  thinpara = 1,
                                  quiet = TRUE) {
  fit <- svsample(
    y,
    draws = draws,
    burnin = burnin,
    thinpara = thinpara,
    quiet = quiet
  )

  phi_draws <- as.numeric(para(fit)[, "phi"])
  psi_draws <- 2 * atanh(phi_draws)

  list(
    fit = fit,

    phi_draws = phi_draws,
    psi_draws = psi_draws,

    phi_mean = mean(phi_draws),
    phi_var = var(phi_draws),
    phi_sd = sd(phi_draws),
    phi_median = median(phi_draws),
    phi_ci_lower = quantile(phi_draws, alpha / 2),
    phi_ci_upper = quantile(phi_draws, 1 - alpha / 2),

    psi_mean = mean(psi_draws),
    psi_var = var(psi_draws),
    psi_sd = sd(psi_draws),
    psi_median = median(psi_draws),
    psi_ci_lower = quantile(psi_draws, alpha / 2),
    psi_ci_upper = quantile(psi_draws, 1 - alpha / 2)
  )
}


# -----------------------------
# Settings
# -----------------------------

set.seed(5)

phi_grid <- seq(0.5, 0.99, length.out = 9)

n_obs <- 253
mu_fixed <- -9
sigma_fixed <- 0.25

draws <- 20000
burnin <- 5000
thinpara <- 1


# -----------------------------
# Storage objects
# -----------------------------

results <- vector("list", length(phi_grid))
simulated_data_list <- vector("list", length(phi_grid))
summary_list <- vector("list", length(phi_grid))


# -----------------------------
# Run simulations and MCMC fits
# -----------------------------

 for (i in seq_along(phi_grid)) {
  phi_true <- phi_grid[i]
  psi_true <- 2 * atanh(phi_true)

  cat("Running", i, "of", length(phi_grid), "with phi =", phi_true, "\n")

  sim <- svsim(
    len = n_obs,
    mu = mu_fixed,
    phi = phi_true,
    sigma = sigma_fixed
  )

  # svsim returns the simulated observations in sim$y
  y <- as.numeric(sim$y)

  post <- psi_posterior_summary(
    y = y,
    draws = draws,
    burnin = burnin,
    thinpara = thinpara,
    quiet = TRUE
  )

  results[[i]] <- list(
    index = i,
    y = y,
    mu_true = mu_fixed,
    phi_true = phi_true,
    sigma_true = sigma_fixed,
    psi_true = psi_true,
    posterior = post
  )

  simulated_data_list[[i]] <- data.frame(
    index = i,
    time = seq_along(y),
    y = y,
    mu_true = mu_fixed,
    phi_true = phi_true,
    sigma_true = sigma_fixed,
    psi_true = psi_true
  )

  summary_list[[i]] <- data.frame(
    index = i,

    mu_true = mu_fixed,
    phi_true = phi_true,
    sigma_true = sigma_fixed,
    psi_true = psi_true,

    phi_mean = post$phi_mean,
    phi_var = post$phi_var,
    phi_sd = post$phi_sd,
    phi_median = post$phi_median,
    phi_ci_lower = post$phi_ci_lower,
    phi_ci_upper = post$phi_ci_upper,

    psi_mean = post$psi_mean,
    psi_var = post$psi_var,
    psi_sd = post$psi_sd,
    psi_median = post$psi_median,
    psi_ci_lower = post$psi_ci_lower,
    psi_ci_upper = post$psi_ci_upper
  )
}


# -----------------------------
# Make Python-friendly data frames
# -----------------------------

summary_df <- do.call(rbind, summary_list)
simulated_data_df <- do.call(rbind, simulated_data_list)

phi_draws_df <- as.data.frame(
  do.call(rbind, lapply(results, function(x) x$posterior$phi_draws))
)

psi_draws_df <- as.data.frame(
  do.call(rbind, lapply(results, function(x) x$posterior$psi_draws))
)

phi_draws_df <- cbind(index = seq_along(phi_grid), phi_true = phi_grid, phi_draws_df)
psi_draws_df <- cbind(index = seq_along(phi_grid), phi_true = phi_grid, psi_draws_df)


# -----------------------------
# Save files for Python
# -----------------------------

write.csv(summary_df, "mcmc_summary.csv", row.names = FALSE)
write.csv(simulated_data_df, "simulated_data.csv", row.names = FALSE)
write.csv(phi_draws_df, "phi_posterior_draws.csv", row.names = FALSE)
write.csv(psi_draws_df, "psi_posterior_draws.csv", row.names = FALSE)

# Optional: save full R object too, in case you want to reopen in R later
saveRDS(results, "full_mcmc_results.rds")


# -----------------------------
# Print summary
# -----------------------------

print(summary_df)

cat("\nSaved files:\n")
cat("mcmc_summary.csv\n")
cat("simulated_data.csv\n")
cat("phi_posterior_draws.csv\n")
cat("psi_posterior_draws.csv\n")
cat("full_mcmc_results.rds\n")