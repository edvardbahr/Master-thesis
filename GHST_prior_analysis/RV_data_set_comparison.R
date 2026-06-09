library(highfrequency)
library(data.table)
library(rumidas)
library(zoo)

# Load S&P 500/SPY realized variance data.
data("rv5", package = "rumidas")
data("SPYRM", package = "highfrequency")

# Keep one date column and one realized-variance column from each source.
# RV/RV5 are realized variances, so sqrt(RV) is daily realized volatility.
rumidas_rv <- data.table(
  date = as.Date(zoo::index(rv5)),
  RV_rumidas = as.numeric(rv5)
)

highfrequency_rv <- data.table(
  date = as.Date(SPYRM$DT),
  RV_highfrequency = as.numeric(SPYRM$RV5)
)

setkey(rumidas_rv, date)
setkey(highfrequency_rv, date)

# Compare only the common calendar range, then use an inner join so each row is
# the same trading day in both sources.
overlap_start <- max(min(rumidas_rv$date, na.rm = TRUE),
                     min(highfrequency_rv$date, na.rm = TRUE))
overlap_end <- min(max(rumidas_rv$date, na.rm = TRUE),
                   max(highfrequency_rv$date, na.rm = TRUE))

rumidas_overlap <- rumidas_rv[date >= overlap_start & date <= overlap_end]
highfrequency_overlap <- highfrequency_rv[date >= overlap_start & date <= overlap_end]

missing_from_rumidas <- highfrequency_overlap[!rumidas_overlap, on = "date"]
missing_from_highfrequency <- rumidas_overlap[!highfrequency_overlap, on = "date"]

rv_comparison <- merge(
  rumidas_overlap,
  highfrequency_overlap,
  by = "date",
  all = FALSE
)

rv_comparison <- rv_comparison[
  is.finite(RV_rumidas) &
    is.finite(RV_highfrequency) &
    RV_rumidas >= 0 &
    RV_highfrequency >= 0
]

rv_comparison[, `:=`(
  vol_rumidas = sqrt(RV_rumidas),
  vol_highfrequency = sqrt(RV_highfrequency),
  diff_RV = RV_highfrequency - RV_rumidas,
  diff_vol = sqrt(RV_highfrequency) - sqrt(RV_rumidas),
  ratio_RV = fifelse(RV_rumidas > 0, RV_highfrequency / RV_rumidas, NA_real_),
  ratio_vol = fifelse(RV_rumidas > 0, sqrt(RV_highfrequency / RV_rumidas), NA_real_)
)]

rv_comparison[, `:=`(
  abs_diff_RV = abs(diff_RV),
  abs_diff_vol = abs(diff_vol),
  pct_diff_vol = 100 * (ratio_vol - 1)
)]

comparison_stats <- rv_comparison[, .(
  overlap_start = min(date),
  overlap_end = max(date),
  matched_days = .N,
  cor_RV = cor(RV_rumidas, RV_highfrequency, use = "complete.obs"),
  cor_vol = cor(vol_rumidas, vol_highfrequency, use = "complete.obs"),
  MAE_RV = mean(abs_diff_RV, na.rm = TRUE),
  RMSE_RV = sqrt(mean(diff_RV^2, na.rm = TRUE)),
  MAE_vol = mean(abs_diff_vol, na.rm = TRUE),
  RMSE_vol = sqrt(mean(diff_vol^2, na.rm = TRUE)),
  mean_ratio_vol = mean(ratio_vol, na.rm = TRUE),
  median_ratio_vol = median(ratio_vol, na.rm = TRUE),
  share_within_10pct_vol = mean(abs(pct_diff_vol) <= 10, na.rm = TRUE)
)]

cat("\nDate coverage\n")
cat(sprintf("rumidas:       %s to %s (%d rows)\n",
            min(rumidas_rv$date), max(rumidas_rv$date), nrow(rumidas_rv)))
cat(sprintf("highfrequency: %s to %s (%d rows)\n",
            min(highfrequency_rv$date), max(highfrequency_rv$date), nrow(highfrequency_rv)))
cat(sprintf("matched overlap: %s to %s (%d rows)\n",
            overlap_start, overlap_end, nrow(rv_comparison)))

cat(sprintf("\nDates in highfrequency but not rumidas inside overlap: %d\n",
            nrow(missing_from_rumidas)))
if (nrow(missing_from_rumidas) > 0) {
  print(missing_from_rumidas[, date])
}

cat(sprintf("\nDates in rumidas but not highfrequency inside overlap: %d\n",
            nrow(missing_from_highfrequency)))
if (nrow(missing_from_highfrequency) > 0) {
  print(missing_from_highfrequency[, date])
}

cat("\nAgreement statistics\n")
print(comparison_stats)

cat("\nLargest absolute volatility differences\n")
print(head(rv_comparison[order(-abs_diff_vol),
  .(date, vol_rumidas, vol_highfrequency, diff_vol, pct_diff_vol)
], 10))

old_par <- par(no.readonly = TRUE)
on.exit(par(old_par), add = TRUE)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(
  rv_comparison$date,
  100 * rv_comparison$vol_rumidas,
  type = "l",
  col = "steelblue",
  lwd = 1.5,
  xlab = "Date",
  ylab = "Daily realized volatility (%)",
  main = "Matched Daily Realized Volatility"
)
lines(
  rv_comparison$date,
  100 * rv_comparison$vol_highfrequency,
  col = "firebrick",
  lwd = 1.5
)
legend(
  "topright",
  legend = c("rumidas rv5", "highfrequency SPYRM RV5"),
  col = c("steelblue", "firebrick"),
  lty = 1,
  lwd = 1.5,
  bty = "n"
)

plot(
  100 * rv_comparison$vol_rumidas,
  100 * rv_comparison$vol_highfrequency,
  pch = 16,
  col = rgb(0, 0, 0, 0.35),
  xlab = "rumidas daily realized volatility (%)",
  ylab = "highfrequency daily realized volatility (%)",
  main = "Same-Day Agreement"
)
abline(0, 1, col = "firebrick", lwd = 2)

plot(
  rv_comparison$date,
  100 * rv_comparison$diff_vol,
  type = "h",
  col = ifelse(rv_comparison$diff_vol >= 0, "steelblue", "firebrick"),
  xlab = "Date",
  ylab = "highfrequency - rumidas (pp)",
  main = "Daily Volatility Difference"
)
abline(h = 0, col = "black")

plot(
  rv_comparison$date,
  rv_comparison$ratio_vol,
  type = "l",
  col = "darkgreen",
  lwd = 1.5,
  xlab = "Date",
  ylab = "highfrequency / rumidas",
  main = "Daily Volatility Ratio"
)
abline(h = 1, col = "black")
