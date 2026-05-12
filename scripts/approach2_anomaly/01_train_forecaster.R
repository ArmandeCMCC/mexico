# 01_train_forecaster.R
# Train forecasting-based anomaly detection model on TRAIN period only
# Model: seasonal mean (per muni, per DOY) + AR(1) on residuals

library(tidyverse)

# CONFIGURATION

# Run tag for versioning outputs
RUN_TAG <- format(Sys.time(), "%Y%m%d_%H%M%S")

# Date boundaries (STRICT - no leakage)
TRAIN_END <- as.Date("2019-12-31")

# Output directory
output_dir <- file.path("runs/anomaly", RUN_TAG)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 01_train_forecaster.R ===\n")
cat(sprintf("Run tag: %s\n", RUN_TAG))
cat(sprintf("Output directory: %s\n", output_dir))

# 1. LOAD PANEL AND RESTRICT TO TRAIN

cat("\n=== Loading Panel ===\n")

panel <- readRDS("data/panel_mex_2017_2021_ntl_ghs.rds")
cat(sprintf("Full panel: %s rows, %d municipalities\n",
            format(nrow(panel), big.mark = ","),
            n_distinct(panel$GID_2)))

# Restrict to TRAIN period only
train_panel <- panel %>%
  filter(date <= TRAIN_END)

cat(sprintf("TRAIN panel (<= %s): %s rows\n",
            TRAIN_END, format(nrow(train_panel), big.mark = ",")))
cat(sprintf("Date range: %s to %s\n", min(train_panel$date), max(train_panel$date)))

# 2. COMPUTE SEASONAL MEAN (per muni, per DOY)

cat("\n=== Computing Seasonal Means ===\n")

# Handle leap day: map Feb 29 to DOY 60 (same as Feb 28)
# This ensures consistent DOY mapping across years
train_panel <- train_panel %>%
  mutate(
    doy = lubridate::yday(date),
    # Remap DOY for non-leap years to align with leap years
    # After Feb 28 in non-leap years, shift DOY by 1 to match leap year DOYs
    is_leap_year = lubridate::leap_year(date),
    doy_adjusted = case_when(
      # Feb 29 in leap years: keep as DOY 60
      is_leap_year & doy == 60 ~ 60L,
      # After Feb 28 in non-leap years: shift by 1 to align
      !is_leap_year & doy >= 60 ~ doy + 1L,
      TRUE ~ doy
    )
  )

# Compute seasonal mean per municipality and adjusted DOY
seasonal_means <- train_panel %>%
  group_by(GID_2, doy_adjusted) %>%
  summarise(
    seasonal_mean = mean(ntl_mean_built, na.rm = TRUE),
    n_obs = n(),
    .groups = "drop"
  )

cat(sprintf("Seasonal means computed: %d muni x DOY combinations\n", nrow(seasonal_means)))

# Check coverage
doy_coverage <- seasonal_means %>%
  group_by(GID_2) %>%
  summarise(n_doys = n(), .groups = "drop")

cat(sprintf("DOY coverage per muni: min=%d, median=%d, max=%d\n",
            min(doy_coverage$n_doys),
            median(doy_coverage$n_doys),
            max(doy_coverage$n_doys)))

# 3. COMPUTE RESIDUALS

cat("\n=== Computing Residuals ===\n")

# Join seasonal means back to training data
train_panel <- train_panel %>%
  left_join(seasonal_means %>% select(GID_2, doy_adjusted, seasonal_mean),
            by = c("GID_2", "doy_adjusted")) %>%
  mutate(residual = ntl_mean_built - seasonal_mean)

cat(sprintf("Residuals computed: mean=%.4f, sd=%.4f\n",
            mean(train_panel$residual, na.rm = TRUE),
            sd(train_panel$residual, na.rm = TRUE)))

# 4. FIT AR(1) PER MUNICIPALITY

cat("\n=== Fitting AR(1) Models ===\n")

# Function to fit AR(1) with fallback
fit_ar1 <- function(residuals) {
  # Need at least 10 observations for reliable AR(1)
  if (length(residuals) < 10 || all(is.na(residuals))) {
    return(list(phi = 0, success = FALSE))
  }

  # Remove NAs
  residuals <- residuals[!is.na(residuals)]
  if (length(residuals) < 10) {
    return(list(phi = 0, success = FALSE))
  }

  # Fit AR(1) using linear regression: residual[t] ~ residual[t-1]
  tryCatch({
    n <- length(residuals)
    y <- residuals[2:n]
    x <- residuals[1:(n-1)]

    fit <- lm(y ~ x - 1)  # No intercept (residuals should be centered)
    phi <- coef(fit)

    # Sanity check: phi should be in (-1, 1) for stationarity
    if (is.na(phi) || abs(phi) >= 1) {
      return(list(phi = 0, success = FALSE))
    }

    return(list(phi = phi, success = TRUE))
  }, error = function(e) {
    return(list(phi = 0, success = FALSE))
  })
}

# Fit AR(1) for each municipality
ar1_models <- train_panel %>%
  arrange(GID_2, date) %>%
  group_by(GID_2) %>%
  summarise(
    ar1_result = list(fit_ar1(residual)),
    .groups = "drop"
  ) %>%
  mutate(
    phi = map_dbl(ar1_result, "phi"),
    ar1_success = map_lgl(ar1_result, "success")
  ) %>%
  select(-ar1_result)

cat(sprintf("AR(1) fitted: %d municipalities\n", nrow(ar1_models)))
cat(sprintf("  Successful fits: %d (%.1f%%)\n",
            sum(ar1_models$ar1_success),
            100 * mean(ar1_models$ar1_success)))
cat(sprintf("  Phi range: [%.4f, %.4f], median=%.4f\n",
            min(ar1_models$phi), max(ar1_models$phi), median(ar1_models$phi)))

# 5. COMPUTE ONE-STEP-AHEAD PREDICTIONS AND ERRORS (TRAIN)

cat("\n=== Computing One-Step-Ahead Predictions ===\n")

# Join AR(1) parameters
train_panel <- train_panel %>%
  left_join(ar1_models, by = "GID_2")

# Compute one-step-ahead predictions
# pred[t] = seasonal_mean[doy(t)] + phi * residual[t-1]
train_panel <- train_panel %>%
  arrange(GID_2, date) %>%
  group_by(GID_2) %>%
  mutate(
    residual_lag1 = lag(residual, 1),
    pred_ntl = seasonal_mean + phi * coalesce(residual_lag1, 0),
    forecast_error = ntl_mean_built - pred_ntl
  ) %>%
  ungroup()

# Drop first day of each muni (no lag available)
train_errors <- train_panel %>%
  filter(!is.na(residual_lag1))

cat(sprintf("Forecast errors computed: %s observations\n",
            format(nrow(train_errors), big.mark = ",")))

# 6. COMPUTE TRAIN-ONLY STATISTICS PER MUNI

cat("\n=== Computing TRAIN-Only Statistics ===\n")

# These will be used to standardize errors during scoring
# CRITICAL: Only computed on TRAIN data to prevent leakage
train_stats <- train_errors %>%
  group_by(GID_2) %>%
  summarise(
    err_mean = mean(forecast_error, na.rm = TRUE),
    err_sd = sd(forecast_error, na.rm = TRUE),
    err_q95 = quantile(abs(forecast_error), 0.95, na.rm = TRUE),
    n_train_errors = n(),
    .groups = "drop"
  ) %>%
  # Handle edge cases where sd is 0 or NA
  mutate(
    err_sd = if_else(is.na(err_sd) | err_sd < 1e-6, 1.0, err_sd)
  )

cat(sprintf("Train stats computed for %d municipalities\n", nrow(train_stats)))
cat(sprintf("  err_mean: [%.4f, %.4f]\n", min(train_stats$err_mean), max(train_stats$err_mean)))
cat(sprintf("  err_sd: [%.4f, %.4f]\n", min(train_stats$err_sd), max(train_stats$err_sd)))
cat(sprintf("  err_q95: [%.4f, %.4f]\n", min(train_stats$err_q95), max(train_stats$err_q95)))

# 7. SAVE ARTIFACTS

cat("\n=== Saving Artifacts ===\n")

# Combined model artifacts (for scoring)
model_artifacts <- list(
  seasonal_means = seasonal_means,
  ar1_params = ar1_models,
  train_stats = train_stats,
  config = list(
    train_end = TRAIN_END,
    run_tag = RUN_TAG,
    created_at = Sys.time()
  )
)

saveRDS(model_artifacts, file.path(output_dir, "forecaster_model.rds"))
cat(sprintf("  Saved: %s/forecaster_model.rds\n", output_dir))

# Summary CSV for inspection
summary_df <- ar1_models %>%
  left_join(train_stats, by = "GID_2") %>%
  arrange(GID_2)

write_csv(summary_df, file.path(output_dir, "forecaster_summary.csv"))
cat(sprintf("  Saved: %s/forecaster_summary.csv\n", output_dir))

# Save run tag for downstream scripts
writeLines(RUN_TAG, file.path(output_dir, "run_tag.txt"))

# Also save to a known location for pipeline orchestration
writeLines(RUN_TAG, "runs/anomaly/latest_run_tag.txt")
cat(sprintf("  Saved: runs/anomaly/latest_run_tag.txt\n"))

# SUMMARY

cat("\n=== Training Summary ===\n")
cat(sprintf("Train period: %s to %s\n", min(train_panel$date), TRAIN_END))
cat(sprintf("Municipalities: %d\n", n_distinct(train_panel$GID_2)))
cat(sprintf("Observations: %s\n", format(nrow(train_panel), big.mark = ",")))
cat(sprintf("AR(1) success rate: %.1f%%\n", 100 * mean(ar1_models$ar1_success)))
cat(sprintf("Median phi: %.4f\n", median(ar1_models$phi)))
cat(sprintf("\nArtifacts saved to: %s\n", output_dir))

cat("\n=== Done ===\n")
