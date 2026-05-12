# 02_score_anomalies.R
# Score ALL periods using TRAIN-only model parameters (no leakage)
# Generate anomaly features for supervised model integration
#
# v1.1 Updates:
# - Run-tagged output folders (data/processed/anomaly_features/<run_tag>/)
# - NA-robust rolling window functions
# - Drop-focused anomaly features (anomaly_drop = pmax(0, -anomaly_score))
# - Fleet rank features per day (national + state level)

library(tidyverse)
library(slider)  # For rolling window calculations

# CONFIGURATION

# Get run tag from latest anomaly run or command line arg
args <- commandArgs(trailingOnly = TRUE)
args <- args[!grepl("^--", args)]  # Filter out flags

if (length(args) > 0) {
  ANOMALY_RUN_TAG <- args[1]
} else {
  ANOMALY_RUN_TAG <- readLines("runs/anomaly/latest_run_tag.txt", warn = FALSE)[1]
}

# Generate new run tag for scoring outputs
SCORING_RUN_TAG <- format(Sys.time(), "%Y%m%d_%H%M%S")

model_dir <- file.path("runs/anomaly", ANOMALY_RUN_TAG)

# Run-tagged output directory (NEW: versioned outputs)
output_dir <- file.path("data/processed/anomaly_features", SCORING_RUN_TAG)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 02_score_anomalies.R ===\n")
cat(sprintf("Anomaly model from: %s\n", model_dir))
cat(sprintf("Scoring run tag: %s\n", SCORING_RUN_TAG))
cat(sprintf("Output directory: %s\n", output_dir))

# 1. LOAD MODEL ARTIFACTS AND FULL PANEL

cat("\n=== Loading Model Artifacts ===\n")

model <- readRDS(file.path(model_dir, "forecaster_model.rds"))
seasonal_means <- model$seasonal_means
ar1_params <- model$ar1_params
train_stats <- model$train_stats

cat(sprintf("Model trained on: <= %s\n", model$config$train_end))
cat(sprintf("Seasonal means: %d entries\n", nrow(seasonal_means)))
cat(sprintf("AR(1) params: %d municipalities\n", nrow(ar1_params)))
cat(sprintf("Train stats: %d municipalities\n", nrow(train_stats)))

cat("\n=== Loading Full Panel ===\n")

panel <- readRDS("data/panel_mex_2017_2021_ntl_ghs.rds")
cat(sprintf("Full panel: %s rows\n", format(nrow(panel), big.mark = ",")))
cat(sprintf("Date range: %s to %s\n", min(panel$date), max(panel$date)))

# 2. PREPARE PANEL FOR SCORING

cat("\n=== Preparing Panel ===\n")

# Add DOY with leap year adjustment (same logic as training)
panel <- panel %>%
  mutate(
    doy = lubridate::yday(date),
    is_leap_year = lubridate::leap_year(date),
    doy_adjusted = case_when(
      is_leap_year & doy == 60 ~ 60L,
      !is_leap_year & doy >= 60 ~ doy + 1L,
      TRUE ~ doy
    )
  )

# Join seasonal means (from TRAIN only)
panel <- panel %>%
  left_join(
    seasonal_means %>% select(GID_2, doy_adjusted, seasonal_mean),
    by = c("GID_2", "doy_adjusted")
  )

# Handle missing seasonal means (new DOY combinations not in TRAIN)
# Fill with municipality mean
muni_means <- panel %>%
  filter(!is.na(seasonal_mean)) %>%
  group_by(GID_2) %>%
  summarise(muni_mean = mean(seasonal_mean), .groups = "drop")

panel <- panel %>%
  left_join(muni_means, by = "GID_2") %>%
  mutate(
    seasonal_mean = coalesce(seasonal_mean, muni_mean)
  ) %>%
  select(-muni_mean)

missing_seasonal <- sum(is.na(panel$seasonal_mean))
if (missing_seasonal > 0) {
  cat(sprintf("WARNING: %d rows still missing seasonal_mean\n", missing_seasonal))
}

# Compute residuals
panel <- panel %>%
  mutate(residual = ntl_mean_built - seasonal_mean)

# 3. COMPUTE ONE-STEP-AHEAD PREDICTIONS

cat("\n=== Computing One-Step-Ahead Predictions ===\n")

# Join AR(1) parameters (from TRAIN only)
panel <- panel %>%
  left_join(ar1_params %>% select(GID_2, phi), by = "GID_2") %>%
  mutate(phi = coalesce(phi, 0))  # Default to 0 for any missing

# Compute predictions: pred[t] = seasonal_mean[doy(t)] + phi * residual[t-1]
panel <- panel %>%
  arrange(GID_2, date) %>%
  group_by(GID_2) %>%
  mutate(
    residual_lag1 = lag(residual, 1),
    pred_ntl = seasonal_mean + phi * coalesce(residual_lag1, 0),
    forecast_error = ntl_mean_built - pred_ntl
  ) %>%
  ungroup()

cat(sprintf("Predictions computed: %s rows\n", format(nrow(panel), big.mark = ",")))

# 4. STANDARDISE ERRORS USING TRAIN STATS

cat("\n=== Standardizing Errors ===\n")

# Join TRAIN-only statistics
panel <- panel %>%
  left_join(train_stats, by = "GID_2")

# Handle missing stats (shouldn't happen, but be safe)
panel <- panel %>%
  mutate(
    err_mean = coalesce(err_mean, 0),
    err_sd = coalesce(err_sd, 1),
    err_q95 = coalesce(err_q95, quantile(abs(forecast_error), 0.95, na.rm = TRUE))
  )

# Compute anomaly score (z-score using TRAIN parameters)
panel <- panel %>%
  mutate(
    anomaly_score = (forecast_error - err_mean) / err_sd,
    anomaly_abs = abs(forecast_error),
    above_q95 = as.integer(anomaly_abs > err_q95)
  )

cat(sprintf("Anomaly score: mean=%.4f, sd=%.4f\n",
            mean(panel$anomaly_score, na.rm = TRUE),
            sd(panel$anomaly_score, na.rm = TRUE)))

# 5. ADD DROP-FOCUSED ANOMALY FEATURES (NEW in v1.1)

cat("\n=== Computing Drop-Focused Features ===\n")

# anomaly_drop captures negative anomalies (NTL drops below expected)
# Higher values = bigger drops = more concerning for outages
panel <- panel %>%
  mutate(
    anomaly_drop = pmax(0, -anomaly_score),
    anomaly_drop_abs = pmax(0, -forecast_error)  # Raw units version
  )

cat(sprintf("Anomaly drop: mean=%.4f, max=%.4f\n",
            mean(panel$anomaly_drop, na.rm = TRUE),
            max(panel$anomaly_drop, na.rm = TRUE)))

# 6. COMPUTE ROLLING WINDOW FEATURES (NA-ROBUST)

cat("\n=== Computing Rolling Window Features (NA-Robust) ===\n")

# NA-robust rolling functions
# These return NA if all values in window are NA, otherwise compute on available data
roll_max_na <- function(x, before = 6) {
  slider::slide_dbl(x, ~{
    vals <- .x[!is.na(.x)]
    if (length(vals) == 0) NA_real_ else max(vals)
  }, .before = before, .complete = FALSE)
}

roll_mean_na <- function(x, before = 6) {
  slider::slide_dbl(x, ~{
    vals <- .x[!is.na(.x)]
    if (length(vals) == 0) NA_real_ else mean(vals)
  }, .before = before, .complete = FALSE)
}

roll_sum_na <- function(x, before = 6) {
  slider::slide_dbl(x, ~{
    vals <- .x[!is.na(.x)]
    if (length(vals) == 0) NA_real_ else sum(vals)
  }, .before = before, .complete = FALSE)
}

# Apply rolling features per municipality
panel <- panel %>%
  arrange(GID_2, date) %>%
  group_by(GID_2) %>%
  mutate(
    # Original anomaly rolling features (7-day lookback including today)
    anomaly_max_7d = roll_max_na(anomaly_abs),
    anomaly_mean_7d = roll_mean_na(anomaly_abs),
    anomaly_above_q95_7d = roll_sum_na(above_q95),

    # Drop-focused rolling features (NEW)
    anomaly_drop_max_7d = roll_max_na(anomaly_drop),
    anomaly_drop_mean_7d = roll_mean_na(anomaly_drop),
    anomaly_drop_above_q95_7d = roll_sum_na(as.integer(anomaly_drop > 2))  # 2 SD threshold
  ) %>%
  ungroup()

cat("Rolling features computed (NA-robust)\n")

# 7. COMPUTE RUN LENGTH (consecutive anomaly days)

cat("\n=== Computing Anomaly Run Length ===\n")

# Run length: consecutive days with anomaly_score > 2 (or anomaly_drop > 2)
compute_run_length <- function(is_anom) {
  n <- length(is_anom)
  result <- integer(n)
  current_run <- 0L

  for (i in seq_len(n)) {
    if (is.na(is_anom[i])) {
      current_run <- 0L
    } else if (is_anom[i]) {
      current_run <- current_run + 1L
    } else {
      current_run <- 0L
    }
    result[i] <- current_run
  }
  result
}

panel <- panel %>%
  arrange(GID_2, date) %>%
  group_by(GID_2) %>%
  mutate(
    anomaly_run_length = compute_run_length(anomaly_score > 2),
    anomaly_drop_run_length = compute_run_length(anomaly_drop > 2)  # NEW
  ) %>%
  ungroup()

cat(sprintf("Run length: max=%d, mean=%.2f\n",
            max(panel$anomaly_run_length, na.rm = TRUE),
            mean(panel$anomaly_run_length, na.rm = TRUE)))

# 8. ADD FLEET RANK FEATURES PER DAY (NEW in v1.1)

cat("\n=== Computing Fleet Rank Features ===\n")

# National rank: how anomalous is this municipality compared to all others today?
panel <- panel %>%
  group_by(date) %>%
  mutate(
    n_munis_day = n(),
    # Rank by anomaly_drop (higher drop = higher rank = more concerning)
    anomaly_drop_rank_national = rank(-anomaly_drop, ties.method = "average", na.last = "keep"),
    anomaly_drop_pctrank_national = anomaly_drop_rank_national / n_munis_day,
    # Also rank by anomaly_score for completeness
    anomaly_score_rank_national = rank(-anomaly_score, ties.method = "average", na.last = "keep"),
    anomaly_score_pctrank_national = anomaly_score_rank_national / n_munis_day
  ) %>%
  ungroup() %>%
  select(-n_munis_day)

# State-level rank: how anomalous within the state?
# Check if state column exists (NAME_1 or similar)
state_col <- NULL
if ("NAME_1" %in% names(panel)) {
  state_col <- "NAME_1"
} else if ("GID_1" %in% names(panel)) {
  state_col <- "GID_1"
}

if (!is.null(state_col)) {
  cat(sprintf("Computing state-level ranks using column: %s\n", state_col))

  panel <- panel %>%
    group_by(date, !!sym(state_col)) %>%
    mutate(
      n_munis_state = n(),
      anomaly_drop_rank_state = rank(-anomaly_drop, ties.method = "average", na.last = "keep"),
      anomaly_drop_pctrank_state = anomaly_drop_rank_state / n_munis_state
    ) %>%
    ungroup() %>%
    select(-n_munis_state)
} else {
  cat("No state column found, skipping state-level ranks\n")
  panel <- panel %>%
    mutate(
      anomaly_drop_rank_state = NA_real_,
      anomaly_drop_pctrank_state = NA_real_
    )
}

cat(sprintf("Fleet ranks computed: national + state\n"))

# 9. SELECT AND SAVE ANOMALY FEATURES

cat("\n=== Saving Anomaly Features ===\n")

# Select columns for output
anomaly_features <- panel %>%
  select(
    GID_2, date,
    # Core anomaly features (v1)
    anomaly_score,
    anomaly_abs,
    # Rolling features (v1)
    anomaly_max_7d,
    anomaly_mean_7d,
    anomaly_above_q95_7d,
    anomaly_run_length,
    # Drop-focused features (NEW v1.1)
    anomaly_drop,
    anomaly_drop_abs,
    anomaly_drop_max_7d,
    anomaly_drop_mean_7d,
    anomaly_drop_above_q95_7d,
    anomaly_drop_run_length,
    # Fleet rank features (NEW v1.1)
    anomaly_drop_rank_national,
    anomaly_drop_pctrank_national,
    anomaly_score_rank_national,
    anomaly_score_pctrank_national,
    anomaly_drop_rank_state,
    anomaly_drop_pctrank_state,
    # Raw forecast components (for diagnostics)
    seasonal_mean,
    pred_ntl,
    forecast_error
  )

# Save as RDS (for R pipeline)
saveRDS(anomaly_features, file.path(output_dir, "anomaly_features.rds"))
cat(sprintf("  Saved: %s/anomaly_features.rds\n", output_dir))

# Save as compressed CSV (for inspection) - using gzip to reduce disk usage
write_csv(anomaly_features, gzfile(file.path(output_dir, "anomaly_features.csv.gz")))
cat(sprintf("  Saved: %s/anomaly_features.csv.gz\n", output_dir))

# Save metadata
metadata <- tibble(
  scoring_run_tag = SCORING_RUN_TAG,
  anomaly_model_run_tag = ANOMALY_RUN_TAG,
  model_train_end = as.character(model$config$train_end),
  scoring_date = as.character(Sys.time()),
  n_rows = nrow(anomaly_features),
  n_munis = n_distinct(anomaly_features$GID_2),
  date_min = as.character(min(anomaly_features$date)),
  date_max = as.character(max(anomaly_features$date)),
  features = paste(setdiff(names(anomaly_features), c("GID_2", "date")), collapse = ", ")
)
write_csv(metadata, file.path(output_dir, "metadata.csv"))
cat(sprintf("  Saved: %s/metadata.csv\n", output_dir))

# Update pointer files
writeLines(SCORING_RUN_TAG, file.path(output_dir, "../latest_run_tag.txt"))
cat(sprintf("  Updated: data/processed/anomaly_features/latest_run_tag.txt\n"))

# SUMMARY STATISTICS

cat("\n=== Feature Summary Statistics ===\n")

feature_cols <- c("anomaly_score", "anomaly_abs", "anomaly_drop",
                  "anomaly_max_7d", "anomaly_mean_7d", "anomaly_drop_max_7d",
                  "anomaly_drop_mean_7d", "anomaly_drop_pctrank_national")

summary_stats <- anomaly_features %>%
  select(all_of(feature_cols)) %>%
  summarise(
    across(
      everything(),
      list(
        mean = ~mean(., na.rm = TRUE),
        sd = ~sd(., na.rm = TRUE),
        na_pct = ~100 * mean(is.na(.))
      ),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  pivot_longer(everything(), names_to = "metric", values_to = "value")

print(head(summary_stats, 24))

# By period
cat("\n=== By Period ===\n")
period_summary <- anomaly_features %>%
  mutate(
    period = case_when(
      date <= as.Date("2019-12-31") ~ "TRAIN",
      date <= as.Date("2020-06-30") ~ "VAL",
      TRUE ~ "TEST"
    )
  ) %>%
  group_by(period) %>%
  summarise(
    n_rows = n(),
    mean_anomaly_score = mean(anomaly_score, na.rm = TRUE),
    mean_anomaly_drop = mean(anomaly_drop, na.rm = TRUE),
    pct_drop_gt2 = 100 * mean(anomaly_drop > 2, na.rm = TRUE),
    .groups = "drop"
  )

print(period_summary)

cat("\n=== Done ===\n")
cat(sprintf("Anomaly features saved to: %s\n", output_dir))
