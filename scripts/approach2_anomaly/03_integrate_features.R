# 03_integrate_features.R
# Join anomaly features to existing model-ready dataset
# Creates enhanced dataset for combined model training
#
# v1.1 Updates:
# - Run-tagged output folders (data/processed/model_ready_anomaly/<run_tag>/)
# - Pointer files for latest run
# - Support for new v1.1 anomaly features (drop-focused, fleet ranks)
#
# v1.2 Updates:
# - WARNING-SAFE LAGS: Create lag1/lag2/lag3 of all anomaly features
#   so predictions at day t use only information from day t-1 or earlier
# - For modeling, use *_lag1 features, not same-day anomaly features

library(tidyverse)

# CONFIGURATION

# Get anomaly features run tag
args <- commandArgs(trailingOnly = TRUE)
args <- args[!grepl("^--", args)]

if (length(args) > 0) {
  ANOMALY_FEATURES_RUN_TAG <- args[1]
} else {
  ANOMALY_FEATURES_RUN_TAG <- readLines("data/processed/anomaly_features/latest_run_tag.txt", warn = FALSE)[1]
}

# Generate new run tag for integrated outputs
INTEGRATION_RUN_TAG <- format(Sys.time(), "%Y%m%d_%H%M%S")

# Input paths
model_ready_path <- "data/model_ready/features_engineered.rds"
anomaly_features_path <- file.path("data/processed/anomaly_features", ANOMALY_FEATURES_RUN_TAG, "anomaly_features.rds")

# Output path (NEW: run-tagged directory)
output_dir <- file.path("data/processed/model_ready_anomaly", INTEGRATION_RUN_TAG)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== 03_integrate_features.R ===\n")
cat(sprintf("Model-ready data: %s\n", model_ready_path))
cat(sprintf("Anomaly features run: %s\n", ANOMALY_FEATURES_RUN_TAG))
cat(sprintf("Anomaly features: %s\n", anomaly_features_path))
cat(sprintf("Integration run tag: %s\n", INTEGRATION_RUN_TAG))
cat(sprintf("Output directory: %s\n", output_dir))

# 1. LOAD DATASETS

cat("\n=== Loading Datasets ===\n")

# Load existing model-ready data (Approach 1 baseline)
model_ready <- readRDS(model_ready_path)
model_ready <- model_ready %>% mutate(date = as.Date(date))  # Ensure Date type
cat(sprintf("Model-ready data: %s rows, %d columns\n",
            format(nrow(model_ready), big.mark = ","),
            ncol(model_ready)))
cat(sprintf("  Date range: %s to %s\n", min(model_ready$date), max(model_ready$date)))

# Load anomaly features
anomaly_features <- readRDS(anomaly_features_path)
anomaly_features <- anomaly_features %>% mutate(date = as.Date(date))  # Ensure Date type
cat(sprintf("Anomaly features: %s rows, %d columns\n",
            format(nrow(anomaly_features), big.mark = ","),
            ncol(anomaly_features)))

# List anomaly feature columns (excluding keys and diagnostics)
anomaly_feature_cols <- setdiff(names(anomaly_features),
                                 c("GID_2", "date", "seasonal_mean", "pred_ntl", "forecast_error"))
cat(sprintf("Anomaly feature columns: %d\n", length(anomaly_feature_cols)))
cat(sprintf("  %s\n", paste(anomaly_feature_cols, collapse = ", ")))

# 2. CHECK KEY ALIGNMENT

cat("\n=== Checking Key Alignment ===\n")

# Keys should be GID_2 and date
model_keys <- model_ready %>% select(GID_2, date) %>% distinct()
anomaly_keys <- anomaly_features %>% select(GID_2, date) %>% distinct()

cat(sprintf("Model-ready keys: %s\n", format(nrow(model_keys), big.mark = ",")))
cat(sprintf("Anomaly keys: %s\n", format(nrow(anomaly_keys), big.mark = ",")))

# Check overlap
in_both <- inner_join(model_keys, anomaly_keys, by = c("GID_2", "date"))
cat(sprintf("Keys in both: %s\n", format(nrow(in_both), big.mark = ",")))

# Keys only in model-ready (shouldn't be many)
only_model <- anti_join(model_keys, anomaly_keys, by = c("GID_2", "date"))
if (nrow(only_model) > 0) {
  cat(sprintf("WARNING: %d keys only in model-ready (missing anomaly features)\n",
              nrow(only_model)))
}

# Keys only in anomaly (can happen if model-ready was filtered)
only_anomaly <- anti_join(anomaly_keys, model_keys, by = c("GID_2", "date"))
if (nrow(only_anomaly) > 0) {
  cat(sprintf("Note: %d keys only in anomaly (not in model-ready)\n",
              nrow(only_anomaly)))
}

# 3. JOIN DATASETS

cat("\n=== Joining Datasets ===\n")

# Select anomaly features to join (exclude diagnostics)
anomaly_to_join <- anomaly_features %>%
  select(GID_2, date, all_of(anomaly_feature_cols))

# Left join to keep all model-ready rows
combined <- model_ready %>%
  left_join(anomaly_to_join, by = c("GID_2", "date"))

cat(sprintf("Combined dataset: %s rows, %d columns\n",
            format(nrow(combined), big.mark = ","),
            ncol(combined)))

# Check for missing anomaly features
missing_anomaly <- sum(is.na(combined$anomaly_score))
if (missing_anomaly > 0) {
  cat(sprintf("WARNING: %d rows missing anomaly_score (%.2f%%)\n",
              missing_anomaly, 100 * missing_anomaly / nrow(combined)))

  # Fill missing with 0 for score-based features (neutral)
  # But keep NA for rank features (they should be NA if anomaly score is NA)
  combined <- combined %>%
    mutate(
      across(
        c(anomaly_score, anomaly_abs, anomaly_max_7d, anomaly_mean_7d,
          anomaly_above_q95_7d, anomaly_run_length,
          anomaly_drop, anomaly_drop_abs, anomaly_drop_max_7d,
          anomaly_drop_mean_7d, anomaly_drop_above_q95_7d, anomaly_drop_run_length),
        ~coalesce(., 0)
      )
    )
  cat("  Filled missing score-based features with 0\n")
}

# 3B. CREATE DATE-AWARE LAGGED FEATURES (v1.3)

cat("\n=== Creating Date-Aware Lagged Features ===\n")

# For each anomaly feature, create lag1/lag2/lag3 versions using DATE-SHIFT JOINS
# This ensures lag1 = previous CALENDAR day, not previous observed row
# Critical for FORECAST mode: avoids information leakage when gaps exist

# Get all anomaly feature columns (same-day)
sameday_anomaly_cols <- anomaly_feature_cols
cat(sprintf("Creating date-aware lags for %d same-day anomaly features\n", length(sameday_anomaly_cols)))

# Build base anomaly table with unique (GID_2, date) keys
anomaly_base <- combined %>%
  select(GID_2, date, all_of(sameday_anomaly_cols)) %>%
  distinct(GID_2, date, .keep_all = TRUE)

# Assert uniqueness
n_keys <- nrow(anomaly_base)
n_unique_keys <- nrow(distinct(anomaly_base, GID_2, date))
if (n_keys != n_unique_keys) {
  stop(sprintf("FATAL: Duplicate (GID_2, date) keys detected: %d rows but %d unique keys",
               n_keys, n_unique_keys))
}
cat(sprintf("Anomaly base table: %s unique (GID_2, date) keys\n", format(n_keys, big.mark = ",")))

# Compute detailed gap diagnostics before creating lags
cat("\n--- Gap Diagnostics (per GID_2) ---\n")

gap_by_muni <- combined %>%
  group_by(GID_2) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(day_gap = as.numeric(date - lag(date, 1))) %>%
  summarise(
    n_days = n(),
    n_gaps = sum(day_gap > 1, na.rm = TRUE),
    has_gap = any(day_gap > 1, na.rm = TRUE),
    median_gap = median(day_gap[day_gap > 1], na.rm = TRUE),
    max_gap = max(day_gap, na.rm = TRUE),
    .groups = "drop"
  )

n_munis_total <- nrow(gap_by_muni)
n_munis_with_gaps <- sum(gap_by_muni$has_gap, na.rm = TRUE)
pct_munis_with_gaps <- 100 * n_munis_with_gaps / n_munis_total
overall_median_gap <- median(gap_by_muni$median_gap, na.rm = TRUE)
overall_max_gap <- max(gap_by_muni$max_gap, na.rm = TRUE)

cat(sprintf("Municipalities with gaps (diff(date) > 1): %d / %d (%.1f%%)\n",
            n_munis_with_gaps, n_munis_total, pct_munis_with_gaps))
cat(sprintf("Overall median gap (among munis with gaps): %.1f days\n",
            ifelse(is.na(overall_median_gap), 0, overall_median_gap)))
cat(sprintf("Overall max gap: %d days\n", overall_max_gap))

# Top 10 worst municipalities by max gap
cat("\nTop 10 municipalities by max gap:\n")
top10_gaps <- gap_by_muni %>%
  filter(has_gap) %>%
  arrange(desc(max_gap), desc(n_gaps)) %>%
  head(10)

if (nrow(top10_gaps) > 0) {
  print(top10_gaps %>% select(GID_2, n_days, n_gaps, median_gap, max_gap))
} else {
  cat("  (No gaps detected - all date sequences are contiguous)\n")
}
cat("\n")

# Pre-record row count for assertion
n_rows_before <- nrow(combined)

# Create lag tables by shifting dates forward
# lag1: data from (date - 1), so we set join_date = date + 1
for (lag_days in 1:3) {
  cat(sprintf("  Creating lag%d features (date + %d shift)...\n", lag_days, lag_days))

  # Create lagged table: shift date forward by lag_days
  # When we join on date, we get the value from lag_days ago
  lag_table <- anomaly_base %>%
    mutate(date = date + lag_days)  # shift forward: value at t becomes available for t+lag_days

  # Rename columns to *_lagN
  lag_suffix <- paste0("_lag", lag_days)
  for (col in sameday_anomaly_cols) {
    names(lag_table)[names(lag_table) == col] <- paste0(col, lag_suffix)
  }

  # Left join back onto combined
  combined <- combined %>%
    left_join(lag_table, by = c("GID_2", "date"))
}

# Assert row count unchanged
n_rows_after <- nrow(combined)
if (n_rows_after != n_rows_before) {
  stop(sprintf("FATAL: Row count changed after lag joins: %d -> %d", n_rows_before, n_rows_after))
}
cat(sprintf("Row count preserved: %s\n", format(n_rows_after, big.mark = ",")))

# Get the new lagged feature column names
lagged_anomaly_cols <- c(
  paste0(sameday_anomaly_cols, "_lag1"),
  paste0(sameday_anomaly_cols, "_lag2"),
  paste0(sameday_anomaly_cols, "_lag3")
)

cat(sprintf("Created %d lagged anomaly features\n", length(lagged_anomaly_cols)))
cat(sprintf("  Lag1: %d features\n", length(sameday_anomaly_cols)))
cat(sprintf("  Lag2: %d features\n", length(sameday_anomaly_cols)))
cat(sprintf("  Lag3: %d features\n", length(sameday_anomaly_cols)))

# Compute NA rates for lagged features (caused by gaps or start of series)
cat("\nNA rates for lagged features (due to date gaps or series start):\n")
for (lag_days in 1:3) {
  lag_col <- paste0(sameday_anomaly_cols[1], "_lag", lag_days)
  if (lag_col %in% names(combined)) {
    na_count <- sum(is.na(combined[[lag_col]]))
    na_pct <- 100 * na_count / nrow(combined)
    cat(sprintf("  lag%d: %d NAs (%.2f%%) - example: %s\n",
                lag_days, na_count, na_pct, lag_col))
  }
}
cat("Note: NAs are NOT backfilled - they represent missing calendar days\n")

# Update anomaly_feature_cols to include lagged versions
# NOTE: For modeling, we should use *_lag1 features, NOT same-day features
anomaly_feature_cols_all <- c(anomaly_feature_cols, lagged_anomaly_cols)
cat(sprintf("Total anomaly features (same-day + lagged): %d\n", length(anomaly_feature_cols_all)))

# 4. VERIFY FEATURE CORRELATIONS

cat("\n=== Feature Correlations with Target ===\n")

# Check correlations with outage indicator
if ("outage_3h_or_more" %in% names(combined)) {
  # Focus on LAG1 features (warning-safe) for correlation analysis
  lag1_cols <- paste0(sameday_anomaly_cols, "_lag1")
  corr_features <- intersect(lag1_cols, names(combined))
  corr_features <- corr_features[sapply(combined[corr_features], is.numeric)]

  cat("Correlation with outage_3h_or_more (LAG1 features - warning-safe):\n")
  correlations <- combined %>%
    select(outage_3h_or_more, all_of(corr_features)) %>%
    cor(use = "pairwise.complete.obs")

  corr_with_target <- correlations["outage_3h_or_more", -1]
  corr_sorted <- sort(corr_with_target, decreasing = TRUE)
  print(round(corr_sorted, 4))

  # Also show same-day correlations for comparison
  cat("\nCorrelation with outage_3h_or_more (same-day features - for reference):\n")
  sameday_corr_features <- intersect(sameday_anomaly_cols, names(combined))
  sameday_corr_features <- sameday_corr_features[sapply(combined[sameday_corr_features], is.numeric)]

  sameday_correlations <- combined %>%
    select(outage_3h_or_more, all_of(sameday_corr_features)) %>%
    cor(use = "pairwise.complete.obs")

  sameday_corr_with_target <- sameday_correlations["outage_3h_or_more", -1]
  sameday_corr_sorted <- sort(sameday_corr_with_target, decreasing = TRUE)
  print(round(sameday_corr_sorted, 4))
}

# 5. SAVE COMBINED DATASET

cat("\n=== Saving Combined Dataset ===\n")

# Save full combined dataset
saveRDS(combined, file.path(output_dir, "model_ready_panel_combined.rds"))
cat(sprintf("  Saved: %s/model_ready_panel_combined.rds\n", output_dir))

# Save column inventory (distinguish same-day vs lagged anomaly features)
col_inventory <- tibble(
  column = names(combined),
  type = map_chr(combined, ~class(.)[1]),
  n_missing = map_int(combined, ~sum(is.na(.))),
  pct_missing = 100 * n_missing / nrow(combined),
  is_anomaly_sameday = column %in% sameday_anomaly_cols,
  is_anomaly_lag1 = column %in% paste0(sameday_anomaly_cols, "_lag1"),
  is_anomaly_lag2 = column %in% paste0(sameday_anomaly_cols, "_lag2"),
  is_anomaly_lag3 = column %in% paste0(sameday_anomaly_cols, "_lag3"),
  is_anomaly_feature = is_anomaly_sameday | is_anomaly_lag1 | is_anomaly_lag2 | is_anomaly_lag3,
  warning_safe = !is_anomaly_sameday | !is_anomaly_feature  # lag features are warning-safe
)
write_csv(col_inventory, file.path(output_dir, "column_inventory.csv"))
cat(sprintf("  Saved: %s/column_inventory.csv\n", output_dir))

# 6. CREATE TRAIN/VAL/TEST SPLITS

cat("\n=== Creating Data Splits ===\n")

# Define split boundaries (same as Approach 1)
TRAIN_END <- as.Date("2019-12-31")
VAL_END <- as.Date("2020-06-30")

train_data <- combined %>% filter(date <= TRAIN_END)
val_data <- combined %>% filter(date > TRAIN_END & date <= VAL_END)
test_data <- combined %>% filter(date > VAL_END)

cat(sprintf("TRAIN: %s rows (%.1f%%)\n",
            format(nrow(train_data), big.mark = ","),
            100 * nrow(train_data) / nrow(combined)))
cat(sprintf("VAL: %s rows (%.1f%%)\n",
            format(nrow(val_data), big.mark = ","),
            100 * nrow(val_data) / nrow(combined)))
cat(sprintf("TEST: %s rows (%.1f%%)\n",
            format(nrow(test_data), big.mark = ","),
            100 * nrow(test_data) / nrow(combined)))

# Save splits
saveRDS(train_data, file.path(output_dir, "train.rds"))
saveRDS(val_data, file.path(output_dir, "val.rds"))
saveRDS(test_data, file.path(output_dir, "test.rds"))

cat("\nSplit files saved\n")

# 7. UPDATE POINTER FILES

cat("\n=== Updating Pointer Files ===\n")

# Update latest run tag pointer
writeLines(INTEGRATION_RUN_TAG, file.path(output_dir, "../latest_run_tag.txt"))
cat(sprintf("  Updated: data/processed/model_ready_anomaly/latest_run_tag.txt\n"))

# Save metadata
metadata <- tibble(
  integration_run_tag = INTEGRATION_RUN_TAG,
  anomaly_features_run_tag = ANOMALY_FEATURES_RUN_TAG,
  model_ready_source = model_ready_path,
  integration_date = as.character(Sys.time()),
  n_rows_total = nrow(combined),
  n_rows_train = nrow(train_data),
  n_rows_val = nrow(val_data),
  n_rows_test = nrow(test_data),
  n_anomaly_features_sameday = length(sameday_anomaly_cols),
  n_anomaly_features_lag1 = length(sameday_anomaly_cols),
  n_anomaly_features_lag2 = length(sameday_anomaly_cols),
  n_anomaly_features_lag3 = length(sameday_anomaly_cols),
  n_anomaly_features_total = length(anomaly_feature_cols_all),
  anomaly_features_sameday = paste(sameday_anomaly_cols, collapse = ", "),
  anomaly_features_lag1 = paste(paste0(sameday_anomaly_cols, "_lag1"), collapse = ", ")
)
write_csv(metadata, file.path(output_dir, "metadata.csv"))
cat(sprintf("  Saved: %s/metadata.csv\n", output_dir))

# 8. SUMMARY STATISTICS

cat("\n=== Anomaly Feature Summary by Split (LAG1 - Warning-Safe) ===\n")

summary_by_split <- combined %>%
  mutate(
    split = case_when(
      date <= TRAIN_END ~ "TRAIN",
      date <= VAL_END ~ "VAL",
      TRUE ~ "TEST"
    )
  ) %>%
  group_by(split) %>%
  summarise(
    n = n(),
    outage_rate = mean(outage_3h_or_more, na.rm = TRUE),
    mean_anomaly_score_lag1 = mean(anomaly_score_lag1, na.rm = TRUE),
    mean_anomaly_drop_lag1 = mean(anomaly_drop_lag1, na.rm = TRUE),
    mean_drop_pctrank_lag1 = mean(anomaly_drop_pctrank_national_lag1, na.rm = TRUE),
    .groups = "drop"
  )

print(summary_by_split)

# Anomaly features by outage status (LAG1)
cat("\n=== Anomaly Features by Outage Status (LAG1 - Warning-Safe) ===\n")

combined %>%
  mutate(outage = factor(outage_3h_or_more, labels = c("No Outage", "Outage"))) %>%
  group_by(outage) %>%
  summarise(
    n = n(),
    mean_anomaly_score_lag1 = mean(anomaly_score_lag1, na.rm = TRUE),
    mean_anomaly_drop_lag1 = mean(anomaly_drop_lag1, na.rm = TRUE),
    mean_drop_max_7d_lag1 = mean(anomaly_drop_max_7d_lag1, na.rm = TRUE),
    mean_drop_pctrank_lag1 = mean(anomaly_drop_pctrank_national_lag1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  print()

# Compare same-day vs lag1 correlation with target
cat("\n=== Same-Day vs Lag1 Predictive Power ===\n")
cat("(Higher absolute correlation = stronger predictive signal)\n\n")

if ("outage_3h_or_more" %in% names(combined)) {
  comparison_cols <- c("anomaly_score", "anomaly_drop", "anomaly_drop_max_7d")
  for (col in comparison_cols) {
    lag1_col <- paste0(col, "_lag1")
    if (col %in% names(combined) && lag1_col %in% names(combined)) {
      corr_sameday <- cor(combined[[col]], combined$outage_3h_or_more, use = "complete.obs")
      corr_lag1 <- cor(combined[[lag1_col]], combined$outage_3h_or_more, use = "complete.obs")
      cat(sprintf("  %s: same-day=%.4f, lag1=%.4f (ratio=%.2f)\n",
                  col, corr_sameday, corr_lag1, corr_lag1/corr_sameday))
    }
  }
}

cat("\n=== Done ===\n")
cat(sprintf("Combined data saved to: %s\n", output_dir))
