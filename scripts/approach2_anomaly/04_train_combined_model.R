# 04_train_combined_model.R
# Train and compare models with different feature sets:
# - Baseline only (replicates Approach 1)
# - Anomaly features only
# - Combined (baseline + anomaly)
#
# v1.2 Updates:
# - WARNING-SAFE MODE: Use only *_lag1 anomaly features for combined models
#   (predictions at day t use only information from day t-1 or earlier)
# - DROP-ONLY VARIANT: Train additional model using only anomaly_drop_* features
# - Model variants: baseline, anomaly_only, combined_lag1, combined_drop_only_lag1

library(tidyverse)
library(tidymodels)
library(xgboost)

# CONFIGURATION

# Parse command line args: [run_tag] [task_mode]
args <- commandArgs(trailingOnly = TRUE)
args <- args[!grepl("^--", args)]

# Arg 1: model_ready_anomaly run tag
if (length(args) > 0 && nzchar(args[1])) {
  MODEL_READY_RUN_TAG <- args[1]
} else {
  tag_file <- "data/processed/model_ready_anomaly/latest_run_tag.txt"
  if (!file.exists(tag_file)) {
    stop("No run tag provided and latest_run_tag.txt not found")
  }
  MODEL_READY_RUN_TAG <- readLines(tag_file, warn = FALSE)[1]
}
if (is.na(MODEL_READY_RUN_TAG) || !nzchar(MODEL_READY_RUN_TAG)) {
  stop("MODEL_READY_RUN_TAG is empty or NA")
}

# Arg 2: task_mode ("nowcast" or "forecast"), default "nowcast"
task_mode <- "nowcast"
if (length(args) > 1) {
  if (args[2] %in% c("nowcast", "forecast")) {
    task_mode <- args[2]
  } else {
    stop("Invalid task_mode: '", args[2], "'. Use 'nowcast' or 'forecast'.")
  }
}

# Run tag for this training run (includes task_mode)
RUN_TAG <- paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_", task_mode)

# Input paths (run-tagged)
combined_data_dir <- file.path("data/processed/model_ready_anomaly", MODEL_READY_RUN_TAG)

# Output directory
output_dir <- file.path("runs/combined", RUN_TAG)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Model hyperparameters (same as Approach 1 for fair comparison)
XGB_PARAMS <- list(
  trees = 500,
  min_n = 20,
  tree_depth = 6,
  learn_rate = 0.05,
  loss_reduction = 0,
  sample_size = 0.8
)

cat("=== 04_train_combined_model.R ===\n")
cat(sprintf("Run tag: %s\n", RUN_TAG))
cat(sprintf("Output directory: %s\n", output_dir))

# 1. LOAD DATA

cat("\n=== Loading Data ===\n")

train_data <- readRDS(file.path(combined_data_dir, "train.rds"))
val_data <- readRDS(file.path(combined_data_dir, "val.rds"))
test_data <- readRDS(file.path(combined_data_dir, "test.rds"))

cat(sprintf("TRAIN: %s rows\n", format(nrow(train_data), big.mark = ",")))
cat(sprintf("VAL: %s rows\n", format(nrow(val_data), big.mark = ",")))
cat(sprintf("TEST: %s rows\n", format(nrow(test_data), big.mark = ",")))

# 2. DEFINE FEATURE SETS (using 00_feature_config.R as source of truth)

cat("\n=== Defining Feature Sets ===\n")

# Source the feature config for EXACT Approach 1 parity
source("scripts/00_feature_config.R")

# Get all column names
all_cols <- names(train_data)

# STEP 1: Detect anomaly features FIRST (before any filtering)
# These are Approach 2 additions, not part of Approach 1
all_anomaly_features <- grep("^anomaly_", all_cols, value = TRUE)
cat(sprintf("Detected %d total anomaly features\n", length(all_anomaly_features)))

# v1.2: Categorize anomaly features
anomaly_sameday <- all_anomaly_features[!grepl("_lag[123]$", all_anomaly_features)]
anomaly_lag1 <- all_anomaly_features[grepl("_lag1$", all_anomaly_features)]
anomaly_lag2 <- all_anomaly_features[grepl("_lag2$", all_anomaly_features)]
anomaly_lag3 <- all_anomaly_features[grepl("_lag3$", all_anomaly_features)]

cat(sprintf("  Same-day anomaly features: %d\n", length(anomaly_sameday)))
cat(sprintf("  Lag1 anomaly features: %d\n", length(anomaly_lag1)))
cat(sprintf("  Lag2 anomaly features: %d\n", length(anomaly_lag2)))
cat(sprintf("  Lag3 anomaly features: %d\n", length(anomaly_lag3)))

# v1.2: Drop-only features (anomaly_drop_* and variants)
anomaly_drop_lag1 <- anomaly_lag1[grepl("anomaly_drop", anomaly_lag1)]
cat(sprintf("  Drop-only lag1 features: %d\n", length(anomaly_drop_lag1)))

# For backwards compatibility, use all anomaly features for exclusion
anomaly_features <- all_anomaly_features

# STEP 2: Get Approach 1 baseline features using config
# Exclude EXCLUDE_COLS (IDs, outcomes, same-day NTL, categoricals, etc.)
potential_features <- setdiff(all_cols, EXCLUDE_COLS)

# Also exclude anomaly features from baseline consideration
potential_features <- setdiff(potential_features, anomaly_features)

# Keep only numeric columns
potential_features <- potential_features[sapply(train_data[potential_features], is.numeric)]

# Use get_feature_set() for operational (forecast-safe) features
# This excludes same-day weather, keeps only lagged weather
baseline_features <- get_feature_set(potential_features, mode = "operational")

# For EXACT Approach 1 parity, also exclude these 3 columns not in original features_used.csv
APPROACH1_EXTRA_EXCLUDE <- c("has_weather", "is_covid_period", "nb_prop_drop_z2_lag1")
baseline_features <- setdiff(baseline_features, APPROACH1_EXTRA_EXCLUDE)

# Apply task_mode filter (FORECAST drops same-day NTL-derived features)
baseline_features <- apply_task_mode_filters(baseline_features, task_mode = task_mode,
                                              strict = TRUE, verbose = TRUE)

# STEP 3: Define feature sets
# Anomaly-only: all detected anomaly features (must be numeric) - FOR REFERENCE
anomaly_only_features <- all_anomaly_features[sapply(train_data[all_anomaly_features], is.numeric)]

# v1.2 WARNING-SAFE FEATURE SETS (use lag1 only)
# These features use information from day t-1 or earlier for predictions at day t

# Lag1 anomaly features only (warning-safe)
anomaly_lag1_features <- anomaly_lag1[sapply(train_data[anomaly_lag1], is.numeric)]

# Drop-only lag1 features (warning-safe, drop-focused)
drop_only_lag1_features <- anomaly_drop_lag1[sapply(train_data[anomaly_drop_lag1], is.numeric)]

# Combined features: baseline + lag1 anomaly (v1.2 primary)
combined_lag1_features <- union(baseline_features, anomaly_lag1_features)

# Drop-only combined: baseline + drop-only lag1 (v1.2 focused variant)
combined_drop_only_lag1_features <- union(baseline_features, drop_only_lag1_features)

# Legacy combined (same-day) - FOR COMPARISON with v1.1
combined_features <- union(baseline_features, anomaly_only_features)

cat(sprintf("\n=== Feature Counts ===\n"))
cat(sprintf("Baseline features (Approach 1 replica): %d\n", length(baseline_features)))
cat(sprintf("\nLegacy (v1.1, same-day - for comparison):\n"))
cat(sprintf("  Anomaly-only (all): %d\n", length(anomaly_only_features)))
cat(sprintf("  Combined (all): %d\n", length(combined_features)))
cat(sprintf("\nv1.2 WARNING-SAFE (lag1 only):\n"))
cat(sprintf("  Anomaly lag1 only: %d\n", length(anomaly_lag1_features)))
cat(sprintf("  Combined (lag1): %d\n", length(combined_lag1_features)))
cat(sprintf("  Drop-only lag1: %d\n", length(drop_only_lag1_features)))
cat(sprintf("  Combined (drop-only lag1): %d\n", length(combined_drop_only_lag1_features)))

# Verify: baseline + lag1 = combined_lag1
stopifnot(setequal(union(baseline_features, anomaly_lag1_features), combined_lag1_features))
cat("\n✓ Verified: baseline ∪ anomaly_lag1 = combined_lag1\n")

# List lag1 anomaly features
cat(sprintf("\nAnomaly LAG1 features (%d):\n", length(anomaly_lag1_features)))
cat(paste("  ", sort(anomaly_lag1_features), collapse = "\n"), "\n")

# List drop-only lag1 features
cat(sprintf("\nDrop-only LAG1 features (%d):\n", length(drop_only_lag1_features)))
cat(paste("  ", sort(drop_only_lag1_features), collapse = "\n"), "\n")

# LEAKAGE CHECK: flag any suspicious columns
cat("\n=== Leakage Check ===\n")

# Check for same-day NTL (no _lag suffix)
sameday_ntl_pattern <- "^ntl_(mean|sum|sd)_(all|built|built_mask)$"
sameday_ntl <- grep(sameday_ntl_pattern, baseline_features, value = TRUE)
if (length(sameday_ntl) > 0) {
  cat("WARNING: Same-day NTL detected in baseline:\n")
  print(sameday_ntl)
} else {
  cat("✓ No same-day NTL in baseline\n")
}

# Check for same-day weather (no _lag suffix)
weather_vars <- c("atm", "dew", "max_dew", "min_dew", "lai_high", "lai_low",
                  "rain", "rh", "skin_temp", "temp", "wdr", "wind_u", "wind_v", "wsp")
sameday_weather <- intersect(weather_vars, baseline_features)
if (length(sameday_weather) > 0) {
  cat("WARNING: Same-day weather detected in baseline:\n")
  print(sameday_weather)
} else {
  cat("✓ No same-day weather in baseline\n")
}

# Check for anomaly features in baseline
anomaly_in_baseline <- grep("^anomaly_", baseline_features, value = TRUE)
if (length(anomaly_in_baseline) > 0) {
  cat("WARNING: Anomaly features detected in baseline:\n")
  print(anomaly_in_baseline)
} else {
  cat("✓ No anomaly features in baseline\n")
}

# APPROACH 1 PARITY CHECK
cat("\n=== Approach 1 Parity Check ===\n")

if (file.exists("data/model_ready/models/features_used.csv")) {
  approach1_feats <- read.csv("data/model_ready/models/features_used.csv")$feature
  cat(sprintf("Approach 1 features: %d\n", length(approach1_feats)))
  cat(sprintf("Our baseline features: %d\n", length(baseline_features)))

  if (setequal(baseline_features, approach1_feats)) {
    cat("✓ EXACT MATCH with Approach 1\n")
  } else {
    diff1 <- setdiff(approach1_feats, baseline_features)
    diff2 <- setdiff(baseline_features, approach1_feats)
    if (length(diff1) > 0) {
      cat(sprintf("In Approach 1 but not baseline (%d):\n", length(diff1)))
      cat(paste("  ", diff1, collapse = "\n"), "\n")
    }
    if (length(diff2) > 0) {
      cat(sprintf("In baseline but not Approach 1 (%d):\n", length(diff2)))
      cat(paste("  ", diff2, collapse = "\n"), "\n")
    }
  }
} else {
  cat("(Approach 1 features_used.csv not found - skipping parity check)\n")
}

# Save feature sets for reference
feature_sets <- list(
  # Approach 1 baseline
  baseline = baseline_features,
  # Legacy v1.1 (same-day, for comparison)
  anomaly_only = anomaly_only_features,
  combined = combined_features,
  # v1.2 warning-safe (lag1 only)
  anomaly_lag1 = anomaly_lag1_features,
  combined_lag1 = combined_lag1_features,
  drop_only_lag1 = drop_only_lag1_features,
  combined_drop_only_lag1 = combined_drop_only_lag1_features
)
saveRDS(feature_sets, file.path(output_dir, "feature_sets.rds"))

# 3. PREPARE DATA FOR MODELING

cat("\n=== Preparing Data ===\n")

# Function to prepare data for a feature set
prepare_model_data <- function(data, features, outcome = "outage_3h_or_more") {
  # Select features and outcome
  cols_to_use <- c(features, outcome)
  df <- data[, cols_to_use, drop = FALSE]

  # Remove columns with all NA
  all_na_cols <- names(df)[sapply(df, function(x) all(is.na(x)))]
  if (length(all_na_cols) > 0) {
    cat(sprintf("  Removing %d all-NA columns\n", length(all_na_cols)))
    df <- df[, !names(df) %in% all_na_cols, drop = FALSE]
  }

  # Convert outcome to factor for classification
  df[[outcome]] <- factor(df[[outcome]], levels = c(0, 1))

  df
}

# 4. MODEL SPECIFICATION

cat("\n=== Model Specification ===\n")

xgb_spec <- boost_tree(
  trees = XGB_PARAMS$trees,
  min_n = XGB_PARAMS$min_n,
  tree_depth = XGB_PARAMS$tree_depth,
  learn_rate = XGB_PARAMS$learn_rate,
  loss_reduction = XGB_PARAMS$loss_reduction,
  sample_size = XGB_PARAMS$sample_size
) %>%
  set_engine("xgboost",
             nthread = parallel::detectCores() - 1,
             colsample_bynode = 0.8) %>%
  set_mode("classification")

cat("XGBoost parameters:\n")
print(XGB_PARAMS)

# 5. TRAIN MODELS

train_model <- function(train_df, val_df, feature_set, model_name) {
  cat(sprintf("\n=== Training: %s ===\n", model_name))

  # Prepare data
  train_prepped <- prepare_model_data(train_df, feature_set)
  val_prepped <- prepare_model_data(val_df, feature_set)

  cat(sprintf("  Features: %d\n", ncol(train_prepped) - 1))
  cat(sprintf("  Training rows: %s\n", format(nrow(train_prepped), big.mark = ",")))

  # Create recipe
  recipe_spec <- recipe(outage_3h_or_more ~ ., data = train_prepped) %>%
    step_zv(all_predictors()) %>%
    step_impute_median(all_numeric_predictors())

  # Create workflow
  wf <- workflow() %>%
    add_recipe(recipe_spec) %>%
    add_model(xgb_spec)

  # Train
  start_time <- Sys.time()
  fitted_wf <- fit(wf, data = train_prepped)
  train_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  cat(sprintf("  Training time: %.1f seconds\n", train_time))

  # Predict on validation
  val_preds <- predict(fitted_wf, val_prepped, type = "prob") %>%
    bind_cols(predict(fitted_wf, val_prepped, type = "class")) %>%
    bind_cols(val_df %>% select(GID_2, date, outage_3h_or_more))

  # Predict on test
  test_prepped <- prepare_model_data(test_data, feature_set)
  test_preds <- predict(fitted_wf, test_prepped, type = "prob") %>%
    bind_cols(predict(fitted_wf, test_prepped, type = "class")) %>%
    bind_cols(test_data %>% select(GID_2, date, outage_3h_or_more))

  list(
    model = fitted_wf,
    val_preds = val_preds,
    test_preds = test_preds,
    train_time = train_time,
    n_features = ncol(train_prepped) - 1
  )
}

# Train all model variants
results <- list()

# Approach 1 baseline (always train)
results$baseline <- train_model(train_data, val_data, baseline_features, "Baseline (Approach 1)")

# Same-day features: only in NOWCAST mode (diagnostic only for forecast)
if (task_mode == "nowcast") {
  results$anomaly_only <- train_model(train_data, val_data, anomaly_only_features, "Anomaly Only (all, same-day)")
  results$combined <- train_model(train_data, val_data, combined_features, "Combined (v1.1, same-day)")
} else {
  cat("\n[FORECAST mode] Skipping same-day variants (anomaly_only, combined_v1.1) - diagnostic only\n")
}

# Anomaly lag1 only (forecast-safe)
results$anomaly_lag1 <- train_model(train_data, val_data, anomaly_lag1_features, "Anomaly (lag1, forecast-safe)")

# v1.2 WARNING-SAFE variants (lag1 features only)
results$combined_lag1 <- train_model(train_data, val_data, combined_lag1_features, "Combined (v1.2 lag1)")
results$combined_drop_only_lag1 <- train_model(train_data, val_data, combined_drop_only_lag1_features, "Combined (v1.2 drop-only lag1)")

# 6. SAVE PREDICTIONS (compressed with minimal columns to save disk)

cat("\n=== Saving Predictions ===\n")

# Minimal columns for predictions (disk space constraint)
minimal_cols <- c("GID_2", "date", "outage_3h_or_more", ".pred_1")

for (name in names(results)) {
  # Skip saving full model (too large, disk space limited)
  # saveRDS(results[[name]]$model, file.path(output_dir, sprintf("model_%s.rds", name)))

  # Save validation predictions (compressed, minimal columns)
  results[[name]]$val_preds %>%
    select(all_of(minimal_cols)) %>%
    write_csv(gzfile(file.path(output_dir, sprintf("predictions_val_%s.csv.gz", name))))

  # Save test predictions (compressed, minimal columns)
  results[[name]]$test_preds %>%
    select(all_of(minimal_cols)) %>%
    write_csv(gzfile(file.path(output_dir, sprintf("predictions_test_%s.csv.gz", name))))

  cat(sprintf("  Saved: predictions_val_%s.csv.gz, predictions_test_%s.csv.gz\n", name, name))
}

# 7. QUICK VALIDATION METRICS

cat("\n=== Validation Set Metrics (Quick Check) ===\n")

compute_quick_metrics <- function(preds, name) {
  truth_numeric <- as.numeric(as.character(preds$outage_3h_or_more))
  prob <- preds$.pred_1

  # ROC AUC
  roc_auc <- yardstick::roc_auc_vec(
    truth = factor(truth_numeric, levels = c(0, 1)),
    estimate = prob,
    event_level = "second"
  )

  # PR AUC
  pr_auc <- yardstick::pr_auc_vec(
    truth = factor(truth_numeric, levels = c(0, 1)),
    estimate = prob,
    event_level = "second"
  )

  # Top-10 precision (daily)
  daily_topk <- preds %>%
    mutate(truth = as.numeric(as.character(outage_3h_or_more))) %>%
    group_by(date) %>%
    arrange(desc(.pred_1)) %>%
    slice_head(n = 10) %>%
    ungroup()

  topk_precision <- mean(daily_topk$truth)
  baseline_rate <- mean(truth_numeric)
  lift_10 <- topk_precision / baseline_rate

  tibble(
    model = name,
    roc_auc = roc_auc,
    pr_auc = pr_auc,
    topk10_precision = topk_precision,
    lift_10 = lift_10
  )
}

# Build metrics only for trained models (dynamic)
val_metrics <- purrr::imap_dfr(results, ~compute_quick_metrics(.x$val_preds, .y))
print(val_metrics %>% mutate(across(where(is.numeric), ~round(., 4))))
write_csv(val_metrics, file.path(output_dir, "validation_metrics_quick.csv"))

# 8. SAVE METADATA

metadata <- list(
  run_tag    = RUN_TAG,
  task_mode  = task_mode,
  created_at = Sys.time(),
  version    = "1.2",
  xgb_params = XGB_PARAMS,
  n_train    = nrow(train_data),
  n_val      = nrow(val_data),
  n_test     = nrow(test_data),
  feature_counts = purrr::map_int(results, ~as.integer(.x$n_features)),
  train_times    = purrr::map_dbl(results, ~as.numeric(.x$train_time))
)

saveRDS(metadata, file.path(output_dir, "metadata.rds"))
writeLines(RUN_TAG, file.path(output_dir, "run_tag.txt"))
dir.create("runs/combined", showWarnings = FALSE, recursive = TRUE)
writeLines(RUN_TAG, file.path("runs/combined", paste0("latest_run_tag_", task_mode, ".txt")))

cat(sprintf("\n=== Done ===\n"))
cat(sprintf("Results saved to: %s\n", output_dir))
cat("\nRun 05_evaluate_combined.R for detailed comparison.\n")
