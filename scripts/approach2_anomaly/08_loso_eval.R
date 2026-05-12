# 08_loso_eval.R
# Leave-One-State-Out (LOSO) Cross-Validation
#
# Tests spatial generalisation by holding out one state at a time.
# This mimics the transfer learning scenario where we train on some regions
# and deploy to new, unseen regions.
#
# CRITICAL: Correct LOSO split design:
#   For each held-out state S:
#     TRAIN: state != S AND date <= 2019-12-31
#     VAL:   state != S AND date in [2020-01-01, 2020-06-30]
#     TEST:  state == S AND date > 2020-06-30 (the actual test period!)
#
#   Evaluation is done WITHIN the held-out state (ranking only S's munis per day).
#
# Outputs are written to runs/ltr/<RUN_TAG>/loso/

# Source utilities only (no execution)
source("scripts/approach2_anomaly/ltr_utils.R")

# CONFIGURATION

args <- commandArgs(trailingOnly = TRUE)
args <- args[!grepl("^--", args)]

MODEL_READY_RUN_TAG <- if (length(args) > 0) args[1] else NULL
RUN_TAG <- format(Sys.time(), "%Y%m%d_%H%M%S")

# Training sample fraction - reduce for memory constraints
# 0.4 = 40% of training data, fits in 24GB with XGBoost
TRAIN_SAMPLE_FRAC <- 0.4

# Output directory
OUTPUT_DIR <- file.path("runs/ltr", RUN_TAG, "loso")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("=== 08_loso_eval.R ===\n")
cat("Leave-One-State-Out Cross-Validation\n")
cat(sprintf("Run tag: %s\n", RUN_TAG))
cat(sprintf("Output directory: %s\n", OUTPUT_DIR))

# 1. LOAD DATA

cat("\n=== Loading Data ===\n")

data <- load_model_ready_data(MODEL_READY_RUN_TAG)

# Combine all data (we'll create our own LOSO splits)
all_data <- bind_rows(
  data$train %>% mutate(orig_split = "train"),
  data$val %>% mutate(orig_split = "val"),
  data$test %>% mutate(orig_split = "test")
)

cat(sprintf("Model-ready data: %s\n", data$data_path))
cat(sprintf("Total data: %s rows\n", format(nrow(all_data), big.mark = ",")))
cat(sprintf("Date range: %s to %s\n", min(all_data$date), max(all_data$date)))

# Define temporal boundaries (must match standard split)
TRAIN_END <- as.Date("2019-12-31")
VAL_START <- as.Date("2020-01-01")
VAL_END <- as.Date("2020-06-30")
TEST_START <- as.Date("2020-07-01")

cat(sprintf("\nTemporal boundaries:\n"))
cat(sprintf("  TRAIN: <= %s\n", TRAIN_END))
cat(sprintf("  VAL: %s to %s\n", VAL_START, VAL_END))
cat(sprintf("  TEST: > %s\n", VAL_END))

# 2. IDENTIFY STATES

cat("\n=== Identifying States ===\n")

# Detect state column
state_col_detected <- detect_state_col(all_data)
result <- add_state_col_if_needed(all_data, state_col_detected)
all_data <- result$df
STATE_COL <- result$state_col

states <- sort(unique(all_data[[STATE_COL]]))
n_states <- length(states)

cat(sprintf("State column: %s\n", STATE_COL))
cat(sprintf("Number of states: %d\n", n_states))

# State summary
state_summary <- all_data %>%
  group_by(across(all_of(STATE_COL))) %>%
  summarise(
    n_munis = n_distinct(GID_2),
    n_obs = n(),
    n_test_period = sum(date > VAL_END),
    outage_rate = mean(outage_3h_or_more),
    .groups = "drop"
  ) %>%
  arrange(desc(n_munis))

cat("\n--- State Summary (top 5 by municipalities) ---\n")
print(head(state_summary, 5))

# 3. GET FEATURES

cat("\n=== Defining Features ===\n")

feature_cols <- get_ltr_features(all_data, mode = "operational")
cat(sprintf("Features: %d\n", length(feature_cols)))

# 4. LOSO CROSS-VALIDATION

cat("\n=== Running LOSO Cross-Validation ===\n")
cat(sprintf("Evaluating %d states...\n\n", n_states))

K_VALUES <- c(1, 2, 5, 10, 20, 50)

loso_results <- list()
loso_start <- Sys.time()

for (i in seq_along(states)) {
  held_out_state <- states[i]

  cat(sprintf("[%2d/%d] State: %s\n", i, n_states, held_out_state))

  # CRITICAL: Correct LOSO split
  # TRAIN: state != S AND date <= TRAIN_END
  # VAL:   state != S AND date in VAL period
  # TEST:  state == S AND date in TEST period (>VAL_END)

  is_held_out <- all_data[[STATE_COL]] == held_out_state

  # Create training fold (full, for counting)
  train_fold_full <- all_data %>%
    filter(!is_held_out, date <= TRAIN_END)

  # Sample training data for memory efficiency
  set.seed(42 + i)  # Reproducible but different seed per fold
  train_fold <- train_fold_full %>%
    sample_frac(TRAIN_SAMPLE_FRAC)
  rm(train_fold_full)

  val_fold_full <- all_data %>%
    filter(!is_held_out, date >= VAL_START, date <= VAL_END)

  # Sample validation data too for memory efficiency
  val_fold <- val_fold_full %>%
    sample_frac(TRAIN_SAMPLE_FRAC)
  rm(val_fold_full)
  gc(verbose = FALSE)

  # TEST is ONLY the held-out state in the test period
  test_fold <- all_data %>%
    filter(is_held_out, date > VAL_END)

  n_train <- nrow(train_fold)
  n_val <- nrow(val_fold)
  n_test <- nrow(test_fold)
  n_test_days <- n_distinct(test_fold$date)
  n_test_munis <- n_distinct(test_fold$GID_2)

  cat(sprintf("       Train: %s rows (%.0f%% sampled, 31 states, <=2019)\n",
              format(n_train, big.mark = ","), TRAIN_SAMPLE_FRAC * 100))
  cat(sprintf("       Val: %s rows (31 states, 2020H1)\n",
              format(n_val, big.mark = ",")))
  cat(sprintf("       Test: %s rows (%d munis, %d days in test period)\n",
              format(n_test, big.mark = ","), n_test_munis, n_test_days))

  # Skip if insufficient data
  if (n_train < 1000 || n_test < 50 || n_test_days < 10) {
    cat("       SKIPPED: insufficient data\n\n")
    next
  }

  # Garbage collection before training
  gc(verbose = FALSE)

  # Train LTR model
  ltr_result <- tryCatch({
    train_ltr(
      train_data = train_fold,
      val_data = val_fold,
      feature_cols = feature_cols,
      nrounds = 300,
      early_stopping = 30,
      seed = 42,
      verbose = 0
    )
  }, error = function(e) {
    cat(sprintf("       ERROR: %s\n\n", e$message))
    return(NULL)
  })

  # Clean up training data to free memory
  rm(train_fold, val_fold)
  gc(verbose = FALSE)

  if (is.null(ltr_result)) next

  # Predict on held-out state's test period
  test_preds <- predict_ltr(ltr_result, test_fold)

  # Evaluate Top-K WITHIN this state only
  # (ranking among the state's municipalities per day)
  # strict_k=FALSE allows small states with <k munis to still contribute
  eval_result <- evaluate_topk(
    test_preds, K_VALUES,
    allocation = "national",  # "national" here means within-state top-K
    n_boot = 500,
    strict_k = FALSE
  )

  # Store results
  loso_results[[held_out_state]] <- list(
    topk = eval_result$topk_metrics %>%
      mutate(
        state = held_out_state,
        n_train = n_train,
        n_val = n_val,
        n_test = n_test,
        n_test_days = n_test_days,
        n_test_munis = n_test_munis,
        train_time_sec = ltr_result$train_time_sec,
        best_iteration = ltr_result$best_iteration
      ),
    bootstrap = eval_result$bootstrap_ci %>%
      mutate(state = held_out_state)
  )

  # Print key metric
  k10 <- eval_result$topk_metrics %>% filter(k == 10)
  cat(sprintf("       Results@10: Hits=%d, Precision=%.4f, Lift=%.2f\n\n",
              k10$n_hits, k10$precision, k10$lift))

  # Save individual fold results
  fold_dir <- file.path(OUTPUT_DIR, paste0("state=", held_out_state))
  dir.create(fold_dir, recursive = TRUE, showWarnings = FALSE)

  test_preds %>%
    select(GID_2, date, outage_3h_or_more, rank_score) %>%
    write_csv(gzfile(file.path(fold_dir, "predictions.csv.gz")))

  write_csv(eval_result$topk_metrics, file.path(fold_dir, "topk_metrics.csv"))

  # Clean up to free memory
  rm(test_fold, test_preds, ltr_result, eval_result)
  gc(verbose = FALSE)
}

loso_elapsed <- as.numeric(difftime(Sys.time(), loso_start, units = "secs"))
cat(sprintf("\nLOSO completed in %.1f seconds (%.1f min)\n",
            loso_elapsed, loso_elapsed / 60))

# 5. AGGREGATE RESULTS

cat("\n=== Aggregating Results ===\n")

# Combine all fold results
all_topk <- bind_rows(lapply(loso_results, function(x) x$topk))
all_bootstrap <- bind_rows(lapply(loso_results, function(x) x$bootstrap))

# Summary by K (across states)
loso_summary <- all_topk %>%
  group_by(k) %>%
  summarise(
    n_states = n(),
    precision_mean = mean(precision, na.rm = TRUE),
    precision_sd = sd(precision, na.rm = TRUE),
    lift_mean = mean(lift, na.rm = TRUE),
    lift_sd = sd(lift, na.rm = TRUE),
    hits_total = sum(n_hits),
    alerts_total = sum(n_alerts),
    .groups = "drop"
  ) %>%
  mutate(
    precision_se = precision_sd / sqrt(n_states),
    lift_se = lift_sd / sqrt(n_states)
  )

cat("\n--- LOSO Summary (mean across states) ---\n")
print(loso_summary %>%
        select(k, n_states, precision_mean, precision_sd, lift_mean, lift_sd) %>%
        mutate(across(where(is.numeric) & !matches("k|n_states"), ~round(., 4))))

# Per-state performance at k=10
state_performance <- all_topk %>%
  filter(k == 10) %>%
  select(state, precision, lift, n_hits, n_test, n_test_munis, n_test_days) %>%
  arrange(desc(lift))

cat("\n--- Per-State Performance at k=10 ---\n")
cat("Top 5 states:\n")
print(head(state_performance, 5))
cat("\nBottom 5 states:\n")
print(tail(state_performance, 5))

# 6. SAVE AGGREGATED RESULTS

cat("\n=== Saving Aggregated Results ===\n")

# All fold results
write_csv(all_topk, file.path(OUTPUT_DIR, "loso_all_folds.csv"))
cat("  Saved: loso_all_folds.csv\n")

# Summary across states
write_csv(loso_summary, file.path(OUTPUT_DIR, "loso_summary.csv"))
cat("  Saved: loso_summary.csv\n")

# Per-state performance at k=10
write_csv(state_performance, file.path(OUTPUT_DIR, "loso_state_performance_k10.csv"))
cat("  Saved: loso_state_performance_k10.csv\n")

# Bootstrap CIs
write_csv(all_bootstrap, file.path(OUTPUT_DIR, "loso_bootstrap_ci.csv"))
cat("  Saved: loso_bootstrap_ci.csv\n")

# Metadata
metadata <- tibble(
  run_tag = RUN_TAG,
  model_ready_run = data$run_tag,
  train_sample_frac = TRAIN_SAMPLE_FRAC,
  n_states_evaluated = length(loso_results),
  n_states_total = n_states,
  n_features = length(feature_cols),
  elapsed_sec = loso_elapsed,
  train_end = as.character(TRAIN_END),
  val_start = as.character(VAL_START),
  val_end = as.character(VAL_END),
  test_start = as.character(TEST_START),
  lift_k10_mean = loso_summary$lift_mean[loso_summary$k == 10],
  lift_k10_sd = loso_summary$lift_sd[loso_summary$k == 10],
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)
write_csv(metadata, file.path(OUTPUT_DIR, "metadata.csv"))
cat("  Saved: metadata.csv\n")

# 7. GENERATE REPORT

report_lines <- c(
  strrep("=", 72),
  "LEAVE-ONE-STATE-OUT (LOSO) EVALUATION REPORT",
  strrep("=", 72),
  sprintf("Run tag: %s", RUN_TAG),
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "LOSO DESIGN",
  strrep("-", 40),
  "For each held-out state S:",
  sprintf("  TRAIN: state != S, date <= %s", TRAIN_END),
  sprintf("  VAL:   state != S, date in [%s, %s]", VAL_START, VAL_END),
  sprintf("  TEST:  state == S, date > %s", VAL_END),
  "",
  "This tests whether a model trained on 31 states generalizes to the",
  "32nd state during the test period. Evaluation ranks only the held-out",
  "state's municipalities (within-state Top-K).",
  "",
  "RESULTS",
  strrep("-", 40),
  sprintf("States evaluated: %d / %d", length(loso_results), n_states),
  sprintf("Total time: %.1f minutes", loso_elapsed / 60),
  "",
  "Performance at k=10 (across states):",
  sprintf("  Mean Lift: %.2f (SD: %.2f)",
          loso_summary$lift_mean[loso_summary$k == 10],
          loso_summary$lift_sd[loso_summary$k == 10]),
  sprintf("  Mean Precision: %.4f (SD: %.4f)",
          loso_summary$precision_mean[loso_summary$k == 10],
          loso_summary$precision_sd[loso_summary$k == 10]),
  sprintf("  Total Hits: %d",
          loso_summary$hits_total[loso_summary$k == 10]),
  "",
  sprintf("Best state: %s (Lift: %.2f)",
          state_performance$state[1], state_performance$lift[1]),
  sprintf("Worst state: %s (Lift: %.2f)",
          tail(state_performance$state, 1), tail(state_performance$lift, 1)),
  "",
  "INTERPRETATION",
  strrep("-", 40),
  "High variance across states indicates the model may not generalize",
  "equally well to all regions. States with lower lift may have different",
  "outage patterns or fewer training examples from similar areas.",
  "",
  "FILES GENERATED",
  strrep("-", 40),
  "loso_all_folds.csv - All K results for all states",
  "loso_summary.csv - Aggregated statistics",
  "loso_state_performance_k10.csv - Per-state results at k=10",
  "loso_bootstrap_ci.csv - Bootstrap CIs per state",
  "state=<S>/predictions.csv.gz - Predictions per state",
  "state=<S>/topk_metrics.csv - Metrics per state",
  "",
  strrep("=", 72)
)

writeLines(report_lines, file.path(OUTPUT_DIR, "evaluation_report.txt"))
cat("  Saved: evaluation_report.txt\n")

# Update pointer
writeLines(RUN_TAG, file.path(OUTPUT_DIR, "run_tag.txt"))

# 8. SUMMARY

cat("\n")
cat(strrep("=", 60), "\n")
cat("LOSO EVALUATION COMPLETE\n")
cat(strrep("=", 60), "\n")
cat(sprintf("States evaluated: %d / %d\n", length(loso_results), n_states))
cat(sprintf("Total time: %.1f minutes\n", loso_elapsed / 60))
cat(sprintf("\nKey Results at k=10:\n"))
cat(sprintf("  Mean Lift: %.2f (SD: %.2f)\n",
            loso_summary$lift_mean[loso_summary$k == 10],
            loso_summary$lift_sd[loso_summary$k == 10]))
cat(sprintf("  Lift range: %.2f to %.2f\n",
            min(state_performance$lift), max(state_performance$lift)))
cat(sprintf("\nOutput: %s\n", OUTPUT_DIR))
cat(strrep("=", 60), "\n")
