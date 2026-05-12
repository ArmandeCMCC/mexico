# 07_learning_to_rank_refactor.R
# Learning-to-Rank model training and evaluation
#
# This script trains an XGBoost rank:pairwise model and evaluates it using
# multiple allocation policies (national, per_state, hybrid).
#
# Outputs are written to runs/ltr/<RUN_TAG>/ (NOT runs/combined/)
#
# Usage:
#   Rscript 07_learning_to_rank_refactor.R [model_ready_run_tag]

# Source utilities (functions only, no execution)
source("scripts/approach2_anomaly/ltr_utils.R")

# CONFIGURATION

args <- commandArgs(trailingOnly = TRUE)
args <- args[!grepl("^--", args)]

MODEL_READY_RUN_TAG <- if (length(args) > 0) args[1] else NULL
RUN_TAG <- format(Sys.time(), "%Y%m%d_%H%M%S")

# Output directory: runs/ltr/ (not runs/combined/)
OUTPUT_DIR <- file.path("runs/ltr", RUN_TAG)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("=== 07_learning_to_rank_refactor.R ===\n")
cat(sprintf("Run tag: %s\n", RUN_TAG))
cat(sprintf("Output directory: %s\n", OUTPUT_DIR))

# 1. LOAD DATA

cat("\n=== Loading Data ===\n")

data <- load_model_ready_data(MODEL_READY_RUN_TAG)

cat(sprintf("Model-ready data: %s\n", data$data_path))
cat(sprintf("TRAIN: %s rows, %d days (<=2019-12-31)\n",
            format(nrow(data$train), big.mark = ","),
            n_distinct(data$train$date)))
cat(sprintf("VAL: %s rows, %d days (2020-01-01 to 2020-06-30)\n",
            format(nrow(data$val), big.mark = ","),
            n_distinct(data$val$date)))
cat(sprintf("TEST: %s rows, %d days (>2020-06-30)\n",
            format(nrow(data$test), big.mark = ","),
            n_distinct(data$test$date)))

# Verify split dates
train_date_range <- range(data$train$date)
val_date_range <- range(data$val$date)
test_date_range <- range(data$test$date)
cat(sprintf("\nDate ranges:\n"))
cat(sprintf("  TRAIN: %s to %s\n", train_date_range[1], train_date_range[2]))
cat(sprintf("  VAL: %s to %s\n", val_date_range[1], val_date_range[2]))
cat(sprintf("  TEST: %s to %s\n", test_date_range[1], test_date_range[2]))

# 2. DEFINE FEATURES

cat("\n=== Defining Features ===\n")

feature_cols <- get_ltr_features(data$train, mode = "operational")
cat(sprintf("Total features: %d\n", length(feature_cols)))

# Show feature breakdown
n_anomaly <- sum(grepl("^anomaly_", feature_cols))
n_ntl <- sum(grepl("ntl", feature_cols, ignore.case = TRUE))
cat(sprintf("  NTL-related: %d\n", n_ntl))
cat(sprintf("  Anomaly (lag1): %d\n", n_anomaly))
cat(sprintf("  Other: %d\n", length(feature_cols) - n_ntl - n_anomaly))

# 3. TRAIN LTR MODEL

cat("\n=== Training Learning-to-Rank Model ===\n")

ltr_result <- train_ltr(
  train_data = data$train,
  val_data = data$val,
  feature_cols = feature_cols,
  nrounds = 500,
  early_stopping = 50,
  seed = 42,
  verbose = 1
)

cat(sprintf("\nTraining completed:\n"))
cat(sprintf("  Time: %.1f seconds\n", ltr_result$train_time_sec))
cat(sprintf("  Best iteration: %d\n", ltr_result$best_iteration))

# 4. PREDICT ON VAL AND TEST

cat("\n=== Generating Predictions ===\n")

val_preds <- predict_ltr(ltr_result, data$val)
test_preds <- predict_ltr(ltr_result, data$test)

cat(sprintf("VAL predictions: %d rows\n", nrow(val_preds)))
cat(sprintf("TEST predictions: %d rows\n", nrow(test_preds)))

# 5. EVALUATE ALL ALLOCATION MODES

cat("\n=== Evaluating Allocation Modes ===\n")

K_VALUES <- c(1, 2, 5, 10, 20, 50)

# National allocation 
cat("\n--- National Allocation (K_total per day) ---\n")

eval_national_val <- evaluate_topk(val_preds, K_VALUES, allocation = "national")
eval_national_test <- evaluate_topk(test_preds, K_VALUES, allocation = "national")

cat("VAL:\n")
print(eval_national_val$topk_metrics %>%
        filter(k == 10) %>%
        select(k, precision, lift, n_hits))

cat("TEST:\n")
print(eval_national_test$topk_metrics %>%
        filter(k == 10) %>%
        select(k, precision, lift, n_hits))

# Per-state allocation 
cat("\n--- Per-State Allocation (k_state per state per day) ---\n")

eval_perstate_val <- evaluate_topk(val_preds, K_VALUES, allocation = "per_state", k_state = 10)
eval_perstate_test <- evaluate_topk(test_preds, K_VALUES, allocation = "per_state", k_state = 10)

cat("VAL (k=10 per state):\n")
print(eval_perstate_val$topk_metrics %>%
        filter(k == 10) %>%
        select(k, precision, lift, n_hits))

cat("TEST (k=10 per state):\n")
print(eval_perstate_test$topk_metrics %>%
        filter(k == 10) %>%
        select(k, precision, lift, n_hits))

# Hybrid State-First allocation 
cat("\n--- Hybrid State-First Allocation (top-K states by max risk, 1 muni per state) ---\n")

eval_hybrid_val <- evaluate_topk(val_preds, K_VALUES, allocation = "hybrid_state_first", K_total = 10)
eval_hybrid_test <- evaluate_topk(test_preds, K_VALUES, allocation = "hybrid_state_first", K_total = 10)

cat("VAL:\n")
print(eval_hybrid_val$topk_metrics %>%
        filter(k == 10) %>%
        select(k, precision, lift, n_hits))

cat("TEST:\n")
print(eval_hybrid_test$topk_metrics %>%
        filter(k == 10) %>%
        select(k, precision, lift, n_hits))

# 6. SAVE OUTPUTS

cat("\n=== Saving Outputs ===\n")

# Predictions
val_preds %>%
  select(GID_2, date, outage_3h_or_more, rank_score) %>%
  write_csv(gzfile(file.path(OUTPUT_DIR, "predictions_val.csv.gz")))

test_preds %>%
  select(GID_2, date, outage_3h_or_more, rank_score) %>%
  write_csv(gzfile(file.path(OUTPUT_DIR, "predictions_test.csv.gz")))

cat("  Saved: predictions_val.csv.gz, predictions_test.csv.gz\n")

# Top-K metrics (all allocations combined)
all_topk_val <- bind_rows(
  eval_national_val$topk_metrics,
  eval_perstate_val$topk_metrics,
  eval_hybrid_val$topk_metrics
)
all_topk_test <- bind_rows(
  eval_national_test$topk_metrics,
  eval_perstate_test$topk_metrics,
  eval_hybrid_test$topk_metrics
)

write_csv(all_topk_val, file.path(OUTPUT_DIR, "topk_metrics_val.csv"))
write_csv(all_topk_test, file.path(OUTPUT_DIR, "topk_metrics_test.csv"))
cat("  Saved: topk_metrics_val.csv, topk_metrics_test.csv\n")

# Bootstrap CIs
all_bootstrap_val <- bind_rows(
  eval_national_val$bootstrap_ci,
  eval_perstate_val$bootstrap_ci,
  eval_hybrid_val$bootstrap_ci
)
all_bootstrap_test <- bind_rows(
  eval_national_test$bootstrap_ci,
  eval_perstate_test$bootstrap_ci,
  eval_hybrid_test$bootstrap_ci
)

write_csv(all_bootstrap_val, file.path(OUTPUT_DIR, "bootstrap_k10_val.csv"))
write_csv(all_bootstrap_test, file.path(OUTPUT_DIR, "bootstrap_k10_test.csv"))
cat("  Saved: bootstrap_k10_val.csv, bootstrap_k10_test.csv\n")

# Feature importance
importance <- xgb.importance(model = ltr_result$model)
write_csv(importance, file.path(OUTPUT_DIR, "feature_importance.csv"))
cat("  Saved: feature_importance.csv\n")

cat("\n--- Top 10 Feature Importance ---\n")
print(head(importance, 10))

# 7. GENERATE EVALUATION REPORT

report_lines <- c(
  strrep("=", 72),
  "LEARNING-TO-RANK EVALUATION REPORT",
  strrep("=", 72),
  sprintf("Run tag: %s", RUN_TAG),
  sprintf("Model-ready data: %s", data$run_tag),
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "DATA SPLITS",
  strrep("-", 40),
  sprintf("TRAIN: %s rows (%s to %s)",
          format(nrow(data$train), big.mark = ","),
          train_date_range[1], train_date_range[2]),
  sprintf("VAL: %s rows (%s to %s)",
          format(nrow(data$val), big.mark = ","),
          val_date_range[1], val_date_range[2]),
  sprintf("TEST: %s rows (%s to %s)",
          format(nrow(data$test), big.mark = ","),
          test_date_range[1], test_date_range[2]),
  "",
  "MODEL",
  strrep("-", 40),
  sprintf("Features: %d", length(feature_cols)),
  sprintf("Best iteration: %d / 500", ltr_result$best_iteration),
  sprintf("Training time: %.1f seconds", ltr_result$train_time_sec),
  "",
  "TOP-K METRICS (TEST SET, k=10)",
  strrep("-", 40),
  sprintf("National:  Lift=%.2f, Hits=%d, Precision=%.4f",
          eval_national_test$topk_metrics$lift[eval_national_test$topk_metrics$k == 10],
          eval_national_test$topk_metrics$n_hits[eval_national_test$topk_metrics$k == 10],
          eval_national_test$topk_metrics$precision[eval_national_test$topk_metrics$k == 10]),
  sprintf("Per-state: Lift=%.2f, Hits=%d, Precision=%.4f",
          eval_perstate_test$topk_metrics$lift[eval_perstate_test$topk_metrics$k == 10],
          eval_perstate_test$topk_metrics$n_hits[eval_perstate_test$topk_metrics$k == 10],
          eval_perstate_test$topk_metrics$precision[eval_perstate_test$topk_metrics$k == 10]),
  sprintf("Hybrid:    Lift=%.2f, Hits=%d, Precision=%.4f",
          eval_hybrid_test$topk_metrics$lift[eval_hybrid_test$topk_metrics$k == 10],
          eval_hybrid_test$topk_metrics$n_hits[eval_hybrid_test$topk_metrics$k == 10],
          eval_hybrid_test$topk_metrics$precision[eval_hybrid_test$topk_metrics$k == 10]),
  "",
  "BOOTSTRAP 95% CI (TEST, k=10, National)",
  strrep("-", 40),
  sprintf("Lift: %.2f [%.2f, %.2f]",
          eval_national_test$bootstrap_ci$lift_mean,
          eval_national_test$bootstrap_ci$lift_ci_lo,
          eval_national_test$bootstrap_ci$lift_ci_hi),
  sprintf("Precision: %.4f [%.4f, %.4f]",
          eval_national_test$bootstrap_ci$precision_mean,
          eval_national_test$bootstrap_ci$precision_ci_lo,
          eval_national_test$bootstrap_ci$precision_ci_hi),
  "",
  "FILES GENERATED",
  strrep("-", 40),
  "predictions_val.csv.gz, predictions_test.csv.gz",
  "topk_metrics_val.csv, topk_metrics_test.csv",
  "bootstrap_k10_val.csv, bootstrap_k10_test.csv",
  "feature_importance.csv",
  "evaluation_report.txt",
  "metadata.csv",
  "",
  strrep("=", 72)
)

writeLines(report_lines, file.path(OUTPUT_DIR, "evaluation_report.txt"))
cat("  Saved: evaluation_report.txt\n")

# Metadata
metadata <- tibble(
  run_tag = RUN_TAG,
  model_ready_run = data$run_tag,
  n_features = length(feature_cols),
  best_iteration = ltr_result$best_iteration,
  train_time_sec = ltr_result$train_time_sec,
  n_train = nrow(data$train),
  n_val = nrow(data$val),
  n_test = nrow(data$test),
  lift_k10_national_test = eval_national_test$topk_metrics$lift[eval_national_test$topk_metrics$k == 10],
  hits_k10_national_test = eval_national_test$topk_metrics$n_hits[eval_national_test$topk_metrics$k == 10],
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)
write_csv(metadata, file.path(OUTPUT_DIR, "metadata.csv"))
cat("  Saved: metadata.csv\n")

# Update latest pointer
writeLines(RUN_TAG, "runs/ltr/latest_run_tag.txt")
cat("  Updated: runs/ltr/latest_run_tag.txt\n")

# 8. SUMMARY

cat("\n")
cat(strrep("=", 60), "\n")
cat("LTR TRAINING COMPLETE\n")
cat(strrep("=", 60), "\n")
cat(sprintf("Output directory: %s\n", OUTPUT_DIR))
cat(sprintf("\nKey Results (TEST, k=10, National):\n"))
cat(sprintf("  Hits: %d\n", eval_national_test$topk_metrics$n_hits[eval_national_test$topk_metrics$k == 10]))
cat(sprintf("  Lift: %.2f [%.2f, %.2f]\n",
            eval_national_test$bootstrap_ci$lift_mean,
            eval_national_test$bootstrap_ci$lift_ci_lo,
            eval_national_test$bootstrap_ci$lift_ci_hi))
cat(strrep("=", 60), "\n")
