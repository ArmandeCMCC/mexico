# 17_ltr_bench_rs_history.R
# Learning-to-Rank (XGBoost rank:pairwise) comparable to the headline calibrated
# classifier: bench_rs_history (83 features), same panel and same splits.
#
# Produces Top-K metrics evaluated on the same test rows as the headline run, so
# benchmark_table_same_run.csv can carry an "exact"-comparability LTR row.
#
# Inputs (must match the headline run):
#   data/model_ready/features_engineered.rds
#   data/model_ready/splits_fixed.rds
#   scripts/00_feature_config.R   (EXCLUDE_COLS_BASE, build_exclude_cols, apply_task_mode_filters)
#   scripts/approach2_anomaly/ltr_utils.R (train_ltr, predict_ltr, evaluate_topk)
#
#
# Usage:
#   Rscript scripts/17_ltr_bench_rs_history.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(xgboost)
  library(jsonlite)
})

set.seed(42)

# CONFIGURATION

PROJECT_DIR <- "/Users/armandeaboudrar-meda/Desktop/CMCC/outages_nightlights/mexico" # make it relative 
HEADLINE_RUN <- file.path(
  PROJECT_DIR, "data", "baselines", "binary_threshold", "ablation_batches",
  "20260513_153833", "20260513_153858_forecast_strict_bench_rs_history"
)

FEATURES_PATH <- file.path(PROJECT_DIR, "data", "model_ready", "features_engineered.rds")
SPLITS_PATH   <- file.path(PROJECT_DIR, "data", "model_ready", "splits_fixed.rds")

OUTPUT_DIR <- file.path(HEADLINE_RUN, "ltr_bench_rs_history")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Top-K values to evaluate
K_VALUES <- c(1, 2, 5, 10, 20, 30, 50, 100)
N_BOOT_CI <- 1000

# XGBoost rank parameters - mirror headline classifier where they overlap and
# match what the prior LTR used for the rank-specific bits.
XGB_PARAMS <- list(
  objective        = "rank:pairwise",
  eval_metric      = "ndcg@10",
  max_depth        = 6,
  eta              = 0.05,
  subsample        = 0.8,
  colsample_bytree = 0.8
)
NROUNDS         <- 500
EARLY_STOPPING  <- 50
SEED            <- 42

cat(strrep("=", 72), "\n", sep = "")
cat("17_ltr_bench_rs_history.R - LTR comparable to headline calibrated model\n")
cat(strrep("=", 72), "\n\n", sep = "")

cat("Headline run: ", HEADLINE_RUN, "\n")
cat("Features:    ", FEATURES_PATH, "\n")
cat("Splits:      ", SPLITS_PATH, "\n")
cat("Output dir:  ", OUTPUT_DIR, "\n\n")

# SOURCE HELPERS

source(file.path(PROJECT_DIR, "scripts", "00_feature_config.R"))
source(file.path(PROJECT_DIR, "scripts", "approach2_anomaly", "ltr_utils.R"))

# LOAD DATA (identical to 05b_binary_threshold_eval_ablation.R lines 749-779)

cat("Loading features...\n")
features <- readRDS(FEATURES_PATH)
cat(sprintf("  rows = %s, cols = %d\n", format(nrow(features), big.mark = ","), ncol(features)))

splits_fixed <- readRDS(SPLITS_PATH)
train_start <- as.Date(splits_fixed$train_range[1])
train_end   <- as.Date(splits_fixed$train_range[2])
val_start   <- as.Date(splits_fixed$val_range[1])
val_end     <- as.Date(splits_fixed$val_range[2])
test_start  <- as.Date(splits_fixed$test_range[1])
test_end    <- as.Date(splits_fixed$test_range[2])

cat("Splits:\n")
cat(sprintf("  Train: %s -> %s\n", train_start, train_end))
cat(sprintf("  Val:   %s -> %s\n", val_start, val_end))
cat(sprintf("  Test:  %s -> %s\n", test_start, test_end))

features <- features %>%
  mutate(
    date = as.Date(date),
    split = case_when(
      date >= train_start & date <= train_end ~ "train",
      date >= val_start   & date <= val_end   ~ "val",
      date >= test_start  & date <= test_end  ~ "test",
      TRUE ~ "other"
    )
  )

train_data <- features %>% filter(split == "train")
val_data   <- features %>% filter(split == "val")
test_data  <- features %>% filter(split == "test")

cat(sprintf("  Train rows = %s\n", format(nrow(train_data), big.mark = ",")))
cat(sprintf("  Val rows   = %s\n", format(nrow(val_data),   big.mark = ",")))
cat(sprintf("  Test rows  = %s\n", format(nrow(test_data),  big.mark = ",")))

# FEATURE SELECTION (identical to headline)

cat("\nApplying feature pipeline identical to headline...\n")
headline_cfg <- fromJSON(file.path(HEADLINE_RUN, "run_config.json"))
inc_regex <- headline_cfg$ablation$feature_include_regex

excl <- build_exclude_cols(task_mode = "forecast", weather_mode = "lagged")
feature_cols <- setdiff(names(features), excl)
feature_cols <- apply_task_mode_filters(feature_cols, task_mode = "forecast",
                                        strict = TRUE, verbose = FALSE)
feature_cols <- feature_cols[grepl(inc_regex, feature_cols, perl = TRUE)]

# Drop all-NA on train, character/factor
feature_cols <- intersect(feature_cols, names(train_data))
na_rate <- sapply(train_data[, feature_cols, drop = FALSE], function(x) mean(is.na(x)))
feature_cols <- setdiff(feature_cols, names(na_rate)[na_rate == 1])
char_cols <- feature_cols[sapply(train_data[, feature_cols, drop = FALSE], is.character)]
fact_cols <- feature_cols[sapply(train_data[, feature_cols, drop = FALSE], is.factor)]
feature_cols <- setdiff(feature_cols, c(char_cols, fact_cols))

cat(sprintf("Feature count: %d\n", length(feature_cols)))

expected_features <- readLines(file.path(HEADLINE_RUN, "features_used.txt"))
stopifnot(identical(sort(feature_cols), sort(expected_features)))
cat("Feature set verified identical to headline (features_used.txt).\n\n")

writeLines(feature_cols, file.path(OUTPUT_DIR, "ltr_features_used.txt"))

# TRAIN LTR

cat(strrep("-", 72), "\n", sep = "")
cat("Training rank:pairwise LTR model\n")
cat(strrep("-", 72), "\n", sep = "")

# train_ltr() needs: train_data, val_data (for early stop), feature_cols
# Both must contain feature_cols + "outage_3h_or_more" + "date" (group var).

ltr_result <- train_ltr(
  train_data    = train_data,
  val_data      = val_data,
  feature_cols  = feature_cols,
  group_col     = "date",
  label_col     = "outage_3h_or_more",
  xgb_params    = XGB_PARAMS,
  nrounds       = NROUNDS,
  early_stopping = EARLY_STOPPING,
  seed          = SEED,
  verbose       = 1
)

cat(sprintf("\nTrain time:    %.1f s\n", ltr_result$train_time_sec))
cat(sprintf("Best iteration: %d\n",       ltr_result$best_iteration))

# PREDICT

cat("\nPredicting on validation...\n")
val_preds  <- predict_ltr(ltr_result, val_data,  group_col = "date", label_col = "outage_3h_or_more")
cat(sprintf("  val rows  = %s\n", format(nrow(val_preds),  big.mark = ",")))

cat("Predicting on test...\n")
test_preds <- predict_ltr(ltr_result, test_data, group_col = "date", label_col = "outage_3h_or_more")
cat(sprintf("  test rows = %s\n", format(nrow(test_preds), big.mark = ",")))

# TOP-K EVALUATION

cat("\nEvaluating Top-K (national allocation) ...\n")

eval_test <- evaluate_topk(
  preds        = test_preds,
  K_values     = K_VALUES,
  group_col    = "date",
  label_col    = "outage_3h_or_more",
  allocation   = "national",
  K_total      = 10,
  n_boot       = N_BOOT_CI,
  ci_level     = 0.95,
  seed         = SEED,
  strict_k     = TRUE
)

# Also compute a val-side version (point estimates only, for completeness)
eval_val <- evaluate_topk(
  preds        = val_preds,
  K_values     = K_VALUES,
  group_col    = "date",
  label_col    = "outage_3h_or_more",
  allocation   = "national",
  K_total      = 10,
  n_boot       = N_BOOT_CI,
  ci_level     = 0.95,
  seed         = SEED,
  strict_k     = TRUE
)

# Combine + tag split
topk_metrics <- bind_rows(
  eval_test$topk_metrics %>% mutate(split = "test"),
  eval_val$topk_metrics  %>% mutate(split = "val")
) %>% select(split, k, allocation, strict_k, precision, recall, lift,
             n_hits, n_alerts, n_days, base_rate)

bootstrap_ci <- bind_rows(
  eval_test$bootstrap_ci %>% mutate(split = "test"),
  eval_val$bootstrap_ci  %>% mutate(split = "val")
) %>% select(split, k, allocation, strict_k, precision_mean, precision_ci_lo, precision_ci_hi,
             lift_mean, lift_ci_lo, lift_ci_hi, n_boot_days)

cat("\nTOP-K (TEST):\n")
print(topk_metrics %>% filter(split == "test") %>%
        select(k, precision, recall, lift, n_hits, n_days) %>%
        mutate(across(c(precision, recall, lift), ~round(.x, 4))))

cat("\nTOP-K bootstrap 95% CI (k=10, TEST):\n")
print(bootstrap_ci %>% filter(split == "test") %>%
        mutate(across(where(is.numeric), ~round(.x, 4))))

# FEATURE IMPORTANCE

importance <- xgb.importance(model = ltr_result$model)
write_csv(importance, file.path(OUTPUT_DIR, "ltr_feature_importance.csv"))

cat("\nTop 10 feature importance (gain):\n")
print(head(importance, 10))

# WRITE OUTPUTS

write_csv(topk_metrics,  file.path(OUTPUT_DIR, "ltr_topk_metrics.csv"))
write_csv(bootstrap_ci,  file.path(OUTPUT_DIR, "ltr_topk_bootstrap_ci.csv"))

# Predictions (compressed)
val_preds %>%
  select(GID_2, date, outage_3h_or_more, rank_score) %>%
  write_csv(gzfile(file.path(OUTPUT_DIR, "ltr_predictions_val.csv.gz")))

test_preds %>%
  select(GID_2, date, outage_3h_or_more, rank_score) %>%
  write_csv(gzfile(file.path(OUTPUT_DIR, "ltr_predictions_test.csv.gz")))

# Run config (mirrors headline run_config.json schema where relevant)
run_config <- list(
  ltr_run_id            = paste0("ltr_bench_rs_history_", format(Sys.time(), "%Y%m%d_%H%M%S")),
  headline_run_dir      = HEADLINE_RUN,
  features_path         = FEATURES_PATH,
  splits_fixed_path     = SPLITS_PATH,
  splits = list(
    train = list(start = format(train_start), end = format(train_end)),
    val   = list(start = format(val_start),   end = format(val_end)),
    test  = list(start = format(test_start),  end = format(test_end))
  ),
  n_train = nrow(train_data),
  n_val   = nrow(val_data),
  n_test  = nrow(test_data),
  test_prevalence = round(mean(test_data$outage_3h_or_more), 6),
  model = list(
    type             = "xgboost",
    objective        = XGB_PARAMS$objective,
    eval_metric      = XGB_PARAMS$eval_metric,
    n_features       = length(feature_cols),
    max_depth        = XGB_PARAMS$max_depth,
    eta              = XGB_PARAMS$eta,
    subsample        = XGB_PARAMS$subsample,
    colsample_bytree = XGB_PARAMS$colsample_bytree,
    nrounds          = NROUNDS,
    early_stopping   = EARLY_STOPPING,
    best_iteration   = ltr_result$best_iteration,
    train_time_sec   = round(ltr_result$train_time_sec, 2),
    seed             = SEED
  ),
  topk = list(
    allocation = "national",
    K_values   = K_VALUES,
    n_boot_ci  = N_BOOT_CI
  ),
  features_match_headline = TRUE,
  comparability = "exact"
)

write_json(run_config, file.path(OUTPUT_DIR, "ltr_run_config.json"),
           pretty = TRUE, auto_unbox = TRUE)

# Plain-text run summary
test_k10 <- topk_metrics %>% filter(split == "test", k == 10)
val_k10  <- topk_metrics %>% filter(split == "val",  k == 10)
ci_k10   <- bootstrap_ci %>% filter(split == "test", k == 10)
summary_lines <- c(
  "LTR rank:pairwise (bench_rs_history feature set) - run summary",
  "",
  paste0("Headline run:        ", basename(HEADLINE_RUN)),
  paste0("Comparability:       exact (same panel, same splits, same 83 features)"),
  paste0("Train rows:          ", format(nrow(train_data), big.mark = ",")),
  paste0("Val rows:            ", format(nrow(val_data),   big.mark = ",")),
  paste0("Test rows:           ", format(nrow(test_data),  big.mark = ",")),
  paste0("Test prevalence:     ", round(mean(test_data$outage_3h_or_more), 6)),
  paste0("Best iteration:      ", ltr_result$best_iteration),
  paste0("Train time (s):      ", round(ltr_result$train_time_sec, 1)),
  "",
  "Top-K (TEST, national allocation):",
  sprintf("  K=10  precision = %.4f  recall = %.4f  lift = %.2f  hits = %d",
          test_k10$precision, test_k10$recall, test_k10$lift, test_k10$n_hits),
  sprintf("  K=10  precision 95%% CI = [%.4f, %.4f]   lift 95%% CI = [%.2f, %.2f]",
          ci_k10$precision_ci_lo, ci_k10$precision_ci_hi,
          ci_k10$lift_ci_lo,      ci_k10$lift_ci_hi),
  "",
  "Top-K (VAL, national allocation):",
  sprintf("  K=10  precision = %.4f  recall = %.4f  lift = %.2f  hits = %d",
          val_k10$precision, val_k10$recall, val_k10$lift, val_k10$n_hits)
)
writeLines(summary_lines, file.path(OUTPUT_DIR, "ltr_run_summary.txt"))

cat("\nWrote:\n")
cat("  ", file.path(OUTPUT_DIR, "ltr_topk_metrics.csv"), "\n")
cat("  ", file.path(OUTPUT_DIR, "ltr_topk_bootstrap_ci.csv"), "\n")
cat("  ", file.path(OUTPUT_DIR, "ltr_feature_importance.csv"), "\n")
cat("  ", file.path(OUTPUT_DIR, "ltr_predictions_test.csv.gz"), "\n")
cat("  ", file.path(OUTPUT_DIR, "ltr_predictions_val.csv.gz"), "\n")
cat("  ", file.path(OUTPUT_DIR, "ltr_features_used.txt"), "\n")
cat("  ", file.path(OUTPUT_DIR, "ltr_run_config.json"), "\n")
cat("  ", file.path(OUTPUT_DIR, "ltr_run_summary.txt"), "\n")

cat("\nDone.\n")
