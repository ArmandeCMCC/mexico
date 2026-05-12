# 05_evaluate_combined.R
# Comprehensive evaluation comparing baseline, anomaly-only, and combined models
# Metrics: ROC AUC, PR AUC, Top-K, Platt calibration, bootstrap CIs
#
# v1.1 Updates:
# - Save reliability curves per model (orig + Platt)
# - Lead-time evaluation (t-1, t-2, t-3 prediction)
# - Fix risk mapping bug (per-model outputs, sanity check)
#
# v1.2 Updates:
# - Support new model variants: combined_lag1, combined_drop_only_lag1
# - Compare v1.1 (same-day) vs v1.2 (warning-safe lag1) performance
# - Dynamically detect available models from prediction files

library(tidyverse)
library(yardstick)

# CONFIGURATION

# Parse args: either [run_tag] or [task_mode] or empty
args <- commandArgs(trailingOnly = TRUE)
args <- args[!grepl("^--", args)]

# Determine task_mode and RUN_TAG
task_mode <- "nowcast"  # default
RUN_TAG <- NULL

if (length(args) > 0) {
  # If arg looks like a timestamp (starts with digit), treat as run_tag
  if (grepl("^[0-9]", args[1])) {
    RUN_TAG <- args[1]
    # Infer task_mode from run_tag suffix if present
    if (grepl("_nowcast$", RUN_TAG)) task_mode <- "nowcast"
    if (grepl("_forecast$", RUN_TAG)) task_mode <- "forecast"
  } else if (args[1] %in% c("nowcast", "forecast")) {
    task_mode <- args[1]
  }
}

# If RUN_TAG not provided, read from pointer file
if (is.null(RUN_TAG)) {
  pointer_file <- file.path("runs/combined", paste0("latest_run_tag_", task_mode, ".txt"))
  if (!file.exists(pointer_file)) {
    stop("Pointer file not found: ", pointer_file,
         "\nRun 04_train_combined_model.R first with task_mode='", task_mode, "'")
  }
  RUN_TAG <- readLines(pointer_file, warn = FALSE)[1]
}

model_dir <- file.path("runs/combined", RUN_TAG)
if (!dir.exists(model_dir)) {
  stop("Model directory not found: ", model_dir)
}

output_dir <- file.path(model_dir, "evaluation")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# K values for Top-K evaluation
K_VALUES <- c(1, 2, 5, 10, 20, 50, 100)

# Bootstrap settings
N_BOOTSTRAP <- 1000
BOOTSTRAP_SEED <- 42

# Lead-time settings
LEAD_TIMES <- c(1, 2, 3)  # Days ahead

cat("=== 05_evaluate_combined.R ===\n")
cat(sprintf("Model directory: %s\n", model_dir))
cat(sprintf("Output directory: %s\n", output_dir))

# 1. LOAD PREDICTIONS

cat("\n=== Loading Predictions ===\n")

# v1.2: Dynamically detect available models from prediction files
all_pred_files <- list.files(model_dir, pattern = "predictions_test_.*\\.csv(\\.gz)?$")
models <- gsub("predictions_test_|\\.csv(\\.gz)?$", "", all_pred_files)
models <- unique(models)

cat(sprintf("Detected %d model variants: %s\n", length(models), paste(models, collapse = ", ")))

# Ensure baseline is always included and first
if (!"baseline" %in% models) {
  stop("ERROR: baseline model predictions not found!")
}
models <- c("baseline", setdiff(models, "baseline"))

test_preds <- list()
val_preds <- list()

for (model in models) {
  # Try compressed first, fall back to uncompressed for backwards compatibility
  test_file <- file.path(model_dir, sprintf("predictions_test_%s.csv.gz", model))
  if (!file.exists(test_file)) {
    test_file <- file.path(model_dir, sprintf("predictions_test_%s.csv", model))
  }
  val_file <- file.path(model_dir, sprintf("predictions_val_%s.csv.gz", model))
  if (!file.exists(val_file)) {
    val_file <- file.path(model_dir, sprintf("predictions_val_%s.csv", model))
  }

  if (!file.exists(test_file)) {
    cat(sprintf("  SKIP %s: test file not found\n", model))
    next
  }

  test_preds[[model]] <- read_csv(test_file, show_col_types = FALSE) %>%
    mutate(
      prob = .pred_1,
      truth = as.numeric(as.character(outage_3h_or_more))
    )

  val_preds[[model]] <- read_csv(val_file, show_col_types = FALSE) %>%
    mutate(
      prob = .pred_1,
      truth = as.numeric(as.character(outage_3h_or_more))
    )

  cat(sprintf("  %s: test=%s, val=%s\n",
              model,
              format(nrow(test_preds[[model]]), big.mark = ","),
              format(nrow(val_preds[[model]]), big.mark = ",")))
}

# Update models list to only include those successfully loaded
models <- names(test_preds)

# Select primary evaluation model (for lead-time, risk mapping, etc.)
# Prefer combined_lag1 > combined_drop_only_lag1 > combined > anomaly_lag1 > anomaly_only
preferred_models <- c("combined_lag1", "combined_drop_only_lag1", "combined", "anomaly_lag1", "anomaly_only")
eval_model <- preferred_models[preferred_models %in% models][1]
if (is.na(eval_model) || length(eval_model) == 0) {
  eval_model <- "baseline"
}
cat(sprintf("\nPrimary evaluation model: %s\n", eval_model))

# 2. OVERALL METRICS

cat("\n=== Computing Overall Metrics ===\n")

compute_overall_metrics <- function(preds, model_name) {
  truth_factor <- factor(preds$truth, levels = c(0, 1))

  # ROC AUC
  roc_auc <- roc_auc_vec(truth_factor, preds$prob, event_level = "second")

  # PR AUC
  pr_auc <- pr_auc_vec(truth_factor, preds$prob, event_level = "second")

  # Brier score
  brier <- mean((preds$prob - preds$truth)^2)

  # Base rate
  base_rate <- mean(preds$truth)

  tibble(
    model = model_name,
    roc_auc = roc_auc,
    pr_auc = pr_auc,
    brier = brier,
    base_rate = base_rate,
    n_obs = nrow(preds),
    n_positives = sum(preds$truth)
  )
}

overall_metrics_test <- bind_rows(
  lapply(models, function(m) compute_overall_metrics(test_preds[[m]], m))
)

overall_metrics_val <- bind_rows(
  lapply(models, function(m) compute_overall_metrics(val_preds[[m]], m))
)

cat("\n--- TEST SET ---\n")
print(overall_metrics_test %>% mutate(across(where(is.numeric), ~round(., 4))))

cat("\n--- VALIDATION SET ---\n")
print(overall_metrics_val %>% mutate(across(where(is.numeric), ~round(., 4))))

write_csv(overall_metrics_test, file.path(output_dir, "overall_metrics_test.csv"))
write_csv(overall_metrics_val, file.path(output_dir, "overall_metrics_val.csv"))

# 3. TOP-K METRICS

cat("\n=== Computing Top-K Metrics ===\n")

compute_topk <- function(preds, k) {
  daily <- preds %>%
    group_by(date) %>%
    arrange(desc(prob)) %>%
    mutate(rank = row_number(), n_obs = n()) %>%
    ungroup() %>%
    filter(rank <= k, n_obs >= k)

  base_rate <- mean(preds$truth)

  tibble(
    k = k,
    precision = sum(daily$truth) / nrow(daily),
    recall = sum(daily$truth) / sum(preds$truth),
    n_alerts = nrow(daily),
    n_hits = sum(daily$truth),
    n_days = n_distinct(daily$date)
  ) %>%
    mutate(lift = precision / base_rate)
}

# Compute topk for TEST
topk_results_test <- list()
for (model in models) {
  topk_results_test[[model]] <- map_dfr(K_VALUES, ~compute_topk(test_preds[[model]], .x)) %>%
    mutate(model = model)
}
topk_test <- bind_rows(topk_results_test)

# Compute topk for VAL
topk_results_val <- list()
for (model in models) {
  topk_results_val[[model]] <- map_dfr(K_VALUES, ~compute_topk(val_preds[[model]], .x)) %>%
    mutate(model = model)
}
topk_val <- bind_rows(topk_results_val)

cat("\n--- Top-K Comparison (VAL) ---\n")
topk_wide_val <- topk_val %>%
  filter(k == 10) %>%
  select(model, precision, lift, n_hits) %>%
  mutate(across(where(is.numeric), ~round(., 4)))
print(topk_wide_val)

cat("\n--- Top-K Comparison (TEST) ---\n")
topk_wide_test <- topk_test %>%
  filter(k == 10) %>%
  select(model, precision, lift, n_hits) %>%
  mutate(across(where(is.numeric), ~round(., 4)))
print(topk_wide_test)

write_csv(topk_val, file.path(output_dir, "topk_metrics_val.csv"))
write_csv(topk_test, file.path(output_dir, "topk_metrics_test.csv"))

# 4. BOOTSTRAP CONFIDENCE INTERVALS

cat("\n=== Computing Bootstrap CIs ===\n")

set.seed(BOOTSTRAP_SEED)

# Pre-compute daily tables for efficient bootstrap
compute_daily_table <- function(preds, k) {
  preds %>%
    group_by(date) %>%
    arrange(desc(prob)) %>%
    mutate(rank = row_number(), n_obs = n()) %>%
    filter(rank <= k, n_obs >= k) %>%
    summarise(
      tp_k = sum(truth),
      n_alerts = n(),
      n_positives_day = sum(truth),
      .groups = "drop"
    )
}

# Bootstrap function (FIXED: use row-index sampling, not match())
bootstrap_topk <- function(preds, k, n_boot = N_BOOTSTRAP) {
  daily <- compute_daily_table(preds, k)
  base_rate <- mean(preds$truth)
  n_days <- nrow(daily)

  boot_results <- replicate(n_boot, {
    # Row-index bootstrap (correctly handles duplicates)
    idx <- sample.int(n_days, size = n_days, replace = TRUE)
    sampled <- daily[idx, , drop = FALSE]
    precision <- sum(sampled$tp_k) / sum(sampled$n_alerts)
    lift <- precision / base_rate
    c(precision = precision, lift = lift)
  })

  tibble(
    k = k,
    precision_mean = mean(boot_results["precision", ]),
    precision_ci_lo = quantile(boot_results["precision", ], 0.025),
    precision_ci_hi = quantile(boot_results["precision", ], 0.975),
    lift_mean = mean(boot_results["lift", ]),
    lift_ci_lo = quantile(boot_results["lift", ], 0.025),
    lift_ci_hi = quantile(boot_results["lift", ], 0.975)
  )
}

# Bootstrap for K=10 (main operational metric) - TEST
bootstrap_k10_test <- list()
for (model in models) {
  bootstrap_k10_test[[model]] <- bootstrap_topk(test_preds[[model]], k = 10) %>%
    mutate(model = model)
}
bootstrap_k10_test_all <- bind_rows(bootstrap_k10_test)

# Bootstrap for K=10 - VAL
bootstrap_k10_val <- list()
for (model in models) {
  bootstrap_k10_val[[model]] <- bootstrap_topk(val_preds[[model]], k = 10) %>%
    mutate(model = model)
}
bootstrap_k10_val_all <- bind_rows(bootstrap_k10_val)

cat("\n--- Bootstrap CIs for K=10 (VAL) ---\n")
print(bootstrap_k10_val_all %>%
        select(model, lift_mean, lift_ci_lo, lift_ci_hi) %>%
        mutate(across(where(is.numeric), ~round(., 2))))

cat("\n--- Bootstrap CIs for K=10 (TEST) ---\n")
print(bootstrap_k10_test_all %>%
        select(model, lift_mean, lift_ci_lo, lift_ci_hi) %>%
        mutate(across(where(is.numeric), ~round(., 2))))

write_csv(bootstrap_k10_val_all, file.path(output_dir, "bootstrap_k10_val.csv"))
write_csv(bootstrap_k10_test_all, file.path(output_dir, "bootstrap_k10_test.csv"))

# 5. PLATT CALIBRATION + RELIABILITY CURVES

cat("\n=== Platt Calibration ===\n")

# Safe logit
logit_safe <- function(p, eps = 1e-7) {
  p <- pmax(pmin(p, 1 - eps), eps)
  log(p / (1 - p))
}

# Calibration metrics with full reliability curve
compute_calibration <- function(df, prob_col = "prob", n_bins = 10) {
  brier <- mean((df[[prob_col]] - df$truth)^2)

  df_binned <- df %>%
    mutate(prob_bin = ntile(!!sym(prob_col), n_bins))

  reliability <- df_binned %>%
    group_by(prob_bin) %>%
    summarise(
      bin_min = min(!!sym(prob_col)),
      bin_max = max(!!sym(prob_col)),
      bin_center = mean(!!sym(prob_col)),
      observed_rate = mean(truth),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(calibration_error = abs(observed_rate - bin_center))

  ece <- sum((reliability$n / nrow(df)) * reliability$calibration_error)

  list(brier = brier, ece = ece, reliability = reliability)
}

# Fit Platt scaling on validation, apply to test
calibration_results <- list()
platt_models <- list()

for (model in models) {
  # Add logit
  val_preds[[model]]$logit_prob <- logit_safe(val_preds[[model]]$prob)
  test_preds[[model]]$logit_prob <- logit_safe(test_preds[[model]]$prob)

  # Fit Platt model on validation
  platt_models[[model]] <- glm(truth ~ logit_prob, data = val_preds[[model]], family = binomial())

  # Apply to test
  test_preds[[model]]$prob_platt <- predict(platt_models[[model]], newdata = test_preds[[model]], type = "response")

  # Compute calibration metrics
  cal_orig <- compute_calibration(test_preds[[model]], "prob")
  cal_platt <- compute_calibration(test_preds[[model]], "prob_platt")

  calibration_results[[model]] <- tibble(
    model = model,
    brier_orig = cal_orig$brier,
    brier_platt = cal_platt$brier,
    ece_orig = cal_orig$ece,
    ece_platt = cal_platt$ece,
    platt_intercept = coef(platt_models[[model]])[1],
    platt_slope = coef(platt_models[[model]])[2]
  )

  # SAVE RELIABILITY CURVES (NEW in v1.1)
  cal_orig$reliability %>%
    mutate(model = model, calibration_type = "original") %>%
    write_csv(file.path(output_dir, sprintf("reliability_%s_orig.csv", model)))

  cal_platt$reliability %>%
    mutate(model = model, calibration_type = "platt") %>%
    write_csv(file.path(output_dir, sprintf("reliability_%s_platt.csv", model)))
}

calibration_all <- bind_rows(calibration_results)

cat("\n--- Calibration Comparison (TEST) ---\n")
print(calibration_all %>% mutate(across(where(is.numeric), ~round(., 5))))

write_csv(calibration_all, file.path(output_dir, "calibration_comparison.csv"))

cat("\nReliability curves saved per model (orig + platt)\n")

# 6. RISK MAPPING PER MODEL (FIXED in v1.1)

cat("\n=== Risk Warning to Confidence Mapping ===\n")

# Compute risk mapping separately for each model
compute_risk_mapping <- function(preds, model_name) {
  preds %>%
    mutate(prob_decile = ntile(prob, 10)) %>%
    group_by(prob_decile) %>%
    summarise(
      prob_range_lo = min(prob),
      prob_range_hi = max(prob),
      mean_predicted = mean(prob),
      observed_rate = mean(truth),
      n_obs = n(),
      n_positives = sum(truth),
      .groups = "drop"
    ) %>%
    mutate(
      model = model_name,
      lift = observed_rate / mean(preds$truth)
    )
}

# Compute and save risk mapping for each model
risk_mapping_list <- list()
for (model in models) {
  risk_mapping_list[[model]] <- compute_risk_mapping(test_preds[[model]], model)
  # Save per-model risk mapping
  write_csv(risk_mapping_list[[model]],
            file.path(output_dir, sprintf("risk_mapping_%s.csv", model)))
}

risk_mapping_all <- bind_rows(risk_mapping_list)

# SANITY CHECK: baseline and eval_model should NOT be identical (skip if eval_model is baseline)
if (eval_model != "baseline" && eval_model %in% names(risk_mapping_list)) {
  baseline_rm <- risk_mapping_list$baseline %>% select(prob_decile, mean_predicted, observed_rate)
  eval_rm <- risk_mapping_list[[eval_model]] %>% select(prob_decile, mean_predicted, observed_rate)

  check_result <- all.equal(baseline_rm, eval_rm, tolerance = 1e-10)
  if (isTRUE(check_result)) {
    warning("BUG DETECTED: baseline and ", eval_model, " risk mappings are identical!")
    cat(sprintf("\nWARNING: Risk mappings for baseline and %s are identical.\n", eval_model))
    cat("This suggests predictions may be the same - investigate!\n")
  } else {
    cat(sprintf("\nRisk mapping sanity check PASSED: baseline and %s differ.\n", eval_model))
  }
} else {
  cat("\nRisk mapping sanity check SKIPPED (eval_model is baseline or not available).\n")
}

cat(sprintf("\n--- Risk Mapping (%s Model, Top Decile) ---\n", eval_model))
print(risk_mapping_list[[eval_model]] %>%
        filter(prob_decile == 10) %>%
        select(prob_range_lo, prob_range_hi, mean_predicted, observed_rate, lift) %>%
        mutate(across(where(is.numeric), ~round(., 4))))

write_csv(risk_mapping_all, file.path(output_dir, "risk_mapping_all.csv"))

# 7. LEAD-TIME EVALUATION (NEW in v1.1)

cat("\n=== Lead-Time Evaluation ===\n")

# Compute prediction performance using prob from t-L to predict outcome at t
compute_leadtime_topk <- function(preds, k, lead) {
  # Create lagged probability within municipality
  preds_lagged <- preds %>%
    arrange(GID_2, date) %>%
    group_by(GID_2) %>%
    mutate(
      prob_lag = lag(prob, lead),
      date_pred = lag(date, lead)
    ) %>%
    ungroup() %>%
    filter(!is.na(prob_lag))

  # Compute top-K using lagged probabilities
  daily <- preds_lagged %>%
    group_by(date) %>%
    arrange(desc(prob_lag)) %>%
    mutate(rank = row_number(), n_obs = n()) %>%
    ungroup() %>%
    filter(rank <= k, n_obs >= k)

  base_rate <- mean(preds_lagged$truth)

  tibble(
    lead_time = lead,
    k = k,
    precision = sum(daily$truth) / nrow(daily),
    recall = sum(daily$truth) / sum(preds_lagged$truth),
    n_alerts = nrow(daily),
    n_hits = sum(daily$truth),
    n_days = n_distinct(daily$date)
  ) %>%
    mutate(lift = precision / base_rate)
}

# Compute lead-time evaluation for eval_model
leadtime_results <- list()
for (lead in LEAD_TIMES) {
  for (k in c(5, 10, 20)) {
    leadtime_results[[paste(lead, k, sep = "_")]] <- compute_leadtime_topk(
      test_preds[[eval_model]], k = k, lead = lead
    ) %>%
      mutate(model = eval_model)
  }
}

leadtime_topk <- bind_rows(leadtime_results)

cat(sprintf("\n--- Lead-Time Top-K Performance (%s Model) ---\n", eval_model))
print(leadtime_topk %>%
        select(lead_time, k, precision, lift, n_hits) %>%
        mutate(across(where(is.numeric), ~round(., 4))))

write_csv(leadtime_topk, file.path(output_dir, "leadtime_topk_test.csv"))

# Event rate by score quantile at different lead times
compute_leadtime_quantile <- function(preds, lead, n_quantiles = 10) {
  preds_lagged <- preds %>%
    arrange(GID_2, date) %>%
    group_by(GID_2) %>%
    mutate(prob_lag = lag(prob, lead)) %>%
    ungroup() %>%
    filter(!is.na(prob_lag))

  preds_lagged %>%
    mutate(prob_quantile = ntile(prob_lag, n_quantiles)) %>%
    group_by(prob_quantile) %>%
    summarise(
      mean_prob = mean(prob_lag),
      event_rate = mean(truth),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(
      lead_time = lead,
      lift = event_rate / mean(preds_lagged$truth)
    )
}

leadtime_quantile <- bind_rows(
  lapply(LEAD_TIMES, function(l) compute_leadtime_quantile(test_preds[[eval_model]], l))
)

cat(sprintf("\n--- Lead-Time Event Rate by Quantile (%s Model) ---\n", eval_model))
print(leadtime_quantile %>%
        filter(prob_quantile == 10) %>%
        select(lead_time, prob_quantile, mean_prob, event_rate, lift) %>%
        mutate(across(where(is.numeric), ~round(., 4))))

write_csv(leadtime_quantile, file.path(output_dir, "leadtime_quantile_event_rate.csv"))

# 8. IMPROVEMENT SUMMARY

cat("\n=== Improvement Summary ===\n")

# Compare all models vs baseline
baseline_metrics <- overall_metrics_test %>% filter(model == "baseline")
baseline_topk10 <- topk_test %>% filter(model == "baseline", k == 10)

# Build improvement comparison for all models vs baseline
improvement_list <- list()
for (model_name in setdiff(models, "baseline")) {
  model_metrics <- overall_metrics_test %>% filter(model == model_name)
  model_topk10 <- topk_test %>% filter(model == model_name, k == 10)

  improvement_list[[model_name]] <- tibble(
    comparison = paste0(model_name, " vs baseline"),
    metric = c("ROC AUC", "PR AUC", "Brier Score", "Lift@10", "Precision@10", "Hits@10"),
    baseline = c(
      baseline_metrics$roc_auc,
      baseline_metrics$pr_auc,
      baseline_metrics$brier,
      baseline_topk10$lift,
      baseline_topk10$precision,
      baseline_topk10$n_hits
    ),
    model_value = c(
      model_metrics$roc_auc,
      model_metrics$pr_auc,
      model_metrics$brier,
      model_topk10$lift,
      model_topk10$precision,
      model_topk10$n_hits
    )
  ) %>%
    mutate(
      absolute_change = model_value - baseline,
      pct_change = 100 * (model_value - baseline) / abs(baseline),
      direction = case_when(
        metric == "Brier Score" & absolute_change < 0 ~ "Improved",
        metric != "Brier Score" & absolute_change > 0 ~ "Improved",
        TRUE ~ "Degraded"
      )
    )
}

improvement <- bind_rows(improvement_list)

cat("\n--- All Models vs Baseline ---\n")
print(improvement %>% mutate(across(where(is.numeric), ~round(., 4))))

write_csv(improvement, file.path(output_dir, "improvement_summary.csv"))

# v1.2: Summary table of Hits@10 for quick comparison
cat("\n--- Quick Hits@10 Comparison (VAL) ---\n")
hits_summary_val <- topk_val %>%
  filter(k == 10) %>%
  select(model, lift, n_hits, precision) %>%
  arrange(desc(n_hits))
print(hits_summary_val %>% mutate(across(where(is.numeric), ~round(., 4))))

cat("\n--- Quick Hits@10 Comparison (TEST) ---\n")
hits_summary_test <- topk_test %>%
  filter(k == 10) %>%
  select(model, lift, n_hits, precision) %>%
  arrange(desc(n_hits))
print(hits_summary_test %>% mutate(across(where(is.numeric), ~round(., 4))))

write_csv(hits_summary_val, file.path(output_dir, "hits10_summary_val.csv"))
write_csv(hits_summary_test, file.path(output_dir, "hits10_summary.csv"))

# 9. GENERATE FINAL REPORT

cat("\n=== Generating Report ===\n")

# v1.2: Find best combined model (highest Hits@10)
best_combined_name <- hits_summary_test$model[hits_summary_test$model != "baseline" & hits_summary_test$model != "anomaly_only"][1]
best_combined_metrics <- overall_metrics_test %>% filter(model == best_combined_name)
best_combined_topk10 <- topk_test %>% filter(model == best_combined_name, k == 10)
best_combined_bootstrap <- bootstrap_k10_test_all %>% filter(model == best_combined_name)

# For risk mapping, use eval_model
risk_model <- eval_model

report <- sprintf("
================================================================================
APPROACH 2: ANOMALY DETECTION + SUPERVISED MODEL
Evaluation Report (v1.2)
================================================================================

Run: %s
Generated: %s

SUMMARY
-------
v1.2 introduces WARNING-SAFE features (lag1 only) and drop-focused variants.
Best model: %s

KEY FINDINGS
------------

1. Hits@10 Ranking (TEST SET)
%s

2. Overall Metrics (Test Set)
   Model                       ROC-AUC    PR-AUC     Brier
   Baseline                    %.4f       %.4f       %.6f
   Best Combined (%s)          %.4f       %.4f       %.6f

3. Bootstrap 95%% CI for Lift@10
   Baseline:     [%.2f, %.2f]
   Best Combined: [%.2f, %.2f]

4. Lead-Time Warning Power (Combined Model, K=10)
   Lead   Precision   Lift     Hits
   t-1    %.4f        %.2f     %d
   t-2    %.4f        %.2f     %d
   t-3    %.4f        %.2f     %d

FILES GENERATED
---------------
- overall_metrics_test.csv, topk_metrics_test.csv, bootstrap_k10_test.csv
- hits10_summary.csv (quick comparison table)
- calibration_comparison.csv, reliability_*.csv, risk_mapping_*.csv
- leadtime_topk_test.csv, improvement_summary.csv

================================================================================
",
RUN_TAG,
format(Sys.time(), "%%Y-%%m-%%d %%H:%%M:%%S"),
best_combined_name,
paste(sprintf("   %-30s Hits=%d  Lift=%.2f", hits_summary_test$model, hits_summary_test$n_hits, hits_summary_test$lift), collapse = "\n"),
baseline_metrics$roc_auc, baseline_metrics$pr_auc, baseline_metrics$brier,
best_combined_name,
best_combined_metrics$roc_auc, best_combined_metrics$pr_auc, best_combined_metrics$brier,
bootstrap_k10_test_all$lift_ci_lo[bootstrap_k10_test_all$model == "baseline"],
bootstrap_k10_test_all$lift_ci_hi[bootstrap_k10_test_all$model == "baseline"],
best_combined_bootstrap$lift_ci_lo, best_combined_bootstrap$lift_ci_hi,
leadtime_topk$precision[leadtime_topk$lead_time == 1 & leadtime_topk$k == 10],
leadtime_topk$lift[leadtime_topk$lead_time == 1 & leadtime_topk$k == 10],
leadtime_topk$n_hits[leadtime_topk$lead_time == 1 & leadtime_topk$k == 10],
leadtime_topk$precision[leadtime_topk$lead_time == 2 & leadtime_topk$k == 10],
leadtime_topk$lift[leadtime_topk$lead_time == 2 & leadtime_topk$k == 10],
leadtime_topk$n_hits[leadtime_topk$lead_time == 2 & leadtime_topk$k == 10],
leadtime_topk$precision[leadtime_topk$lead_time == 3 & leadtime_topk$k == 10],
leadtime_topk$lift[leadtime_topk$lead_time == 3 & leadtime_topk$k == 10],
leadtime_topk$n_hits[leadtime_topk$lead_time == 3 & leadtime_topk$k == 10]
)

cat(report)
writeLines(report, file.path(output_dir, "evaluation_report.txt"))

# 10. GENERATE MANIFEST (DISK USAGE TRACKING)

cat("\n=== Generating Manifest ===\n")

# List all files in the run directory and compute sizes
run_files <- list.files(model_dir, recursive = TRUE, full.names = TRUE)
manifest <- tibble(
  path = run_files,
  relative_path = gsub(paste0(model_dir, "/"), "", run_files),
  size_bytes = file.size(run_files)
) %>%
  mutate(size_mb = round(size_bytes / 1024 / 1024, 3)) %>%
  arrange(desc(size_mb))

write_csv(manifest, file.path(output_dir, "manifest_files.csv"))
cat(sprintf("  Saved: %s/manifest_files.csv\n", output_dir))
cat(sprintf("  Total files: %d, Total size: %.1f MB\n", nrow(manifest), sum(manifest$size_mb)))

cat(sprintf("\n=== Done ===\n"))
cat(sprintf("Evaluation saved to: %s\n", output_dir))
