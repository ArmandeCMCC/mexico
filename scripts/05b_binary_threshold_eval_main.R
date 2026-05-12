# 05b_binary_threshold_eval.R
# PURPOSE:
#   Clean, reproducible binary-threshold baseline evaluation
#   Uses existing engineered features and splits (does NOT modify them)
#   Produces classic binary classifier evaluation: ROC/PR curves, calibration, Brier score, etc.
#
# IMPORTANT:
#   - Uses SAME exclude_cols logic as 05_model_training.R
#   - Outputs to SAFE folder: data/baselines/binary_threshold/<run_id>/
#   - Does NOT overwrite any existing model_ready outputs
#   - Threshold is chosen on VALIDATION only; TEST is evaluated at that threshold
#
# OUTPUTS (data/baselines/binary_threshold/<run_id>/):
#   - metrics_summary.csv                  All metrics in one file
#   - roc_curve.csv                        ROC curve data (FPR, TPR, threshold)
#   - pr_curve.csv                         PR curve data (recall, precision, threshold)
#   - calibration_bins_equal.csv           Equal-width calibration bins
#   - calibration_bins_quantile.csv        Quantile-based calibration bins (more informative)
#   - calibration_metrics.csv              Intercept, slope, ECE, Brier score, log-loss
#   - calibration_comparison_test.csv      Comparison of raw vs platt vs isotonic vs beta (TEST)
#   - calibration_bins_equal_all_methods.csv    Combined bins for all calibration methods
#   - calibration_bins_quantile_all_methods.csv Combined bins for all calibration methods
#   - threshold_sweep.csv                  Metrics across thresholds (raw only, legacy)
#   - threshold_sweep_calibrated.csv       Metrics across thresholds for ALL methods (raw, platt, beta, isotonic)
#   - decision_curve.csv                   Net benefit vs threshold for ALL calibration methods
#   - alerts_per_day_val.csv               Alerts per day at best_threshold (val)
#   - alerts_per_day_test.csv              Alerts per day at best_threshold (test)
#   - test_at_val_threshold.csv            Test metrics evaluated at VAL-optimal threshold
#   - platt_calibration.csv                Platt-scaled calibration (optional)
#   - operating_thresholds_table.csv       Operating points for no-fixed-budget analysis
#   - benchmark_table_same_run.csv         Unified comparison: binary (raw/platt/beta) + LTR
#   - benchmark_uncertainty_bootstrap.csv  Bootstrap CIs for benchmark metrics at reference policy
#   - rolling_temporal_windows.csv         Per-window metrics for temporal stability analysis
#   - temporal_stability_summary.csv       Variance/CV summary across rolling windows
#   - deployment_decision_table.csv        Deployment-ready summary with CIs and variance
#   - feature_importance.csv               XGBoost gain importance
#   - run_config.json                      Full run configuration for reproducibility
#
# Usage:
#   Rscript scripts/05b_binary_threshold_eval.R [task_mode] [strict_mode] [bootstrap_n]
#   task_mode: "forecast" (default) or "nowcast"
#   strict_mode: "strict" (default) or "standard" - controls calibration-threshold separation
#   bootstrap_n: integer (default 500) - number of bootstrap iterations
#
# Environment variables (override CLI args):
#   STRICT_MODE: "true" or "false"
#   BOOTSTRAP_N: integer
#
# Examples:
#   Rscript scripts/05b_binary_threshold_eval.R forecast strict 500    # strict mode, 500 bootstrap
#   Rscript scripts/05b_binary_threshold_eval.R forecast standard 50   # standard mode, 50 bootstrap (fast)
#   STRICT_MODE=false BOOTSTRAP_N=100 Rscript scripts/05b_binary_threshold_eval.R forecast

# CONFIGURATION

set.seed(42)

args <- commandArgs(trailingOnly = TRUE)
args <- args[!grepl("^--", args)]

# Task mode from CLI or default (forecast = strict t-1 only)
task_mode <- if (length(args) > 0) args[1] else "forecast"
stopifnot(task_mode %in% c("nowcast", "forecast"))

# CLI/ENV CONFIGURABLE PARAMETERS 

# Strict mode: CLI arg 2 or env var STRICT_MODE
strict_mode_arg <- if (length(args) > 1) args[2] else NULL
strict_mode_env <- Sys.getenv("STRICT_MODE", unset = NA)

if (!is.null(strict_mode_arg) && strict_mode_arg %in% c("strict", "standard")) {
  STRICT_CALIBRATION_THRESHOLD_SEPARATION <- (strict_mode_arg == "strict")
} else if (!is.na(strict_mode_env)) {
  STRICT_CALIBRATION_THRESHOLD_SEPARATION <- (tolower(strict_mode_env) == "true")
} else {
  STRICT_CALIBRATION_THRESHOLD_SEPARATION <- TRUE  # default
}

# Bootstrap N: CLI arg 3 or env var BOOTSTRAP_N
bootstrap_n_arg <- if (length(args) > 2) suppressWarnings(as.integer(args[3])) else NA
bootstrap_n_env <- suppressWarnings(as.integer(Sys.getenv("BOOTSTRAP_N", unset = NA)))

if (!is.na(bootstrap_n_arg) && bootstrap_n_arg > 0) {
  BOOTSTRAP_N <- bootstrap_n_arg
} else if (!is.na(bootstrap_n_env) && bootstrap_n_env > 0) {
  BOOTSTRAP_N <- bootstrap_n_env
} else {
  BOOTSTRAP_N <- 500  # default
}

# Weather mode: CLI arg 4 or env var WEATHER_MODE
# "lagged"  = drop same-day weather (truly forecast-safe; primary benchmark)
# "sameday" = keep same-day weather (assumes weather forecast available; sensitivity)
weather_mode_arg <- if (length(args) > 3) args[4] else NULL
weather_mode_env <- Sys.getenv("WEATHER_MODE", unset = NA)

if (!is.null(weather_mode_arg) && weather_mode_arg %in% c("lagged", "sameday")) {
  weather_mode <- weather_mode_arg
} else if (!is.na(weather_mode_env) && weather_mode_env %in% c("lagged", "sameday")) {
  weather_mode <- weather_mode_env
} else {
  weather_mode <- "lagged"  # default = primary benchmark (forecast-safe)
}
stopifnot(weather_mode %in% c("lagged", "sameday"))

# Build run ID with mode suffix for easy identification
mode_suffix <- if (STRICT_CALIBRATION_THRESHOLD_SEPARATION) "strict" else "standard"
RUN_ID <- paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_",
                 task_mode, "_", mode_suffix, "_", weather_mode)

# Calibration settings
N_CALIBRATION_BINS <- 10
PLATT_SCALING <- TRUE  # Whether to compute Platt-scaled calibration

# DEFAULT CALIBRATED METHOD SPEC 
# Platt is the default because:
#   - Simple 2-parameter logistic model, less prone to overfitting than isotonic
#   - Well-studied theoretical properties (Platt 1999)
#   - Matches calibration slope interpretation directly
# Beta is secondary (sensitivity analysis) - more flexible but similar to Platt for most cases
# Isotonic is diagnostic only - excluded from headline benchmark due to step-function behavior
DEFAULT_CALIBRATED_METHOD <- "platt"
SECONDARY_CALIBRATED_METHOD <- "beta"
BENCHMARK_BINARY_METHODS <- c("raw", "platt", "beta")  # frozen spec for benchmark table

# STRICT CALIBRATION-THRESHOLD SEPARATION 
# When TRUE: splits validation into two time-ordered parts
#   - First half: fit calibrators (Platt/Beta/Isotonic)
#   - Second half: select thresholds for calibrated methods
# This reduces optimism from using the same data for both calibration and threshold selection
# When FALSE: preserves original behavior (calibration and threshold selection on full validation)
# NOTE: Already set above via CLI/env parsing
STRICT_SPLIT_RATIO <- 0.5  # fraction of validation used for calibration fitting

# RAW METHOD THRESHOLD SELECTION IN STRICT MODE
# Design decision: Raw method uses FULL validation for threshold selection even in strict mode.
# Justification:
#   - Raw has no calibration step, so no fit leakage exists
#   - Using full validation maximizes sample size for threshold selection
#   - This is defensible because the only source of optimism (calibration-threshold overlap)
#     does not apply to raw probabilities
# Alternative: Set RAW_USE_THRESHOLD_SELECT_ONLY=TRUE to use threshold-select subset for symmetry
# This is a sensitivity analysis option, not recommended as default.
RAW_USE_THRESHOLD_SELECT_ONLY <- FALSE

# BOOTSTRAP UNCERTAINTY SETTINGS
# NOTE: BOOTSTRAP_N already set above via CLI/env parsing
BOOTSTRAP_SEED <- 42       # reproducibility
BOOTSTRAP_CI_LEVEL <- 0.95 # confidence level for intervals

# Log configuration
message("Configuration:")
message("  Task mode: ", task_mode)
message("  Weather mode: ", weather_mode,
        if (weather_mode == "lagged") "  (forecast-safe; same-day weather DROPPED)"
        else                            "  (sensitivity; same-day weather KEPT)")
message("  Strict calibration-threshold separation: ", STRICT_CALIBRATION_THRESHOLD_SEPARATION)
message("  Bootstrap N: ", BOOTSTRAP_N)
message("  Run ID: ", RUN_ID)

# Threshold sweep settings - full range from spammy (low t) to selective (high t)
# Covers: spammy regime (thousands/day) to very selective (~1/day)
THRESHOLD_GRID <- unique(round(c(
  seq(0.0001, 0.01,  by = 0.0001),  # very low (spammy, for completeness)
  seq(0.01,   0.10,  by = 0.001),   # low range
  seq(0.10,   0.90,  by = 0.01),    # mid range
  seq(0.90,   0.99,  by = 0.005),   # high range (operational)
  seq(0.99,   0.999, by = 0.001)    # very high (very low budgets)
), 6))

# XGBoost hyperparameters (same as 05_model_training.R)
xgb_params <- list(
  trees          = 500,
  tree_depth     = 6,
  min_n          = 10,
  loss_reduction = 0,
  sample_size    = 0.8,
  mtry           = 0.8,
  learn_rate     = 0.05
)

# LOAD PACKAGES

message("Loading packages...")
suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(tidymodels)
  library(xgboost)
  library(yardstick)
  library(jsonlite)
})

# PATH HANDLING

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (dir.exists("data") || dir.exists("scripts")) return(normalizePath(getwd()))
  warning("Could not determine project directory. Using current directory.")
  normalizePath(getwd())
}

project_dir <- detect_project_dir()
data_dir <- file.path(project_dir, "data")
model_ready_dir <- file.path(data_dir, "model_ready")

# Output directory anchored to project_dir (SAFE - does not touch model_ready)
OUTPUT_DIR <- file.path(project_dir, "data", "baselines", "binary_threshold", RUN_ID)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

message("\n", strrep("=", 70))
message("05b_binary_threshold_eval.R - Binary Threshold Baseline Evaluation")
message(strrep("=", 70))
message("Run ID: ", RUN_ID)
message("Task mode: ", task_mode)
message("Output directory: ", OUTPUT_DIR)
message("Project directory: ", project_dir)

# Input paths (read-only)
features_path <- file.path(model_ready_dir, "features_engineered.rds")
splits_fixed_path <- file.path(model_ready_dir, "splits_fixed.rds")

# EXCLUDE_COLS (sourced from 00_feature_config.R - single source of truth)

# Source the centralised feature configuration. This provides:
#   - EXCLUDE_COLS_BASE, EXCLUDE_SAMEDAY_NTL, EXCLUDE_SAMEDAY_WEATHER
#   - build_exclude_cols(task_mode, weather_mode)
#   - apply_task_mode_filters(features, task_mode, ...)
source(file.path(project_dir, "scripts", "00_feature_config.R"))

# Build exclude_cols from the centralized config (single source of truth)
exclude_cols <- build_exclude_cols(task_mode = task_mode, weather_mode = weather_mode)

message(sprintf("Exclude lists: base=%d, sameday_ntl=%d, sameday_weather=%d, total_applied=%d",
                length(EXCLUDE_COLS_BASE),
                length(EXCLUDE_SAMEDAY_NTL),
                length(EXCLUDE_SAMEDAY_WEATHER),
                length(exclude_cols)))

# HELPER FUNCTIONS

get_nthread <- function() {
  n <- suppressWarnings(parallel::detectCores())
  if (is.na(n) || n < 2) return(1L)
  max(1L, as.integer(n - 1))
}

# Cohen's kappa from confusion-matrix counts
# Uses double precision to avoid integer overflow with large N
kappa_from_counts <- function(tp, fp, tn, fn) {
  tp <- as.double(tp); fp <- as.double(fp)
  tn <- as.double(tn); fn <- as.double(fn)
  N  <- tp + fp + tn + fn
  po <- (tp + tn) / N
  pe <- ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (N^2)
  k  <- (po - pe) / (1 - pe)
  # handle edge cases (e.g., pe == 1)
  k[is.nan(k) | is.infinite(k)] <- NA_real_
  k
}

# NOTE: apply_task_mode_filters() is sourced from 00_feature_config.R (single source of truth).
# Do NOT redefine it here - the centralised version is stricter (also catches `nb_*` features
# without lag) and any divergence is a drift bug.

# Proper scoring rule: log-loss (cross entropy) 
compute_logloss <- function(truth, prob, eps = 1e-15) {
  ok <- !is.na(truth) & !is.na(prob)
  if (!any(ok)) return(NA_real_)
  y <- as.numeric(truth[ok] == "1")
  p <- pmin(pmax(prob[ok], eps), 1 - eps)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

# Brier score
compute_brier <- function(truth, prob) {
  ok <- !is.na(truth) & !is.na(prob)
  if (!any(ok)) return(NA_real_)
  y <- as.numeric(truth[ok] == "1")
  mean((prob[ok] - y)^2)
}

# Expected Calibration Error (ECE) - equal-width bins
compute_ece_equal <- function(truth, prob, n_bins = 10) {
  y <- as.numeric(truth == "1")
  breaks <- seq(0, 1, length.out = n_bins + 1)
  bin_idx <- cut(prob, breaks = breaks, include.lowest = TRUE, labels = FALSE)

  ece <- 0
  n_total <- length(prob)

  for (b in 1:n_bins) {
    in_bin <- bin_idx == b
    n_bin <- sum(in_bin, na.rm = TRUE)
    if (n_bin > 0) {
      avg_pred <- mean(prob[in_bin], na.rm = TRUE)
      avg_true <- mean(y[in_bin], na.rm = TRUE)
      ece <- ece + (n_bin / n_total) * abs(avg_pred - avg_true)
    }
  }

  ece
}

# Expected Calibration Error (ECE) - quantile bins (more informative for imbalanced data)
# Returns list with ece value and whether fallback occurred
compute_ece_quantile <- function(truth, prob, n_bins = 10, verbose = FALSE) {
 y <- as.numeric(truth == "1")

 # Use quantiles of predicted probabilities
 breaks <- quantile(prob, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE)
 n_unique_breaks <- length(unique(breaks))
 breaks <- unique(breaks)  # handle ties

 fallback <- FALSE
 if (length(breaks) < 3) {

   # Not enough variation - fallback to equal-width
   if (verbose) message("  [ECE quantile] Fallback to equal-width: only ", n_unique_breaks, " unique quantile breaks")
   return(list(ece = compute_ece_equal(truth, prob, n_bins), fallback = TRUE, n_bins_actual = n_bins))
 }

 actual_n_bins <- length(breaks) - 1
 if (actual_n_bins < n_bins && verbose) {
   message("  [ECE quantile] Reduced from ", n_bins, " to ", actual_n_bins, " bins due to ties in predicted probabilities")
 }

 bin_idx <- cut(prob, breaks = breaks, include.lowest = TRUE, labels = FALSE)

 ece <- 0
 n_total <- length(prob)

 for (b in 1:actual_n_bins) {
   in_bin <- which(bin_idx == b)
   n_bin <- length(in_bin)
   if (n_bin > 0) {
     avg_pred <- mean(prob[in_bin], na.rm = TRUE)
     avg_true <- mean(y[in_bin], na.rm = TRUE)
     ece <- ece + (n_bin / n_total) * abs(avg_pred - avg_true)
   }
 }

 list(ece = ece, fallback = (actual_n_bins < n_bins), n_bins_actual = actual_n_bins)
}

# Calibration bins - equal-width
compute_calibration_bins_equal <- function(truth, prob, n_bins = 10) {
  y <- as.numeric(truth == "1")
  breaks <- seq(0, 1, length.out = n_bins + 1)
  bin_idx <- cut(prob, breaks = breaks, include.lowest = TRUE, labels = FALSE)

  bins <- tibble(
    bin = 1:n_bins,
    bin_lower = head(breaks, -1),
    bin_upper = tail(breaks, -1),
    bin_midpoint = (bin_lower + bin_upper) / 2
  )

  bin_stats <- tibble(prob = prob, truth = y, bin = bin_idx) %>%
    group_by(bin) %>%
    summarise(
      n = n(),
      mean_predicted = mean(prob, na.rm = TRUE),
      mean_observed = mean(truth, na.rm = TRUE),
      .groups = "drop"
    )

  bins %>%
    left_join(bin_stats, by = "bin") %>%
    mutate(
      n = if_else(is.na(n), 0L, as.integer(n)),
      mean_predicted = if_else(is.na(mean_predicted), bin_midpoint, mean_predicted),
      bin_type = "equal_width"
    )
}

# Calibration bins - quantile-based (more informative for imbalanced data)
compute_calibration_bins_quantile <- function(truth, prob, n_bins = 10) {
  y <- as.numeric(truth == "1")

  # Use quantiles of predicted probabilities
  breaks <- quantile(prob, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE)
  breaks <- unique(breaks)  # handle ties
  actual_n_bins <- length(breaks) - 1

  if (actual_n_bins < 2) {
    # Fallback if not enough variation
    return(compute_calibration_bins_equal(truth, prob, n_bins) %>%
             mutate(bin_type = "quantile_fallback"))
  }

  bin_idx <- cut(prob, breaks = breaks, include.lowest = TRUE, labels = FALSE)

  bins <- tibble(
    bin = 1:actual_n_bins,
    bin_lower = head(breaks, -1),
    bin_upper = tail(breaks, -1),
    bin_midpoint = (bin_lower + bin_upper) / 2
  )

  bin_stats <- tibble(prob = prob, truth = y, bin = bin_idx) %>%
    group_by(bin) %>%
    summarise(
      n = n(),
      mean_predicted = mean(prob, na.rm = TRUE),
      mean_observed = mean(truth, na.rm = TRUE),
      .groups = "drop"
    )

  bins %>%
    left_join(bin_stats, by = "bin") %>%
    mutate(
      n = if_else(is.na(n), 0L, as.integer(n)),
      mean_predicted = if_else(is.na(mean_predicted), bin_midpoint, mean_predicted),
      bin_type = "quantile"
    )
}

# Calibration intercept and slope (via logistic regression)
compute_calibration_coeffs <- function(truth, prob) {
  y <- as.numeric(truth == "1")
  logit_prob <- qlogis(pmax(pmin(prob, 0.999), 0.001))

  fit <- tryCatch({
    glm(y ~ logit_prob, family = binomial(link = "logit"))
  }, error = function(e) NULL)

  if (is.null(fit)) {
    return(list(intercept = NA_real_, slope = NA_real_))
  }

  coefs <- coef(fit)
  list(intercept = coefs[1], slope = coefs[2])
}

# Platt scaling (train on val, apply to test)
fit_platt_scaling <- function(val_prob, val_truth) {
  y <- as.numeric(val_truth == "1")
  logit_prob <- qlogis(pmax(pmin(val_prob, 0.999), 0.001))

  fit <- tryCatch({
    glm(y ~ logit_prob, family = binomial(link = "logit"))
  }, error = function(e) NULL)

  fit
}

apply_platt_scaling <- function(prob, platt_model) {
  if (is.null(platt_model)) return(prob)
  logit_prob <- qlogis(pmax(pmin(prob, 0.999), 0.001))
  predict(platt_model, newdata = data.frame(logit_prob = logit_prob), type = "response")
}

# Isotonic calibration: fit on VAL, apply to any probs via interpolation 
fit_isotonic_calibration <- function(val_prob, val_truth, eps = 1e-15) {
  y <- as.numeric(val_truth == "1")
  p <- pmin(pmax(val_prob, eps), 1 - eps)

  iso <- tryCatch({
    stats::isoreg(p, y)
  }, error = function(e) NULL)

  if (is.null(iso)) return(NULL)

  # store fitted curve as (x, yhat) points
  # IMPORTANT: handle duplicate x values robustly for approx()
  x <- iso$x
  yhat <- iso$yf
  df <- data.frame(x = x, yhat = yhat) %>%
    dplyr::group_by(x) %>%
    dplyr::summarise(yhat = mean(yhat), .groups = "drop") %>%
    dplyr::arrange(x)

  list(x = df$x, yhat = df$yhat)
}

apply_isotonic_calibration <- function(prob, iso_model) {
  if (is.null(iso_model)) return(prob)
  stats::approx(
    x = iso_model$x,
    y = iso_model$yhat,
    xout = prob,
    rule = 2  # clamp to endpoints outside range
  )$y
}

# Beta calibration (Kull et al style):
# logit(y) = a*log(p) + b*log(1-p) + c
fit_beta_calibration <- function(val_prob, val_truth, eps = 1e-15) {
  y <- as.numeric(val_truth == "1")
  p <- pmin(pmax(val_prob, eps), 1 - eps)

  x1 <- log(p)
  x2 <- log(1 - p)

  fit <- tryCatch(
    glm(y ~ x1 + x2, family = binomial(link = "logit")),
    error = function(e) NULL
  )
  fit
}

apply_beta_calibration <- function(prob, beta_model, eps = 1e-15) {
  if (is.null(beta_model)) return(prob)
  p <- pmin(pmax(prob, eps), 1 - eps)
  x1 <- log(p)
  x2 <- log(1 - p)
  plogis(predict(beta_model, newdata = data.frame(x1 = x1, x2 = x2), type = "link"))
}

# Bundle calibration metrics for one (truth, prob) 
compute_calibration_suite <- function(truth, prob, n_bins = 10) {
  brier <- compute_brier(truth, prob)
  logloss <- compute_logloss(truth, prob)
  ece_equal <- compute_ece_equal(truth, prob, n_bins)

  ece_quant_res <- compute_ece_quantile(truth, prob, n_bins, verbose = FALSE)
  cal_coef <- compute_calibration_coeffs(truth, prob)

  list(
    brier = brier,
    logloss = logloss,
    ece_equal = ece_equal,
    ece_quant = ece_quant_res$ece,
    ece_quant_fallback = ece_quant_res$fallback,
    cal_intercept = cal_coef$intercept,
    cal_slope = cal_coef$slope
  )
}

# Threshold sweep with alerts count
compute_threshold_sweep <- function(truth, prob, dates, thresholds) {
  n_days <- n_distinct(dates)

  results <- map_dfr(thresholds, function(t) {
    pred <- factor(if_else(prob >= t, "1", "0"), levels = c("1", "0"))

    tp <- sum(truth == "1" & pred == "1", na.rm = TRUE)
    fp <- sum(truth == "0" & pred == "1", na.rm = TRUE)
    tn <- sum(truth == "0" & pred == "0", na.rm = TRUE)
    fn <- sum(truth == "1" & pred == "0", na.rm = TRUE)

    n_alerts <- tp + fp
    alerts_per_day <- n_alerts / n_days

    precision <- if (n_alerts > 0) tp / n_alerts else NA_real_
    recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
    f1 <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
      2 * precision * recall / (precision + recall)
    } else NA_real_

    tibble(
      threshold = t,
      tp = tp,
      fp = fp,
      tn = tn,
      fn = fn,
      n_alerts = n_alerts,
      alerts_per_day = alerts_per_day,
      precision = precision,
      recall = recall,
      specificity = if ((tn + fp) > 0) tn / (tn + fp) else NA_real_,
      f1 = f1,
      accuracy = (tp + tn) / (tp + fp + tn + fn),
      kappa = kappa_from_counts(tp, fp, tn, fn)
    )
  })

  results
}

# Decision curve analysis (net benefit)
compute_decision_curve <- function(truth, prob, thresholds) {
  ok <- !is.na(truth) & !is.na(prob)
  y <- as.numeric(truth[ok] == "1")
  prob <- prob[ok]
  n <- length(y)
  if (n == 0) {
    return(tibble(threshold = thresholds, net_benefit = NA_real_, treat_all = NA_real_, treat_none = 0))
  }
  prevalence <- mean(y)

  results <- map_dfr(thresholds, function(t) {
    pred <- prob >= t
    tp <- sum(y == 1 & pred, na.rm = TRUE)
    fp <- sum(y == 0 & pred, na.rm = TRUE)

    odds_threshold <- t / (1 - t)
    net_benefit <- (tp / n) - (fp / n) * odds_threshold
    treat_all_net_benefit <- prevalence - (1 - prevalence) * odds_threshold

    tibble(
      threshold = t,
      net_benefit = net_benefit,
      treat_all = treat_all_net_benefit,
      treat_none = 0
    )
  })

  results
}

# Compute alerts per day at a given threshold
compute_alerts_per_day <- function(prob, dates, threshold) {
  tibble(date = dates, prob = prob) %>%
    group_by(date) %>%
    summarise(
      alerts = sum(prob >= threshold, na.rm = TRUE),
      n_municipalities = sum(!is.na(prob)),
      .groups = "drop"
    )
}

# LOAD DATA

message("\n", strrep("=", 60))
message("STEP 1: Loading data")
message(strrep("=", 60))

if (!file.exists(features_path)) {
  stop("Features file not found: ", features_path, "\nRun 04_feature_engineering_improved.R first.")
}
if (!file.exists(splits_fixed_path)) {
  stop("Splits file not found: ", splits_fixed_path, "\nRun 03_training_prep.R first.")
}

features <- readRDS(features_path)
message("Loaded features: ", format(nrow(features), big.mark = ","), " rows, ", ncol(features), " columns")

splits_fixed <- readRDS(splits_fixed_path)

train_start <- as.Date(splits_fixed$train_range[1])
train_end   <- as.Date(splits_fixed$train_range[2])
val_start   <- as.Date(splits_fixed$val_range[1])
val_end     <- as.Date(splits_fixed$val_range[2])
test_start  <- as.Date(splits_fixed$test_range[1])
test_end    <- as.Date(splits_fixed$test_range[2])

message("Split ranges:")
message("  Train: ", train_start, " to ", train_end)
message("  Val:   ", val_start, " to ", val_end)
message("  Test:  ", test_start, " to ", test_end)

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

message("\nSplit sizes:")
message("  Train: ", format(nrow(train_data), big.mark = ","), " rows")
message("  Val:   ", format(nrow(val_data), big.mark = ","), " rows")
message("  Test:  ", format(nrow(test_data), big.mark = ","), " rows")

# Keep dates for alerts-per-day calculation
val_dates  <- val_data$date
test_dates <- test_data$date

# FEATURE SELECTION

message("\n", strrep("=", 60))
message("STEP 2: Feature selection")
message(strrep("=", 60))

all_cols <- names(features)
feature_cols <- setdiff(all_cols, exclude_cols)

# Apply task mode filter
feature_cols <- apply_task_mode_filters(feature_cols, task_mode = task_mode,
                                         strict = TRUE, verbose = TRUE)

# Remove all-NA columns (based on training only)
feature_cols <- intersect(feature_cols, names(train_data))
na_rate_train <- sapply(train_data[, feature_cols, drop = FALSE], function(x) mean(is.na(x)))
all_na_cols <- names(na_rate_train)[na_rate_train == 1]
if (length(all_na_cols) > 0) {
  message("Removing ", length(all_na_cols), " all-NA columns")
  feature_cols <- setdiff(feature_cols, all_na_cols)
}

# Remove character/factor columns
char_cols <- feature_cols[sapply(train_data[, feature_cols, drop = FALSE], is.character)]
factor_cols <- feature_cols[sapply(train_data[, feature_cols, drop = FALSE], is.factor)]
if (length(char_cols) > 0) {
  message("Removing ", length(char_cols), " character columns")
  feature_cols <- setdiff(feature_cols, char_cols)
}
if (length(factor_cols) > 0) {
  message("Removing ", length(factor_cols), " factor columns")
  feature_cols <- setdiff(feature_cols, factor_cols)
}

message("Final feature count: ", length(feature_cols))

# PREPARE DATA

message("\n", strrep("=", 60))
message("STEP 3: Preparing modeling data")
message(strrep("=", 60))

model_cols <- c("outage_3h_or_more", feature_cols)

prep_model_df <- function(df) {
  df %>%
    select(all_of(model_cols)) %>%
    mutate(
      outage_3h_or_more = factor(as.character(outage_3h_or_more), levels = c("1", "0"))
    ) %>%
    mutate(across(where(is.logical), as.integer))
}

train_model <- prep_model_df(train_data)
val_model   <- prep_model_df(val_data)
test_model  <- prep_model_df(test_data)

n_pos <- sum(train_model$outage_3h_or_more == "1", na.rm = TRUE)
n_neg <- sum(train_model$outage_3h_or_more == "0", na.rm = TRUE)
train_prevalence <- n_pos / (n_pos + n_neg)
message("Training prevalence: ", sprintf("%.4f%%", 100 * train_prevalence))

# TRAIN MODEL

message("\n", strrep("=", 60))
message("STEP 4: Training XGBoost classifier")
message(strrep("=", 60))

# Guard against division by zero
scale_pos_weight <- n_neg / max(1, n_pos)
message("scale_pos_weight: ", sprintf("%.2f", scale_pos_weight))

n_features <- length(feature_cols)
mtry_count <- max(1, floor(xgb_params$mtry * n_features))

model_recipe <- recipe(outage_3h_or_more ~ ., data = train_model) %>%
  step_zv(all_predictors())

xgb_spec <- boost_tree(
  trees          = xgb_params$trees,
  tree_depth     = xgb_params$tree_depth,
  min_n          = xgb_params$min_n,
  loss_reduction = xgb_params$loss_reduction,
  sample_size    = xgb_params$sample_size,
  mtry           = mtry_count,
  learn_rate     = xgb_params$learn_rate
) %>%
  set_engine(
    "xgboost",
    scale_pos_weight = scale_pos_weight,
    nthread          = get_nthread(),
    verbosity        = 0
  ) %>%
  set_mode("classification")

xgb_workflow <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(xgb_spec)

train_start_time <- Sys.time()
xgb_fit <- fit(xgb_workflow, data = train_model)
train_end_time <- Sys.time()
train_duration <- difftime(train_end_time, train_start_time, units = "mins")
message("Training completed in ", sprintf("%.2f", train_duration), " minutes")

# PREDICTIONS

message("\n", strrep("=", 60))
message("STEP 5: Generating predictions")
message(strrep("=", 60))

val_prob <- predict(xgb_fit, new_data = val_model, type = "prob")
test_prob <- predict(xgb_fit, new_data = test_model, type = "prob")

# Include dates for alerts-per-day calculation
val_predictions <- bind_cols(
  val_prob,
  val_model %>% select(outage_3h_or_more)
) %>%
  rename(truth = outage_3h_or_more) %>%
  mutate(prob = .pred_1, date = val_dates)

test_predictions <- bind_cols(
  test_prob,
  test_model %>% select(outage_3h_or_more)
) %>%
  rename(truth = outage_3h_or_more) %>%
  mutate(prob = .pred_1, date = test_dates)

message("Validation predictions: ", nrow(val_predictions), " (", n_distinct(val_dates), " days)")
message("Test predictions: ", nrow(test_predictions), " (", n_distinct(test_dates), " days)")

# STRICT CALIBRATION-THRESHOLD SEPARATION (optional)

# When STRICT_CALIBRATION_THRESHOLD_SEPARATION is TRUE:
#   - Split validation into two time-ordered parts
#   - First part: fit calibrators (Platt/Beta/Isotonic)
#   - Second part: select thresholds for calibrated methods
# This reduces optimism from using the same data for both calibration and threshold selection

if (STRICT_CALIBRATION_THRESHOLD_SEPARATION) {
  message("\n*** STRICT MODE: Splitting validation for calibration vs threshold selection ***")

  # Get unique dates sorted
  val_unique_dates <- sort(unique(val_predictions$date))
  n_val_dates <- length(val_unique_dates)

  # STRICT MODE GUARDRAILS 
  # Minimum requirements for valid strict mode operation
  MIN_STRICT_VAL_DATES <- 4  # minimum validation dates to allow split
  MIN_STRICT_SPLIT_DATES <- 2  # minimum dates per split subset
  MIN_STRICT_EVENTS_PER_SPLIT <- 10  # minimum positive events per split (warning only)

  if (n_val_dates < MIN_STRICT_VAL_DATES) {
    stop("STRICT MODE GUARDRAIL: Validation has only ", n_val_dates, " unique dates. ",
         "Minimum required: ", MIN_STRICT_VAL_DATES, ". ",
         "Use standard mode (--strict_mode standard) or provide more validation data.")
  }

  # Time-ordered split (first STRICT_SPLIT_RATIO for calibration fitting)
  n_calib_dates <- floor(n_val_dates * STRICT_SPLIT_RATIO)

  # Ensure both subsets have minimum dates
  if (n_calib_dates < MIN_STRICT_SPLIT_DATES) {
    warning("STRICT MODE: Calibration-fit subset would have only ", n_calib_dates,
            " dates. Adjusting to minimum of ", MIN_STRICT_SPLIT_DATES, ".")
    n_calib_dates <- MIN_STRICT_SPLIT_DATES
  }
  if ((n_val_dates - n_calib_dates) < MIN_STRICT_SPLIT_DATES) {
    warning("STRICT MODE: Threshold-select subset would have only ", (n_val_dates - n_calib_dates),
            " dates. Adjusting calibration split to allow minimum threshold-select dates.")
    n_calib_dates <- n_val_dates - MIN_STRICT_SPLIT_DATES
  }

  calib_fit_dates <- val_unique_dates[1:n_calib_dates]
  threshold_select_dates <- val_unique_dates[(n_calib_dates + 1):n_val_dates]

  # Create subsets
  val_predictions_calib_fit <- val_predictions %>%
    filter(date %in% calib_fit_dates)
  val_predictions_threshold_select <- val_predictions %>%
    filter(date %in% threshold_select_dates)

  # Check for non-empty subsets (should not happen after guardrails, but be defensive)
  if (nrow(val_predictions_calib_fit) == 0) {
    stop("STRICT MODE GUARDRAIL: Calibration-fit subset is empty after filtering. Check data integrity.")
  }
  if (nrow(val_predictions_threshold_select) == 0) {
    stop("STRICT MODE GUARDRAIL: Threshold-select subset is empty after filtering. Check data integrity.")
  }

  # Warn about low event counts (does not stop execution)
  calib_fit_events <- sum(val_predictions_calib_fit$truth == "1")
  threshold_select_events <- sum(val_predictions_threshold_select$truth == "1")
  if (calib_fit_events < MIN_STRICT_EVENTS_PER_SPLIT) {
    warning("STRICT MODE: Calibration-fit subset has only ", calib_fit_events,
            " positive events (recommended minimum: ", MIN_STRICT_EVENTS_PER_SPLIT,
            "). Calibration estimates may be unstable.")
  }
  if (threshold_select_events < MIN_STRICT_EVENTS_PER_SPLIT) {
    warning("STRICT MODE: Threshold-select subset has only ", threshold_select_events,
            " positive events (recommended minimum: ", MIN_STRICT_EVENTS_PER_SPLIT,
            "). Threshold selection may be unstable.")
  }

  message("  Calibration-fit split: ", nrow(val_predictions_calib_fit), " rows (",
          length(calib_fit_dates), " days: ",
          min(calib_fit_dates), " to ", max(calib_fit_dates), ")")
  message("  Threshold-select split: ", nrow(val_predictions_threshold_select), " rows (",
          length(threshold_select_dates), " days: ",
          min(threshold_select_dates), " to ", max(threshold_select_dates), ")")

  # Store split info for metadata
  strict_split_info <- list(
    enabled = TRUE,
    ratio = STRICT_SPLIT_RATIO,
    calib_fit_dates_range = c(as.character(min(calib_fit_dates)), as.character(max(calib_fit_dates))),
    calib_fit_n_days = length(calib_fit_dates),
    calib_fit_n_rows = nrow(val_predictions_calib_fit),
    threshold_select_dates_range = c(as.character(min(threshold_select_dates)), as.character(max(threshold_select_dates))),
    threshold_select_n_days = length(threshold_select_dates),
    threshold_select_n_rows = nrow(val_predictions_threshold_select)
  )
} else {
  message("\n*** STANDARD MODE: Using full validation for both calibration and threshold selection ***")

  # Use full validation for both (original behavior)
  val_predictions_calib_fit <- val_predictions
  val_predictions_threshold_select <- val_predictions

  strict_split_info <- list(
    enabled = FALSE,
    ratio = NA_real_,
    note = "Full validation used for both calibration fitting and threshold selection"
  )
}

# Print probability quantiles to confirm threshold grid coverage
prob_quantiles <- c(0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
val_pq <- quantile(val_predictions$prob, probs = prob_quantiles, na.rm = TRUE)
test_pq <- quantile(test_predictions$prob, probs = prob_quantiles, na.rm = TRUE)

message("\nPredicted probability quantiles:")
message("  VAL:  p50=", sprintf("%.4f", val_pq[1]),
        ", p90=", sprintf("%.4f", val_pq[2]),
        ", p95=", sprintf("%.4f", val_pq[3]),
        ", p99=", sprintf("%.4f", val_pq[4]),
        ", p99.9=", sprintf("%.4f", val_pq[5]),
        ", max=", sprintf("%.4f", val_pq[6]))
message("  TEST: p50=", sprintf("%.4f", test_pq[1]),
        ", p90=", sprintf("%.4f", test_pq[2]),
        ", p95=", sprintf("%.4f", test_pq[3]),
        ", p99=", sprintf("%.4f", test_pq[4]),
        ", p99.9=", sprintf("%.4f", test_pq[5]),
        ", max=", sprintf("%.4f", test_pq[6]))

# ROC CURVE

message("\n", strrep("=", 60))
message("STEP 6: Computing ROC curve")
message(strrep("=", 60))

roc_val <- roc_curve(val_predictions, truth, prob, event_level = "first")
roc_test <- roc_curve(test_predictions, truth, prob, event_level = "first")

roc_all <- bind_rows(
  roc_val %>% mutate(split = "validation"),
  roc_test %>% mutate(split = "test")
)

write_csv(roc_all, file.path(OUTPUT_DIR, "roc_curve.csv"))
message("Saved: roc_curve.csv")

# ROC AUC
roc_auc_val <- roc_auc_vec(val_predictions$truth, val_predictions$prob, event_level = "first")
roc_auc_test <- roc_auc_vec(test_predictions$truth, test_predictions$prob, event_level = "first")
message("ROC AUC - Val: ", sprintf("%.4f", roc_auc_val), ", Test: ", sprintf("%.4f", roc_auc_test))

# PR CURVE

message("\n", strrep("=", 60))
message("STEP 7: Computing PR curve")
message(strrep("=", 60))

pr_val <- pr_curve(val_predictions, truth, prob, event_level = "first")
pr_test <- pr_curve(test_predictions, truth, prob, event_level = "first")

pr_all <- bind_rows(
  pr_val %>% mutate(split = "validation"),
  pr_test %>% mutate(split = "test")
)

write_csv(pr_all, file.path(OUTPUT_DIR, "pr_curve.csv"))
message("Saved: pr_curve.csv")

# PR AUC
pr_auc_val <- pr_auc_vec(val_predictions$truth, val_predictions$prob, event_level = "first")
pr_auc_test <- pr_auc_vec(test_predictions$truth, test_predictions$prob, event_level = "first")
message("PR AUC - Val: ", sprintf("%.4f", pr_auc_val), ", Test: ", sprintf("%.4f", pr_auc_test))

# CALIBRATION DIAGNOSTICS

message("\n", strrep("=", 60))
message("STEP 8: Computing calibration diagnostics")
message(strrep("=", 60))

# Equal-width calibration bins
cal_bins_equal_val <- compute_calibration_bins_equal(val_predictions$truth, val_predictions$prob, N_CALIBRATION_BINS) %>%
  mutate(split = "validation")
cal_bins_equal_test <- compute_calibration_bins_equal(test_predictions$truth, test_predictions$prob, N_CALIBRATION_BINS) %>%
  mutate(split = "test")

cal_bins_equal_all <- bind_rows(cal_bins_equal_val, cal_bins_equal_test)
write_csv(cal_bins_equal_all, file.path(OUTPUT_DIR, "calibration_bins_equal.csv"))
message("Saved: calibration_bins_equal.csv")

# Quantile-based calibration bins (more informative for imbalanced data)
cal_bins_quant_val <- compute_calibration_bins_quantile(val_predictions$truth, val_predictions$prob, N_CALIBRATION_BINS) %>%
  mutate(split = "validation")
cal_bins_quant_test <- compute_calibration_bins_quantile(test_predictions$truth, test_predictions$prob, N_CALIBRATION_BINS) %>%
  mutate(split = "test")

cal_bins_quant_all <- bind_rows(cal_bins_quant_val, cal_bins_quant_test)
write_csv(cal_bins_quant_all, file.path(OUTPUT_DIR, "calibration_bins_quantile.csv"))
message("Saved: calibration_bins_quantile.csv")

# Brier score
brier_val <- compute_brier(val_predictions$truth, val_predictions$prob)
brier_test <- compute_brier(test_predictions$truth, test_predictions$prob)
message("Brier score - Val: ", sprintf("%.6f", brier_val), ", Test: ", sprintf("%.6f", brier_test))

# Log-loss (cross entropy)
logloss_val <- compute_logloss(val_predictions$truth, val_predictions$prob)
logloss_test <- compute_logloss(test_predictions$truth, test_predictions$prob)
message("Log-loss - Val: ", sprintf("%.6f", logloss_val), ", Test: ", sprintf("%.6f", logloss_test))

# ECE - equal-width bins
ece_equal_val <- compute_ece_equal(val_predictions$truth, val_predictions$prob, N_CALIBRATION_BINS)
ece_equal_test <- compute_ece_equal(test_predictions$truth, test_predictions$prob, N_CALIBRATION_BINS)
message("ECE (equal-width) - Val: ", sprintf("%.6f", ece_equal_val), ", Test: ", sprintf("%.6f", ece_equal_test))

# ECE - quantile bins (more informative for imbalanced data)
ece_quant_val_result <- compute_ece_quantile(val_predictions$truth, val_predictions$prob, N_CALIBRATION_BINS, verbose = TRUE)
ece_quant_test_result <- compute_ece_quantile(test_predictions$truth, test_predictions$prob, N_CALIBRATION_BINS, verbose = TRUE)
ece_quant_val <- ece_quant_val_result$ece
ece_quant_test <- ece_quant_test_result$ece
message("ECE (quantile) - Val: ", sprintf("%.6f", ece_quant_val),
        if (ece_quant_val_result$fallback) " [fallback/reduced bins]" else "",
        ", Test: ", sprintf("%.6f", ece_quant_test),
        if (ece_quant_test_result$fallback) " [fallback/reduced bins]" else "")

# Calibration coefficients (intercept, slope)
cal_coef_val <- compute_calibration_coeffs(val_predictions$truth, val_predictions$prob)
cal_coef_test <- compute_calibration_coeffs(test_predictions$truth, test_predictions$prob)
message("Calibration intercept - Val: ", sprintf("%.4f", cal_coef_val$intercept),
        ", Test: ", sprintf("%.4f", cal_coef_test$intercept))
message("Calibration slope - Val: ", sprintf("%.4f", cal_coef_val$slope),
        ", Test: ", sprintf("%.4f", cal_coef_test$slope))

# Save calibration metrics (with logloss)
calibration_metrics <- tibble(
  split = c("validation", "test"),
  brier_score = c(brier_val, brier_test),
  logloss = c(logloss_val, logloss_test),
  ece_equal = c(ece_equal_val, ece_equal_test),
  ece_quantile = c(ece_quant_val, ece_quant_test),
  calibration_intercept = c(cal_coef_val$intercept, cal_coef_test$intercept),
  calibration_slope = c(cal_coef_val$slope, cal_coef_test$slope)
)
write_csv(calibration_metrics, file.path(OUTPUT_DIR, "calibration_metrics.csv"))
message("Saved: calibration_metrics.csv")

# PLATT SCALING (OPTIONAL)

# Initialise Platt-scaled metrics (will be populated if PLATT_SCALING is TRUE)
logloss_test_platt <- NA_real_
brier_test_platt <- NA_real_
ece_equal_test_platt <- NA_real_
ece_quant_test_platt <- NA_real_
cal_int_test_platt <- NA_real_
cal_slope_test_platt <- NA_real_

platt_model <- NULL
platt_results <- NULL
test_prob_platt <- NULL
cal_bins_platt_equal <- NULL
cal_bins_platt_quant <- NULL

if (PLATT_SCALING) {
  message("\n", strrep("=", 60))
  message("STEP 9: Platt scaling")
  message(strrep("=", 60))

  # Fit Platt scaling on validation set (or calibration-fit subset in strict mode)
  platt_model <- fit_platt_scaling(val_predictions_calib_fit$prob, val_predictions_calib_fit$truth)

  if (!is.null(platt_model)) {
    # Apply to test set
    test_prob_platt <- apply_platt_scaling(test_predictions$prob, platt_model)

    # Compute calibration suite
    suite_platt <- compute_calibration_suite(test_predictions$truth, test_prob_platt, N_CALIBRATION_BINS)
    logloss_test_platt <- suite_platt$logloss
    brier_test_platt <- suite_platt$brier
    ece_equal_test_platt <- suite_platt$ece_equal
    ece_quant_test_platt <- suite_platt$ece_quant
    cal_int_test_platt <- suite_platt$cal_intercept
    cal_slope_test_platt <- suite_platt$cal_slope

    message("Platt-scaled test metrics:")
    message("  Log-loss: ", sprintf("%.6f", logloss_test_platt), " (raw ", sprintf("%.6f", logloss_test), ")")
    message("  Brier score: ", sprintf("%.6f", brier_test_platt), " (raw ", sprintf("%.6f", brier_test), ")")
    message("  ECE (equal): ", sprintf("%.6f", ece_equal_test_platt), " (raw ", sprintf("%.6f", ece_equal_test), ")")
    message("  ECE (quantile): ", sprintf("%.6f", ece_quant_test_platt),
            if (suite_platt$ece_quant_fallback) " [fallback]" else "",
            " (raw ", sprintf("%.6f", ece_quant_test), ")")
    message("  Calibration intercept: ", sprintf("%.4f", cal_int_test_platt),
            " (raw ", sprintf("%.4f", cal_coef_test$intercept), ")")
    message("  Calibration slope: ", sprintf("%.4f", cal_slope_test_platt),
            " (raw ", sprintf("%.4f", cal_coef_test$slope), ")")

    # Calibration bins after Platt scaling (both types)
    cal_bins_platt_equal <- compute_calibration_bins_equal(test_predictions$truth, test_prob_platt, N_CALIBRATION_BINS) %>%
      mutate(split = "test_platt_scaled")
    cal_bins_platt_quant <- compute_calibration_bins_quantile(test_predictions$truth, test_prob_platt, N_CALIBRATION_BINS) %>%
      mutate(split = "test_platt_scaled")

    platt_results <- tibble(
      split = "test_platt_scaled",
      brier_score = brier_test_platt,
      logloss = logloss_test_platt,
      ece_equal = ece_equal_test_platt,
      ece_quantile = ece_quant_test_platt,
      calibration_intercept = cal_int_test_platt,
      calibration_slope = cal_slope_test_platt,
      platt_intercept = coef(platt_model)[1],
      platt_slope = coef(platt_model)[2]
    )

    write_csv(platt_results, file.path(OUTPUT_DIR, "platt_calibration.csv"))
    write_csv(cal_bins_platt_equal, file.path(OUTPUT_DIR, "calibration_bins_platt_equal.csv"))
    write_csv(cal_bins_platt_quant, file.path(OUTPUT_DIR, "calibration_bins_platt_quantile.csv"))
    message("Saved: platt_calibration.csv, calibration_bins_platt_*.csv")
  } else {
    message("Platt scaling failed - skipping")
  }
}

# ISOTONIC + BETA CALIBRATION (fit on VAL, eval on TEST)

# Initialise calibrated metrics
logloss_test_isotonic <- NA_real_
brier_test_isotonic <- NA_real_
ece_equal_test_isotonic <- NA_real_
ece_quant_test_isotonic <- NA_real_
cal_int_test_isotonic <- NA_real_
cal_slope_test_isotonic <- NA_real_

logloss_test_beta <- NA_real_
brier_test_beta <- NA_real_
ece_equal_test_beta <- NA_real_
ece_quant_test_beta <- NA_real_
cal_int_test_beta <- NA_real_
cal_slope_test_beta <- NA_real_

test_prob_iso <- NULL
test_prob_beta <- NULL
cal_bins_test_iso_equal <- NULL
cal_bins_test_iso_quant <- NULL
cal_bins_test_beta_equal <- NULL
cal_bins_test_beta_quant <- NULL

message("\n", strrep("=", 60))
message("STEP 9b: Isotonic + Beta calibration")
message(strrep("=", 60))

# Isotonic 
# Fit on calibration-fit subset (or full validation if strict mode OFF)
iso_model <- fit_isotonic_calibration(val_predictions_calib_fit$prob, val_predictions_calib_fit$truth)

if (!is.null(iso_model)) {
  test_prob_iso <- apply_isotonic_calibration(test_predictions$prob, iso_model)

  suite_iso <- compute_calibration_suite(test_predictions$truth, test_prob_iso, N_CALIBRATION_BINS)
  logloss_test_isotonic <- suite_iso$logloss
  brier_test_isotonic <- suite_iso$brier
  ece_equal_test_isotonic <- suite_iso$ece_equal
  ece_quant_test_isotonic <- suite_iso$ece_quant
  cal_int_test_isotonic <- suite_iso$cal_intercept
  cal_slope_test_isotonic <- suite_iso$cal_slope

  message("Isotonic-calibrated test metrics:")
  message("  Log-loss: ", sprintf("%.6f", logloss_test_isotonic), " (raw ", sprintf("%.6f", logloss_test), ")")
  message("  Brier:    ", sprintf("%.6f", brier_test_isotonic), " (raw ", sprintf("%.6f", brier_test), ")")
  message("  ECE eq:   ", sprintf("%.6f", ece_equal_test_isotonic), " (raw ", sprintf("%.6f", ece_equal_test), ")")
  message("  ECE q:    ", sprintf("%.6f", ece_quant_test_isotonic), " (raw ", sprintf("%.6f", ece_quant_test), ")")
  message("  Cal int:  ", sprintf("%.4f", cal_int_test_isotonic), " (raw ", sprintf("%.4f", cal_coef_test$intercept), ")")
  message("  Cal slope:", sprintf("%.4f", cal_slope_test_isotonic), " (raw ", sprintf("%.4f", cal_coef_test$slope), ")")

  # Calibration bins for isotonic
  cal_bins_test_iso_equal <- compute_calibration_bins_equal(test_predictions$truth, test_prob_iso, N_CALIBRATION_BINS) %>%
    mutate(split = "test", method = "isotonic")
  cal_bins_test_iso_quant <- compute_calibration_bins_quantile(test_predictions$truth, test_prob_iso, N_CALIBRATION_BINS) %>%
    mutate(split = "test", method = "isotonic")
} else {
  message("Isotonic calibration failed - skipping")
}

# Beta 
# Fit on calibration-fit subset (or full validation if strict mode OFF)
beta_model <- fit_beta_calibration(val_predictions_calib_fit$prob, val_predictions_calib_fit$truth)

if (!is.null(beta_model)) {
  test_prob_beta <- apply_beta_calibration(test_predictions$prob, beta_model)

  suite_beta <- compute_calibration_suite(test_predictions$truth, test_prob_beta, N_CALIBRATION_BINS)
  logloss_test_beta <- suite_beta$logloss
  brier_test_beta <- suite_beta$brier
  ece_equal_test_beta <- suite_beta$ece_equal
  ece_quant_test_beta <- suite_beta$ece_quant
  cal_int_test_beta <- suite_beta$cal_intercept
  cal_slope_test_beta <- suite_beta$cal_slope

  message("Beta-calibrated test metrics:")
  message("  Log-loss: ", sprintf("%.6f", logloss_test_beta), " (raw ", sprintf("%.6f", logloss_test), ")")
  message("  Brier:    ", sprintf("%.6f", brier_test_beta), " (raw ", sprintf("%.6f", brier_test), ")")
  message("  ECE eq:   ", sprintf("%.6f", ece_equal_test_beta), " (raw ", sprintf("%.6f", ece_equal_test), ")")
  message("  ECE q:    ", sprintf("%.6f", ece_quant_test_beta), " (raw ", sprintf("%.6f", ece_quant_test), ")")
  message("  Cal int:  ", sprintf("%.4f", cal_int_test_beta), " (raw ", sprintf("%.4f", cal_coef_test$intercept), ")")
  message("  Cal slope:", sprintf("%.4f", cal_slope_test_beta), " (raw ", sprintf("%.4f", cal_coef_test$slope), ")")

  # Calibration bins for beta
  cal_bins_test_beta_equal <- compute_calibration_bins_equal(test_predictions$truth, test_prob_beta, N_CALIBRATION_BINS) %>%
    mutate(split = "test", method = "beta")
  cal_bins_test_beta_quant <- compute_calibration_bins_quantile(test_predictions$truth, test_prob_beta, N_CALIBRATION_BINS) %>%
    mutate(split = "test", method = "beta")
} else {
  message("Beta calibration failed - skipping")
}

# COMBINED CALIBRATION BINS (all methods)

message("\n", strrep("=", 60))
message("STEP 9c: Saving combined calibration bins")
message(strrep("=", 60))

# Build list of equal-width bins
bins_equal_list <- list(
  cal_bins_equal_test %>% mutate(method = "raw")
)
if (!is.null(cal_bins_platt_equal)) {
  bins_equal_list <- append(bins_equal_list, list(cal_bins_platt_equal %>% mutate(method = "platt") %>% mutate(split = "test")))
}
if (!is.null(cal_bins_test_iso_equal)) {
  bins_equal_list <- append(bins_equal_list, list(cal_bins_test_iso_equal))
}
if (!is.null(cal_bins_test_beta_equal)) {
  bins_equal_list <- append(bins_equal_list, list(cal_bins_test_beta_equal))
}

# Build list of quantile bins
bins_quant_list <- list(
  cal_bins_quant_test %>% mutate(method = "raw")
)
if (!is.null(cal_bins_platt_quant)) {
  bins_quant_list <- append(bins_quant_list, list(cal_bins_platt_quant %>% mutate(method = "platt") %>% mutate(split = "test")))
}
if (!is.null(cal_bins_test_iso_quant)) {
  bins_quant_list <- append(bins_quant_list, list(cal_bins_test_iso_quant))
}
if (!is.null(cal_bins_test_beta_quant)) {
  bins_quant_list <- append(bins_quant_list, list(cal_bins_test_beta_quant))
}

cal_bins_all_methods_equal <- dplyr::bind_rows(bins_equal_list)
cal_bins_all_methods_quant <- dplyr::bind_rows(bins_quant_list)

write_csv(cal_bins_all_methods_equal, file.path(OUTPUT_DIR, "calibration_bins_equal_all_methods.csv"))
write_csv(cal_bins_all_methods_quant, file.path(OUTPUT_DIR, "calibration_bins_quantile_all_methods.csv"))
message("Saved: calibration_bins_equal_all_methods.csv")
message("Saved: calibration_bins_quantile_all_methods.csv")

# CALIBRATION COMPARISON TABLE (TEST only)

message("\n", strrep("=", 60))
message("STEP 9d: Calibration comparison table")
message(strrep("=", 60))

calibration_comparison <- tibble(
  split = "test",
  method = c("raw", "platt", "isotonic", "beta"),
  logloss = c(
    logloss_test,
    logloss_test_platt,
    logloss_test_isotonic,
    logloss_test_beta
  ),
  brier = c(
    brier_test,
    brier_test_platt,
    brier_test_isotonic,
    brier_test_beta
  ),
  ece_equal = c(
    ece_equal_test,
    ece_equal_test_platt,
    ece_equal_test_isotonic,
    ece_equal_test_beta
  ),
  ece_quantile = c(
    ece_quant_test,
    ece_quant_test_platt,
    ece_quant_test_isotonic,
    ece_quant_test_beta
  ),
  cal_intercept = c(
    cal_coef_test$intercept,
    cal_int_test_platt,
    cal_int_test_isotonic,
    cal_int_test_beta
  ),
  cal_slope = c(
    cal_coef_test$slope,
    cal_slope_test_platt,
    cal_slope_test_isotonic,
    cal_slope_test_beta
  )
)

write_csv(calibration_comparison, file.path(OUTPUT_DIR, "calibration_comparison_test.csv"))
message("Saved: calibration_comparison_test.csv")

# Print comparison table
message("\nCalibration comparison (TEST set):")
message("  Method     | Log-loss  | Brier     | ECE-eq    | ECE-q     | Intercept | Slope")
message("  -----------+-----------+-----------+-----------+-----------+-----------+-------")
for (i in 1:nrow(calibration_comparison)) {
  row <- calibration_comparison[i, ]
  message(sprintf("  %-10s | %9.6f | %9.6f | %9.6f | %9.6f | %9.4f | %6.4f",
                  row$method,
                  ifelse(is.na(row$logloss), NA, row$logloss),
                  ifelse(is.na(row$brier), NA, row$brier),
                  ifelse(is.na(row$ece_equal), NA, row$ece_equal),
                  ifelse(is.na(row$ece_quantile), NA, row$ece_quantile),
                  ifelse(is.na(row$cal_intercept), NA, row$cal_intercept),
                  ifelse(is.na(row$cal_slope), NA, row$cal_slope)))
}

# PREPARE CALIBRATED PROBABILITIES BY METHOD

# NOTE: For threshold selection on calibrated probabilities, we apply calibrators
# (fitted on validation) to validation probs too. This is technically "in-sample"
# for the calibrator, but is standard practice for threshold selection comparison
# when the goal is to find optimal thresholds for each calibration method.

message("\n", strrep("=", 60))
message("STEP 9e: Preparing calibrated probabilities by method")
message(strrep("=", 60))

# Build named lists of probabilities for each method
# Only include methods that were successfully fitted
#
# In STRICT mode, we have two validation subsets:
#   - val_predictions (full): used for diagnostics/sweeps
#   - val_predictions_threshold_select: used for threshold selection (calibrated methods only)
# Calibrators were fitted on val_predictions_calib_fit (first half of validation)

val_probs_by_method <- list(raw = val_predictions$prob)
test_probs_by_method <- list(raw = test_predictions$prob)

# Also create threshold-selection subset probabilities (for operating thresholds in strict mode)
val_threshold_select_probs_by_method <- list(raw = val_predictions_threshold_select$prob)

# Platt
if (!is.null(platt_model)) {
  val_probs_by_method$platt <- apply_platt_scaling(val_predictions$prob, platt_model)
  val_threshold_select_probs_by_method$platt <- apply_platt_scaling(val_predictions_threshold_select$prob, platt_model)
  test_probs_by_method$platt <- test_prob_platt
  message("  Added: platt")
}

# Beta
if (!is.null(beta_model)) {
  val_probs_by_method$beta <- apply_beta_calibration(val_predictions$prob, beta_model)
  val_threshold_select_probs_by_method$beta <- apply_beta_calibration(val_predictions_threshold_select$prob, beta_model)
  test_probs_by_method$beta <- test_prob_beta
  message("  Added: beta")
}

# Isotonic
if (!is.null(iso_model)) {
  val_probs_by_method$isotonic <- apply_isotonic_calibration(val_predictions$prob, iso_model)
  val_threshold_select_probs_by_method$isotonic <- apply_isotonic_calibration(val_predictions_threshold_select$prob, iso_model)
  test_probs_by_method$isotonic <- test_prob_iso
  message("  Added: isotonic")
}

methods_available <- names(val_probs_by_method)
message("Methods available for threshold analysis: ", paste(methods_available, collapse = ", "))

if (STRICT_CALIBRATION_THRESHOLD_SEPARATION) {
  message("  [STRICT MODE] Calibrators fitted on calib-fit subset, threshold selection will use threshold-select subset")
}

# THRESHOLD SWEEP (with alerts per day)

message("\n", strrep("=", 60))
message("STEP 10: Threshold sweep")
message(strrep("=", 60))

message("Threshold grid: ", length(THRESHOLD_GRID), " values from ",
        sprintf("%.4f", min(THRESHOLD_GRID)), " to ", sprintf("%.2f", max(THRESHOLD_GRID)))

sweep_val <- compute_threshold_sweep(val_predictions$truth, val_predictions$prob,
                                      val_predictions$date, THRESHOLD_GRID) %>%
  mutate(split = "validation")
sweep_test <- compute_threshold_sweep(test_predictions$truth, test_predictions$prob,
                                       test_predictions$date, THRESHOLD_GRID) %>%
  mutate(split = "test")

sweep_all <- bind_rows(sweep_val, sweep_test)
write_csv(sweep_all, file.path(OUTPUT_DIR, "threshold_sweep.csv"))
message("Saved: threshold_sweep.csv")

# CALIBRATED THRESHOLD SWEEP (all methods)

message("\n", strrep("=", 60))
message("STEP 10b: Calibrated threshold sweeps (all methods)")
message(strrep("=", 60))

# Compute threshold sweep for all available methods
calibrated_sweep_list <- list()

for (method in methods_available) {
  message("  Computing sweep for method: ", method)

  val_prob_m <- val_probs_by_method[[method]]
  test_prob_m <- test_probs_by_method[[method]]

  sweep_val_m <- compute_threshold_sweep(val_predictions$truth, val_prob_m,
                                          val_predictions$date, THRESHOLD_GRID) %>%
    mutate(split = "validation", method = method)

  sweep_test_m <- compute_threshold_sweep(test_predictions$truth, test_prob_m,
                                           test_predictions$date, THRESHOLD_GRID) %>%
    mutate(split = "test", method = method)

  calibrated_sweep_list <- append(calibrated_sweep_list, list(sweep_val_m, sweep_test_m))
}

threshold_sweep_calibrated <- bind_rows(calibrated_sweep_list)

# Guardrails: calibrated threshold sweep must include method column and full row count 
expected_sweep_rows <- length(THRESHOLD_GRID) * 2L * length(methods_available)  # val + test
required_sweep_cols <- c(
  "split", "method", "threshold",
  "tp", "fp", "tn", "fn",
  "n_alerts", "alerts_per_day",
  "precision", "recall", "specificity", "f1", "accuracy", "kappa"
)

stopifnot(all(required_sweep_cols %in% names(threshold_sweep_calibrated)))
stopifnot(setequal(unique(threshold_sweep_calibrated$split), c("validation", "test")))
stopifnot(setequal(unique(threshold_sweep_calibrated$method), methods_available))
stopifnot(nrow(threshold_sweep_calibrated) == expected_sweep_rows)

write_csv(threshold_sweep_calibrated, file.path(OUTPUT_DIR, "threshold_sweep_calibrated.csv"))
message("Saved: threshold_sweep_calibrated.csv")
message("  Methods included: ", paste(methods_available, collapse = ", "))

# Find optimal F1 threshold on VALIDATION ONLY (legacy RAW method output)
# NOTE: This uses FULL validation intentionally for backward compatibility with metrics_summary.csv.
# The RAW_USE_THRESHOLD_SELECT_ONLY flag affects operating_thresholds_table (operational decisions)
# but NOT this legacy best_threshold metric, which is diagnostic only.
opt_f1_val <- sweep_val %>%
  filter(!is.na(f1)) %>%
  filter(f1 == max(f1, na.rm = TRUE)) %>%
  slice(1)

best_threshold <- opt_f1_val$threshold

message("\n*** THRESHOLD SELECTION (on validation only) ***")
message("  Best threshold (max F1): ", sprintf("%.4f", best_threshold))
message("  Val F1 at best_t: ", sprintf("%.4f", opt_f1_val$f1))
message("  Val precision at best_t: ", sprintf("%.4f", opt_f1_val$precision))
message("  Val recall at best_t: ", sprintf("%.4f", opt_f1_val$recall))
message("  Val kappa at best_t: ", sprintf("%.4f", opt_f1_val$kappa))
message("  Val alerts/day at best_t: ", sprintf("%.1f", opt_f1_val$alerts_per_day))

# Alternative threshold selection: best recall subject to alerts/day <= K
# This matches the operational framing better
ALERT_BUDGET_TARGETS <- c(10, 25, 50)

message("\n*** ALTERNATIVE THRESHOLDS (best recall @ alerts/day <= K) ***")
for (k_budget in ALERT_BUDGET_TARGETS) {
  # Filter candidates, handling empty sets gracefully
  candidates <- sweep_val %>%
    filter(alerts_per_day <= k_budget, !is.na(recall))

  if (nrow(candidates) > 0) {
    # Safe max() - we know there's at least one row
    best_candidate <- candidates %>%
      filter(recall == max(recall)) %>%
      slice(1)

    message(sprintf("  K=%d: threshold=%.4f, recall=%.4f, precision=%.4f, alerts/day=%.1f",
                    k_budget,
                    best_candidate$threshold,
                    best_candidate$recall,
                    best_candidate$precision,
                    best_candidate$alerts_per_day))
  } else {
    message(sprintf("  K=%d: no threshold achieves <= %d alerts/day", k_budget, k_budget))
  }
}

# TEST METRICS AT VAL-OPTIMAL THRESHOLD

message("\n", strrep("=", 60))
message("STEP 11: Test metrics at VAL-optimal threshold")
message(strrep("=", 60))

# Get test metrics at the VAL-chosen threshold
test_at_val_t <- sweep_test %>% filter(threshold == best_threshold)

if (nrow(test_at_val_t) == 0) {
  # Find closest threshold if exact match not found
  test_at_val_t <- sweep_test %>%
    mutate(diff = abs(threshold - best_threshold)) %>%
    filter(diff == min(diff)) %>%
    select(-diff) %>%
    slice(1)
}

message("*** TEST PERFORMANCE (at VAL-optimal threshold ", sprintf("%.4f", best_threshold), ") ***")
message("  Test F1: ", sprintf("%.4f", test_at_val_t$f1))
message("  Test precision: ", sprintf("%.4f", test_at_val_t$precision))
message("  Test recall: ", sprintf("%.4f", test_at_val_t$recall))
message("  Test kappa: ", sprintf("%.4f", test_at_val_t$kappa))
message("  Test alerts/day: ", sprintf("%.1f", test_at_val_t$alerts_per_day))
message("  Test total alerts: ", test_at_val_t$n_alerts)
message("  Test TP: ", test_at_val_t$tp, ", FP: ", test_at_val_t$fp,
        ", TN: ", test_at_val_t$tn, ", FN: ", test_at_val_t$fn)

# Save test at val threshold
write_csv(test_at_val_t, file.path(OUTPUT_DIR, "test_at_val_threshold.csv"))
message("Saved: test_at_val_threshold.csv")

# ALERTS PER DAY (at best threshold)

message("\n", strrep("=", 60))
message("STEP 12: Alerts per day at best threshold")
message(strrep("=", 60))

alerts_val <- compute_alerts_per_day(val_predictions$prob, val_predictions$date, best_threshold)
alerts_test <- compute_alerts_per_day(test_predictions$prob, test_predictions$date, best_threshold)

write_csv(alerts_val, file.path(OUTPUT_DIR, "alerts_per_day_val.csv"))
write_csv(alerts_test, file.path(OUTPUT_DIR, "alerts_per_day_test.csv"))
message("Saved: alerts_per_day_val.csv, alerts_per_day_test.csv")

message("\nAlerts per day summary (at threshold ", sprintf("%.4f", best_threshold), "):")
message("  Validation:")
message("    Mean alerts/day: ", sprintf("%.1f", mean(alerts_val$alerts)))
message("    Median alerts/day: ", sprintf("%.1f", median(alerts_val$alerts)))
message("    Min/Max alerts/day: ", min(alerts_val$alerts), " / ", max(alerts_val$alerts))
message("  Test:")
message("    Mean alerts/day: ", sprintf("%.1f", mean(alerts_test$alerts)))
message("    Median alerts/day: ", sprintf("%.1f", median(alerts_test$alerts)))
message("    Min/Max alerts/day: ", min(alerts_test$alerts), " / ", max(alerts_test$alerts))

# DECISION CURVE (all methods)

message("\n", strrep("=", 60))
message("STEP 13: Decision curve analysis (all methods)")
message(strrep("=", 60))

# Compute decision curves for all available calibration methods
# Decision curves are particularly meaningful for calibrated probabilities
# since threshold = risk threshold interpretation requires calibration

decision_curve_list <- list()

for (method in methods_available) {
  message("  Computing decision curve for method: ", method)

  val_prob_m <- val_probs_by_method[[method]]
  test_prob_m <- test_probs_by_method[[method]]

  dc_val_m <- compute_decision_curve(val_predictions$truth, val_prob_m, THRESHOLD_GRID) %>%
    mutate(split = "validation", method = method)

  dc_test_m <- compute_decision_curve(test_predictions$truth, test_prob_m, THRESHOLD_GRID) %>%
    mutate(split = "test", method = method)

  decision_curve_list <- append(decision_curve_list, list(dc_val_m, dc_test_m))
}

dc_all <- bind_rows(decision_curve_list)

# Guardrails: decision curve must include method column and full row count 
expected_dc_rows <- length(THRESHOLD_GRID) * 2L * length(methods_available)  # val + test
required_dc_cols <- c("threshold", "net_benefit", "treat_all", "treat_none", "split", "method")

stopifnot(all(required_dc_cols %in% names(dc_all)))
stopifnot(setequal(unique(dc_all$split), c("validation", "test")))
stopifnot(setequal(unique(dc_all$method), methods_available))
stopifnot(nrow(dc_all) == expected_dc_rows)

write_csv(dc_all, file.path(OUTPUT_DIR, "decision_curve.csv"))
message("Saved: decision_curve.csv")
message("  Methods included: ", paste(methods_available, collapse = ", "))

# OPERATING THRESHOLDS TABLE (no-fixed-budget analysis)

message("\n", strrep("=", 60))
message("STEP 13b: Operating thresholds table (flexible budget)")
message(strrep("=", 60))

# This table summarises practical operating points for a NO-FIXED-BUDGET setting
# Thresholds are selected on VALIDATION, then evaluated on TEST
# Includes multiple policy types:
#   - precision_floor: maximise recall subject to precision >= target
#   - alerts_per_day_cap: maximize recall subject to alerts/day <= target
#   - f1_max: standard F1 optimization (reference only)

# Helper function to select threshold for a given policy
select_threshold_for_policy <- function(sweep_df, policy_type, target) {
  if (policy_type == "precision_floor") {
    # Maximize recall subject to precision >= target
    candidates <- sweep_df %>%
      filter(!is.na(precision), !is.na(recall), precision >= target)
    if (nrow(candidates) == 0) return(NULL)
    best <- candidates %>% filter(recall == max(recall, na.rm = TRUE)) %>% slice(1)
  } else if (policy_type == "alerts_per_day_cap") {
    # Maximize recall subject to alerts_per_day <= target
    candidates <- sweep_df %>%
      filter(!is.na(recall), alerts_per_day <= target)
    if (nrow(candidates) == 0) return(NULL)
    best <- candidates %>% filter(recall == max(recall, na.rm = TRUE)) %>% slice(1)
  } else if (policy_type == "f1_max") {
    # Maximize F1
    candidates <- sweep_df %>% filter(!is.na(f1))
    if (nrow(candidates) == 0) return(NULL)
    best <- candidates %>% filter(f1 == max(f1, na.rm = TRUE)) %>% slice(1)
  } else {
    return(NULL)
  }
  best
}

# Define policy configurations
policy_configs <- list(
  list(type = "precision_floor", target = 0.05),
  list(type = "precision_floor", target = 0.10),
  list(type = "precision_floor", target = 0.15),
  list(type = "alerts_per_day_cap", target = 5),
  list(type = "alerts_per_day_cap", target = 10),
  list(type = "alerts_per_day_cap", target = 20),
  list(type = "alerts_per_day_cap", target = 30),
  list(type = "f1_max", target = NA)
)

# Build operating thresholds table
operating_thresholds_rows <- list()

# In STRICT mode, compute threshold sweeps on the threshold-select subset for operating threshold selection
# This ensures calibrators were fitted on different data than threshold selection
if (STRICT_CALIBRATION_THRESHOLD_SEPARATION) {
  message("  [STRICT MODE] Computing threshold sweeps on threshold-select subset for operating thresholds...")

  # Build threshold sweeps for threshold-select subset
  threshold_select_sweep_list <- list()
  for (m in methods_available) {
    prob_m <- val_threshold_select_probs_by_method[[m]]
    sweep_m <- compute_threshold_sweep(
      val_predictions_threshold_select$truth,
      prob_m,
      val_predictions_threshold_select$date,
      THRESHOLD_GRID
    ) %>%
      mutate(split = "validation_threshold_select", method = m)
    threshold_select_sweep_list <- append(threshold_select_sweep_list, list(sweep_m))
  }
  threshold_select_sweeps <- bind_rows(threshold_select_sweep_list)
}

for (method in methods_available) {
  message("  Processing method: ", method)

  # Get validation sweep for threshold selection:
  # - STRICT mode + calibrated method: use threshold-select subset
  # - STRICT mode + raw: use full validation (no calibration bias) UNLESS RAW_USE_THRESHOLD_SELECT_ONLY
  # - STANDARD mode: use full validation
  # The RAW_USE_THRESHOLD_SELECT_ONLY flag allows symmetry sensitivity analysis
  use_strict_sweep <- STRICT_CALIBRATION_THRESHOLD_SEPARATION &&
                      (method != "raw" || RAW_USE_THRESHOLD_SELECT_ONLY)

  if (use_strict_sweep) {
    val_sweep_m <- threshold_select_sweeps %>%
      filter(method == !!method)
    selection_source <- "validation_threshold_select"
  } else {
    val_sweep_m <- threshold_sweep_calibrated %>%
      filter(split == "validation", method == !!method)
    selection_source <- "validation_full"
  }

  test_sweep_m <- threshold_sweep_calibrated %>%
    filter(split == "test", method == !!method)

  for (policy in policy_configs) {
    policy_type <- policy$type
    policy_target <- policy$target

    # Select threshold on validation
    val_best <- select_threshold_for_policy(val_sweep_m, policy_type, policy_target)

    if (is.null(val_best)) {
      # No valid threshold found for this policy
      row <- tibble(
        method = method,
        policy_type = policy_type,
        policy_target = policy_target,
        selected_split = selection_source,  # consistent with successful selection
        selected_threshold = NA_real_,
        val_precision = NA_real_,
        val_recall = NA_real_,
        val_alerts_per_day = NA_real_,
        val_f1 = NA_real_,
        test_precision = NA_real_,
        test_recall = NA_real_,
        test_alerts_per_day = NA_real_,
        test_f1 = NA_real_,
        test_kappa = NA_real_,
        test_tp = NA_integer_,
        test_fp = NA_integer_,
        test_tn = NA_integer_,
        test_fn = NA_integer_,
        notes = "No threshold satisfies policy constraint"
      )
    } else {
      selected_t <- val_best$threshold

      # Get test metrics at selected threshold
      test_at_t <- test_sweep_m %>% filter(threshold == selected_t)
      if (nrow(test_at_t) == 0) {
        # Find closest threshold
        test_at_t <- test_sweep_m %>%
          mutate(diff = abs(threshold - selected_t)) %>%
          filter(diff == min(diff)) %>%
          select(-diff) %>%
          slice(1)
      }

      # Build notes with provenance
      note_base <- if (method == "raw") {
        "raw probabilities"
      } else {
        paste0(method, " calibration")
      }
      note_source <- if (use_strict_sweep) {
        "; threshold selected on threshold-select subset (strict mode)"
      } else {
        "; threshold selected on full validation"
      }

      row <- tibble(
        method = method,
        policy_type = policy_type,
        policy_target = policy_target,
        selected_split = selection_source,
        selected_threshold = selected_t,
        val_precision = val_best$precision,
        val_recall = val_best$recall,
        val_alerts_per_day = val_best$alerts_per_day,
        val_f1 = val_best$f1,
        test_precision = test_at_t$precision,
        test_recall = test_at_t$recall,
        test_alerts_per_day = test_at_t$alerts_per_day,
        test_f1 = test_at_t$f1,
        test_kappa = test_at_t$kappa,
        test_tp = as.integer(test_at_t$tp),
        test_fp = as.integer(test_at_t$fp),
        test_tn = as.integer(test_at_t$tn),
        test_fn = as.integer(test_at_t$fn),
        notes = paste0(note_base, note_source)
      )
    }

    operating_thresholds_rows <- append(operating_thresholds_rows, list(row))
  }
}

operating_thresholds_table <- bind_rows(operating_thresholds_rows)
write_csv(operating_thresholds_table, file.path(OUTPUT_DIR, "operating_thresholds_table.csv"))
message("Saved: operating_thresholds_table.csv")

# Print summary
message("\nOperating thresholds summary (selected on validation):")
for (method in methods_available) {
  message("  Method: ", method)
  method_rows <- operating_thresholds_table %>% filter(method == !!method)
  for (i in 1:nrow(method_rows)) {
    r <- method_rows[i, ]
    if (!is.na(r$selected_threshold)) {
      message(sprintf("    %s (target=%.2f): t=%.4f, test recall=%.4f, test prec=%.4f, alerts/day=%.1f",
                      r$policy_type,
                      ifelse(is.na(r$policy_target), 0, r$policy_target),
                      r$selected_threshold,
                      r$test_recall,
                      r$test_precision,
                      r$test_alerts_per_day))
    }
  }
}
# BENCHMARK TABLE (Binary + LTR comparison)

message("\n", strrep("=", 60))
message("STEP 13c: Benchmark table (same run conditions)")
message(strrep("=", 60))

# This table compares:
#   1) Binary raw
#   2) Binary + Platt
#   3) Binary + Beta
#   4) LTR (Top-K metrics only, secondary comparator)
#
# Binary rows include probability-quality metrics + operating point metrics
# LTR row includes only Top-K metrics (not probability-calibrated)

# Build binary method rows
benchmark_rows <- list()

# Reference policy for operating point comparison
reference_policy_type <- "alerts_per_day_cap"
reference_policy_target <- 10

# CLEAN benchmark set: keep headline table stable and uncluttered 
# Single-source: use BENCHMARK_BINARY_METHODS constant defined in configuration
benchmark_methods <- intersect(BENCHMARK_BINARY_METHODS, methods_available)
if (length(benchmark_methods) == 0) {
  stop("FATAL: No benchmark methods available among {", paste(BENCHMARK_BINARY_METHODS, collapse = ", "), "}. ",
       "Check calibration step outputs.")
}
if (!setequal(benchmark_methods, BENCHMARK_BINARY_METHODS)) {
  warning("Benchmark methods subset: expected {", paste(BENCHMARK_BINARY_METHODS, collapse = ", "), "}, ",
          "got {", paste(benchmark_methods, collapse = ", "), "}. Some calibrators may have failed.")
}

for (method in benchmark_methods) {
  # Get calibration metrics from calibration_comparison (test only)
  cal_row <- calibration_comparison %>% filter(method == !!method)

  # Get operating point metrics
  op_row <- operating_thresholds_table %>%
    filter(method == !!method,
           policy_type == reference_policy_type,
           policy_target == reference_policy_target)

  # Build benchmark row
  row <- tibble(
    model_type = "binary",
    method = method,
    comparison_scope = "primary_probability_calibrated",

    # Discrimination metrics
    roc_auc_test = roc_auc_test,
    pr_auc_test = pr_auc_test,

    # Calibration metrics (vary by method)
    logloss_test = ifelse(nrow(cal_row) > 0, cal_row$logloss, NA_real_),
    brier_test = ifelse(nrow(cal_row) > 0, cal_row$brier, NA_real_),
    ece_equal_test = ifelse(nrow(cal_row) > 0, cal_row$ece_equal, NA_real_),
    ece_quantile_test = ifelse(nrow(cal_row) > 0, cal_row$ece_quantile, NA_real_),

    # Operating point metrics
    operating_policy_type = reference_policy_type,
    operating_policy_target = reference_policy_target,
    operating_threshold = ifelse(nrow(op_row) > 0, op_row$selected_threshold, NA_real_),
    test_precision_at_policy = ifelse(nrow(op_row) > 0, op_row$test_precision, NA_real_),
    test_recall_at_policy = ifelse(nrow(op_row) > 0, op_row$test_recall, NA_real_),
    test_f1_at_policy = ifelse(nrow(op_row) > 0, op_row$test_f1, NA_real_),
    test_alerts_per_day_at_policy = ifelse(nrow(op_row) > 0, op_row$test_alerts_per_day, NA_real_),

    # LTR-specific metrics (NA for binary)
    lift_k10 = NA_real_,
    precision_k10 = NA_real_,
    recall_k10 = NA_real_,
    hits_k10 = NA_integer_,

    # Metadata
    comparability_flag = "exact",
    provenance = OUTPUT_DIR,
    notes = ifelse(method == "raw",
                   "Raw XGBoost probabilities, not recommended for threshold selection",
                   paste0(method, " calibration fitted on validation set"))
  )

  benchmark_rows <- append(benchmark_rows, list(row))
}

# Add LTR row
# Search for most recent LTR results
ltr_path <- file.path(project_dir, "runs", "ltr")
ltr_runs <- if (dir.exists(ltr_path)) {
  dirs <- list.dirs(ltr_path, full.names = TRUE, recursive = FALSE)
  dirs <- dirs[order(basename(dirs), decreasing = TRUE)]  # Most recent first
  dirs
} else {
  character(0)
}

ltr_row_added <- FALSE
if (length(ltr_runs) > 0) {
  # Try to find a compatible LTR run
  for (ltr_run_dir in ltr_runs) {
    ltr_metrics_file <- file.path(ltr_run_dir, "topk_metrics_test.csv")
    if (file.exists(ltr_metrics_file)) {
      ltr_metrics <- read_csv(ltr_metrics_file, show_col_types = FALSE)

      # Get K=10 national metrics
      ltr_k10 <- ltr_metrics %>%
        filter(k == 10, allocation == "national")

      if (nrow(ltr_k10) > 0) {
        # Check metadata for comparability
        ltr_metadata_file <- file.path(ltr_run_dir, "metadata.csv")
        comparability <- "not_exact"
        ltr_notes <- "LTR is a Top-K policy model, not probability-calibrated; thresholds not directly comparable"

        if (file.exists(ltr_metadata_file)) {
          ltr_meta <- read_csv(ltr_metadata_file, show_col_types = FALSE)
          # Check if same data size
          if (nrow(ltr_meta) > 0 && ltr_meta$n_test[1] == nrow(test_predictions)) {
            comparability <- "same_split_size"
            ltr_notes <- paste0(ltr_notes, "; same test split size")
          }
        }

        ltr_row <- tibble(
          model_type = "ltr",
          method = "rank:pairwise",
          comparison_scope = "secondary_topk_policy",

          # Discrimination metrics (NA - not directly comparable)
          roc_auc_test = NA_real_,
          pr_auc_test = NA_real_,

          # Calibration metrics (NA - LTR scores not probabilities)
          logloss_test = NA_real_,
          brier_test = NA_real_,
          ece_equal_test = NA_real_,
          ece_quantile_test = NA_real_,

          # Operating point metrics (NA - LTR uses fixed K not threshold)
          operating_policy_type = "topk_national",
          operating_policy_target = 10,
          operating_threshold = NA_real_,
          test_precision_at_policy = ltr_k10$precision,
          test_recall_at_policy = ltr_k10$recall,
          test_f1_at_policy = NA_real_,
          test_alerts_per_day_at_policy = ltr_k10$k,

          # LTR-specific metrics
          lift_k10 = ltr_k10$lift,
          precision_k10 = ltr_k10$precision,
          recall_k10 = ltr_k10$recall,
          hits_k10 = as.integer(ltr_k10$n_hits),

          # Metadata
          comparability_flag = comparability,
          provenance = ltr_run_dir,
          notes = ltr_notes
        )

        benchmark_rows <- append(benchmark_rows, list(ltr_row))
        ltr_row_added <- TRUE
        message("  Added LTR row from: ", basename(ltr_run_dir))
        break
      }
    }
  }
}

if (!ltr_row_added) {
  # Add placeholder LTR row
  message("  No LTR results found - adding placeholder row")
  ltr_row <- tibble(
    model_type = "ltr",
    method = "rank:pairwise",
    comparison_scope = "secondary_topk_policy",
    roc_auc_test = NA_real_,
    pr_auc_test = NA_real_,
    logloss_test = NA_real_,
    brier_test = NA_real_,
    ece_equal_test = NA_real_,
    ece_quantile_test = NA_real_,
    operating_policy_type = NA_character_,
    operating_policy_target = NA_real_,
    operating_threshold = NA_real_,
    test_precision_at_policy = NA_real_,
    test_recall_at_policy = NA_real_,
    test_f1_at_policy = NA_real_,
    test_alerts_per_day_at_policy = NA_real_,
    lift_k10 = NA_real_,
    precision_k10 = NA_real_,
    recall_k10 = NA_real_,
    hits_k10 = NA_integer_,
    comparability_flag = "not_found",
    provenance = NA_character_,
    notes = "No LTR run found in runs/ltr/"
  )
  benchmark_rows <- append(benchmark_rows, list(ltr_row))
}

benchmark_table <- bind_rows(benchmark_rows)

# Guardrail: benchmark binary rows should be exactly BENCHMARK_BINARY_METHODS subset only 
binary_methods_in_benchmark <- benchmark_table %>%
  filter(model_type == "binary") %>%
  pull(method) %>%
  unique()

# Verify against single-source constant
if (!setequal(binary_methods_in_benchmark, benchmark_methods)) {
  stop("GUARDRAIL FAIL: Benchmark table methods {", paste(binary_methods_in_benchmark, collapse = ", "), "} ",
       "do not match expected benchmark_methods {", paste(benchmark_methods, collapse = ", "), "}.")
}
if ("isotonic" %in% binary_methods_in_benchmark) {
  stop("GUARDRAIL FAIL: Isotonic should not appear in benchmark table (diagnostic only).")
}
message("  Guardrail passed: benchmark methods = {", paste(binary_methods_in_benchmark, collapse = ", "), "}")

write_csv(benchmark_table, file.path(OUTPUT_DIR, "benchmark_table_same_run.csv"))
message("Saved: benchmark_table_same_run.csv")

# Print benchmark summary
message("\nBenchmark comparison:")
message("  Binary methods (primary - probability calibrated):")
for (method in benchmark_methods) {
  bm_row <- benchmark_table %>% filter(model_type == "binary", method == !!method)
  if (nrow(bm_row) > 0) {
    message(sprintf("    %s: logloss=%.4f, ece=%.4f, recall@%s=%d: %.3f",
                    method,
                    bm_row$logloss_test,
                    bm_row$ece_equal_test,
                    reference_policy_type,
                    as.integer(reference_policy_target),
                    ifelse(is.na(bm_row$test_recall_at_policy), 0, bm_row$test_recall_at_policy)))
  }
}
message("  LTR (secondary - Top-K policy):")
ltr_bm <- benchmark_table %>% filter(model_type == "ltr")
if (nrow(ltr_bm) > 0 && !is.na(ltr_bm$lift_k10)) {
  message(sprintf("    rank:pairwise: lift@10=%.2f, precision@10=%.3f, recall@10=%.3f",
                  ltr_bm$lift_k10,
                  ltr_bm$precision_k10,
                  ltr_bm$recall_k10))
} else {
  message("    (no results available)")
}

# BOOTSTRAP UNCERTAINTY FOR BENCHMARK METRICS

message("\n", strrep("=", 60))
message("STEP 13d: Bootstrap confidence intervals for benchmark metrics")
message(strrep("=", 60))

# Compute bootstrap CIs on TEST set for the default and secondary calibrated methods
# Uses the operating threshold selected on validation (respecting strict mode if enabled)

# Helper function to compute metrics at a threshold
compute_metrics_at_threshold <- function(truth, prob, threshold) {
  pred <- prob >= threshold
  y <- as.numeric(truth == "1")

  tp <- sum(y == 1 & pred, na.rm = TRUE)
  fp <- sum(y == 0 & pred, na.rm = TRUE)
  tn <- sum(y == 0 & !pred, na.rm = TRUE)
  fn <- sum(y == 1 & !pred, na.rm = TRUE)

  n_alerts <- tp + fp
  precision <- if (n_alerts > 0) tp / n_alerts else NA_real_
  recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  f1 <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
    2 * precision * recall / (precision + recall)
  } else NA_real_

  c(precision = precision, recall = recall, f1 = f1, n_alerts = n_alerts)
}

# Bootstrap function for a single method + policy
# NOTE: This uses ROW-LEVEL bootstrap which may underestimate uncertainty under temporal
# clustering (outage events are often spatially and temporally correlated).
# For realistic operational uncertainty, prefer the rolling temporal backtests output.
# Reference: Politis & Romano (1994) "The Stationary Bootstrap", JASA;
#            Lahiri (2003) "Resampling Methods for Dependent Data"
BOOTSTRAP_TYPE <- "row"  # "row" = i.i.d. row sampling; future: "block" for day-level

bootstrap_metrics <- function(test_df, prob_vec, threshold, n_days,
                              n_boot = BOOTSTRAP_N, seed = BOOTSTRAP_SEED,
                              bootstrap_type = BOOTSTRAP_TYPE) {
  set.seed(seed)
  n <- nrow(test_df)

  boot_results <- matrix(NA_real_, nrow = n_boot, ncol = 5)
  colnames(boot_results) <- c("precision", "recall", "f1", "n_alerts", "alerts_per_day")

  for (b in 1:n_boot) {
    if (bootstrap_type == "row") {
      # Standard i.i.d. row-level bootstrap (may be optimistic under temporal dependence)
      idx <- sample(1:n, n, replace = TRUE)
      boot_truth <- test_df$truth[idx]
      boot_prob <- prob_vec[idx]
      boot_n_days <- n_days  # approximation: same effective days
    } else {
      # Future: block/day bootstrap would sample dates with replacement
      stop("Block bootstrap not yet implemented. Use bootstrap_type='row'.")
    }

    metrics <- compute_metrics_at_threshold(boot_truth, boot_prob, threshold)
    alerts_per_day <- metrics["n_alerts"] / boot_n_days
    boot_results[b, ] <- c(metrics, alerts_per_day = alerts_per_day)
  }

  # Compute CIs
  alpha <- 1 - BOOTSTRAP_CI_LEVEL
  ci_low <- function(x) quantile(x, probs = alpha / 2, na.rm = TRUE)
  ci_high <- function(x) quantile(x, probs = 1 - alpha / 2, na.rm = TRUE)

  tibble(
    metric = c("precision", "recall", "f1", "n_alerts", "alerts_per_day"),
    mean = colMeans(boot_results, na.rm = TRUE),
    sd = apply(boot_results, 2, sd, na.rm = TRUE),
    ci_lower = apply(boot_results, 2, ci_low),
    ci_upper = apply(boot_results, 2, ci_high),
    bootstrap_type = bootstrap_type,
    bootstrap_note = ifelse(bootstrap_type == "row",
      "Row-level bootstrap; CIs may be narrow under temporal clustering. See rolling_temporal_windows.csv for temporal variance.",
      "Block bootstrap respects temporal structure.")
  )
}

# Run bootstrap for benchmark methods at reference policy
bootstrap_rows <- list()
reference_policy_row <- operating_thresholds_table %>%
  filter(policy_type == reference_policy_type, policy_target == reference_policy_target)

# Get test n_days for alerts_per_day calculation
test_n_days <- n_distinct(test_predictions$date)

message("  Bootstrap configuration: type=", BOOTSTRAP_TYPE, ", N=", BOOTSTRAP_N, ", test_days=", test_n_days)
if (BOOTSTRAP_TYPE == "row") {
  message("  WARNING: Row-level bootstrap may underestimate uncertainty under temporal clustering.")
  message("           See rolling_temporal_windows.csv for temporal variance estimates.")
}

for (method in benchmark_methods) {
  message("  Bootstrapping method: ", method)

  # Get threshold and test probabilities for this method
  method_row <- reference_policy_row %>% filter(method == !!method)
  if (nrow(method_row) == 0 || is.na(method_row$selected_threshold)) {
    message("    Skipping - no valid threshold")
    next
  }

  threshold_m <- method_row$selected_threshold
  test_prob_m <- test_probs_by_method[[method]]

  if (is.null(test_prob_m)) {
    message("    Skipping - no test probabilities")
    next
  }

  # Run bootstrap (pass n_days for alerts_per_day calculation)
  boot_ci <- bootstrap_metrics(test_predictions, test_prob_m, threshold_m, n_days = test_n_days)
  boot_ci <- boot_ci %>%
    mutate(
      method = method,
      policy_type = reference_policy_type,
      policy_target = reference_policy_target,
      threshold = threshold_m,
      n_bootstrap = BOOTSTRAP_N,
      ci_level = BOOTSTRAP_CI_LEVEL,
      test_n_days = test_n_days
    )

  bootstrap_rows <- append(bootstrap_rows, list(boot_ci))
}

if (length(bootstrap_rows) > 0) {
  benchmark_uncertainty_bootstrap <- bind_rows(bootstrap_rows)
  write_csv(benchmark_uncertainty_bootstrap, file.path(OUTPUT_DIR, "benchmark_uncertainty_bootstrap.csv"))
  message("Saved: benchmark_uncertainty_bootstrap.csv")

  # Print summary for default method
  default_boot <- benchmark_uncertainty_bootstrap %>% filter(method == DEFAULT_CALIBRATED_METHOD)
  if (nrow(default_boot) > 0) {
    message("\nBootstrap CIs for default method (", DEFAULT_CALIBRATED_METHOD, ") at policy ", reference_policy_type, "=", reference_policy_target, ":")
    for (i in 1:nrow(default_boot)) {
      r <- default_boot[i, ]
      message(sprintf("  %s: %.4f [%.4f, %.4f]", r$metric, r$mean, r$ci_lower, r$ci_upper))
    }
  }
} else {
  message("  No bootstrap results computed (no valid thresholds)")
  benchmark_uncertainty_bootstrap <- NULL
}

# ROLLING TEMPORAL BACKTESTS (Temporal Uncertainty)

message("\n", strrep("=", 60))
message("STEP 14: Rolling temporal backtests")
message(strrep("=", 60))

# Evaluate performance stability across time windows on the test set
# This provides operational uncertainty estimates independent of bootstrap
# NOTE: Windows are NON-OVERLAPPING sequential blocks (not sliding/rolling).
# "Rolling" here refers to the rolling-forward evaluation style, not overlapping windows.
# Each window covers approximately ROLLING_WINDOW_DAYS consecutive days.

# Parameters for sequential time windows
ROLLING_WINDOW_DAYS <- 30  # approximate window size in days (non-overlapping)
MIN_WINDOW_EVENTS <- 5     # minimum positive events for reliable metrics

# Get test dates
test_unique_dates <- sort(unique(test_predictions$date))
n_test_dates <- length(test_unique_dates)

message("Test period: ", min(test_unique_dates), " to ", max(test_unique_dates))
message("Total test days: ", n_test_dates)

# Create rolling windows
rolling_windows <- list()
window_start <- 1
window_id <- 1

while (window_start <= n_test_dates) {
  window_end <- min(window_start + ROLLING_WINDOW_DAYS - 1, n_test_dates)
  window_dates <- test_unique_dates[window_start:window_end]

  rolling_windows[[window_id]] <- list(
    window_id = window_id,
    start_date = min(window_dates),
    end_date = max(window_dates),
    n_days = length(window_dates),
    dates = window_dates
  )

  window_start <- window_end + 1
  window_id <- window_id + 1
}

message("Created ", length(rolling_windows), " rolling windows of ~", ROLLING_WINDOW_DAYS, " days each")

# Evaluate each method at reference policy threshold across rolling windows
rolling_results <- list()

for (method in benchmark_methods) {
  # Get threshold for this method at reference policy
  method_row <- reference_policy_row %>% filter(method == !!method)
  if (nrow(method_row) == 0 || is.na(method_row$selected_threshold)) {
    message("  Skipping ", method, " - no valid threshold")
    next
  }

  threshold_m <- method_row$selected_threshold
  test_prob_m <- test_probs_by_method[[method]]

  if (is.null(test_prob_m)) {
    message("  Skipping ", method, " - no test probabilities")
    next
  }

  message("  Processing method: ", method, " (threshold=", round(threshold_m, 4), ")")

  for (win in rolling_windows) {
    # Filter to this window
    win_mask <- test_predictions$date %in% win$dates
    win_truth <- test_predictions$truth[win_mask]
    win_prob <- test_prob_m[win_mask]
    win_n_days <- win$n_days

    # Count events
    n_positives <- sum(win_truth == "1")
    n_rows <- length(win_truth)

    # Skip if too few events
    if (n_positives < MIN_WINDOW_EVENTS) {
      rolling_results <- append(rolling_results, list(tibble(
        method = method,
        window_id = win$window_id,
        start_date = win$start_date,
        end_date = win$end_date,
        n_days = win_n_days,
        n_rows = n_rows,
        n_positives = n_positives,
        threshold = threshold_m,
        precision = NA_real_,
        recall = NA_real_,
        f1 = NA_real_,
        alerts_per_day = NA_real_,
        note = "insufficient_events"
      )))
      next
    }

    # Compute metrics
    metrics <- compute_metrics_at_threshold(win_truth, win_prob, threshold_m)
    alerts_per_day <- metrics["n_alerts"] / win_n_days

    rolling_results <- append(rolling_results, list(tibble(
      method = method,
      window_id = win$window_id,
      start_date = win$start_date,
      end_date = win$end_date,
      n_days = win_n_days,
      n_rows = n_rows,
      n_positives = n_positives,
      threshold = threshold_m,
      precision = metrics["precision"],
      recall = metrics["recall"],
      f1 = metrics["f1"],
      alerts_per_day = alerts_per_day,
      note = "ok"
    )))
  }
}

rolling_temporal_df <- bind_rows(rolling_results)
write_csv(rolling_temporal_df, file.path(OUTPUT_DIR, "rolling_temporal_windows.csv"))
message("Saved: rolling_temporal_windows.csv")

# Compute temporal stability summary
# NOTE: This measures FIXED-THRESHOLD performance stability across time windows.
# The threshold is fixed at the value selected on validation; we then evaluate
# how performance varies across test windows at that fixed threshold.
# This is the operationally relevant question: "Will my deployed threshold perform consistently?"
# For threshold selection stability (would we select the same threshold each window?),
# see the per-window threshold analysis if implemented, or rolling validation backtests.
# Reference: Gama et al. (2014) "A Survey on Concept Drift Adaptation", ACM Computing Surveys

temporal_stability_summary <- rolling_temporal_df %>%
  filter(note == "ok") %>%
  group_by(method) %>%
  summarise(
    n_windows = n(),
    n_windows_valid = sum(!is.na(f1)),
    # Fixed threshold used (should be constant within method)
    threshold_used = first(threshold),
    # Precision stats
    precision_mean = mean(precision, na.rm = TRUE),
    precision_sd = sd(precision, na.rm = TRUE),
    precision_min = min(precision, na.rm = TRUE),
    precision_max = max(precision, na.rm = TRUE),
    precision_cv = if_else(precision_mean > 0, precision_sd / precision_mean, NA_real_),
    # Recall stats
    recall_mean = mean(recall, na.rm = TRUE),
    recall_sd = sd(recall, na.rm = TRUE),
    recall_min = min(recall, na.rm = TRUE),
    recall_max = max(recall, na.rm = TRUE),
    recall_cv = if_else(recall_mean > 0, recall_sd / recall_mean, NA_real_),
    # F1 stats
    f1_mean = mean(f1, na.rm = TRUE),
    f1_sd = sd(f1, na.rm = TRUE),
    f1_min = min(f1, na.rm = TRUE),
    f1_max = max(f1, na.rm = TRUE),
    f1_cv = if_else(f1_mean > 0, f1_sd / f1_mean, NA_real_),
    # Alerts per day stats
    alerts_per_day_mean = mean(alerts_per_day, na.rm = TRUE),
    alerts_per_day_sd = sd(alerts_per_day, na.rm = TRUE),
    alerts_per_day_min = min(alerts_per_day, na.rm = TRUE),
    alerts_per_day_max = max(alerts_per_day, na.rm = TRUE),
    # Stability interpretation
    stability_note = "Fixed-threshold performance stability across rolling windows",
    .groups = "drop"
  )

write_csv(temporal_stability_summary, file.path(OUTPUT_DIR, "temporal_stability_summary.csv"))
message("Saved: temporal_stability_summary.csv")

# Print summary for default method
default_temporal <- temporal_stability_summary %>% filter(method == DEFAULT_CALIBRATED_METHOD)
if (nrow(default_temporal) > 0) {
  message("\nTemporal stability for default method (", DEFAULT_CALIBRATED_METHOD, "):")
  message(sprintf("  F1:        %.4f ± %.4f (CV=%.2f%%, range=[%.4f, %.4f])",
                  default_temporal$f1_mean, default_temporal$f1_sd,
                  default_temporal$f1_cv * 100,
                  default_temporal$f1_min, default_temporal$f1_max))
  message(sprintf("  Precision: %.4f ± %.4f (CV=%.2f%%)",
                  default_temporal$precision_mean, default_temporal$precision_sd,
                  default_temporal$precision_cv * 100))
  message(sprintf("  Recall:    %.4f ± %.4f (CV=%.2f%%)",
                  default_temporal$recall_mean, default_temporal$recall_sd,
                  default_temporal$recall_cv * 100))
  message(sprintf("  Alerts/day: %.1f ± %.1f (range=[%.1f, %.1f])",
                  default_temporal$alerts_per_day_mean, default_temporal$alerts_per_day_sd,
                  default_temporal$alerts_per_day_min, default_temporal$alerts_per_day_max))
}

# DEPLOYMENT DECISION TABLE

message("\n", strrep("=", 60))
message("STEP 15: Deployment decision table")
message(strrep("=", 60))

# Build a deployment-ready summary combining:
# - Operating threshold from operating_thresholds_table
# - Point estimates (precision, recall, F1, alerts/day) on test
# - Bootstrap CIs
# - Temporal variance from rolling windows
# - Deployment readiness flag

DEPLOYMENT_PRECISION_FLOOR <- 0.05  # minimum acceptable precision for deployment

deployment_rows <- list()

for (method in c(DEFAULT_CALIBRATED_METHOD, SECONDARY_CALIBRATED_METHOD)) {
  message("  Building deployment row for: ", method)

  # Get operating threshold info
  method_op <- reference_policy_row %>% filter(method == !!method)

  if (nrow(method_op) == 0 || is.na(method_op$selected_threshold)) {
    message("    Skipping - no valid threshold")
    next
  }

  # Point estimates from operating thresholds table
  threshold <- method_op$selected_threshold
  precision_point <- method_op$test_precision
  recall_point <- method_op$test_recall
  f1_point <- method_op$test_f1
  alerts_per_day_point <- method_op$test_alerts_per_day

  # Bootstrap CIs (if available)
  if (!is.null(benchmark_uncertainty_bootstrap)) {
    boot_method <- benchmark_uncertainty_bootstrap %>% filter(method == !!method)

    precision_ci_lower <- boot_method %>% filter(metric == "precision") %>% pull(ci_lower)
    precision_ci_upper <- boot_method %>% filter(metric == "precision") %>% pull(ci_upper)
    recall_ci_lower <- boot_method %>% filter(metric == "recall") %>% pull(ci_lower)
    recall_ci_upper <- boot_method %>% filter(metric == "recall") %>% pull(ci_upper)
    f1_ci_lower <- boot_method %>% filter(metric == "f1") %>% pull(ci_lower)
    f1_ci_upper <- boot_method %>% filter(metric == "f1") %>% pull(ci_upper)

    # Handle empty results
    precision_ci_lower <- if (length(precision_ci_lower) == 0) NA_real_ else precision_ci_lower
    precision_ci_upper <- if (length(precision_ci_upper) == 0) NA_real_ else precision_ci_upper
    recall_ci_lower <- if (length(recall_ci_lower) == 0) NA_real_ else recall_ci_lower
    recall_ci_upper <- if (length(recall_ci_upper) == 0) NA_real_ else recall_ci_upper
    f1_ci_lower <- if (length(f1_ci_lower) == 0) NA_real_ else f1_ci_lower
    f1_ci_upper <- if (length(f1_ci_upper) == 0) NA_real_ else f1_ci_upper
  } else {
    precision_ci_lower <- NA_real_
    precision_ci_upper <- NA_real_
    recall_ci_lower <- NA_real_
    recall_ci_upper <- NA_real_
    f1_ci_lower <- NA_real_
    f1_ci_upper <- NA_real_
  }

  # Temporal variance (if available)
  temporal_method <- temporal_stability_summary %>% filter(method == !!method)

  if (nrow(temporal_method) > 0) {
    precision_temporal_sd <- temporal_method$precision_sd
    recall_temporal_sd <- temporal_method$recall_sd
    f1_temporal_sd <- temporal_method$f1_sd
    f1_temporal_cv <- temporal_method$f1_cv
    n_temporal_windows <- temporal_method$n_windows_valid
  } else {
    precision_temporal_sd <- NA_real_
    recall_temporal_sd <- NA_real_
    f1_temporal_sd <- NA_real_
    f1_temporal_cv <- NA_real_
    n_temporal_windows <- NA_integer_
  }

  # Deployment readiness:
  # - Precision CI lower bound > floor
  # - OR if no bootstrap, point estimate > floor
  deployment_ready <- FALSE
  if (!is.na(precision_ci_lower)) {
    deployment_ready <- precision_ci_lower > DEPLOYMENT_PRECISION_FLOOR
  } else if (!is.na(precision_point)) {
    deployment_ready <- precision_point > DEPLOYMENT_PRECISION_FLOOR
  }

  deployment_rows <- append(deployment_rows, list(tibble(
    method = method,
    role = if (method == DEFAULT_CALIBRATED_METHOD) "default" else "sensitivity",
    policy_type = reference_policy_type,
    policy_target = reference_policy_target,
    threshold = threshold,
    # Point estimates
    precision = precision_point,
    recall = recall_point,
    f1 = f1_point,
    alerts_per_day = alerts_per_day_point,
    # Bootstrap CIs
    precision_ci_lower = precision_ci_lower,
    precision_ci_upper = precision_ci_upper,
    recall_ci_lower = recall_ci_lower,
    recall_ci_upper = recall_ci_upper,
    f1_ci_lower = f1_ci_lower,
    f1_ci_upper = f1_ci_upper,
    # Temporal variance
    precision_temporal_sd = precision_temporal_sd,
    recall_temporal_sd = recall_temporal_sd,
    f1_temporal_sd = f1_temporal_sd,
    f1_temporal_cv = f1_temporal_cv,
    n_temporal_windows = n_temporal_windows,
    # Deployment flag
    deployment_precision_floor = DEPLOYMENT_PRECISION_FLOOR,
    deployment_ready = deployment_ready
  )))
}

if (length(deployment_rows) > 0) {
  deployment_decision_table <- bind_rows(deployment_rows)
  write_csv(deployment_decision_table, file.path(OUTPUT_DIR, "deployment_decision_table.csv"))
  message("Saved: deployment_decision_table.csv")

  # Print summary
  message("\nDeployment Decision Summary:")
  for (i in 1:nrow(deployment_decision_table)) {
    r <- deployment_decision_table[i, ]
    ready_str <- if (r$deployment_ready) "YES" else "NO"
    message(sprintf("  %s (%s): threshold=%.4f, precision=%.4f [%.4f, %.4f], ready=%s",
                    r$method, r$role, r$threshold,
                    r$precision, r$precision_ci_lower, r$precision_ci_upper,
                    ready_str))
  }
} else {
  message("  No deployment rows computed (no valid methods)")
  deployment_decision_table <- NULL
}

# FEATURE IMPORTANCE

message("\n", strrep("=", 60))
message("STEP 16: Feature importance")
message(strrep("=", 60))

xgb_engine <- extract_fit_engine(xgb_fit)
importance_df <- xgboost::xgb.importance(model = xgb_engine) %>%
  as_tibble() %>%
  arrange(desc(Gain))

write_csv(importance_df, file.path(OUTPUT_DIR, "feature_importance.csv"))
message("Saved: feature_importance.csv")
message("Top 10 features by Gain:")
print(head(importance_df, 10))

# METRICS SUMMARY

message("\n", strrep("=", 60))
message("STEP 17: Saving metrics summary")
message(strrep("=", 60))

val_prevalence <- mean(val_predictions$truth == "1", na.rm = TRUE)
test_prevalence <- mean(test_predictions$truth == "1", na.rm = TRUE)

metrics_summary <- tibble(
  split = c("validation", "test"),
  n_rows = c(nrow(val_predictions), nrow(test_predictions)),
  n_days = c(n_distinct(val_predictions$date), n_distinct(test_predictions$date)),
  n_positive = c(sum(val_predictions$truth == "1"), sum(test_predictions$truth == "1")),
  prevalence = c(val_prevalence, test_prevalence),
  roc_auc = c(roc_auc_val, roc_auc_test),
  pr_auc = c(pr_auc_val, pr_auc_test),
  # Raw calibration metrics
  brier_score = c(brier_val, brier_test),
  logloss = c(logloss_val, logloss_test),
  ece_equal = c(ece_equal_val, ece_equal_test),
  ece_quantile = c(ece_quant_val, ece_quant_test),
  ece_quantile_fallback = c(ece_quant_val_result$fallback, ece_quant_test_result$fallback),
  calibration_intercept = c(cal_coef_val$intercept, cal_coef_test$intercept),
  calibration_slope = c(cal_coef_val$slope, cal_coef_test$slope),
  # Platt-scaled metrics (test only, NA for validation since calibrators fit on val)
  logloss_platt = c(NA_real_, logloss_test_platt),
  brier_platt = c(NA_real_, brier_test_platt),
  ece_equal_platt = c(NA_real_, ece_equal_test_platt),
  ece_quantile_platt = c(NA_real_, ece_quant_test_platt),
  cal_intercept_platt = c(NA_real_, cal_int_test_platt),
  cal_slope_platt = c(NA_real_, cal_slope_test_platt),
  # Isotonic-calibrated metrics (test only)
  logloss_isotonic = c(NA_real_, logloss_test_isotonic),
  brier_isotonic = c(NA_real_, brier_test_isotonic),
  ece_equal_isotonic = c(NA_real_, ece_equal_test_isotonic),
  ece_quantile_isotonic = c(NA_real_, ece_quant_test_isotonic),
  cal_intercept_isotonic = c(NA_real_, cal_int_test_isotonic),
  cal_slope_isotonic = c(NA_real_, cal_slope_test_isotonic),
  # Beta-calibrated metrics (test only)
  logloss_beta = c(NA_real_, logloss_test_beta),
  brier_beta = c(NA_real_, brier_test_beta),
  ece_equal_beta = c(NA_real_, ece_equal_test_beta),
  ece_quantile_beta = c(NA_real_, ece_quant_test_beta),
  cal_intercept_beta = c(NA_real_, cal_int_test_beta),
  cal_slope_beta = c(NA_real_, cal_slope_test_beta),
  # Threshold-based metrics
  best_threshold_from_val = best_threshold,
  f1_at_best_t = c(opt_f1_val$f1, test_at_val_t$f1),
  precision_at_best_t = c(opt_f1_val$precision, test_at_val_t$precision),
  recall_at_best_t = c(opt_f1_val$recall, test_at_val_t$recall),
  kappa_at_best_t = c(
    kappa_from_counts(opt_f1_val$tp, opt_f1_val$fp, opt_f1_val$tn, opt_f1_val$fn),
    kappa_from_counts(test_at_val_t$tp, test_at_val_t$fp, test_at_val_t$tn, test_at_val_t$fn)
  ),
  alerts_per_day_at_best_t = c(opt_f1_val$alerts_per_day, test_at_val_t$alerts_per_day)
)

write_csv(metrics_summary, file.path(OUTPUT_DIR, "metrics_summary.csv"))
message("Saved: metrics_summary.csv")

# RUN CONFIGURATION

message("\n", strrep("=", 60))
message("STEP 18: Saving run configuration")
message(strrep("=", 60))

run_config <- list(
  run_id = RUN_ID,
  task_mode = task_mode,
  weather_mode = weather_mode,
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  seed = 42,

  data = list(
    features_path = features_path,
    splits_fixed_path = splits_fixed_path,
    train_rows = nrow(train_data),
    val_rows = nrow(val_data),
    test_rows = nrow(test_data),
    val_days = n_distinct(val_predictions$date),
    test_days = n_distinct(test_predictions$date),
    train_prevalence = train_prevalence,
    val_prevalence = val_prevalence,
    test_prevalence = test_prevalence
  ),

  splits = list(
    train = list(start = as.character(train_start), end = as.character(train_end)),
    val = list(start = as.character(val_start), end = as.character(val_end)),
    test = list(start = as.character(test_start), end = as.character(test_end))
  ),

  model = list(
    type = "xgboost",
    n_features = length(feature_cols),
    scale_pos_weight = scale_pos_weight,
    params = xgb_params,
    training_time_min = as.numeric(train_duration)
  ),

  threshold = list(
    best_threshold = best_threshold,
    selected_on = "validation",
    metric_optimized = "f1",
    threshold_grid_size = length(THRESHOLD_GRID),
    threshold_grid_min = min(THRESHOLD_GRID),
    threshold_grid_max = max(THRESHOLD_GRID)
  ),

  calibration = list(
    n_bins = N_CALIBRATION_BINS,
    platt_scaling = PLATT_SCALING,
    isotonic_calibration = !is.null(iso_model),
    beta_calibration = !is.null(beta_model),
    # Default calibrated method specification
    default_calibrated_method = DEFAULT_CALIBRATED_METHOD,
    secondary_calibrated_method = SECONDARY_CALIBRATED_METHOD,
    benchmark_binary_methods = BENCHMARK_BINARY_METHODS,
    # Strict calibration-threshold separation
    strict_separation = strict_split_info,
    # Raw method threshold selection in strict mode
    raw_use_threshold_select_only = RAW_USE_THRESHOLD_SELECT_ONLY
  ),

  results = list(
    # Discrimination
    roc_auc_val = roc_auc_val,
    roc_auc_test = roc_auc_test,
    pr_auc_val = pr_auc_val,
    pr_auc_test = pr_auc_test,
    # Raw calibration
    logloss_val = logloss_val,
    logloss_test = logloss_test,
    brier_val = brier_val,
    brier_test = brier_test,
    ece_equal_val = ece_equal_val,
    ece_equal_test = ece_equal_test,
    ece_quant_val = ece_quant_val,
    ece_quant_test = ece_quant_test,
    ece_quant_val_fallback = ece_quant_val_result$fallback,
    ece_quant_test_fallback = ece_quant_test_result$fallback,
    # Platt-calibrated (test only)
    logloss_test_platt = logloss_test_platt,
    brier_test_platt = brier_test_platt,
    ece_equal_test_platt = ece_equal_test_platt,
    ece_quant_test_platt = ece_quant_test_platt,
    cal_intercept_test_platt = cal_int_test_platt,
    cal_slope_test_platt = cal_slope_test_platt,
    # Isotonic-calibrated (test only)
    logloss_test_isotonic = logloss_test_isotonic,
    brier_test_isotonic = brier_test_isotonic,
    ece_equal_test_isotonic = ece_equal_test_isotonic,
    ece_quant_test_isotonic = ece_quant_test_isotonic,
    cal_intercept_test_isotonic = cal_int_test_isotonic,
    cal_slope_test_isotonic = cal_slope_test_isotonic,
    # Beta-calibrated (test only)
    logloss_test_beta = logloss_test_beta,
    brier_test_beta = brier_test_beta,
    ece_equal_test_beta = ece_equal_test_beta,
    ece_quant_test_beta = ece_quant_test_beta,
    cal_intercept_test_beta = cal_int_test_beta,
    cal_slope_test_beta = cal_slope_test_beta,
    # Threshold-based
    f1_val_at_best_t = opt_f1_val$f1,
    f1_test_at_best_t = test_at_val_t$f1,
    kappa_val_at_best_t = kappa_from_counts(opt_f1_val$tp, opt_f1_val$fp, opt_f1_val$tn, opt_f1_val$fn),
    kappa_test_at_best_t = kappa_from_counts(test_at_val_t$tp, test_at_val_t$fp, test_at_val_t$tn, test_at_val_t$fn),
    alerts_per_day_val = opt_f1_val$alerts_per_day,
    alerts_per_day_test = test_at_val_t$alerts_per_day
  )
)

write_json(run_config, file.path(OUTPUT_DIR, "run_config.json"), pretty = TRUE, auto_unbox = TRUE)
message("Saved: run_config.json")

# Save feature list
writeLines(feature_cols, file.path(OUTPUT_DIR, "features_used.txt"))
message("Saved: features_used.txt")

# FINAL SUMMARY

message("\n", strrep("=", 70))
message("BINARY THRESHOLD BASELINE EVALUATION COMPLETE")
message(strrep("=", 70))

message("\nRun ID: ", RUN_ID)
message("Task mode: ", task_mode)
message("Output directory: ", OUTPUT_DIR)

message("\nThreshold Selection:")
message("  Best threshold (from VAL): ", sprintf("%.4f", best_threshold))

message("\nValidation Results (raw):")
message("  ROC AUC: ", sprintf("%.4f", roc_auc_val))
message("  PR AUC:  ", sprintf("%.4f", pr_auc_val))
message("  Log-loss: ", sprintf("%.6f", logloss_val))
message("  Brier:   ", sprintf("%.6f", brier_val))
message("  ECE (equal/quantile): ", sprintf("%.6f / %.6f", ece_equal_val, ece_quant_val))
message("  F1 at best_t: ", sprintf("%.4f", opt_f1_val$f1))
message("  Kappa at best_t: ", sprintf("%.4f", kappa_from_counts(opt_f1_val$tp, opt_f1_val$fp, opt_f1_val$tn, opt_f1_val$fn)))
message("  Alerts/day at best_t: ", sprintf("%.1f", opt_f1_val$alerts_per_day))

message("\nTest Results (raw, at VAL-optimal threshold):")
message("  ROC AUC: ", sprintf("%.4f", roc_auc_test))
message("  PR AUC:  ", sprintf("%.4f", pr_auc_test))
message("  Log-loss: ", sprintf("%.6f", logloss_test))
message("  Brier:   ", sprintf("%.6f", brier_test))
message("  ECE (equal/quantile): ", sprintf("%.6f / %.6f", ece_equal_test, ece_quant_test))
message("  F1 at best_t: ", sprintf("%.4f", test_at_val_t$f1))
message("  Kappa at best_t: ", sprintf("%.4f", kappa_from_counts(test_at_val_t$tp, test_at_val_t$fp, test_at_val_t$tn, test_at_val_t$fn)))
message("  Alerts/day at best_t: ", sprintf("%.1f", test_at_val_t$alerts_per_day))

message("\nTest Results (calibrated - best log-loss):")
best_cal_method <- calibration_comparison %>%
  filter(!is.na(logloss)) %>%
  filter(logloss == min(logloss, na.rm = TRUE)) %>%
  pull(method)
best_cal_logloss <- min(calibration_comparison$logloss, na.rm = TRUE)
message("  Best method: ", best_cal_method, " (log-loss: ", sprintf("%.6f", best_cal_logloss), ")")

message("\nFiles generated:")
message("  - metrics_summary.csv")
message("  - roc_curve.csv")
message("  - pr_curve.csv")
message("  - calibration_bins_equal.csv")
message("  - calibration_bins_quantile.csv")
message("  - calibration_metrics.csv")
message("  - calibration_comparison_test.csv")
message("  - calibration_bins_equal_all_methods.csv")
message("  - calibration_bins_quantile_all_methods.csv")
message("  - threshold_sweep.csv")
message("  - threshold_sweep_calibrated.csv  [NEW: all methods]")
message("  - test_at_val_threshold.csv")
message("  - alerts_per_day_val.csv")
message("  - alerts_per_day_test.csv")
message("  - decision_curve.csv  [UPDATED: includes method column]")
message("  - operating_thresholds_table.csv  [NEW: flexible budget analysis]")
message("  - benchmark_table_same_run.csv  [NEW: binary vs LTR comparison]")
message("  - benchmark_uncertainty_bootstrap.csv  [NEW: bootstrap CIs for benchmark metrics]")
message("  - rolling_temporal_windows.csv  [NEW: per-window temporal metrics]")
message("  - temporal_stability_summary.csv  [NEW: temporal variance summary]")
message("  - deployment_decision_table.csv  [NEW: deployment readiness summary]")
if (PLATT_SCALING && !is.null(platt_results)) {
  message("  - platt_calibration.csv")
  message("  - calibration_bins_platt_equal.csv")
  message("  - calibration_bins_platt_quantile.csv")
}
message("  - feature_importance.csv")
message("  - run_config.json")
message("  - features_used.txt")

message("\nScript completed at: ", Sys.time())
message(strrep("=", 70))
