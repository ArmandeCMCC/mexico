# 15_logistic_baseline.R

# Logistic regression baseline for binary outage prediction.
# Uses the same 82 features, same train/val/test split, and
# same Platt calibration framework as the XGBoost benchmark.
#
# Produces comparable evaluation to 05b for fair comparison.
#
# Usage:
#   Rscript scripts/15_logistic_baseline.R [xgb_run_id]

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tibble)
  library(tidyr)
  library(jsonlite)
  library(yardstick)
  library(glmnet)
})

set.seed(42)

# HELPERS

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0 || is.na(x)) y else x

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (dir.exists("data") || dir.exists("scripts")) return(normalizePath(getwd()))
  normalizePath(getwd())
}

arg_or_default <- function(args, i, default) {
  if (length(args) >= i && nzchar(args[[i]])) args[[i]] else default
}

apply_platt_with_params <- function(prob, intercept, slope, eps = 1e-15) {
  p <- pmin(pmax(prob, eps), 1 - eps)
  plogis(intercept + slope * qlogis(p))
}

compute_logloss <- function(actual, prob, eps = 1e-15) {
  ok <- !is.na(actual) & !is.na(prob)
  if (!any(ok)) return(NA_real_)
  y <- as.numeric(actual[ok])
  p <- pmin(pmax(prob[ok], eps), 1 - eps)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

compute_brier <- function(actual, prob) {
  ok <- !is.na(actual) & !is.na(prob)
  if (!any(ok)) return(NA_real_)
  mean((prob[ok] - as.numeric(actual[ok]))^2)
}

compute_ece_equal <- function(actual, prob, n_bins = 10) {
  y <- as.numeric(actual)
  breaks <- seq(0, 1, length.out = n_bins + 1)
  bin_idx <- cut(prob, breaks = breaks, include.lowest = TRUE, labels = FALSE)

  ece <- 0
  n_total <- length(prob)

  for (b in seq_len(n_bins)) {
    in_bin <- which(bin_idx == b)
    n_bin <- length(in_bin)
    if (n_bin > 0) {
      avg_pred <- mean(prob[in_bin], na.rm = TRUE)
      avg_true <- mean(y[in_bin], na.rm = TRUE)
      ece <- ece + (n_bin / n_total) * abs(avg_pred - avg_true)
    }
  }

  ece
}

compute_calibration_bins_equal <- function(actual, prob, n_bins = 10) {
  y <- as.numeric(actual)
  breaks <- seq(0, 1, length.out = n_bins + 1)
  bin_idx <- cut(prob, breaks = breaks, include.lowest = TRUE, labels = FALSE)

  bins <- tibble(
    bin = seq_len(n_bins),
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

kappa_from_counts <- function(tp, fp, tn, fn) {
  tp <- as.double(tp); fp <- as.double(fp)
  tn <- as.double(tn); fn <- as.double(fn)
  N  <- tp + fp + tn + fn
  po <- (tp + tn) / N
  pe <- ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (N^2)
  k  <- (po - pe) / (1 - pe)
  k[is.nan(k) | is.infinite(k)] <- NA_real_
  k
}

prep_model_df <- function(df, model_features) {
  df %>%
    select(all_of(c("outage_3h_or_more", model_features))) %>%
    mutate(
      outage_3h_or_more = as.integer(outage_3h_or_more)
    ) %>%
    mutate(across(where(is.logical), as.integer))
}

# ARGUMENTS / PATHS

args <- commandArgs(trailingOnly = TRUE)
xgb_run_id <- arg_or_default(args, 1, "20260223_125430_forecast_strict")

project_dir <- detect_project_dir()

# XGBoost run directory (reference)
xgb_run_dir <- file.path(project_dir, "data", "baselines", "binary_threshold", xgb_run_id)

# Logistic output directory (new timestamp-based ID)
logistic_run_id <- paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_logistic_baseline")
logistic_out_dir <- file.path(project_dir, "data", "baselines", "logistic_threshold", logistic_run_id)
dir.create(logistic_out_dir, recursive = TRUE, showWarnings = FALSE)

# Input paths
features_path <- file.path(project_dir, "data", "model_ready", "features_engineered.rds")
splits_path   <- file.path(project_dir, "data", "model_ready", "splits_fixed.rds")
features_used_path <- file.path(xgb_run_dir, "features_used.txt")
xgb_config_path    <- file.path(xgb_run_dir, "run_config.json")

message(strrep("=", 70))
message("15_logistic_baseline.R")
message(strrep("=", 70))
message("XGBoost reference run: ", xgb_run_id)
message("Logistic run ID:       ", logistic_run_id)
message("Output directory:      ", logistic_out_dir)
message("Features path:         ", features_path)
message("Splits path:           ", splits_path)

# Validate required files
required_files <- c(features_path, splits_path, features_used_path, xgb_config_path)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop("Missing required files:\n", paste0("  - ", missing_files, collapse = "\n"))
}

# STEP 1: LOAD DATA

message("\n", strrep("-", 60))
message("STEP 1: Loading data")
message(strrep("-", 60))

feature_cols <- read_lines(features_used_path)
feature_cols <- feature_cols[nzchar(feature_cols)]
feature_cols <- unique(feature_cols)
message("  Features loaded: ", length(feature_cols))

features <- readRDS(features_path)
splits_fixed <- readRDS(splits_path)

train_start <- as.Date(splits_fixed$train_range[1])
train_end   <- as.Date(splits_fixed$train_range[2])
val_start   <- as.Date(splits_fixed$val_range[1])
val_end     <- as.Date(splits_fixed$val_range[2])
test_start  <- as.Date(splits_fixed$test_range[1])
test_end    <- as.Date(splits_fixed$test_range[2])

message("  Train: ", train_start, " to ", train_end)
message("  Val:   ", val_start, " to ", val_end)
message("  Test:  ", test_start, " to ", test_end)

features <- features %>%
  mutate(
    date = as.Date(date),
    GID_2 = as.character(GID_2),
    split = case_when(
      date >= train_start & date <= train_end ~ "train",
      date >= val_start   & date <= val_end   ~ "val",
      date >= test_start  & date <= test_end  ~ "test",
      TRUE ~ "other"
    )
  )

missing_feats <- setdiff(feature_cols, names(features))
if (length(missing_feats) > 0) {
  stop("Missing features from features_engineered.rds:\n",
       paste0("  - ", head(missing_feats, 30), collapse = "\n"))
}

# STEP 2: PREPARE DATA

message("\n", strrep("-", 60))
message("STEP 2: Preparing model data")
message(strrep("-", 60))

train_raw <- features %>% filter(split == "train")
val_raw   <- features %>% filter(split == "val")
test_raw  <- features %>% filter(split == "test")

train_df <- prep_model_df(train_raw, feature_cols)
val_df   <- prep_model_df(val_raw,   feature_cols)
test_df  <- prep_model_df(test_raw,  feature_cols)

# Remove rows with any NA in predictors (glm requires complete cases)
train_complete <- complete.cases(train_df)
val_complete   <- complete.cases(val_df)
test_complete  <- complete.cases(test_df)

message("  Train rows: ", nrow(train_df), " (complete: ", sum(train_complete), ")")
message("  Val rows:   ", nrow(val_df),   " (complete: ", sum(val_complete), ")")
message("  Test rows:  ", nrow(test_df),  " (complete: ", sum(test_complete), ")")

train_df <- train_df[train_complete, , drop = FALSE]
val_df   <- val_df[val_complete, , drop = FALSE]
test_df  <- test_df[test_complete, , drop = FALSE]

# Keep date vectors aligned with complete-case subsets for later use
val_dates  <- val_raw$date[val_complete]
test_dates <- test_raw$date[test_complete]

train_prevalence <- mean(train_df$outage_3h_or_more, na.rm = TRUE)
val_prevalence   <- mean(val_df$outage_3h_or_more, na.rm = TRUE)
test_prevalence  <- mean(test_df$outage_3h_or_more, na.rm = TRUE)

message("  Train prevalence: ", sprintf("%.4f", train_prevalence),
        " (", sum(train_df$outage_3h_or_more), " positives)")
message("  Val prevalence:   ", sprintf("%.4f", val_prevalence))
message("  Test prevalence:  ", sprintf("%.4f", test_prevalence))

# STEP 3: FIT LOGISTIC REGRESSION

message("\n", strrep("-", 60))
message("STEP 3: Fitting logistic regression")
message(strrep("-", 60))

y_train <- train_df$outage_3h_or_more
X_train <- train_df %>% select(all_of(feature_cols))

# Remove zero-variance columns (glm will fail on them)
col_vars <- sapply(X_train, var, na.rm = TRUE)
zv_cols <- names(col_vars[col_vars == 0 | is.na(col_vars)])
if (length(zv_cols) > 0) {
  message("  Removing ", length(zv_cols), " zero-variance columns: ",
          paste(head(zv_cols, 5), collapse = ", "),
          if (length(zv_cols) > 5) "..." else "")
  feature_cols_use <- setdiff(feature_cols, zv_cols)
  X_train <- X_train %>% select(all_of(feature_cols_use))
} else {
  feature_cols_use <- feature_cols
}

message("  Features used after ZV removal: ", length(feature_cols_use))

# Primary attempt: standard glm
fit_success <- FALSE
glm_model <- NULL
ridge_model <- NULL
model_type <- "glm"

train_fit_df <- bind_cols(tibble(outage_3h_or_more = y_train), X_train)

message("  Attempting standard glm(family=binomial)...")
t_start <- proc.time()

glm_result <- tryCatch({
  fit <- glm(
    outage_3h_or_more ~ .,
    family = binomial(link = "logit"),
    data = train_fit_df,
    control = glm.control(maxit = 100)
  )

  if (!fit$converged) {
    message("  WARNING: glm did not converge. Falling back to ridge regression.")
    NULL
  } else {
    fit
  }
}, warning = function(w) {
  message("  WARNING during glm fit: ", conditionMessage(w))
  # Still try to return the fit if it exists
  tryCatch(
    suppressWarnings(glm(
      outage_3h_or_more ~ .,
      family = binomial(link = "logit"),
      data = train_fit_df,
      control = glm.control(maxit = 100)
    )),
    error = function(e) NULL
  )
}, error = function(e) {
  message("  ERROR during glm fit: ", conditionMessage(e))
  NULL
})

if (!is.null(glm_result) && glm_result$converged) {
  glm_model <- glm_result
  fit_success <- TRUE
  model_type <- "glm"
  t_elapsed <- (proc.time() - t_start)[["elapsed"]]
  message("  glm converged in ", sprintf("%.1f", t_elapsed), "s")
  message("  Coefficients: ", length(coef(glm_model)), " (including intercept)")
  n_na_coef <- sum(is.na(coef(glm_model)))
  if (n_na_coef > 0) {
    message("  WARNING: ", n_na_coef, " coefficients are NA (aliased/collinear features)")
  }
}

# Fallback: ridge regression via glmnet
if (!fit_success) {
  message("  Falling back to ridge regression (glmnet, alpha=0)...")
  t_start <- proc.time()

  X_mat <- as.matrix(X_train)

  cv_fit <- tryCatch({
    cv.glmnet(
      x = X_mat,
      y = y_train,
      family = "binomial",
      alpha = 0,
      nfolds = 5,
      type.measure = "deviance"
    )
  }, error = function(e) {
    stop("Both glm and glmnet failed. glmnet error: ", conditionMessage(e))
  })

  ridge_model <- cv_fit
  model_type <- "ridge"
  fit_success <- TRUE
  t_elapsed <- (proc.time() - t_start)[["elapsed"]]
  message("  Ridge CV completed in ", sprintf("%.1f", t_elapsed), "s")
  message("  Best lambda: ", sprintf("%.6f", cv_fit$lambda.min))
  message("  Lambda 1SE:  ", sprintf("%.6f", cv_fit$lambda.1se))
  message("  Using lambda.min for predictions")
}

stopifnot(fit_success)

# STEP 4: PREDICT ON VAL AND TEST

message("\n", strrep("-", 60))
message("STEP 4: Generating predictions")
message(strrep("-", 60))

if (model_type == "glm") {
  val_prob_raw <- predict(glm_model,
                          newdata = val_df %>% select(all_of(feature_cols_use)),
                          type = "response")
  test_prob_raw <- predict(glm_model,
                           newdata = test_df %>% select(all_of(feature_cols_use)),
                           type = "response")
} else {
  # Ridge (glmnet)
  val_prob_raw <- as.numeric(predict(ridge_model,
                                     newx = as.matrix(val_df %>% select(all_of(feature_cols_use))),
                                     s = "lambda.min",
                                     type = "response"))
  test_prob_raw <- as.numeric(predict(ridge_model,
                                      newx = as.matrix(test_df %>% select(all_of(feature_cols_use))),
                                      s = "lambda.min",
                                      type = "response"))
}

# Clamp probabilities to (0, 1)
eps <- 1e-15
val_prob_raw  <- pmin(pmax(val_prob_raw, eps), 1 - eps)
test_prob_raw <- pmin(pmax(test_prob_raw, eps), 1 - eps)

message("  Val  predictions: n=", length(val_prob_raw),
        ", mean=", sprintf("%.6f", mean(val_prob_raw)),
        ", median=", sprintf("%.6f", median(val_prob_raw)))
message("  Test predictions: n=", length(test_prob_raw),
        ", mean=", sprintf("%.6f", mean(test_prob_raw)),
        ", median=", sprintf("%.6f", median(test_prob_raw)))

# STEP 5: PLATT CALIBRATION (strict separation)

message("\n", strrep("-", 60))
message("STEP 5: Platt calibration (strict val split)")
message(strrep("-", 60))

# Strict separation: first half of val for calibration, second half for threshold
val_unique_dates <- sort(unique(val_dates))
n_val_dates <- length(val_unique_dates)
n_calib_dates <- floor(n_val_dates * 0.5)

calib_fit_dates <- val_unique_dates[1:n_calib_dates]
threshold_select_dates <- val_unique_dates[(n_calib_dates + 1):n_val_dates]

calib_mask     <- val_dates %in% calib_fit_dates
threshold_mask <- val_dates %in% threshold_select_dates

val_y <- val_df$outage_3h_or_more

message("  Calibration-fit split: ", sum(calib_mask), " rows (",
        length(calib_fit_dates), " days: ",
        min(calib_fit_dates), " to ", max(calib_fit_dates), ")")
message("  Threshold-select split: ", sum(threshold_mask), " rows (",
        length(threshold_select_dates), " days: ",
        min(threshold_select_dates), " to ", max(threshold_select_dates), ")")

# Fit Platt on calibration-fit subset
calib_y    <- val_y[calib_mask]
calib_prob <- val_prob_raw[calib_mask]
logit_calib <- qlogis(pmin(pmax(calib_prob, eps), 1 - eps))

platt_fit <- tryCatch({
  glm(calib_y ~ logit_calib, family = binomial(link = "logit"))
}, error = function(e) {
  warning("Platt calibration failed: ", conditionMessage(e))
  NULL
})

if (!is.null(platt_fit)) {
  platt_intercept <- as.numeric(coef(platt_fit)[1])
  platt_slope     <- as.numeric(coef(platt_fit)[2])
  message("  Platt intercept: ", sprintf("%.6f", platt_intercept))
  message("  Platt slope:     ", sprintf("%.6f", platt_slope))

  # Apply Platt to all probabilities
  val_prob_platt  <- apply_platt_with_params(val_prob_raw, platt_intercept, platt_slope)
  test_prob_platt <- apply_platt_with_params(test_prob_raw, platt_intercept, platt_slope)
} else {
  message("  Platt calibration failed. Using raw probabilities.")
  platt_intercept <- NA_real_
  platt_slope     <- NA_real_
  val_prob_platt  <- val_prob_raw
  test_prob_platt <- test_prob_raw
}

# STEP 6: THRESHOLD SELECTION (on threshold-select half of val)

message("\n", strrep("-", 60))
message("STEP 6: Threshold selection (F1 optimization)")
message(strrep("-", 60))

threshold_y    <- val_y[threshold_mask]
threshold_prob <- val_prob_platt[threshold_mask]
threshold_dates_sub <- val_dates[threshold_mask]
n_threshold_days <- length(unique(threshold_dates_sub))

threshold_grid <- unique(round(c(
  seq(0.0001, 0.01,  by = 0.0001),
  seq(0.01,   0.10,  by = 0.001),
  seq(0.10,   0.90,  by = 0.01),
  seq(0.90,   0.99,  by = 0.005),
  seq(0.99,   0.999, by = 0.001)
), 6))

sweep_results <- lapply(threshold_grid, function(t) {
  pred <- as.integer(threshold_prob >= t)

  tp <- sum(threshold_y == 1L & pred == 1L, na.rm = TRUE)
  fp <- sum(threshold_y == 0L & pred == 1L, na.rm = TRUE)
  tn <- sum(threshold_y == 0L & pred == 0L, na.rm = TRUE)
  fn <- sum(threshold_y == 1L & pred == 0L, na.rm = TRUE)

  n_alerts <- tp + fp
  precision <- if (n_alerts > 0) tp / n_alerts else NA_real_
  recall    <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  f1 <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
    2 * precision * recall / (precision + recall)
  } else NA_real_

  tibble(
    threshold = t,
    tp = tp, fp = fp, tn = tn, fn = fn,
    n_alerts = n_alerts,
    alerts_per_day = n_alerts / n_threshold_days,
    precision = precision,
    recall = recall,
    f1 = f1,
    specificity = if ((tn + fp) > 0) tn / (tn + fp) else NA_real_,
    accuracy = (tp + tn) / (tp + fp + tn + fn),
    kappa = kappa_from_counts(tp, fp, tn, fn)
  )
})
sweep_df <- bind_rows(sweep_results)

best_idx <- which.max(sweep_df$f1)
best_threshold <- sweep_df$threshold[best_idx]
best_f1_val <- sweep_df$f1[best_idx]

message("  Thresholds evaluated: ", nrow(sweep_df))
message("  Best threshold: ", sprintf("%.6f", best_threshold))
message("  Best F1 (val threshold-select): ", sprintf("%.4f", best_f1_val))

# STEP 7: EVALUATION ON TEST

message("\n", strrep("-", 60))
message("STEP 7: Test set evaluation")
message(strrep("-", 60))

test_y <- test_df$outage_3h_or_more
n_test_days <- length(unique(test_dates))

# Discrimination metrics (using raw probabilities, same as XGBoost)
truth_fac <- factor(if_else(test_y == 1L, "1", "0"), levels = c("1", "0"))

roc_auc_test <- roc_auc_vec(truth = truth_fac, estimate = test_prob_raw, event_level = "first")
pr_auc_test  <- pr_auc_vec(truth = truth_fac, estimate = test_prob_raw, event_level = "first")

message("  ROC-AUC (raw): ", sprintf("%.4f", roc_auc_test))
message("  PR-AUC  (raw): ", sprintf("%.4f", pr_auc_test))

# Also compute discrimination on Platt-calibrated probabilities
roc_auc_test_platt <- roc_auc_vec(truth = truth_fac, estimate = test_prob_platt, event_level = "first")
pr_auc_test_platt  <- pr_auc_vec(truth = truth_fac, estimate = test_prob_platt, event_level = "first")

message("  ROC-AUC (platt): ", sprintf("%.4f", roc_auc_test_platt))
message("  PR-AUC  (platt): ", sprintf("%.4f", pr_auc_test_platt))

# Calibration metrics (on Platt-calibrated probabilities) 
brier_test   <- compute_brier(test_y, test_prob_platt)
logloss_test <- compute_logloss(test_y, test_prob_platt)
ece_test     <- compute_ece_equal(test_y, test_prob_platt, n_bins = 10)

message("  Brier score (platt): ", sprintf("%.6f", brier_test))
message("  Log-loss    (platt): ", sprintf("%.6f", logloss_test))
message("  ECE equal   (platt): ", sprintf("%.6f", ece_test))

# Threshold-based metrics 
test_pred <- as.integer(test_prob_platt >= best_threshold)
tp <- sum(test_y == 1L & test_pred == 1L, na.rm = TRUE)
fp <- sum(test_y == 0L & test_pred == 1L, na.rm = TRUE)
tn <- sum(test_y == 0L & test_pred == 0L, na.rm = TRUE)
fn <- sum(test_y == 1L & test_pred == 0L, na.rm = TRUE)

test_precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
test_recall    <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
test_f1 <- if (!is.na(test_precision) && !is.na(test_recall) &&
               (test_precision + test_recall) > 0) {
  2 * test_precision * test_recall / (test_precision + test_recall)
} else NA_real_
test_kappa <- kappa_from_counts(tp, fp, tn, fn)
test_alerts_per_day <- (tp + fp) / n_test_days

message("  Threshold: ", sprintf("%.6f", best_threshold))
message("  Precision: ", sprintf("%.4f", test_precision %||% 0))
message("  Recall:    ", sprintf("%.4f", test_recall %||% 0))
message("  F1:        ", sprintf("%.4f", test_f1 %||% 0))
message("  Kappa:     ", sprintf("%.4f", test_kappa %||% 0))
message("  Alerts/day:", sprintf("%.2f", test_alerts_per_day))

# STEP 8: ROC AND PR CURVE DATA

message("\n", strrep("-", 60))
message("STEP 8: Computing ROC and PR curves")
message(strrep("-", 60))

# Build predictions tibble for yardstick
test_pred_tbl <- tibble(
  truth = truth_fac,
  prob  = test_prob_raw
)

roc_data <- roc_curve(test_pred_tbl, truth, prob, event_level = "first") %>%
  mutate(split = "test")
pr_data <- pr_curve(test_pred_tbl, truth, prob, event_level = "first") %>%
  mutate(split = "test")

message("  ROC curve points: ", nrow(roc_data))
message("  PR curve points:  ", nrow(pr_data))

# STEP 9: CALIBRATION BINS

message("\n", strrep("-", 60))
message("STEP 9: Calibration bins (Platt-calibrated)")
message(strrep("-", 60))

cal_bins <- compute_calibration_bins_equal(test_y, test_prob_platt, n_bins = 10) %>%
  mutate(split = "test_platt_scaled")

message("  Calibration bins computed (10 equal-width)")

# STEP 10: DECISION TABLE (deployment format)

message("\n", strrep("-", 60))
message("STEP 10: Decision table")
message(strrep("-", 60))

decision_table <- tibble(
  method = "platt",
  role = "default",
  policy_type = "f1_optimized",
  policy_target = NA_character_,
  threshold = best_threshold,
  precision = test_precision,
  recall = test_recall,
  f1 = test_f1,
  alerts_per_day = test_alerts_per_day,
  n_test_rows = nrow(test_df),
  n_test_days = n_test_days,
  deployment_ready = !is.na(test_precision) && test_precision > 0
)

message("  Decision table built")

# STEP 11: XGBOOST COMPARISON TABLE

message("\n", strrep("-", 60))
message("STEP 11: XGBoost comparison")
message(strrep("-", 60))

xgb_config <- read_json(xgb_config_path)

# Extract XGBoost results from run_config.json
xgb_results <- xgb_config$results
xgb_roc_auc <- xgb_results$roc_auc_test %||% NA_real_
xgb_pr_auc  <- xgb_results$pr_auc_test %||% NA_real_
xgb_brier   <- xgb_results$brier_test_platt %||% (xgb_results$brier_test %||% NA_real_)
xgb_logloss <- xgb_results$logloss_test_platt %||% (xgb_results$logloss_test %||% NA_real_)
xgb_ece     <- xgb_results$ece_equal_test_platt %||% (xgb_results$ece_equal_test %||% NA_real_)
xgb_f1      <- xgb_results$f1_test_at_best_t %||% NA_real_

comparison <- tibble(
  metric = c("roc_auc", "pr_auc", "brier_score", "logloss", "ece_equal",
             "f1", "precision", "recall", "alerts_per_day"),
  xgboost_value = c(
    as.numeric(xgb_roc_auc),
    as.numeric(xgb_pr_auc),
    as.numeric(xgb_brier),
    as.numeric(xgb_logloss),
    as.numeric(xgb_ece),
    as.numeric(xgb_f1),
    NA_real_,
    NA_real_,
    NA_real_
  ),
  logistic_value = c(
    roc_auc_test,
    pr_auc_test,
    brier_test,
    logloss_test,
    ece_test,
    test_f1,
    test_precision,
    test_recall,
    test_alerts_per_day
  )
) %>%
  mutate(
    delta = logistic_value - xgboost_value
  )

# Try to fill in XGBoost precision/recall/alerts from deployment decision table
xgb_deploy_path <- file.path(xgb_run_dir, "deployment_decision_table.csv")
if (file.exists(xgb_deploy_path)) {
  xgb_deploy <- read_csv(xgb_deploy_path, show_col_types = FALSE)
  xgb_default <- xgb_deploy %>% filter(role == "default", method == "platt") %>% slice(1)
  if (nrow(xgb_default) > 0) {
    comparison$xgboost_value[comparison$metric == "precision"] <- as.numeric(xgb_default$precision)
    comparison$xgboost_value[comparison$metric == "recall"]    <- as.numeric(xgb_default$recall)
    comparison$xgboost_value[comparison$metric == "f1"]        <- as.numeric(xgb_default$f1)
    comparison$xgboost_value[comparison$metric == "alerts_per_day"] <- as.numeric(xgb_default$alerts_per_day)
    comparison <- comparison %>% mutate(delta = logistic_value - xgboost_value)
  }
}

message("  Comparison table built (", nrow(comparison), " metrics)")
for (i in seq_len(nrow(comparison))) {
  r <- comparison[i, ]
  delta_str <- if (!is.na(r$delta)) sprintf("%+.4f", r$delta) else "NA"
  message("    ", r$metric, ": XGB=",
          if (!is.na(r$xgboost_value)) sprintf("%.4f", r$xgboost_value) else "NA",
          " | LR=", sprintf("%.4f", r$logistic_value),
          " | delta=", delta_str)
}

# STEP 12: SAVE ALL OUTPUTS

message("\n", strrep("-", 60))
message("STEP 12: Saving outputs")
message(strrep("-", 60))

# 1. Metrics summary
metrics_summary <- tibble(
  split = c("validation", "test"),
  n_rows = c(nrow(val_df), nrow(test_df)),
  n_days = c(length(unique(val_dates)), n_test_days),
  n_positive = c(sum(val_y), sum(test_y)),
  prevalence = c(val_prevalence, test_prevalence),
  roc_auc = c(NA_real_, roc_auc_test),
  pr_auc = c(NA_real_, pr_auc_test),
  brier_score = c(NA_real_, brier_test),
  logloss = c(NA_real_, logloss_test),
  ece_equal = c(NA_real_, ece_test),
  best_threshold = c(best_threshold, best_threshold),
  f1_at_best_t = c(best_f1_val, test_f1),
  precision_at_best_t = c(NA_real_, test_precision),
  recall_at_best_t = c(NA_real_, test_recall),
  kappa_at_best_t = c(NA_real_, test_kappa),
  alerts_per_day_at_best_t = c(NA_real_, test_alerts_per_day)
)
write_csv(metrics_summary, file.path(logistic_out_dir, "metrics_summary.csv"))
message("  Saved: metrics_summary.csv")

# 2. ROC curve
write_csv(roc_data, file.path(logistic_out_dir, "roc_curve.csv"))
message("  Saved: roc_curve.csv")

# 3. PR curve
write_csv(pr_data, file.path(logistic_out_dir, "pr_curve.csv"))
message("  Saved: pr_curve.csv")

# 4. Calibration bins
write_csv(cal_bins, file.path(logistic_out_dir, "calibration_bins_platt_equal.csv"))
message("  Saved: calibration_bins_platt_equal.csv")

# 5. Platt calibration params
platt_params_out <- tibble(
  split = "test_platt_scaled",
  platt_intercept = platt_intercept,
  platt_slope = platt_slope,
  brier_score = brier_test,
  logloss = logloss_test,
  ece_equal = ece_test
)
write_csv(platt_params_out, file.path(logistic_out_dir, "platt_calibration.csv"))
message("  Saved: platt_calibration.csv")

# 6. Comparison table
write_csv(comparison, file.path(logistic_out_dir, "logistic_vs_xgboost_comparison.csv"))
message("  Saved: logistic_vs_xgboost_comparison.csv")

# 7. Run config
run_config <- list(
  run_id = logistic_run_id,
  xgb_reference_run_id = xgb_run_id,
  model_type = model_type,
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  seed = 42,

  data = list(
    features_path = features_path,
    splits_fixed_path = splits_path,
    train_rows = nrow(train_df),
    val_rows = nrow(val_df),
    test_rows = nrow(test_df),
    n_features_original = length(feature_cols),
    n_features_used = length(feature_cols_use),
    n_zv_removed = length(zv_cols),
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
    type = model_type,
    family = "binomial",
    link = "logit",
    weighting = "none",
    ridge_lambda = if (model_type == "ridge") ridge_model$lambda.min else NULL
  ),

  calibration = list(
    method = "platt",
    strict_separation = TRUE,
    strict_split_ratio = 0.5,
    calib_fit_dates_range = c(as.character(min(calib_fit_dates)), as.character(max(calib_fit_dates))),
    calib_fit_n_days = length(calib_fit_dates),
    calib_fit_n_rows = sum(calib_mask),
    threshold_select_dates_range = c(as.character(min(threshold_select_dates)), as.character(max(threshold_select_dates))),
    threshold_select_n_days = length(threshold_select_dates),
    threshold_select_n_rows = sum(threshold_mask),
    platt_intercept = platt_intercept,
    platt_slope = platt_slope
  ),

  threshold = list(
    best_threshold = best_threshold,
    selected_on = "validation_threshold_select_half",
    metric_optimized = "f1",
    threshold_grid_size = length(threshold_grid),
    threshold_grid_min = min(threshold_grid),
    threshold_grid_max = max(threshold_grid)
  ),

  results = list(
    roc_auc_test = roc_auc_test,
    pr_auc_test = pr_auc_test,
    roc_auc_test_platt = roc_auc_test_platt,
    pr_auc_test_platt = pr_auc_test_platt,
    brier_test = brier_test,
    logloss_test = logloss_test,
    ece_equal_test = ece_test,
    f1_test = test_f1,
    precision_test = test_precision,
    recall_test = test_recall,
    kappa_test = test_kappa,
    alerts_per_day_test = test_alerts_per_day
  )
)

write_json(run_config, file.path(logistic_out_dir, "run_config.json"), pretty = TRUE, auto_unbox = TRUE)
message("  Saved: run_config.json")

# 8. Feature list used
writeLines(feature_cols_use, file.path(logistic_out_dir, "features_used.txt"))
message("  Saved: features_used.txt")

# 9. Decision table
write_csv(decision_table, file.path(logistic_out_dir, "deployment_decision_table.csv"))
message("  Saved: deployment_decision_table.csv")

# FINAL SUMMARY

message("\n", strrep("=", 70))
message("LOGISTIC BASELINE COMPLETE")
message(strrep("=", 70))
message("Run ID:         ", logistic_run_id)
message("Model type:     ", model_type)
message("Output dir:     ", logistic_out_dir)
message("XGBoost ref:    ", xgb_run_id)
message("")
message("Test Results:")
message("  ROC-AUC:      ", sprintf("%.4f", roc_auc_test))
message("  PR-AUC:       ", sprintf("%.4f", pr_auc_test))
message("  Brier:        ", sprintf("%.6f", brier_test))
message("  Log-loss:     ", sprintf("%.6f", logloss_test))
message("  ECE:          ", sprintf("%.6f", ece_test))
message("  Threshold:    ", sprintf("%.6f", best_threshold))
message("  Precision:    ", sprintf("%.4f", test_precision %||% 0))
message("  Recall:       ", sprintf("%.4f", test_recall %||% 0))
message("  F1:           ", sprintf("%.4f", test_f1 %||% 0))
message("  Alerts/day:   ", sprintf("%.2f", test_alerts_per_day))
message("")
message("Outputs saved:")
message("  - metrics_summary.csv")
message("  - roc_curve.csv")
message("  - pr_curve.csv")
message("  - calibration_bins_platt_equal.csv")
message("  - platt_calibration.csv")
message("  - logistic_vs_xgboost_comparison.csv")
message("  - deployment_decision_table.csv")
message("  - run_config.json")
message("  - features_used.txt")
message(strrep("=", 70))
