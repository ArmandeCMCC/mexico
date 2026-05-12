# 14_two_part_count_severity.R

# Two-part decomposition of expected outage burden:
#   E[count]   = P(outage) x E[n_outages | outage=1]
#   E[minutes] = P(outage) x E[total_length_min | outage=1]
#
# Section 0: Load artifacts from strict run
# Section 1: Rebuild P(outage) identically to benchmark
# Section 2a: Count model (Poisson & NB GLM on positives)
# Section 2b: Severity model (log-normal OLS on positives)
# Section 3: Decomposed expected burden (all rows)
#
# Usage:
#   Rscript scripts/14_two_part_count_severity.R [run_id]

suppressPackageStartupMessages({
  library(MASS)
  library(dplyr)
  library(readr)
  library(tibble)
  library(tidyr)
  library(tidymodels)
  library(jsonlite)
})

set.seed(42)

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0 || is.na(x)) y else x

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (dir.exists("data") || dir.exists("scripts")) return(normalizePath(getwd()))
  normalizePath(getwd())
}

get_nthread <- function() {
  override <- suppressWarnings(as.integer(Sys.getenv("XGB_NTHREAD", "")))
  if (!is.na(override) && override >= 1) return(as.integer(override))
  n <- suppressWarnings(parallel::detectCores())
  if (is.na(n) || n < 2) return(1L)
  max(1L, as.integer(n - 1))
}

apply_platt_with_params <- function(prob, intercept, slope, eps = 1e-15) {
  p <- pmin(pmax(prob, eps), 1 - eps)
  plogis(intercept + slope * qlogis(p))
}

prep_model_df <- function(df, model_features) {
  df %>%
    select(all_of(c("outage_3h_or_more", model_features))) %>%
    mutate(
      outage_3h_or_more = factor(as.character(outage_3h_or_more), levels = c("1", "0"))
    ) %>%
    mutate(across(where(is.logical), as.integer))
}

# SECTION 0: LOAD ARTIFACTS FROM STRICT RUN

args <- commandArgs(trailingOnly = TRUE)
run_id <- if (length(args) >= 1 && nzchar(args[1])) args[1] else "20260223_125430_forecast_strict"

project_dir <- detect_project_dir()
run_dir <- file.path(project_dir, "data", "baselines", "binary_threshold", run_id)
out_dir <- file.path(run_dir, "two_part_extension")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

features_path <- file.path(project_dir, "data", "model_ready", "features_engineered.rds")
splits_path   <- file.path(project_dir, "data", "model_ready", "splits_fixed.rds")
panel_path    <- file.path(project_dir, "data", "panel_mex_2017_2021_ntl_ghs.rds")

run_config_path    <- file.path(run_dir, "run_config.json")
features_used_path <- file.path(run_dir, "features_used.txt")
platt_path         <- file.path(run_dir, "platt_calibration.csv")

# Optional path overrides
features_override <- Sys.getenv("FEATURES_PATH_OVERRIDE", "")
splits_override   <- Sys.getenv("SPLITS_PATH_OVERRIDE", "")
panel_override    <- Sys.getenv("PANEL_PATH_OVERRIDE", "")
if (nzchar(features_override)) features_path <- features_override
if (nzchar(splits_override))   splits_path   <- splits_override
if (nzchar(panel_override))    panel_path    <- panel_override

message(strrep("=", 70))
message("14_two_part_count_severity.R")
message(strrep("=", 70))
message("Run ID:       ", run_id)
message("Output dir:   ", out_dir)
message("Features:     ", features_path)
message("Splits:       ", splits_path)
message("Panel:        ", panel_path)
message("Run config:   ", run_config_path)

# Check required files
required_files <- c(run_config_path, features_used_path, platt_path,
                    features_path, splits_path, panel_path)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop("Missing required files:\n", paste0("  - ", missing_files, collapse = "\n"))
}

# Load run config 
run_config <- jsonlite::read_json(run_config_path)
xgb_params <- run_config$model$params
message("Loaded XGBoost params: tree_depth=", xgb_params$tree_depth,
        ", trees=", xgb_params$trees, ", learn_rate=", xgb_params$learn_rate)

# Load feature list 
feature_cols <- read_lines(features_used_path)
feature_cols <- feature_cols[nzchar(feature_cols)]
feature_cols <- unique(feature_cols)
message("Features loaded: ", length(feature_cols))

# Load Platt calibration params
platt_tbl <- read_csv(platt_path, show_col_types = FALSE) %>% slice(1)
platt_intercept <- as.numeric(platt_tbl$platt_intercept)
platt_slope     <- as.numeric(platt_tbl$platt_slope)
if (!is.finite(platt_intercept) || !is.finite(platt_slope)) {
  stop("Invalid Platt parameters in platt_calibration.csv")
}
message("Platt params: intercept=", sprintf("%.6f", platt_intercept),
        ", slope=", sprintf("%.6f", platt_slope))

# Load features and splits
message("\nLoading features_engineered.rds ...")
features <- readRDS(features_path)
splits_fixed <- readRDS(splits_path)

train_start <- as.Date(splits_fixed$train_range[1])
train_end   <- as.Date(splits_fixed$train_range[2])
val_start   <- as.Date(splits_fixed$val_range[1])
val_end     <- as.Date(splits_fixed$val_range[2])
test_start  <- as.Date(splits_fixed$test_range[1])
test_end    <- as.Date(splits_fixed$test_range[2])

message("Splits: train ", train_start, " to ", train_end,
        " | val ", val_start, " to ", val_end,
        " | test ", test_start, " to ", test_end)

features <- features %>%
  mutate(
    date  = as.Date(date),
    GID_2 = as.character(GID_2),
    split = case_when(
      date >= train_start & date <= train_end ~ "train",
      date >= val_start   & date <= val_end   ~ "val",
      date >= test_start  & date <= test_end  ~ "test",
      TRUE ~ "other"
    )
  )

# Join total_length_min from panel (not in features_engineered) 
message("Loading panel for total_length_min ...")
panel <- readRDS(panel_path)

panel_severity <- panel %>%
  transmute(
    GID_2 = as.character(GID_2),
    date  = as.Date(date),
    total_length_min_panel = as.numeric(total_length_min)
  ) %>%
  distinct(GID_2, date, .keep_all = TRUE)

features <- features %>%
  left_join(panel_severity, by = c("GID_2", "date"))

n_joined <- sum(!is.na(features$total_length_min_panel))
message("Joined total_length_min for ", format(n_joined, big.mark = ","),
        " / ", format(nrow(features), big.mark = ","), " rows")

# Verify missing features
missing_feats <- setdiff(feature_cols, names(features))
if (length(missing_feats) > 0) {
  stop("Missing features from features_engineered.rds:\n",
       paste0("  - ", head(missing_feats, 30), collapse = "\n"))
}

# Split data
train_data <- features %>% filter(split == "train")
test_data  <- features %>% filter(split == "test")

message("Train rows: ", format(nrow(train_data), big.mark = ","),
        " | Test rows: ", format(nrow(test_data), big.mark = ","))

# SECTION 1: REBUILD P(outage) IDENTICALLY TO BENCHMARK

message("\n", strrep("=", 70))
message("SECTION 1: Rebuilding P(outage) from benchmark XGBoost")
message(strrep("=", 70))

train_model <- prep_model_df(train_data, feature_cols)
test_model  <- prep_model_df(test_data, feature_cols)

n_pos <- sum(train_model$outage_3h_or_more == "1", na.rm = TRUE)
n_neg <- sum(train_model$outage_3h_or_more == "0", na.rm = TRUE)
scale_pos_weight <- n_neg / max(1, n_pos)
message("Training: n_pos=", format(n_pos, big.mark = ","),
        ", n_neg=", format(n_neg, big.mark = ","),
        ", scale_pos_weight=", sprintf("%.2f", scale_pos_weight))

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

message("Fitting XGBoost (trees=", xgb_params$trees, ", depth=", xgb_params$tree_depth, ") ...")
t0 <- Sys.time()
xgb_fit <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(xgb_spec) %>%
  fit(data = train_model)
t1 <- Sys.time()
message("XGBoost training completed in ", sprintf("%.1f", difftime(t1, t0, units = "mins")), " minutes")

# Predict on test 
test_prob_raw <- predict(xgb_fit, new_data = test_model, type = "prob") %>%
  pull(.pred_1)

# Apply Platt scaling
test_prob_calibrated <- apply_platt_with_params(test_prob_raw, platt_intercept, platt_slope)

# Sanity check: ROC-AUC 
truth_fac <- factor(
  if_else(as.integer(as.character(test_data$outage_3h_or_more)) == 1L, "1", "0"),
  levels = c("1", "0")
)

test_roc_auc <- yardstick::roc_auc_vec(
  truth    = truth_fac,
  estimate = test_prob_raw,
  event_level = "first"
)

benchmark_roc <- 0.879
roc_diff <- abs(test_roc_auc - benchmark_roc)

if (roc_diff <= 0.005) {
  message("Sanity check PASSED: test ROC-AUC = ", sprintf("%.4f", test_roc_auc),
          " (benchmark = ", benchmark_roc, ", diff = ", sprintf("%.4f", roc_diff), ")")
} else {
  message("WARNING: test ROC-AUC = ", sprintf("%.4f", test_roc_auc),
          " differs from benchmark ", benchmark_roc,
          " by ", sprintf("%.4f", roc_diff),
          ". Possible nthread nondeterminism. Proceeding anyway.")
}

# Extract feature importance for GLM fallback 
xgb_engine <- extract_fit_engine(xgb_fit)
importance_df <- xgboost::xgb.importance(model = xgb_engine) %>%
  as_tibble() %>%
  arrange(desc(Gain))

top20_features <- head(importance_df$Feature, 20)
message("Top-20 features by XGBoost gain (GLM fallback): ",
        paste(head(top20_features, 5), collapse = ", "), " ...")

# SECTION 2a: COUNT MODEL (Poisson & NB GLM on positives)

message("\n", strrep("=", 70))
message("SECTION 2a: Count model (n_outages | outage=1)")
message(strrep("=", 70))

# Subset to positives only
train_pos <- train_data %>%
  filter(as.integer(outage_3h_or_more) == 1L | as.character(outage_3h_or_more) == "1")
test_pos <- test_data %>%
  filter(as.integer(outage_3h_or_more) == 1L | as.character(outage_3h_or_more) == "1")

message("Positives: train=", format(nrow(train_pos), big.mark = ","),
        ", test=", format(nrow(test_pos), big.mark = ","))

# Prepare GLM data frames (response + features only)
prep_glm_df <- function(df, response_col, feat_cols) {
  out <- df %>%
    select(all_of(c(response_col, feat_cols))) %>%
    mutate(across(where(is.logical), as.integer)) %>%
    drop_na()
  out
}

# Try full feature set first, fall back to top-20 if convergence fails
use_fallback <- FALSE

train_count_df <- prep_glm_df(train_pos, "n_outages", feature_cols)
test_count_df  <- prep_glm_df(test_pos, "n_outages", feature_cols)

message("Count model training data: ", nrow(train_count_df), " rows, ",
        length(feature_cols), " features")
message("Count response (train): mean=", sprintf("%.2f", mean(train_count_df$n_outages)),
        ", var=", sprintf("%.2f", var(train_count_df$n_outages)),
        ", max=", max(train_count_df$n_outages))

# Poisson GLM 
message("\nFitting Poisson GLM ...")
poisson_fit <- tryCatch({
  glm(n_outages ~ ., family = poisson, data = train_count_df)
}, warning = function(w) {
  message("  Poisson GLM warning: ", conditionMessage(w))
  suppressWarnings(glm(n_outages ~ ., family = poisson, data = train_count_df))
}, error = function(e) {
  message("  Poisson GLM FAILED with ", length(feature_cols),
          " features: ", conditionMessage(e))
  message("  Falling back to top-20 features ...")
  use_fallback <<- TRUE
  NULL
})

if (is.null(poisson_fit) || use_fallback) {
  use_fallback <- TRUE
  fallback_feats <- intersect(top20_features, feature_cols)
  message("  Using ", length(fallback_feats), " features for GLMs")
  train_count_df <- prep_glm_df(train_pos, "n_outages", fallback_feats)
  test_count_df  <- prep_glm_df(test_pos, "n_outages", fallback_feats)

  poisson_fit <- tryCatch({
    glm(n_outages ~ ., family = poisson, data = train_count_df)
  }, error = function(e) {
    stop("Poisson GLM failed even with top-20 features: ", conditionMessage(e))
  })
}

message("  Poisson AIC: ", sprintf("%.1f", AIC(poisson_fit)))

# Negative Binomial GLM 
message("Fitting Negative Binomial GLM ...")
nb_fit <- tryCatch({
  MASS::glm.nb(n_outages ~ ., data = train_count_df)
}, warning = function(w) {
  message("  NB GLM warning: ", conditionMessage(w))
  suppressWarnings(MASS::glm.nb(n_outages ~ ., data = train_count_df))
}, error = function(e) {
  message("  NB GLM FAILED: ", conditionMessage(e))
  if (!use_fallback) {
    message("  Falling back to top-20 features for NB ...")
    fallback_feats <- intersect(top20_features, feature_cols)
    train_count_fb <- prep_glm_df(train_pos, "n_outages", fallback_feats)
    tryCatch(
      MASS::glm.nb(n_outages ~ ., data = train_count_fb),
      error = function(e2) {
        message("  NB GLM failed even with top-20: ", conditionMessage(e2))
        NULL
      }
    )
  } else {
    NULL
  }
})

if (!is.null(nb_fit)) {
  message("  NB AIC: ", sprintf("%.1f", AIC(nb_fit)),
          ", theta: ", sprintf("%.4f", nb_fit$theta))
}

# Evaluate count models on test positives 
pred_poisson_test <- predict(poisson_fit, newdata = test_count_df, type = "response")
actual_count_test <- test_count_df$n_outages

overdispersion_test <- var(actual_count_test) / mean(actual_count_test)

count_eval_poisson <- tibble(
  model       = "poisson",
  AIC         = AIC(poisson_fit),
  overdispersion_var_over_mean = overdispersion_test,
  MAE         = mean(abs(pred_poisson_test - actual_count_test)),
  RMSE        = sqrt(mean((pred_poisson_test - actual_count_test)^2)),
  correlation = cor(pred_poisson_test, actual_count_test, use = "complete.obs"),
  n_train     = nrow(train_count_df),
  n_test      = nrow(test_count_df),
  n_features  = ncol(train_count_df) - 1L,
  used_fallback_features = use_fallback
)

count_eval <- count_eval_poisson

if (!is.null(nb_fit)) {
  pred_nb_test <- predict(nb_fit, newdata = test_count_df, type = "response")
  count_eval_nb <- tibble(
    model       = "negative_binomial",
    AIC         = AIC(nb_fit),
    overdispersion_var_over_mean = overdispersion_test,
    MAE         = mean(abs(pred_nb_test - actual_count_test)),
    RMSE        = sqrt(mean((pred_nb_test - actual_count_test)^2)),
    correlation = cor(pred_nb_test, actual_count_test, use = "complete.obs"),
    n_train     = nrow(train_count_df),
    n_test      = nrow(test_count_df),
    n_features  = ncol(train_count_df) - 1L,
    used_fallback_features = use_fallback
  )
  count_eval <- bind_rows(count_eval, count_eval_nb)
}

message("\nCount model comparison:")
message("  Poisson  - MAE: ", sprintf("%.3f", count_eval_poisson$MAE),
        ", RMSE: ", sprintf("%.3f", count_eval_poisson$RMSE),
        ", cor: ", sprintf("%.3f", count_eval_poisson$correlation))
if (!is.null(nb_fit)) {
  message("  NB       - MAE: ", sprintf("%.3f", count_eval_nb$MAE),
          ", RMSE: ", sprintf("%.3f", count_eval_nb$RMSE),
          ", cor: ", sprintf("%.3f", count_eval_nb$correlation))
}
message("  Overdispersion (var/mean, test positives): ", sprintf("%.3f", overdispersion_test))

write_csv(count_eval, file.path(out_dir, "count_model_comparison.csv"))
message("Saved: count_model_comparison.csv")

# Select best count model (prefer NB if AIC lower and it converged)
if (!is.null(nb_fit) && AIC(nb_fit) < AIC(poisson_fit)) {
  selected_count_model <- nb_fit
  selected_count_name  <- "negative_binomial"
  message("Selected count model: Negative Binomial (lower AIC)")
} else {
  selected_count_model <- poisson_fit
  selected_count_name  <- "poisson"
  message("Selected count model: Poisson",
          if (is.null(nb_fit)) " (NB did not converge)" else " (lower or equal AIC)")
}

# SECTION 2b: SEVERITY MODEL (log-normal OLS on positives)

message("\n", strrep("=", 70))
message("SECTION 2b: Severity model (total_length_min | outage=1)")
message(strrep("=", 70))

# Use the feature set consistent with count model
severity_feats <- if (use_fallback) intersect(top20_features, feature_cols) else feature_cols

train_sev_df <- train_pos %>%
  select(all_of(c("total_length_min_panel", severity_feats))) %>%
  rename(total_length_min = total_length_min_panel) %>%
  mutate(across(where(is.logical), as.integer)) %>%
  filter(!is.na(total_length_min), total_length_min > 0) %>%
  mutate(log_total_length = log1p(total_length_min))

test_sev_df <- test_pos %>%
  select(all_of(c("total_length_min_panel", severity_feats))) %>%
  rename(total_length_min = total_length_min_panel) %>%
  mutate(across(where(is.logical), as.integer)) %>%
  filter(!is.na(total_length_min), total_length_min > 0) %>%
  mutate(log_total_length = log1p(total_length_min))

message("Severity model data: train=", nrow(train_sev_df), ", test=", nrow(test_sev_df))
message("Response (train): mean total_length_min=",
        sprintf("%.1f", mean(train_sev_df$total_length_min)),
        ", median=", sprintf("%.1f", median(train_sev_df$total_length_min)),
        ", max=", sprintf("%.1f", max(train_sev_df$total_length_min)))

if (nrow(train_sev_df) < 50) {
  stop("Too few positive rows with valid total_length_min for severity model: ",
       nrow(train_sev_df))
}

# Build formula excluding total_length_min and log_total_length from predictors
sev_formula <- as.formula(paste("log_total_length ~",
                                paste(severity_feats, collapse = " + ")))

message("Fitting log-normal OLS ...")
lm_severity <- tryCatch({
  lm(sev_formula, data = train_sev_df)
}, error = function(e) {
  message("  OLS with ", length(severity_feats), " features FAILED: ", conditionMessage(e))
  if (!use_fallback) {
    message("  Falling back to top-20 features for severity model ...")
    fallback_feats_sev <- intersect(top20_features, feature_cols)
    sev_formula_fb <- as.formula(paste("log_total_length ~",
                                       paste(fallback_feats_sev, collapse = " + ")))
    train_sev_fb <- train_pos %>%
      select(all_of(c("total_length_min_panel", fallback_feats_sev))) %>%
      rename(total_length_min = total_length_min_panel) %>%
      mutate(across(where(is.logical), as.integer)) %>%
      filter(!is.na(total_length_min), total_length_min > 0) %>%
      mutate(log_total_length = log1p(total_length_min))
    lm(sev_formula_fb, data = train_sev_fb)
  } else {
    stop("OLS severity model failed with fallback features: ", conditionMessage(e))
  }
})

# Duan smearing estimator
train_residuals <- residuals(lm_severity)
smearing_factor <- mean(exp(train_residuals))
message("Duan smearing factor: ", sprintf("%.6f", smearing_factor))

# Log-scale R-squared 
log_r_squared <- summary(lm_severity)$r.squared
message("Log-scale R-squared: ", sprintf("%.4f", log_r_squared))

# Evaluate on test positives
pred_log_test <- predict(lm_severity, newdata = test_sev_df)
pred_minutes_test <- expm1(pred_log_test) * smearing_factor
actual_minutes_test <- test_sev_df$total_length_min

# Winsorise at 99th percentile for evaluation
winsor_cap <- quantile(actual_minutes_test, 0.99, na.rm = TRUE)
actual_winsorized <- pmin(actual_minutes_test, winsor_cap)
pred_winsorized   <- pmin(pred_minutes_test, winsor_cap)

n_na_pred <- sum(is.na(pred_minutes_test))
n_eval_sev <- length(pred_minutes_test) - n_na_pred
message("  Severity predictions: ", n_na_pred, "/", length(pred_minutes_test),
        " NA (lagged features missing for early test rows)")

mae_original    <- mean(abs(pred_minutes_test - actual_minutes_test), na.rm = TRUE)
rmse_original   <- sqrt(mean((pred_minutes_test - actual_minutes_test)^2, na.rm = TRUE))
mae_winsorized  <- mean(abs(pred_winsorized - actual_winsorized), na.rm = TRUE)
rmse_winsorized <- sqrt(mean((pred_winsorized - actual_winsorized)^2, na.rm = TRUE))

severity_summary <- tibble(
  model               = "log_normal_ols",
  r_squared_log_scale = log_r_squared,
  smearing_factor     = smearing_factor,
  MAE_original        = mae_original,
  RMSE_original       = rmse_original,
  MAE_winsorized_99   = mae_winsorized,
  RMSE_winsorized_99  = rmse_winsorized,
  winsorize_cap_min   = winsor_cap,
  correlation         = cor(pred_minutes_test, actual_minutes_test, use = "complete.obs"),
  n_train             = nrow(train_sev_df),
  n_test              = nrow(test_sev_df),
  n_test_evaluated    = n_eval_sev,
  n_test_na_excluded  = n_na_pred,
  n_features          = length(severity_feats),
  used_fallback_features = use_fallback
)

message("\nSeverity model evaluation (test positives):")
message("  R-squared (log): ", sprintf("%.4f", log_r_squared))
message("  MAE (original):  ", sprintf("%.1f", mae_original), " min")
message("  RMSE (original): ", sprintf("%.1f", rmse_original), " min")
message("  MAE (winsor 99): ", sprintf("%.1f", mae_winsorized), " min")
message("  RMSE (winsor 99): ", sprintf("%.1f", rmse_winsorized), " min")
message("  Correlation:     ", sprintf("%.4f", severity_summary$correlation))

write_csv(severity_summary, file.path(out_dir, "severity_model_summary.csv"))
message("Saved: severity_model_summary.csv")

# SECTION 3: DECOMPOSED EXPECTED BURDEN (all test rows)

message("\n", strrep("=", 70))
message("SECTION 3: Decomposed expected burden on full test set")
message(strrep("=", 70))

# Predict conditional expectations for ALL test rows
# (these are E[Y|outage=1] regardless of whether outage actually happened)

# Count: E[n_outages | outage=1] 
# Need to match columns used in count model
count_model_feats <- names(coef(selected_count_model))[-1]  # exclude intercept
count_pred_df <- test_data %>%
  select(any_of(c(count_model_feats))) %>%
  mutate(across(where(is.logical), as.integer))

# Handle potential missing features by filling with 0
for (f in count_model_feats) {
  if (!f %in% names(count_pred_df)) {
    count_pred_df[[f]] <- 0
  }
}

pred_count_given_pos <- tryCatch(
  predict(selected_count_model, newdata = count_pred_df, type = "response"),
  error = function(e) {
    message("WARNING: count prediction failed for some rows: ", conditionMessage(e))
    rep(NA_real_, nrow(count_pred_df))
  }
)

# Severity: E[total_length_min | outage=1] 
sev_model_feats <- names(coef(lm_severity))[-1]
sev_pred_df <- test_data %>%
  select(any_of(c(sev_model_feats))) %>%
  mutate(across(where(is.logical), as.integer))

for (f in sev_model_feats) {
  if (!f %in% names(sev_pred_df)) {
    sev_pred_df[[f]] <- 0
  }
}

pred_log_minutes_given_pos <- tryCatch(
  predict(lm_severity, newdata = sev_pred_df),
  error = function(e) {
    message("WARNING: severity prediction failed for some rows: ", conditionMessage(e))
    rep(NA_real_, nrow(sev_pred_df))
  }
)
pred_minutes_given_pos <- expm1(pred_log_minutes_given_pos) * smearing_factor

# Compose E[count] and E[minutes] 
E_count   <- test_prob_calibrated * pred_count_given_pos
E_minutes <- test_prob_calibrated * pred_minutes_given_pos

# Build per-row output
burden_df <- tibble(
  GID_2                    = as.character(test_data$GID_2),
  date                     = as.Date(test_data$date),
  prob_calibrated          = test_prob_calibrated,
  pred_count_given_pos     = pred_count_given_pos,
  pred_minutes_given_pos   = pred_minutes_given_pos,
  E_count                  = E_count,
  E_minutes                = E_minutes,
  actual_outage            = as.integer(as.character(test_data$outage_3h_or_more)),
  actual_n_outages         = as.integer(test_data$n_outages),
  actual_total_length_min  = as.numeric(test_data$total_length_min_panel)
)

message("Burden predictions: ", format(nrow(burden_df), big.mark = ","), " rows")
message("E[count] summary: mean=", sprintf("%.4f", mean(E_count, na.rm = TRUE)),
        ", max=", sprintf("%.4f", max(E_count, na.rm = TRUE)))
message("E[minutes] summary: mean=", sprintf("%.4f", mean(E_minutes, na.rm = TRUE)),
        ", max=", sprintf("%.2f", max(E_minutes, na.rm = TRUE)))

write_csv(burden_df, file.path(out_dir, "decomposed_burden_test.csv"))
message("Saved: decomposed_burden_test.csv")

# Calibration bins: deciles of E[count] and E[minutes] 
make_calibration_bins <- function(df, pred_col, actual_col, label, n_bins = 10) {
  df_valid <- df %>%
    filter(!is.na(.data[[pred_col]]), !is.na(.data[[actual_col]]))

  if (nrow(df_valid) < n_bins) {
    message("  Too few valid rows for calibration bins (", label, "): ", nrow(df_valid))
    return(tibble())
  }

  df_valid %>%
    mutate(decile = ntile(.data[[pred_col]], n_bins)) %>%
    group_by(decile) %>%
    summarise(
      n           = n(),
      mean_pred   = mean(.data[[pred_col]], na.rm = TRUE),
      mean_actual = mean(.data[[actual_col]], na.rm = TRUE),
      min_pred    = min(.data[[pred_col]], na.rm = TRUE),
      max_pred    = max(.data[[pred_col]], na.rm = TRUE),
      .groups     = "drop"
    ) %>%
    mutate(
      metric     = label,
      ratio      = if_else(mean_pred > 0, mean_actual / mean_pred, NA_real_)
    )
}

cal_count   <- make_calibration_bins(burden_df, "E_count", "actual_n_outages", "E_count")
cal_minutes <- make_calibration_bins(burden_df, "E_minutes", "actual_total_length_min", "E_minutes")
cal_bins <- bind_rows(cal_count, cal_minutes)

if (nrow(cal_bins) > 0) {
  write_csv(cal_bins, file.path(out_dir, "burden_calibration_bins.csv"))
  message("Saved: burden_calibration_bins.csv")
} else {
  message("WARNING: No calibration bins computed (insufficient valid data)")
}

# Aggregate by month and state
# Extract state from GID_2 (format: MEX.{state_id}.{muni_id}_1)
burden_agg <- burden_df %>%
  mutate(
    month = format(date, "%Y-%m"),
    state = sub("^([^.]+\\.[^.]+)\\..*$", "\\1", GID_2)
  )

# Monthly aggregation
monthly_agg <- burden_agg %>%
  group_by(month) %>%
  summarise(
    n_rows            = n(),
    sum_E_count       = sum(E_count, na.rm = TRUE),
    sum_actual_count  = sum(actual_n_outages, na.rm = TRUE),
    sum_E_minutes     = sum(E_minutes, na.rm = TRUE),
    sum_actual_minutes = sum(actual_total_length_min, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(level = "monthly", group = month)

# State aggregation
state_agg <- burden_agg %>%
  group_by(state) %>%
  summarise(
    n_rows            = n(),
    sum_E_count       = sum(E_count, na.rm = TRUE),
    sum_actual_count  = sum(actual_n_outages, na.rm = TRUE),
    sum_E_minutes     = sum(E_minutes, na.rm = TRUE),
    sum_actual_minutes = sum(actual_total_length_min, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(level = "state", group = state)

agg_summary <- bind_rows(monthly_agg, state_agg)

# Compute correlations
cor_count_monthly <- if (nrow(monthly_agg) >= 3) {
  cor(monthly_agg$sum_E_count, monthly_agg$sum_actual_count, use = "complete.obs")
} else NA_real_

cor_count_state <- if (nrow(state_agg) >= 3) {
  cor(state_agg$sum_E_count, state_agg$sum_actual_count, use = "complete.obs")
} else NA_real_

cor_minutes_monthly <- if (nrow(monthly_agg) >= 3) {
  cor(monthly_agg$sum_E_minutes, monthly_agg$sum_actual_minutes, use = "complete.obs")
} else NA_real_

cor_minutes_state <- if (nrow(state_agg) >= 3) {
  cor(state_agg$sum_E_minutes, state_agg$sum_actual_minutes, use = "complete.obs")
} else NA_real_

message("\nAggregate correlations (sum predicted vs sum actual):")
message("  Count   - monthly: ", sprintf("%.4f", cor_count_monthly),
        ", by state: ", sprintf("%.4f", cor_count_state))
message("  Minutes - monthly: ", sprintf("%.4f", cor_minutes_monthly),
        ", by state: ", sprintf("%.4f", cor_minutes_state))

# Add correlation summary rows
cor_summary <- tibble(
  level = "correlation",
  group = c("count_monthly", "count_state", "minutes_monthly", "minutes_state"),
  n_rows = c(nrow(monthly_agg), nrow(state_agg), nrow(monthly_agg), nrow(state_agg)),
  sum_E_count = c(cor_count_monthly, cor_count_state, NA_real_, NA_real_),
  sum_actual_count = NA_real_,
  sum_E_minutes = c(NA_real_, NA_real_, cor_minutes_monthly, cor_minutes_state),
  sum_actual_minutes = NA_real_
)

agg_summary <- bind_rows(agg_summary, cor_summary)

write_csv(agg_summary, file.path(out_dir, "burden_aggregate_summary.csv"))
message("Saved: burden_aggregate_summary.csv")

# FINAL: SAVE CONFIG

message("\n", strrep("=", 70))
message("Saving two-part run configuration")
message(strrep("=", 70))

two_part_config <- list(
  run_id              = run_id,
  script              = "14_two_part_count_severity.R",
  timestamp           = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  seed                = 42,
  binary_model = list(
    type              = "xgboost",
    params            = xgb_params,
    n_features        = n_features,
    scale_pos_weight  = scale_pos_weight,
    test_roc_auc      = test_roc_auc,
    benchmark_roc_auc = benchmark_roc,
    platt_intercept   = platt_intercept,
    platt_slope       = platt_slope
  ),
  count_model = list(
    selected          = selected_count_name,
    n_features        = ncol(train_count_df) - 1L,
    used_fallback     = use_fallback,
    poisson_AIC       = AIC(poisson_fit),
    nb_AIC            = if (!is.null(nb_fit)) AIC(nb_fit) else NA_real_,
    nb_theta          = if (!is.null(nb_fit)) nb_fit$theta else NA_real_,
    overdispersion    = overdispersion_test,
    test_MAE          = count_eval %>% filter(model == selected_count_name) %>% pull(MAE),
    test_RMSE         = count_eval %>% filter(model == selected_count_name) %>% pull(RMSE),
    test_correlation  = count_eval %>% filter(model == selected_count_name) %>% pull(correlation)
  ),
  severity_model = list(
    type                   = "log_normal_ols",
    n_features             = length(severity_feats),
    used_fallback          = use_fallback,
    smearing_factor        = smearing_factor,
    r_squared_log_scale    = log_r_squared,
    test_MAE_original      = mae_original,
    test_RMSE_original     = rmse_original,
    test_MAE_winsorized_99 = mae_winsorized,
    test_RMSE_winsorized_99 = rmse_winsorized,
    winsorize_cap_min      = winsor_cap,
    test_correlation       = severity_summary$correlation
  ),
  aggregate_correlations = list(
    count_monthly   = cor_count_monthly,
    count_state     = cor_count_state,
    minutes_monthly = cor_minutes_monthly,
    minutes_state   = cor_minutes_state
  ),
  data = list(
    train_rows       = nrow(train_data),
    test_rows        = nrow(test_data),
    train_positives  = nrow(train_pos),
    test_positives   = nrow(test_pos),
    severity_train   = nrow(train_sev_df),
    severity_test    = nrow(test_sev_df)
  ),
  output_dir = out_dir,
  output_files = c(
    "count_model_comparison.csv",
    "severity_model_summary.csv",
    "decomposed_burden_test.csv",
    "burden_calibration_bins.csv",
    "burden_aggregate_summary.csv",
    "two_part_run_config.json"
  )
)

write_json(two_part_config, file.path(out_dir, "two_part_run_config.json"),
           pretty = TRUE, auto_unbox = TRUE)
message("Saved: two_part_run_config.json")

# SUMMARY

message("\n", strrep("=", 70))
message("TWO-PART MODEL COMPLETE")
message(strrep("=", 70))
message("\nOutputs saved to: ", out_dir)
message("  - count_model_comparison.csv")
message("  - severity_model_summary.csv")
message("  - decomposed_burden_test.csv")
message("  - burden_calibration_bins.csv")
message("  - burden_aggregate_summary.csv")
message("  - two_part_run_config.json")
message("\nKey results:")
message("  Binary model ROC-AUC: ", sprintf("%.4f", test_roc_auc))
message("  Count model (", selected_count_name, "): MAE=",
        sprintf("%.3f", count_eval %>% filter(model == selected_count_name) %>% pull(MAE)))
message("  Severity model: R2(log)=", sprintf("%.4f", log_r_squared),
        ", MAE=", sprintf("%.1f", mae_original), " min")
message("  Aggregate count cor (monthly): ", sprintf("%.4f", cor_count_monthly))
message("  Aggregate minutes cor (monthly): ", sprintf("%.4f", cor_minutes_monthly))
message("\nScript completed at: ", Sys.time())
