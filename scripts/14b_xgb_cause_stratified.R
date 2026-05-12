# 14b_xgb_cause_stratified.R
# XGBoost-based two-part decomposition + cause stratification.
#
#   - Replace GLM/OLS in script 14 with XGBoost (count:poisson,
#     reg:squarederror on log-duration)
#   - Fit cause-stratified severity models (Environmental vs
#     Technical) and evaluate per-cause performance
#
# Sections:
#   0. Load artifacts from binary run + cause join from panel
#   1. Rebuild binary XGBoost (P(outage)) identical to benchmark
#   2. XGBoost count model on positives (count:poisson)
#   3a. XGBoost severity OVERALL on positives (reg:squarederror,
#       log1p(total_length_min))
#   3b. XGBoost severity ENVIRONMENTAL-only
#   3c. XGBoost severity TECHNICAL-only
#   4. Cause-stratified evaluation: predict each test positive
#      with the model matching its actual cause
#   5. Decomposed expected burden (XGB-based)
#   6. Save artifacts
#
# Usage:
#   Rscript scripts/14b_xgb_cause_stratified.R [run_id]
#   default run_id = 20260416_153241_forecast_strict


suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tibble)
  library(tidyr)
  library(tidymodels)
  library(jsonlite)
  library(xgboost)
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

# Build a numeric matrix for xgb.train from a data frame and feature list.
# Logical -> integer; missing features filled with NA (xgboost handles NA).
make_xgb_matrix <- function(df, feat_cols) {
  X <- df %>%
    select(any_of(feat_cols)) %>%
    mutate(across(where(is.logical), as.integer))
  for (f in feat_cols) {
    if (!f %in% names(X)) X[[f]] <- NA_real_
  }
  X <- X[, feat_cols, drop = FALSE]
  for (j in seq_len(ncol(X))) {
    if (!is.numeric(X[[j]])) X[[j]] <- as.numeric(X[[j]])
  }
  as.matrix(X)
}

# SECTION 0: SETUP

args <- commandArgs(trailingOnly = TRUE)
run_id <- if (length(args) >= 1 && nzchar(args[1])) args[1] else "20260416_153241_forecast_strict"

project_dir <- detect_project_dir()
run_dir <- file.path(project_dir, "data", "baselines", "binary_threshold", run_id)
out_dir <- file.path(run_dir, "two_part_xgb_cause")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

features_path <- file.path(project_dir, "data", "model_ready", "features_engineered.rds")
splits_path   <- file.path(project_dir, "data", "model_ready", "splits_fixed.rds")
panel_path    <- file.path(project_dir, "data", "panel_mex_2017_2021_ntl_ghs.rds")

run_config_path    <- file.path(run_dir, "run_config.json")
features_used_path <- file.path(run_dir, "features_used.txt")
platt_path         <- file.path(run_dir, "platt_calibration.csv")

required_files <- c(run_config_path, features_used_path, platt_path,
                    features_path, splits_path, panel_path)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop("Missing required files:\n", paste0("  - ", missing_files, collapse = "\n"))
}

message(strrep("=", 70))
message("14b_xgb_cause_stratified.R")
message(strrep("=", 70))
message("Run ID:       ", run_id)
message("Output dir:   ", out_dir)

run_config <- jsonlite::read_json(run_config_path)
xgb_params <- run_config$model$params
message("Loaded XGB params: tree_depth=", xgb_params$tree_depth,
        ", trees=", xgb_params$trees, ", learn_rate=", xgb_params$learn_rate)

feature_cols <- read_lines(features_used_path)
feature_cols <- unique(feature_cols[nzchar(feature_cols)])
n_features <- length(feature_cols)
message("Features loaded: ", n_features)

platt_tbl <- read_csv(platt_path, show_col_types = FALSE) %>% slice(1)
platt_intercept <- as.numeric(platt_tbl$platt_intercept)
platt_slope     <- as.numeric(platt_tbl$platt_slope)
message("Platt: intercept=", sprintf("%.6f", platt_intercept),
        ", slope=", sprintf("%.6f", platt_slope))

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

# Join total_length_min AND classification_general from panel
message("Loading panel for total_length_min + classification_general ...")
panel <- readRDS(panel_path)

panel_join <- panel %>%
  transmute(
    GID_2 = as.character(GID_2),
    date  = as.Date(date),
    total_length_min_panel = as.numeric(total_length_min),
    classification_general_panel = as.character(classification_general)
  ) %>%
  distinct(GID_2, date, .keep_all = TRUE)

features <- features %>%
  left_join(panel_join, by = c("GID_2", "date"))

n_joined_dur <- sum(!is.na(features$total_length_min_panel))
n_joined_cls <- sum(!is.na(features$classification_general_panel))
message("Joined total_length_min: ", format(n_joined_dur, big.mark = ","),
        " rows; classification: ", format(n_joined_cls, big.mark = ","), " rows")

# Verify all required features present
missing_feats <- setdiff(feature_cols, names(features))
if (length(missing_feats) > 0) {
  stop("Missing features from features_engineered.rds:\n",
       paste0("  - ", head(missing_feats, 30), collapse = "\n"))
}

train_data <- features %>% filter(split == "train")
test_data  <- features %>% filter(split == "test")

message("Train rows: ", format(nrow(train_data), big.mark = ","),
        " | Test rows: ", format(nrow(test_data), big.mark = ","))

# SECTION 1: REBUILD BINARY XGBOOST (P(outage))

message("\n", strrep("=", 70))
message("SECTION 1: Rebuild P(outage) via XGBoost (binary)")
message(strrep("=", 70))

train_model <- prep_model_df(train_data, feature_cols)
test_model  <- prep_model_df(test_data, feature_cols)

n_pos_train <- sum(train_model$outage_3h_or_more == "1", na.rm = TRUE)
n_neg_train <- sum(train_model$outage_3h_or_more == "0", na.rm = TRUE)
scale_pos_weight <- n_neg_train / max(1, n_pos_train)
message("Train: pos=", format(n_pos_train, big.mark = ","),
        ", neg=", format(n_neg_train, big.mark = ","),
        ", scale_pos_weight=", sprintf("%.2f", scale_pos_weight))

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

message("Fitting binary XGBoost (trees=", xgb_params$trees,
        ", depth=", xgb_params$tree_depth, ") ...")
t0 <- Sys.time()
xgb_fit <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(xgb_spec) %>%
  fit(data = train_model)
t1 <- Sys.time()
message("Binary XGB trained in ", sprintf("%.1f", difftime(t1, t0, units = "mins")), " min")

test_prob_raw <- predict(xgb_fit, new_data = test_model, type = "prob") %>% pull(.pred_1)
test_prob_calibrated <- apply_platt_with_params(test_prob_raw, platt_intercept, platt_slope)

truth_fac <- factor(
  if_else(as.integer(as.character(test_data$outage_3h_or_more)) == 1L, "1", "0"),
  levels = c("1", "0")
)
test_roc_auc <- yardstick::roc_auc_vec(truth = truth_fac, estimate = test_prob_raw,
                                        event_level = "first")
benchmark_roc <- 0.920  # new benchmark with history features
roc_diff <- abs(test_roc_auc - benchmark_roc)
if (roc_diff <= 0.01) {
  message("Sanity check PASSED: test ROC-AUC = ", sprintf("%.4f", test_roc_auc),
          " (benchmark = ", benchmark_roc, ")")
} else {
  message("WARNING: test ROC-AUC = ", sprintf("%.4f", test_roc_auc),
          " differs from benchmark ", benchmark_roc,
          " by ", sprintf("%.4f", roc_diff))
}

# SECTION 2: XGBOOST COUNT MODEL (count:poisson) ON POSITIVES

message("\n", strrep("=", 70))
message("SECTION 2: XGBoost count model (count:poisson) on positives")
message(strrep("=", 70))

train_pos <- train_data %>%
  filter(as.character(outage_3h_or_more) == "1")
test_pos <- test_data %>%
  filter(as.character(outage_3h_or_more) == "1")

message("Positives: train=", format(nrow(train_pos), big.mark = ","),
        ", test=", format(nrow(test_pos), big.mark = ","))

X_train_pos <- make_xgb_matrix(train_pos, feature_cols)
X_test_pos  <- make_xgb_matrix(test_pos,  feature_cols)
y_count_train <- as.numeric(train_pos$n_outages)
y_count_test  <- as.numeric(test_pos$n_outages)

message("n_outages train: mean=", sprintf("%.3f", mean(y_count_train)),
        ", var=", sprintf("%.3f", var(y_count_train)),
        ", max=", max(y_count_train))

dtrain_count <- xgb.DMatrix(data = X_train_pos, label = y_count_train)
dtest_count  <- xgb.DMatrix(data = X_test_pos,  label = y_count_test)

count_params <- list(
  objective        = "count:poisson",
  eval_metric      = "poisson-nloglik",
  eta              = xgb_params$learn_rate,
  max_depth        = xgb_params$tree_depth,
  min_child_weight = xgb_params$min_n,
  gamma            = xgb_params$loss_reduction,
  subsample        = xgb_params$sample_size,
  colsample_bytree = xgb_params$mtry,
  nthread          = get_nthread(),
  verbosity        = 0
)

message("Fitting XGBoost count model ...")
set.seed(42)
t0 <- Sys.time()
count_xgb <- xgb.train(
  params  = count_params,
  data    = dtrain_count,
  nrounds = xgb_params$trees,
  watchlist = list(train = dtrain_count, test = dtest_count),
  verbose = 0
)
t1 <- Sys.time()
message("Count XGB trained in ", sprintf("%.1f", difftime(t1, t0, units = "secs")), " sec")

pred_count_xgb_test <- predict(count_xgb, dtest_count)
overdispersion_test <- var(y_count_test) / mean(y_count_test)

count_eval <- tibble(
  model       = "xgb_poisson",
  MAE         = mean(abs(pred_count_xgb_test - y_count_test)),
  RMSE        = sqrt(mean((pred_count_xgb_test - y_count_test)^2)),
  correlation = cor(pred_count_xgb_test, y_count_test, use = "complete.obs"),
  overdispersion_var_over_mean = overdispersion_test,
  n_train     = length(y_count_train),
  n_test      = length(y_count_test),
  n_features  = n_features
)

message("XGBoost count model (test positives):")
message("  MAE: ", sprintf("%.3f", count_eval$MAE),
        ", RMSE: ", sprintf("%.3f", count_eval$RMSE),
        ", cor: ", sprintf("%.3f", count_eval$correlation))
message("  Overdispersion (var/mean): ", sprintf("%.3f", overdispersion_test))

write_csv(count_eval, file.path(out_dir, "count_model_xgb_summary.csv"))

# SECTION 3: XGBOOST SEVERITY MODELS

message("\n", strrep("=", 70))
message("SECTION 3: XGBoost severity models on log1p(total_length_min)")
message(strrep("=", 70))

# Filter positives with valid duration
train_pos_sev <- train_pos %>%
  filter(!is.na(total_length_min_panel), total_length_min_panel > 0) %>%
  mutate(log_dur = log1p(total_length_min_panel))
test_pos_sev <- test_pos %>%
  filter(!is.na(total_length_min_panel), total_length_min_panel > 0) %>%
  mutate(log_dur = log1p(total_length_min_panel))

message("Positives with valid duration: train=", format(nrow(train_pos_sev), big.mark = ","),
        ", test=", format(nrow(test_pos_sev), big.mark = ","))
message("total_length_min train: mean=", sprintf("%.1f", mean(train_pos_sev$total_length_min_panel)),
        ", median=", sprintf("%.1f", median(train_pos_sev$total_length_min_panel)),
        ", max=", sprintf("%.1f", max(train_pos_sev$total_length_min_panel)))

# Helper: train an xgb.train regression model and return predictions on test
fit_xgb_severity <- function(train_df, test_df, feat_cols, label_name) {
  X_tr <- make_xgb_matrix(train_df, feat_cols)
  y_tr <- train_df$log_dur
  X_te <- make_xgb_matrix(test_df, feat_cols)
  y_te <- test_df$log_dur

  dtr <- xgb.DMatrix(data = X_tr, label = y_tr)
  dte <- xgb.DMatrix(data = X_te, label = y_te)

  params <- list(
    objective        = "reg:squarederror",
    eval_metric      = "rmse",
    eta              = xgb_params$learn_rate,
    max_depth        = xgb_params$tree_depth,
    min_child_weight = xgb_params$min_n,
    gamma            = xgb_params$loss_reduction,
    subsample        = xgb_params$sample_size,
    colsample_bytree = xgb_params$mtry,
    nthread          = get_nthread(),
    verbosity        = 0
  )

  set.seed(42)
  t0 <- Sys.time()
  fit <- xgb.train(
    params    = params,
    data      = dtr,
    nrounds   = xgb_params$trees,
    watchlist = list(train = dtr, test = dte),
    verbose   = 0
  )
  t1 <- Sys.time()
  message(sprintf("  [%s] trained in %.1f sec on n=%d positives",
                  label_name, difftime(t1, t0, units = "secs"), nrow(train_df)))

  pred_log_te <- predict(fit, dte)
  resid_tr    <- y_tr - predict(fit, dtr)
  smear       <- mean(exp(resid_tr))  # Duan smearing

  list(
    fit         = fit,
    pred_log_te = pred_log_te,
    smearing    = smear,
    n_train     = nrow(train_df),
    n_test      = nrow(test_df)
  )
}

# 3a. OVERALL severity model (all positives)
message("\n--- 3a. OVERALL severity model ---")
sev_overall <- fit_xgb_severity(train_pos_sev, test_pos_sev, feature_cols, "OVERALL")

# 3b. ENVIRONMENTAL-only
train_pos_env <- train_pos_sev %>%
  filter(!is.na(classification_general_panel),
         classification_general_panel == "Environmental")
test_pos_env  <- test_pos_sev %>%
  filter(!is.na(classification_general_panel),
         classification_general_panel == "Environmental")
message("\n--- 3b. ENVIRONMENTAL severity (n_train=", nrow(train_pos_env),
        ", n_test=", nrow(test_pos_env), ") ---")
sev_env <- fit_xgb_severity(train_pos_env, test_pos_env, feature_cols, "ENV")

# 3c. TECHNICAL-only
train_pos_tech <- train_pos_sev %>%
  filter(!is.na(classification_general_panel),
         classification_general_panel == "Technical")
test_pos_tech  <- test_pos_sev %>%
  filter(!is.na(classification_general_panel),
         classification_general_panel == "Technical")
message("\n--- 3c. TECHNICAL severity (n_train=", nrow(train_pos_tech),
        ", n_test=", nrow(test_pos_tech), ") ---")
sev_tech <- fit_xgb_severity(train_pos_tech, test_pos_tech, feature_cols, "TECH")

# Per-model evaluation on respective test sets
eval_severity <- function(pred_log, actual_minutes, smear, label, n_train, n_test) {
  pred_min <- expm1(pred_log) * smear
  cap99 <- quantile(actual_minutes, 0.99, na.rm = TRUE)
  actual_w <- pmin(actual_minutes, cap99)
  pred_w   <- pmin(pred_min, cap99)
  tibble(
    model              = label,
    n_train            = n_train,
    n_test             = n_test,
    smearing_factor    = smear,
    MAE_original       = mean(abs(pred_min - actual_minutes), na.rm = TRUE),
    RMSE_original      = sqrt(mean((pred_min - actual_minutes)^2, na.rm = TRUE)),
    MAE_winsorized_99  = mean(abs(pred_w - actual_w), na.rm = TRUE),
    RMSE_winsorized_99 = sqrt(mean((pred_w - actual_w)^2, na.rm = TRUE)),
    correlation        = cor(pred_min, actual_minutes, use = "complete.obs"),
    winsorize_cap_min  = cap99
  )
}

sev_eval_overall <- eval_severity(sev_overall$pred_log_te,
                                  test_pos_sev$total_length_min_panel,
                                  sev_overall$smearing, "xgb_overall",
                                  sev_overall$n_train, sev_overall$n_test)
sev_eval_env     <- eval_severity(sev_env$pred_log_te,
                                  test_pos_env$total_length_min_panel,
                                  sev_env$smearing, "xgb_environmental",
                                  sev_env$n_train, sev_env$n_test)
sev_eval_tech    <- eval_severity(sev_tech$pred_log_te,
                                  test_pos_tech$total_length_min_panel,
                                  sev_tech$smearing, "xgb_technical",
                                  sev_tech$n_train, sev_tech$n_test)

severity_summary <- bind_rows(sev_eval_overall, sev_eval_env, sev_eval_tech)
write_csv(severity_summary, file.path(out_dir, "severity_model_xgb_summary.csv"))

message("\nSeverity model summary:")
print(severity_summary %>% select(model, n_test, MAE_original, RMSE_original,
                                  MAE_winsorized_99, correlation))

# SECTION 4: CAUSE-STRATIFIED EVALUATION (ORACLE DIAGNOSTIC)
# WARNING: This is an ORACLE evaluation, NOT a deployable model.
# It selects the Environmental or Technical model using the actual
# realised cause of the outage — which is unknown at decision time.
#
# This Section 4 result represents the UPPER BOUND on the lift
# achievable IF cause were perfectly predicted. To make this
# operational, a separate cause classifier (e.g., XGB multi:softprob
# trained on positives to predict P(cause | x, outage=1)) is needed,
# and the burden composition becomes:
#     E[duration | x] = sum_c P(c | x, outage=1) x E_c[duration | x]
# That mixture model is NOT implemented here (deferred).
#
# Section 5 (full-test burden) uses only the OVERALL severity model
# and is a deployable composition with the binary classifier.

message("\n", strrep("=", 70))
message("SECTION 4: Cause-stratified evaluation [ORACLE — not deployable]")
message("  Selects env/tech model using REALIZED cause (unknown at decision time).")
message("  Reports the UPPER BOUND on lift if cause were perfectly predicted.")
message("  Section 5 burden uses the OVERALL severity model (deployable).")
message(strrep("=", 70))

# Build test_pos_sev with cause label and predictions from all 3 models
X_test_all <- make_xgb_matrix(test_pos_sev, feature_cols)
dtest_all  <- xgb.DMatrix(data = X_test_all)

pred_log_overall <- predict(sev_overall$fit, dtest_all)
pred_log_env     <- predict(sev_env$fit,     dtest_all)
pred_log_tech    <- predict(sev_tech$fit,    dtest_all)

pred_min_overall <- expm1(pred_log_overall) * sev_overall$smearing
pred_min_env     <- expm1(pred_log_env)     * sev_env$smearing
pred_min_tech    <- expm1(pred_log_tech)    * sev_tech$smearing

# Cause-matched prediction: use env model if env, tech model if tech, else overall
cause <- test_pos_sev$classification_general_panel
pred_min_matched <- case_when(
  cause == "Environmental" ~ pred_min_env,
  cause == "Technical"     ~ pred_min_tech,
  TRUE                     ~ pred_min_overall
)

actual <- test_pos_sev$total_length_min_panel

cause_eval <- tibble(
  cause = cause,
  actual = actual,
  pred_overall = pred_min_overall,
  pred_matched = pred_min_matched
)

# Per-cause eval, using overall vs matched model
per_cause_summary <- cause_eval %>%
  group_by(cause) %>%
  summarise(
    n               = n(),
    MAE_overall     = mean(abs(pred_overall - actual), na.rm = TRUE),
    RMSE_overall    = sqrt(mean((pred_overall - actual)^2, na.rm = TRUE)),
    MAE_matched     = mean(abs(pred_matched - actual), na.rm = TRUE),
    RMSE_matched    = sqrt(mean((pred_matched - actual)^2, na.rm = TRUE)),
    cor_overall     = cor(pred_overall, actual, use = "complete.obs"),
    cor_matched     = cor(pred_matched, actual, use = "complete.obs"),
    .groups = "drop"
  ) %>%
  mutate(
    MAE_lift_pct  = 100 * (MAE_overall - MAE_matched) / MAE_overall,
    RMSE_lift_pct = 100 * (RMSE_overall - RMSE_matched) / RMSE_overall
  )

# Add overall (all positives) row
all_row <- cause_eval %>%
  summarise(
    cause = "ALL_positives",
    n               = n(),
    MAE_overall     = mean(abs(pred_overall - actual), na.rm = TRUE),
    RMSE_overall    = sqrt(mean((pred_overall - actual)^2, na.rm = TRUE)),
    MAE_matched     = mean(abs(pred_matched - actual), na.rm = TRUE),
    RMSE_matched    = sqrt(mean((pred_matched - actual)^2, na.rm = TRUE)),
    cor_overall     = cor(pred_overall, actual, use = "complete.obs"),
    cor_matched     = cor(pred_matched, actual, use = "complete.obs")
  ) %>%
  mutate(
    MAE_lift_pct  = 100 * (MAE_overall - MAE_matched) / MAE_overall,
    RMSE_lift_pct = 100 * (RMSE_overall - RMSE_matched) / RMSE_overall
  )

per_cause_summary <- bind_rows(per_cause_summary, all_row)

message("\nPer-cause evaluation (overall vs cause-matched model):")
print(per_cause_summary %>% mutate(across(where(is.numeric), ~round(.x, 2))))

write_csv(per_cause_summary, file.path(out_dir, "cause_stratified_eval.csv"))

# SECTION 5: DECOMPOSED EXPECTED BURDEN (XGB-based)

message("\n", strrep("=", 70))
message("SECTION 5: Decomposed expected burden on full test set (XGB)")
message(strrep("=", 70))

X_test_full <- make_xgb_matrix(test_data, feature_cols)
dtest_full  <- xgb.DMatrix(data = X_test_full)

E_count_given_pos   <- predict(count_xgb, dtest_full)
E_log_dur_given_pos <- predict(sev_overall$fit, dtest_full)
E_min_given_pos     <- expm1(E_log_dur_given_pos) * sev_overall$smearing

E_count   <- test_prob_calibrated * E_count_given_pos
E_minutes <- test_prob_calibrated * E_min_given_pos

burden_df <- tibble(
  GID_2                   = as.character(test_data$GID_2),
  date                    = as.Date(test_data$date),
  prob_calibrated         = test_prob_calibrated,
  pred_count_given_pos    = E_count_given_pos,
  pred_minutes_given_pos  = E_min_given_pos,
  E_count                 = E_count,
  E_minutes               = E_minutes,
  actual_outage           = as.integer(as.character(test_data$outage_3h_or_more)),
  actual_n_outages        = as.integer(test_data$n_outages),
  actual_total_length_min = as.numeric(test_data$total_length_min_panel)
)

write_csv(burden_df, file.path(out_dir, "decomposed_burden_xgb_test.csv"))

# Calibration bins (deciles)
make_cal_bins <- function(df, pred_col, actual_col, label, n_bins = 10) {
  df %>%
    filter(!is.na(.data[[pred_col]]), !is.na(.data[[actual_col]])) %>%
    mutate(decile = ntile(.data[[pred_col]], n_bins)) %>%
    group_by(decile) %>%
    summarise(
      n           = n(),
      mean_pred   = mean(.data[[pred_col]], na.rm = TRUE),
      mean_actual = mean(.data[[actual_col]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(metric = label,
           ratio = if_else(mean_pred > 0, mean_actual / mean_pred, NA_real_))
}

cal_count   <- make_cal_bins(burden_df, "E_count", "actual_n_outages", "E_count")
cal_minutes <- make_cal_bins(burden_df, "E_minutes", "actual_total_length_min", "E_minutes")
write_csv(bind_rows(cal_count, cal_minutes),
          file.path(out_dir, "burden_calibration_bins_xgb.csv"))

# Aggregate correlations (monthly + state)
burden_agg <- burden_df %>%
  mutate(
    month = format(date, "%Y-%m"),
    state = sub("^([^.]+\\.[^.]+)\\..*$", "\\1", GID_2)
  )
monthly_agg <- burden_agg %>% group_by(month) %>% summarise(
  sum_E_count = sum(E_count, na.rm = TRUE),
  sum_actual_count = sum(actual_n_outages, na.rm = TRUE),
  sum_E_minutes = sum(E_minutes, na.rm = TRUE),
  sum_actual_minutes = sum(actual_total_length_min, na.rm = TRUE), .groups = "drop")
state_agg <- burden_agg %>% group_by(state) %>% summarise(
  sum_E_count = sum(E_count, na.rm = TRUE),
  sum_actual_count = sum(actual_n_outages, na.rm = TRUE),
  sum_E_minutes = sum(E_minutes, na.rm = TRUE),
  sum_actual_minutes = sum(actual_total_length_min, na.rm = TRUE), .groups = "drop")

cor_count_monthly   <- cor(monthly_agg$sum_E_count, monthly_agg$sum_actual_count, use = "complete.obs")
cor_count_state     <- cor(state_agg$sum_E_count, state_agg$sum_actual_count, use = "complete.obs")
cor_minutes_monthly <- cor(monthly_agg$sum_E_minutes, monthly_agg$sum_actual_minutes, use = "complete.obs")
cor_minutes_state   <- cor(state_agg$sum_E_minutes, state_agg$sum_actual_minutes, use = "complete.obs")

message("Aggregate correlations (XGB stage 2):")
message("  Count   - monthly: ", sprintf("%.4f", cor_count_monthly),
        ", state: ", sprintf("%.4f", cor_count_state))
message("  Minutes - monthly: ", sprintf("%.4f", cor_minutes_monthly),
        ", state: ", sprintf("%.4f", cor_minutes_state))

write_csv(
  bind_rows(
    monthly_agg %>% mutate(level = "monthly", group = month) %>% select(-month),
    state_agg   %>% mutate(level = "state",   group = state) %>% select(-state)
  ),
  file.path(out_dir, "burden_aggregate_summary_xgb.csv")
)

# FINAL: SAVE CONFIG

config <- list(
  run_id     = run_id,
  script     = "14b_xgb_cause_stratified.R",
  timestamp  = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  seed       = 42,
  binary_model = list(
    type            = "xgboost",
    params          = xgb_params,
    n_features      = n_features,
    test_roc_auc    = test_roc_auc,
    benchmark_roc_auc = benchmark_roc,
    platt_intercept = platt_intercept,
    platt_slope     = platt_slope
  ),
  count_model = list(
    type        = "xgb_count_poisson",
    objective   = "count:poisson",
    n_features  = n_features,
    test_MAE    = count_eval$MAE,
    test_RMSE   = count_eval$RMSE,
    test_corr   = count_eval$correlation,
    overdispersion = overdispersion_test
  ),
  severity_models = list(
    overall      = as.list(sev_eval_overall),
    environmental= as.list(sev_eval_env),
    technical    = as.list(sev_eval_tech)
  ),
  cause_stratified_lift = list(
    per_cause = per_cause_summary
  ),
  aggregate_correlations = list(
    count_monthly   = cor_count_monthly,
    count_state     = cor_count_state,
    minutes_monthly = cor_minutes_monthly,
    minutes_state   = cor_minutes_state
  ),
  data = list(
    train_rows        = nrow(train_data),
    test_rows         = nrow(test_data),
    train_positives   = nrow(train_pos),
    test_positives    = nrow(test_pos),
    sev_train_overall = nrow(train_pos_sev),
    sev_test_overall  = nrow(test_pos_sev),
    sev_train_env     = nrow(train_pos_env),
    sev_test_env      = nrow(test_pos_env),
    sev_train_tech    = nrow(train_pos_tech),
    sev_test_tech     = nrow(test_pos_tech)
  ),
  output_dir = out_dir
)

write_json(config, file.path(out_dir, "two_part_xgb_run_config.json"),
           pretty = TRUE, auto_unbox = TRUE)

message("\n", strrep("=", 70))
message("14b COMPLETE")
message(strrep("=", 70))
message("Outputs saved to: ", out_dir)
message("\nKey results:")
message("  Binary ROC-AUC:    ", sprintf("%.4f", test_roc_auc))
message("  Count XGB MAE:     ", sprintf("%.3f", count_eval$MAE),
        " (vs benchmark from script 14)")
message("  Severity overall MAE: ", sprintf("%.1f", sev_eval_overall$MAE_original), " min")
message("  Severity ENV MAE:     ", sprintf("%.1f", sev_eval_env$MAE_original), " min")
message("  Severity TECH MAE:    ", sprintf("%.1f", sev_eval_tech$MAE_original), " min")
message("  Cause-matched lift (ALL positives): MAE ",
        sprintf("%.2f%%", all_row$MAE_lift_pct[1]))
message("  Aggregate count cor (monthly): ", sprintf("%.4f", cor_count_monthly))
message("  Aggregate minutes cor (monthly): ", sprintf("%.4f", cor_minutes_monthly))
message("\nScript completed at: ", Sys.time())
