# 13_cause_specific_eval.R

# Minimal cause-specific evaluation (dominant-cause panel):
# - one-vs-rest ROC-AUC / PR-AUC by cause
# - precision / recall at selected policy threshold
# - bootstrap CIs + support counts
#
# Default behavior:
# - Rebuild test scores from the strict forecast setup
# - Use deployment default policy threshold from run artifacts
# - Evaluate causes: Environmental, Technical, Planned, Other
#
# Usage:
#   Rscript scripts/13_cause_specific_eval.R
#   Rscript scripts/13_cause_specific_eval.R <run_id> <method> <policy_type> <policy_target> <n_boot>
#
# Example:
#   Rscript scripts/13_cause_specific_eval.R 20260223_125430_forecast_strict platt alerts_per_day_cap 10 400

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(tibble)
  library(purrr)
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

arg_or_default <- function(args, i, default) {
  if (length(args) >= i && nzchar(args[[i]])) args[[i]] else default
}

safe_as_numeric <- function(x) suppressWarnings(as.numeric(x))

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

normalize_cause <- function(x) {
  x <- str_trim(as.character(x))
  x <- ifelse(x == "", NA_character_, x)
  # Keep dominant-cause labels stable; map unknown strings to "Other"
  out <- case_when(
    is.na(x) ~ NA_character_,
    x == "Environmental" ~ "Environmental",
    x == "Technical" ~ "Technical",
    x == "Planned" ~ "Planned",
    x == "Other" ~ "Other",
    TRUE ~ "Other"
  )
  factor(out, levels = c("Environmental", "Technical", "Planned", "Other"))
}

compute_binary_metrics <- function(y_true01, prob, pred01) {
  truth_fac <- factor(if_else(y_true01 == 1L, "1", "0"), levels = c("1", "0"))

  n_pos <- sum(y_true01 == 1L, na.rm = TRUE)
  n_neg <- sum(y_true01 == 0L, na.rm = TRUE)

  tp <- sum(y_true01 == 1L & pred01 == 1L, na.rm = TRUE)
  fp <- sum(y_true01 == 0L & pred01 == 1L, na.rm = TRUE)
  fn <- sum(y_true01 == 1L & pred01 == 0L, na.rm = TRUE)

  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_

  roc_auc <- if (n_pos > 0 && n_neg > 0) {
    yardstick::roc_auc_vec(truth = truth_fac, estimate = prob, event_level = "first")
  } else {
    NA_real_
  }
  pr_auc <- if (n_pos > 0 && n_neg > 0) {
    yardstick::pr_auc_vec(truth = truth_fac, estimate = prob, event_level = "first")
  } else {
    NA_real_
  }

  tibble(
    n_positive = n_pos,
    n_negative = n_neg,
    prevalence = n_pos / (n_pos + n_neg),
    tp = tp,
    fp = fp,
    fn = fn,
    precision = precision,
    recall = recall,
    roc_auc = roc_auc,
    pr_auc = pr_auc
  )
}

run_bootstrap <- function(df, cause_name, threshold, n_boot = 400, seed = 42,
                          method = "row") {
  # method: "row" = row-level i.i.d., "block" = ISO week-block bootstrap
  set.seed(seed)
  n <- nrow(df)

  empty_result <- tibble(
    roc_auc_ci_lower = NA_real_, roc_auc_ci_upper = NA_real_,
    pr_auc_ci_lower = NA_real_, pr_auc_ci_upper = NA_real_,
    precision_ci_lower = NA_real_, precision_ci_upper = NA_real_,
    recall_ci_lower = NA_real_, recall_ci_upper = NA_real_,
    n_boot = n_boot,
    n_boot_valid_roc = 0L,
    n_boot_valid_pr = 0L,
    bootstrap_method = method
  )

  if (n == 0) return(empty_result)

  out <- matrix(NA_real_, nrow = n_boot, ncol = 4)
  colnames(out) <- c("roc_auc", "pr_auc", "precision", "recall")

  if (method == "block") {
    # Week-block bootstrap: pre-index rows by ISO week, then sample week indices
    iso_weeks <- format(as.Date(df$date), "%G-W%V")
    unique_weeks <- unique(iso_weeks)
    n_weeks <- length(unique_weeks)
    # Pre-compute row indices for each week (avoids repeated subsetting)
    week_row_idx <- split(seq_len(n), iso_weeks)

    # Pre-extract vectors to avoid repeated df column access
    outage_vec <- as.integer(df$outage_3h_or_more == 1L)
    cause_vec <- as.character(df$cause_std)
    prob_vec <- df$prob_policy

    for (b in seq_len(n_boot)) {
      sampled_weeks <- sample(unique_weeks, n_weeks, replace = TRUE)
      idx <- unlist(week_row_idx[sampled_weeks], use.names = FALSE)
      b_outage <- outage_vec[idx]
      b_cause <- cause_vec[idx]
      b_prob <- prob_vec[idx]
      y <- as.integer(b_outage == 1L & b_cause == cause_name)
      pred <- as.integer(b_prob >= threshold)
      m <- compute_binary_metrics(y_true01 = y, prob = b_prob, pred01 = pred)
      out[b, ] <- c(m$roc_auc, m$pr_auc, m$precision, m$recall)
    }
  } else {
    # Row-level bootstrap (original)
    for (b in seq_len(n_boot)) {
      idx <- sample.int(n, n, replace = TRUE)
      bdf <- df[idx, , drop = FALSE]
      y <- as.integer(bdf$outage_3h_or_more == 1L & bdf$cause_std == cause_name)
      pred <- as.integer(bdf$prob_policy >= threshold)
      m <- compute_binary_metrics(y_true01 = y, prob = bdf$prob_policy, pred01 = pred)
      out[b, ] <- c(m$roc_auc, m$pr_auc, m$precision, m$recall)
    }
  }

  ci <- function(x, p) as.numeric(stats::quantile(x, probs = p, na.rm = TRUE, names = FALSE))

  tibble(
    roc_auc_ci_lower = ci(out[, "roc_auc"], 0.025),
    roc_auc_ci_upper = ci(out[, "roc_auc"], 0.975),
    pr_auc_ci_lower = ci(out[, "pr_auc"], 0.025),
    pr_auc_ci_upper = ci(out[, "pr_auc"], 0.975),
    precision_ci_lower = ci(out[, "precision"], 0.025),
    precision_ci_upper = ci(out[, "precision"], 0.975),
    recall_ci_lower = ci(out[, "recall"], 0.025),
    recall_ci_upper = ci(out[, "recall"], 0.975),
    n_boot = n_boot,
    n_boot_valid_roc = sum(!is.na(out[, "roc_auc"])),
    n_boot_valid_pr = sum(!is.na(out[, "pr_auc"])),
    bootstrap_method = method
  )
}

# Backwards-compatible wrapper
bootstrap_ci <- function(df, cause_name, threshold, n_boot = 400, seed = 42) {
  run_bootstrap(df, cause_name, threshold, n_boot, seed, method = "row") %>%
    select(-bootstrap_method)
}

compute_cause_table <- function(df, threshold, n_boot, seed,
                                boot_method = "row") {
  causes <- c("Environmental", "Technical", "Planned", "Other")

  map_dfr(seq_along(causes), function(i) {
    cause_name <- causes[[i]]
    y <- as.integer(df$outage_3h_or_more == 1L & df$cause_std == cause_name)
    pred <- as.integer(df$prob_policy >= threshold)
    point <- compute_binary_metrics(y_true01 = y, prob = df$prob_policy, pred01 = pred)
    ci <- run_bootstrap(df = df, cause_name = cause_name, threshold = threshold,
                        n_boot = n_boot, seed = seed + i, method = boot_method)

    bind_cols(
      tibble(
        cause = cause_name,
        analysis_priority = if_else(cause_name %in% c("Environmental", "Technical"),
                                    "primary", "exploratory")
      ),
      point,
      ci
    )
  })
}

compare_env_tech <- function(df, threshold, n_boot, seed,
                             boot_method = "row") {
  get_vec <- function(cause_name) {
    y <- as.integer(df$outage_3h_or_more == 1L & df$cause_std == cause_name)
    pred <- as.integer(df$prob_policy >= threshold)
    m <- compute_binary_metrics(y_true01 = y, prob = df$prob_policy, pred01 = pred)
    c(roc_auc = m$roc_auc, pr_auc = m$pr_auc, precision = m$precision, recall = m$recall)
  }

  env <- get_vec("Environmental")
  tech <- get_vec("Technical")
  point_diff <- tech - env

  set.seed(seed)
  n <- nrow(df)
  boot_diff <- matrix(NA_real_, nrow = n_boot, ncol = 4)
  colnames(boot_diff) <- names(point_diff)

  if (boot_method == "block") {
    iso_weeks <- format(as.Date(df$date), "%G-W%V")
    unique_weeks <- unique(iso_weeks)
    n_weeks <- length(unique_weeks)
    week_row_idx <- split(seq_len(n), iso_weeks)

    for (b in seq_len(n_boot)) {
      sampled_weeks <- sample(unique_weeks, n_weeks, replace = TRUE)
      idx <- unlist(week_row_idx[sampled_weeks], use.names = FALSE)
      bdf <- df[idx, , drop = FALSE]
      env_b <- get_vec_from_df(bdf, "Environmental", threshold)
      tech_b <- get_vec_from_df(bdf, "Technical", threshold)
      boot_diff[b, ] <- tech_b - env_b
    }
  } else {
    for (b in seq_len(n_boot)) {
      idx <- sample.int(n, n, replace = TRUE)
      bdf <- df[idx, , drop = FALSE]
      env_b <- get_vec_from_df(bdf, "Environmental", threshold)
      tech_b <- get_vec_from_df(bdf, "Technical", threshold)
      boot_diff[b, ] <- tech_b - env_b
    }
  }

  ci <- function(x, p) as.numeric(stats::quantile(x, probs = p, na.rm = TRUE, names = FALSE))

  tibble(
    metric = names(point_diff),
    technical_minus_environmental = as.numeric(point_diff),
    ci_lower = map_dbl(seq_along(point_diff), ~ ci(boot_diff[, .x], 0.025)),
    ci_upper = map_dbl(seq_along(point_diff), ~ ci(boot_diff[, .x], 0.975)),
    n_boot = n_boot,
    bootstrap_method = boot_method
  )
}

get_vec_from_df <- function(df, cause_name, threshold) {
  y <- as.integer(df$outage_3h_or_more == 1L & df$cause_std == cause_name)
  pred <- as.integer(df$prob_policy >= threshold)
  m <- compute_binary_metrics(y_true01 = y, prob = df$prob_policy, pred01 = pred)
  c(roc_auc = m$roc_auc, pr_auc = m$pr_auc, precision = m$precision, recall = m$recall)
}

# ARGUMENTS / PATHS

args <- commandArgs(trailingOnly = TRUE)
run_id <- arg_or_default(args, 1, "20260223_125430_forecast_strict")
policy_method <- arg_or_default(args, 2, "platt")
policy_type <- arg_or_default(args, 3, "alerts_per_day_cap")
policy_target_arg <- arg_or_default(args, 4, "10")
n_boot <- as.integer(arg_or_default(args, 5, "400"))

policy_target_num <- safe_as_numeric(policy_target_arg)
if (!is.finite(policy_target_num)) policy_target_num <- NA_real_
if (!is.finite(n_boot) || n_boot < 50) n_boot <- 400L

project_dir <- detect_project_dir()
run_dir <- file.path(project_dir, "data", "baselines", "binary_threshold", run_id)
out_dir <- file.path(run_dir, "cause_specific_eval")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

features_path <- file.path(project_dir, "data", "model_ready", "features_engineered.rds")
splits_path <- file.path(project_dir, "data", "model_ready", "splits_fixed.rds")
panel_path <- file.path(project_dir, "data", "panel_mex_2017_2021_ntl_ghs.rds")

# Optional path overrides
features_override <- Sys.getenv("FEATURES_PATH_OVERRIDE", "")
splits_override <- Sys.getenv("SPLITS_PATH_OVERRIDE", "")
panel_override <- Sys.getenv("PANEL_PATH_OVERRIDE", "")
if (nzchar(features_override)) features_path <- features_override
if (nzchar(splits_override)) splits_path <- splits_override
if (nzchar(panel_override)) panel_path <- panel_override

deployment_path <- file.path(run_dir, "deployment_decision_table.csv")
platt_path <- file.path(run_dir, "platt_calibration.csv")
alerts_test_path <- file.path(run_dir, "alerts_per_day_test.csv")
features_used_path <- file.path(run_dir, "features_used.txt")

message(strrep("=", 70))
message("13_cause_specific_eval.R")
message(strrep("=", 70))
message("Run ID: ", run_id)
message("Method: ", policy_method)
message("Policy: ", policy_type, " (target=", policy_target_arg, ")")
message("Bootstrap N: ", n_boot)
message("Output dir: ", out_dir)
message("Features path: ", features_path)
message("Splits path: ", splits_path)
message("Panel path: ", panel_path)

required_files <- c(deployment_path, alerts_test_path, features_used_path, features_path, splits_path, panel_path)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop("Missing required files:\n", paste0(" - ", missing_files, collapse = "\n"))
}

# LOAD POLICY SETTINGS FROM STRICT RUN ARTIFACTS

deployment_tbl <- read_csv(deployment_path, show_col_types = FALSE)

policy_rows <- deployment_tbl %>%
  filter(method == policy_method, policy_type == policy_type)

if (!is.na(policy_target_num)) {
  policy_rows <- policy_rows %>%
    filter(!is.na(policy_target), abs(as.numeric(policy_target) - policy_target_num) < 1e-9)
}

if (nrow(policy_rows) == 0) {
  policy_rows <- deployment_tbl %>% filter(role == "default", method == policy_method)
}
if (nrow(policy_rows) == 0) {
  stop("Could not find policy row in deployment_decision_table.csv for method=", policy_method,
       ", policy_type=", policy_type, ", policy_target=", policy_target_arg)
}

policy_row <- policy_rows %>% slice(1)
policy_threshold <- as.numeric(policy_row$threshold)
if (!is.finite(policy_threshold)) stop("Selected policy threshold is missing/invalid.")

message("Selected threshold from deployment table: ", sprintf("%.6f", policy_threshold))

test_dates <- read_csv(alerts_test_path, show_col_types = FALSE) %>%
  transmute(date = as.Date(date)) %>%
  pull(date) %>%
  unique()

# REBUILD TEST SCORES (RAW) USING THE SAME FEATURE SET

feature_cols <- read_lines(features_used_path)
feature_cols <- feature_cols[nzchar(feature_cols)]
feature_cols <- unique(feature_cols)

features <- readRDS(features_path)
splits_fixed <- readRDS(splits_path)

train_start <- as.Date(splits_fixed$train_range[1])
train_end <- as.Date(splits_fixed$train_range[2])
val_start <- as.Date(splits_fixed$val_range[1])
val_end <- as.Date(splits_fixed$val_range[2])
test_start <- as.Date(splits_fixed$test_range[1])
test_end <- as.Date(splits_fixed$test_range[2])

features <- features %>%
  mutate(
    date = as.Date(date),
    GID_2 = as.character(GID_2),
    split = case_when(
      date >= train_start & date <= train_end ~ "train",
      date >= val_start & date <= val_end ~ "val",
      date >= test_start & date <= test_end ~ "test",
      TRUE ~ "other"
    )
  )

missing_feats <- setdiff(feature_cols, names(features))
if (length(missing_feats) > 0) {
  stop("Missing features from features_engineered.rds:\n",
       paste0(" - ", head(missing_feats, 30), collapse = "\n"),
       if (length(missing_feats) > 30) "\n ... (truncated)" else "")
}

train_data <- features %>% filter(split == "train")
test_data <- features %>% filter(split == "test", date %in% test_dates)

if (nrow(test_data) == 0) stop("No test rows found after applying run test dates.")

prep_model_df <- function(df, model_features) {
  df %>%
    select(all_of(c("outage_3h_or_more", model_features))) %>%
    mutate(
      outage_3h_or_more = factor(as.character(outage_3h_or_more), levels = c("1", "0"))
    ) %>%
    mutate(across(where(is.logical), as.integer))
}

train_model <- prep_model_df(train_data, feature_cols)
test_model <- prep_model_df(test_data, feature_cols)

n_pos <- sum(train_model$outage_3h_or_more == "1", na.rm = TRUE)
n_neg <- sum(train_model$outage_3h_or_more == "0", na.rm = TRUE)
scale_pos_weight <- n_neg / max(1, n_pos)

run_config_path <- file.path(run_dir, "run_config.json")
if (!file.exists(run_config_path)) {
  stop("run_config.json not found in run directory: ", run_config_path)
}
run_config <- jsonlite::read_json(run_config_path)
xgb_params <- run_config$model$params
message("Loaded XGBoost params from run_config.json: tree_depth=", xgb_params$tree_depth)

n_features <- length(feature_cols)
mtry_count <- max(1, floor(xgb_params$mtry * n_features))

model_recipe <- recipe(outage_3h_or_more ~ ., data = train_model) %>%
  step_zv(all_predictors())

xgb_spec <- boost_tree(
  trees = xgb_params$trees,
  tree_depth = xgb_params$tree_depth,
  min_n = xgb_params$min_n,
  loss_reduction = xgb_params$loss_reduction,
  sample_size = xgb_params$sample_size,
  mtry = mtry_count,
  learn_rate = xgb_params$learn_rate
) %>%
  set_engine(
    "xgboost",
    scale_pos_weight = scale_pos_weight,
    nthread = get_nthread(),
    verbosity = 0
  ) %>%
  set_mode("classification")

xgb_fit <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(xgb_spec) %>%
  fit(data = train_model)

test_prob_raw <- predict(xgb_fit, new_data = test_model, type = "prob") %>%
  pull(.pred_1)

# APPLY SELECTED POLICY SCORE (RAW OR PLATT)

if (policy_method == "raw") {
  test_prob_policy <- test_prob_raw
} else if (policy_method == "platt") {
  if (!file.exists(platt_path)) stop("platt_calibration.csv not found in run directory: ", platt_path)
  platt_params <- read_csv(platt_path, show_col_types = FALSE) %>% slice(1)
  platt_intercept <- as.numeric(platt_params$platt_intercept)
  platt_slope <- as.numeric(platt_params$platt_slope)
  if (!is.finite(platt_intercept) || !is.finite(platt_slope)) {
    stop("Invalid Platt parameters in platt_calibration.csv")
  }
  test_prob_policy <- apply_platt_with_params(test_prob_raw, platt_intercept, platt_slope)
} else {
  stop("Minimal script supports method='raw' or method='platt'. Requested: ", policy_method)
}

pred_df <- test_data %>%
  transmute(
    GID_2 = as.character(GID_2),
    date = as.Date(date),
    outage_3h_or_more_model = as.integer(outage_3h_or_more),
    prob_raw = test_prob_raw,
    prob_policy = test_prob_policy,
    pred_policy = as.integer(prob_policy >= policy_threshold)
  )

# JOIN DOMINANT-CAUSE LABELS FROM PANEL

panel <- readRDS(panel_path)
required_panel_cols <- c("GID_2", "date", "outage_3h_or_more", "classification_general")
missing_panel_cols <- setdiff(required_panel_cols, names(panel))
if (length(missing_panel_cols) > 0) {
  stop("Panel missing required columns:\n", paste0(" - ", missing_panel_cols, collapse = "\n"))
}

label_cols <- c(required_panel_cols, intersect("is_mixed_cause_general", names(panel)))
panel_labels <- panel %>%
  select(all_of(label_cols)) %>%
  mutate(
    GID_2 = as.character(GID_2),
    date = as.Date(date),
    outage_3h_or_more = as.integer(outage_3h_or_more)
  ) %>%
  distinct(GID_2, date, .keep_all = TRUE) %>%
  rename(outage_3h_or_more_label = outage_3h_or_more)

eval_df <- pred_df %>%
  left_join(panel_labels, by = c("GID_2", "date")) %>%
  mutate(
    outage_3h_or_more = coalesce(outage_3h_or_more_label, outage_3h_or_more_model),
    cause_std = normalize_cause(classification_general),
    cause_std = if_else(is.na(cause_std), NA_character_, as.character(cause_std)),
    cause_std = if_else(!is.na(cause_std) & !(cause_std %in% c("Environmental", "Technical", "Planned", "Other")),
                        "Other", cause_std)
  )

n_missing_cause_positive <- sum(eval_df$outage_3h_or_more == 1L & is.na(eval_df$cause_std), na.rm = TRUE)

# Do not force missing-cause positives into "rest" negatives.
eval_main <- eval_df %>%
  filter(!(outage_3h_or_more == 1L & is.na(cause_std)))

if (nrow(eval_main) == 0) stop("No rows left for cause evaluation.")

# MAIN CAUSE TABLE + CIs

add_cause_table_metadata <- function(tbl, method, policy_row, policy_type,
                                     policy_target_arg, policy_threshold,
                                     n_eval, n_dates, n_missing, note) {
  tbl %>%
    mutate(
      method = method,
      policy_type = policy_row$policy_type %||% policy_type,
      policy_target = policy_row$policy_target %||% policy_target_arg,
      policy_threshold = policy_threshold,
      n_eval_rows = n_eval,
      n_test_dates = n_dates,
      n_missing_cause_positive_excluded = n_missing,
      notes = note
    )
}

# Row-level bootstrap (original) 
message("\nComputing row-level bootstrap CIs...")
cause_table <- compute_cause_table(
  df = eval_main, threshold = policy_threshold,
  n_boot = n_boot, seed = 42, boot_method = "row"
) %>%
  add_cause_table_metadata(
    policy_method, policy_row, policy_type, policy_target_arg,
    policy_threshold, nrow(eval_main), n_distinct(eval_main$date),
    n_missing_cause_positive,
    "One-vs-rest over all municipality-nights in test window; row-level bootstrap."
  )

# Week-block bootstrap 
message("Computing week-block bootstrap CIs...")
n_iso_weeks <- n_distinct(format(as.Date(eval_main$date), "%G-W%V"))
message("  ISO weeks in test period: ", n_iso_weeks)

cause_table_block <- compute_cause_table(
  df = eval_main, threshold = policy_threshold,
  n_boot = n_boot, seed = 42, boot_method = "block"
) %>%
  add_cause_table_metadata(
    policy_method, policy_row, policy_type, policy_target_arg,
    policy_threshold, nrow(eval_main), n_distinct(eval_main$date),
    n_missing_cause_positive,
    paste0("One-vs-rest; week-block bootstrap (", n_iso_weeks, " ISO weeks).")
  )

support_table <- eval_main %>%
  filter(outage_3h_or_more == 1L) %>%
  mutate(cause_std = if_else(is.na(cause_std), "Missing", cause_std)) %>%
  count(cause_std, name = "n_positive_nights") %>%
  mutate(share_positive_nights = n_positive_nights / sum(n_positive_nights)) %>%
  arrange(desc(n_positive_nights))

# Env vs Tech comparison with both bootstrap methods
env_tech_row <- compare_env_tech(
  df = eval_main, threshold = policy_threshold,
  n_boot = n_boot, seed = 123, boot_method = "row"
)
env_tech_block <- compare_env_tech(
  df = eval_main, threshold = policy_threshold,
  n_boot = n_boot, seed = 123, boot_method = "block"
)
env_tech_comparison <- bind_rows(env_tech_row, env_tech_block) %>%
  mutate(
    method = policy_method,
    policy_type = policy_row$policy_type %||% policy_type,
    policy_target = policy_row$policy_target %||% policy_target_arg,
    policy_threshold = policy_threshold
  ) %>%
  select(method, policy_type, policy_target, policy_threshold, everything())

# Optional sensitivity: exclude mixed-cause positives if column exists
has_mixed_flag <- "is_mixed_cause_general" %in% names(eval_main)
if (has_mixed_flag) {
  eval_nomixed <- eval_main %>%
    filter(!(outage_3h_or_more == 1L & as.integer(is_mixed_cause_general) == 1L))

  cause_table_nomixed <- compute_cause_table(
    df = eval_nomixed,
    threshold = policy_threshold,
    n_boot = n_boot,
    seed = 1042
  ) %>%
    mutate(
      method = policy_method,
      policy_type = policy_row$policy_type %||% policy_type,
      policy_target = policy_row$policy_target %||% policy_target_arg,
      policy_threshold = policy_threshold,
      n_eval_rows = nrow(eval_nomixed),
      n_test_dates = n_distinct(eval_nomixed$date),
      n_missing_cause_positive_excluded = n_missing_cause_positive,
      notes = "Sensitivity: mixed-cause positives excluded; row-level bootstrap."
    )

  write_csv(cause_table_nomixed, file.path(out_dir, "cause_metrics_excluding_mixed.csv"))

  # Delta table: main vs no-mixed comparison
  delta_df <- cause_table %>%
    select(cause, roc_auc_main = roc_auc, pr_auc_main = pr_auc,
           precision_main = precision, recall_main = recall) %>%
    left_join(
      cause_table_nomixed %>%
        select(cause, roc_auc_nomixed = roc_auc, pr_auc_nomixed = pr_auc,
               precision_nomixed = precision, recall_nomixed = recall),
      by = "cause"
    ) %>%
    mutate(
      delta_roc_auc = roc_auc_nomixed - roc_auc_main,
      delta_pr_auc = pr_auc_nomixed - pr_auc_main,
      delta_precision = precision_nomixed - precision_main,
      delta_recall = recall_nomixed - recall_main
    )
  write_csv(delta_df, file.path(out_dir, "cause_metrics_mixed_sensitivity_delta.csv"))
} else {
  write_lines(
    "Mixed-cause sensitivity skipped: column is_mixed_cause_general not present in panel.",
    file.path(out_dir, "cause_metrics_excluding_mixed_NOTE.txt")
  )
}

write_csv(cause_table, file.path(out_dir, "cause_metrics_main.csv"))
write_csv(cause_table_block, file.path(out_dir, "cause_metrics_main_block_bootstrap.csv"))
write_csv(support_table, file.path(out_dir, "cause_support_counts.csv"))
write_csv(env_tech_comparison, file.path(out_dir, "env_vs_tech_comparison.csv"))

summary_lines <- c(
  paste0("run_id: ", run_id),
  paste0("method: ", policy_method),
  paste0("policy_type: ", policy_row$policy_type %||% policy_type),
  paste0("policy_target: ", policy_row$policy_target %||% policy_target_arg),
  paste0("policy_threshold: ", sprintf("%.6f", policy_threshold)),
  paste0("n_eval_rows: ", nrow(eval_main)),
  paste0("n_test_dates: ", n_distinct(eval_main$date)),
  paste0("n_iso_weeks: ", n_iso_weeks),
  paste0("n_missing_cause_positive_excluded: ", n_missing_cause_positive),
  "bootstrap_types: row, block (ISO week)",
  "bootstrap_note: block bootstrap produces more honest CIs under temporal dependence."
)
write_lines(summary_lines, file.path(out_dir, "cause_eval_run_summary.txt"))

message("\nSaved outputs:")
message(" - ", file.path(out_dir, "cause_metrics_main.csv"))
message(" - ", file.path(out_dir, "cause_metrics_main_block_bootstrap.csv"))
message(" - ", file.path(out_dir, "cause_support_counts.csv"))
message(" - ", file.path(out_dir, "env_vs_tech_comparison.csv"))
if (has_mixed_flag) {
  message(" - ", file.path(out_dir, "cause_metrics_excluding_mixed.csv"))
  message(" - ", file.path(out_dir, "cause_metrics_mixed_sensitivity_delta.csv"))
} else {
  message(" - ", file.path(out_dir, "cause_metrics_excluding_mixed_NOTE.txt"))
}
message(" - ", file.path(out_dir, "cause_eval_run_summary.txt"))
