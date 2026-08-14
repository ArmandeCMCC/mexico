# Phase 6: corrected two-part count and cumulative-minutes models.
# Existing analysis scripts and outputs are read-only inputs.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(patchwork)
  library(readr)
  library(tibble)
  library(tidyr)
  library(xgboost)
})

set.seed(20260813)

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))
  stop("Run from the project root or scripts directory.")
}

parse_arg <- function(flag, default = NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  hit <- which(args == flag)
  if (!length(hit)) return(default)
  if (hit[1] == length(args)) stop("Missing value after ", flag)
  args[hit[1] + 1L]
}

project_dir <- detect_project_dir()
run_id <- parse_arg(
  "--run-id", paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_phase6_count_duration")
)
n_boot <- as.integer(parse_arg("--n-bootstrap", "1000"))
seed <- as.integer(parse_arg("--seed", "20260813"))
if (!is.finite(n_boot) || n_boot < 1000L) stop("Phase 6 requires at least 1,000 resamples.")

phase2_dir <- file.path(
  project_dir, "data", "revision", "20260813_144700_phase2_canonical_comparison"
)
phase4_dir <- file.path(
  project_dir, "data", "revision", "20260813_214000_phase4_classifier_uncertainty"
)
features_path <- file.path(project_dir, "data", "model_ready", "features_engineered.rds")
panel_path <- file.path(project_dir, "data", "panel_mex_2017_2021_ntl_ghs.rds")
feature_list_path <- file.path(phase2_dir, "feature_sets", "full_83_xgb.txt")
phase2_predictions_path <- file.path(phase2_dir, "paired_predictions_test.rds")
phase2_config_path <- file.path(phase2_dir, "run_config.json")
phase2_qa_path <- file.path(phase2_dir, "independent_qa_checks.csv")
block_counts_path <- file.path(phase4_dir, "bootstrap_block_counts.rds")
phase4_qa_path <- file.path(phase4_dir, "independent_qa_checks.csv")
required_inputs <- c(
  features_path, panel_path, feature_list_path, phase2_predictions_path,
  phase2_config_path, phase2_qa_path, block_counts_path, phase4_qa_path
)
missing_inputs <- required_inputs[!file.exists(required_inputs)]
if (length(missing_inputs)) stop("Missing required input(s): ", paste(missing_inputs, collapse = ", "))

output_dir <- file.path(project_dir, "data", "revision", run_id)
figure_dir <- file.path(project_dir, "figures", "revision", run_id)
model_dir <- file.path(output_dir, "models")
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to overwrite non-empty run directory: ", output_dir)
}
if (dir.exists(figure_dir) && length(list.files(figure_dir, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to overwrite non-empty figure directory: ", figure_dir)
}
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
log_path <- file.path(output_dir, "run.log")
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
log_message <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", msg, "\n",
      file = log_path, append = TRUE, sep = "")
}

phase2_qa <- read_csv(phase2_qa_path, show_col_types = FALSE)
phase4_qa <- read_csv(phase4_qa_path, show_col_types = FALSE)
if (any(phase2_qa$status == "FAIL") || any(phase4_qa$status == "FAIL")) {
  stop("A definitive upstream run contains failed independent checks.")
}

headline_features <- readLines(feature_list_path, warn = FALSE)
headline_features <- unique(headline_features[nzchar(headline_features)])
if (length(headline_features) != 83L) stop("Expected exactly 83 frozen headline features.")
phase2_config <- read_json(phase2_config_path, simplifyVector = TRUE)
xgb_frozen <- phase2_config$xgboost
max_rounds <- as.integer(xgb_frozen$trees)
early_stopping_rounds <- 30L
detected_cores <- suppressWarnings(parallel::detectCores())
nthread <- if (is.na(detected_cores) || detected_cores < 2L) 1L else as.integer(detected_cores - 1L)

train_start <- as.Date("2017-01-01")
train_end <- as.Date("2019-12-31")
validation_start <- as.Date("2020-01-01")
validation_end <- as.Date("2020-06-30")
test_start <- as.Date("2020-07-01")
test_end <- as.Date("2021-12-31")

make_matrix <- function(df, features) {
  x <- as.data.table(df)[, ..features]
  logical_cols <- names(x)[vapply(x, is.logical, logical(1))]
  if (length(logical_cols)) x[, (logical_cols) := lapply(.SD, as.integer), .SDcols = logical_cols]
  nonnumeric <- names(x)[!vapply(x, is.numeric, logical(1))]
  if (length(nonnumeric)) x[, (nonnumeric) := lapply(.SD, as.numeric), .SDcols = nonnumeric]
  as.matrix(x)
}

base_params <- list(
  eta = as.numeric(xgb_frozen$learn_rate),
  max_depth = as.integer(xgb_frozen$tree_depth),
  min_child_weight = as.numeric(xgb_frozen$min_n),
  gamma = as.numeric(xgb_frozen$loss_reduction),
  subsample = as.numeric(xgb_frozen$sample_size),
  colsample_bytree = as.numeric(xgb_frozen$mtry),
  nthread = nthread,
  seed = seed,
  verbosity = 0
)

fit_builtin <- function(objective, eval_metric, dtrain, dvalidation, label,
                        extra_params = list()) {
  params <- modifyList(base_params, c(
    list(objective = objective, eval_metric = eval_metric), extra_params
  ))
  set.seed(seed)
  tuned <- xgb.train(
    params = params, data = dtrain, nrounds = max_rounds,
    evals = list(validation = dvalidation),
    early_stopping_rounds = early_stopping_rounds, verbose = 0
  )
  best_iteration <- as.integer(tuned$best_iteration %||% max_rounds)
  set.seed(seed)
  final <- xgb.train(params = params, data = dtrain, nrounds = best_iteration, verbose = 0)
  log_message(label, ": selected ", best_iteration, " boosting rounds")
  list(model = final, best_iteration = best_iteration, params = params)
}

`%||%` <- function(x, y) if (is.null(x) || !length(x) || is.na(x)) y else x

metric_set <- function(actual, predicted, type, winsor_cap = NA_real_,
                       tail90 = NA_real_, tail95 = NA_real_) {
  err <- predicted - actual
  out <- c(
    mae = mean(abs(err)), rmse = sqrt(mean(err^2)),
    median_absolute_error = median(abs(err)), mean_bias = mean(err),
    pearson = suppressWarnings(cor(predicted, actual, method = "pearson")),
    spearman = suppressWarnings(cor(predicted, actual, method = "spearman")),
    calibration_ratio = sum(predicted) / sum(actual)
  )
  if (type == "severity") {
    actual_w <- pmin(actual, winsor_cap)
    predicted_w <- pmin(predicted, winsor_cap)
    idx90 <- actual >= tail90
    idx95 <- actual >= tail95
    out <- c(
      out,
      winsorized_mae_99 = mean(abs(predicted_w - actual_w)),
      winsorized_rmse_99 = sqrt(mean((predicted_w - actual_w)^2)),
      tail90_mae = mean(abs(err[idx90])), tail90_bias = mean(err[idx90]),
      tail95_mae = mean(abs(err[idx95])), tail95_bias = mean(err[idx95])
    )
  }
  out
}

log_message("Phase 6 corrected count and duration models")
log_message("Run ID: ", run_id)
log_message("Loading panel targets and canonical model-ready features...")

panel <- as.data.table(readRDS(panel_path))
panel[, date := as.Date(date)]
panel_targets <- unique(panel[, .(
  GID_2 = as.character(GID_2), date, total_length_min = as.numeric(total_length_min),
  state_name = as.character(NAME_1)
)], by = c("GID_2", "date"))
rm(panel); gc(verbose = FALSE)

needed <- c("GID_2", "date", "outage_3h_or_more", "n_outages", headline_features)
features <- as.data.table(readRDS(features_path))
missing_features <- setdiff(needed, names(features))
if (length(missing_features)) stop("Missing model-ready columns: ", paste(missing_features, collapse = ", "))
dat <- features[, ..needed]
rm(features); gc(verbose = FALSE)
dat[, `:=`(
  GID_2 = as.character(GID_2), date = as.Date(date),
  outage_3h_or_more = as.integer(outage_3h_or_more), n_outages = as.integer(n_outages)
)]
if (anyDuplicated(dat, by = c("GID_2", "date"))) stop("Duplicate model-ready keys.")
setkey(dat, GID_2, date)
setkey(panel_targets, GID_2, date)
dat[panel_targets, `:=`(
  total_length_min = i.total_length_min, state_name = i.state_name
)]
if (anyNA(dat$total_length_min) || anyNA(dat$state_name)) stop("Panel target join is incomplete.")

split_name <- fifelse(
  dat$date >= train_start & dat$date <= train_end, "train",
  fifelse(dat$date >= validation_start & dat$date <= validation_end, "validation",
          fifelse(dat$date >= test_start & dat$date <= test_end, "test", "other"))
)
dat[, split := split_name]
support <- dat[split %in% c("train", "validation", "test"), .(
  n_rows = .N, n_positive = sum(outage_3h_or_more), n_nights = uniqueN(date),
  n_municipalities = uniqueN(GID_2), total_events = sum(n_outages),
  total_minutes = sum(total_length_min)
), by = split][match(c("train", "validation", "test"), split)]
write_csv(as_tibble(support), file.path(output_dir, "split_target_support.csv"))

train_pos <- dat[split == "train" & outage_3h_or_more == 1L]
validation_pos <- dat[split == "validation" & outage_3h_or_more == 1L]
test_pos <- dat[split == "test" & outage_3h_or_more == 1L]
if (!identical(c(nrow(train_pos), nrow(validation_pos), nrow(test_pos)),
               c(15951L, 1695L, 5350L))) {
  stop("Positive split support drifted from the audited panel.")
}
if (any(train_pos$n_outages < 1L) || any(test_pos$total_length_min <= 0)) {
  stop("Conditional outcomes violate positive-support assumptions.")
}

X_train <- make_matrix(train_pos, headline_features)
X_validation <- make_matrix(validation_pos, headline_features)
X_test <- make_matrix(test_pos, headline_features)

# Count hurdle: model excess events so E[N | Y=1] is structurally at least one.
y_count_train <- train_pos$n_outages - 1
y_count_validation <- validation_pos$n_outages - 1
y_count_test <- test_pos$n_outages
dtrain_count <- xgb.DMatrix(X_train, label = y_count_train, missing = NA)
dvalidation_count <- xgb.DMatrix(X_validation, label = y_count_validation, missing = NA)
dtest <- xgb.DMatrix(X_test, missing = NA)

poisson_fit <- fit_builtin(
  "count:poisson", "poisson-nloglik", dtrain_count, dvalidation_count,
  "Poisson excess-count model"
)
poisson_validation_excess <- pmax(predict(poisson_fit$model, dvalidation_count), 0)
poisson_test_count <- 1 + pmax(predict(poisson_fit$model, dtest), 0)

# NB2 dispersion is estimated on held-out validation residuals from the Poisson mean.
alpha_numerator <- sum((y_count_validation - poisson_validation_excess)^2 - y_count_validation)
alpha_denominator <- sum(poisson_validation_excess^2)
nb_alpha <- max(alpha_numerator / alpha_denominator, 1e-6)
nb_theta <- 1 / nb_alpha
log_message("NB2 validation dispersion alpha: ", sprintf("%.6f", nb_alpha))

nb_objective <- function(pred, dtrain) {
  y <- getinfo(dtrain, "label")
  mu <- exp(pmin(pmax(pred, -20), 20))
  theta <- nb_theta
  list(
    grad = theta * (mu - y) / (mu + theta),
    hess = theta * mu * (theta + y) / (mu + theta)^2
  )
}
nb_metric <- function(pred, dtrain) {
  y <- getinfo(dtrain, "label")
  mu <- exp(pmin(pmax(pred, -20), 20))
  theta <- nb_theta
  nll <- -mean(
    lgamma(y + theta) - lgamma(theta) - lgamma(y + 1) +
      theta * log(theta / (theta + mu)) + y * log(mu / (theta + mu))
  )
  list(metric = "nb_nloglik", value = nll)
}
nb_params <- modifyList(base_params, list(
  base_score = log(max(mean(y_count_train), 1e-6)),
  disable_default_eval_metric = 1
))
set.seed(seed)
nb_tuned <- xgb.train(
  params = nb_params, data = dtrain_count, nrounds = max_rounds,
  evals = list(validation = dvalidation_count), objective = nb_objective,
  custom_metric = nb_metric, maximize = FALSE,
  early_stopping_rounds = early_stopping_rounds, verbose = 0
)
nb_best_iteration <- as.integer(nb_tuned$best_iteration %||% max_rounds)
set.seed(seed)
nb_model <- xgb.train(
  params = nb_params, data = dtrain_count, nrounds = nb_best_iteration,
  objective = nb_objective, custom_metric = nb_metric, verbose = 0
)
nb_validation_excess <- exp(predict(nb_model, dvalidation_count, outputmargin = TRUE))
nb_test_count <- 1 + exp(predict(nb_model, dtest, outputmargin = TRUE))
log_message("NB2 excess-count model: selected ", nb_best_iteration, " boosting rounds")

# Cumulative minutes conditional on a positive municipality-night.
y_minutes_train <- train_pos$total_length_min
y_minutes_validation <- validation_pos$total_length_min
y_minutes_test <- test_pos$total_length_min
dtrain_log <- xgb.DMatrix(X_train, label = log1p(y_minutes_train), missing = NA)
dvalidation_log <- xgb.DMatrix(X_validation, label = log1p(y_minutes_validation), missing = NA)
dtrain_minutes <- xgb.DMatrix(X_train, label = y_minutes_train, missing = NA)
dvalidation_minutes <- xgb.DMatrix(X_validation, label = y_minutes_validation, missing = NA)

log_fit <- fit_builtin(
  "reg:squarederror", "rmse", dtrain_log, dvalidation_log,
  "Squared-error log1p duration model"
)
log_validation_margin <- predict(log_fit$model, dvalidation_log)
log_test_margin <- predict(log_fit$model, dtest)
duan_smearing <- mean(exp(log1p(y_minutes_validation) - log_validation_margin))
log_validation_minutes <- pmax(duan_smearing * exp(log_validation_margin) - 1, 0)
log_test_minutes <- pmax(duan_smearing * exp(log_test_margin) - 1, 0)
log_test_old_formula <- pmax(expm1(log_test_margin) * duan_smearing, 0)

gamma_fit <- fit_builtin(
  "reg:gamma", "gamma-nloglik", dtrain_minutes, dvalidation_minutes,
  "Gamma-log duration model"
)
gamma_validation_minutes <- pmax(predict(gamma_fit$model, dvalidation_minutes), 0)
gamma_test_minutes <- pmax(predict(gamma_fit$model, dtest), 0)

tweedie_powers <- c(1.1, 1.3, 1.5, 1.7, 1.9)
tweedie_candidates <- vector("list", length(tweedie_powers))
tweedie_tuning <- vector("list", length(tweedie_powers))
for (i in seq_along(tweedie_powers)) {
  power <- tweedie_powers[i]
  fit <- fit_builtin(
    "reg:tweedie", paste0("tweedie-nloglik@", power),
    dtrain_minutes, dvalidation_minutes,
    paste0("Tweedie-log duration model (power ", power, ")"),
    list(tweedie_variance_power = power)
  )
  pred_validation <- pmax(predict(fit$model, dvalidation_minutes), 0)
  tweedie_candidates[[i]] <- list(fit = fit, pred_validation = pred_validation)
  tweedie_tuning[[i]] <- tibble(
    variance_power = power, best_iteration = fit$best_iteration,
    validation_mae = mean(abs(pred_validation - y_minutes_validation)),
    validation_rmse = sqrt(mean((pred_validation - y_minutes_validation)^2)),
    validation_mean_bias = mean(pred_validation - y_minutes_validation)
  )
}
tweedie_tuning <- bind_rows(tweedie_tuning) %>%
  arrange(validation_mae, validation_rmse, variance_power)
selected_tweedie_power <- tweedie_tuning$variance_power[1]
selected_tweedie_index <- match(selected_tweedie_power, tweedie_powers)
tweedie_fit <- tweedie_candidates[[selected_tweedie_index]]$fit
tweedie_validation_minutes <- tweedie_candidates[[selected_tweedie_index]]$pred_validation
tweedie_test_minutes <- pmax(predict(tweedie_fit$model, dtest), 0)
write_csv(tweedie_tuning, file.path(output_dir, "tweedie_power_selection.csv"))

count_validation_predictions <- tibble(
  GID_2 = validation_pos$GID_2, date = validation_pos$date,
  actual_count = validation_pos$n_outages,
  poisson_excess = poisson_validation_excess,
  poisson_count = 1 + poisson_validation_excess,
  nb2_excess = nb_validation_excess,
  nb2_count = 1 + nb_validation_excess
)
count_test_predictions <- tibble(
  GID_2 = test_pos$GID_2, date = test_pos$date,
  actual_count = y_count_test,
  poisson_count = poisson_test_count, nb2_count = nb_test_count
)
severity_validation_predictions <- tibble(
  GID_2 = validation_pos$GID_2, date = validation_pos$date,
  actual_minutes = y_minutes_validation,
  log1p_duan_minutes = log_validation_minutes,
  gamma_minutes = gamma_validation_minutes,
  tweedie_minutes = tweedie_validation_minutes
)
conditional_test_predictions <- tibble(
  GID_2 = test_pos$GID_2, date = test_pos$date,
  iso_week = format(test_pos$date, "%G-W%V"),
  actual_count = y_count_test, actual_minutes = y_minutes_test,
  poisson_count = poisson_test_count, nb2_count = nb_test_count,
  log1p_margin = log_test_margin, log1p_duan_minutes = log_test_minutes,
  log1p_old_formula_minutes = log_test_old_formula,
  gamma_minutes = gamma_test_minutes, tweedie_minutes = tweedie_test_minutes
)
write_csv(count_validation_predictions, file.path(output_dir, "count_validation_predictions.csv"))
write_csv(severity_validation_predictions, file.path(output_dir, "severity_validation_predictions.csv"))
write_csv(conditional_test_predictions, file.path(output_dir, "conditional_test_predictions.csv"))

count_model_columns <- c(poisson = "poisson_count", negative_binomial = "nb2_count")
severity_model_columns <- c(
  log1p_duan = "log1p_duan_minutes", gamma_log = "gamma_minutes",
  tweedie_log = "tweedie_minutes"
)
count_validation_mae <- c(
  poisson = mean(abs(count_validation_predictions$poisson_count -
                       count_validation_predictions$actual_count)),
  negative_binomial = mean(abs(count_validation_predictions$nb2_count -
                                 count_validation_predictions$actual_count))
)
severity_validation_mae <- c(
  log1p_duan = mean(abs(log_validation_minutes - y_minutes_validation)),
  gamma_log = mean(abs(gamma_validation_minutes - y_minutes_validation)),
  tweedie_log = mean(abs(tweedie_validation_minutes - y_minutes_validation))
)
selected_count_model <- names(which.min(count_validation_mae))[1]
selected_severity_model <- names(which.min(severity_validation_mae))[1]
log_message("Selected count method by validation MAE: ", selected_count_model)
log_message("Selected severity method by validation MAE: ", selected_severity_model)

count_point <- bind_rows(lapply(names(count_model_columns), function(model_id) {
  vals <- metric_set(
    conditional_test_predictions$actual_count,
    conditional_test_predictions[[count_model_columns[[model_id]]]], "count"
  )
  tibble(model_id = model_id, metric = names(vals), point_estimate = as.numeric(vals))
}))

winsor_cap <- as.numeric(quantile(y_minutes_train, 0.99, names = FALSE))
tail90 <- as.numeric(quantile(y_minutes_train, 0.90, names = FALSE))
tail95 <- as.numeric(quantile(y_minutes_train, 0.95, names = FALSE))
severity_point <- bind_rows(lapply(names(severity_model_columns), function(model_id) {
  vals <- metric_set(
    conditional_test_predictions$actual_minutes,
    conditional_test_predictions[[severity_model_columns[[model_id]]]], "severity",
    winsor_cap, tail90, tail95
  )
  tibble(model_id = model_id, metric = names(vals), point_estimate = as.numeric(vals))
}))
write_csv(count_point, file.path(output_dir, "count_point_metrics.csv"))
write_csv(severity_point, file.path(output_dir, "severity_point_metrics.csv"))

# Paired ISO-week bootstrap uses the definitive Phase 4 draw matrix.
block_counts <- readRDS(block_counts_path)
iso_weeks <- colnames(block_counts)
if (is.null(iso_weeks) || nrow(block_counts) < n_boot) stop("Phase 4 block draws are not labeled.")
block_counts <- block_counts[seq_len(n_boot), , drop = FALSE]
test_block <- match(conditional_test_predictions$iso_week, iso_weeks)
if (anyNA(test_block)) stop("Conditional test rows do not map to Phase 4 ISO-week blocks.")

bootstrap_metrics <- function(actual, predicted, type) {
  out <- vector("list", n_boot)
  for (b in seq_len(n_boot)) {
    idx <- rep(seq_along(actual), times = block_counts[b, test_block])
    vals <- metric_set(
      actual[idx], predicted[idx], type,
      winsor_cap = winsor_cap, tail90 = tail90, tail95 = tail95
    )
    out[[b]] <- as_tibble_row(c(bootstrap_id = b, vals))
  }
  bind_rows(out)
}

log_message("Computing paired ISO-week confidence intervals...")
count_draws <- bind_rows(lapply(names(count_model_columns), function(model_id) {
  bootstrap_metrics(
    conditional_test_predictions$actual_count,
    conditional_test_predictions[[count_model_columns[[model_id]]]], "count"
  ) %>% mutate(model_id = model_id, .after = bootstrap_id)
}))
severity_draws <- bind_rows(lapply(names(severity_model_columns), function(model_id) {
  bootstrap_metrics(
    conditional_test_predictions$actual_minutes,
    conditional_test_predictions[[severity_model_columns[[model_id]]]], "severity"
  ) %>% mutate(model_id = model_id, .after = bootstrap_id)
}))
write_csv(count_draws, file.path(output_dir, "count_bootstrap_draws.csv"))
write_csv(severity_draws, file.path(output_dir, "severity_bootstrap_draws.csv"))

summarize_draws <- function(draws, points) {
  draws %>%
    pivot_longer(-c(bootstrap_id, model_id), names_to = "metric", values_to = "value") %>%
    group_by(model_id, metric) %>%
    summarise(
      bootstrap_mean = mean(value, na.rm = TRUE),
      bootstrap_se = sd(value, na.rm = TRUE),
      ci_lower = quantile(value, 0.025, na.rm = TRUE, names = FALSE),
      ci_upper = quantile(value, 0.975, na.rm = TRUE, names = FALSE),
      n_valid = sum(is.finite(value)), .groups = "drop"
    ) %>%
    left_join(points, by = c("model_id", "metric")) %>%
    select(model_id, metric, point_estimate, everything())
}
count_uncertainty <- summarize_draws(count_draws, count_point)
severity_uncertainty <- summarize_draws(severity_draws, severity_point)
write_csv(count_uncertainty, file.path(output_dir, "count_uncertainty_table.csv"))
write_csv(severity_uncertainty, file.path(output_dir, "severity_uncertainty_table.csv"))

paired_difference <- function(draws, points, pairs) {
  long <- draws %>% pivot_longer(
    -c(bootstrap_id, model_id), names_to = "metric", values_to = "value"
  )
  bind_rows(lapply(seq_len(nrow(pairs)), function(i) {
    expanded <- pairs$expanded[i]; reference <- pairs$reference[i]
    delta <- long %>% filter(model_id %in% c(expanded, reference)) %>%
      select(bootstrap_id, model_id, metric, value) %>%
      pivot_wider(names_from = model_id, values_from = value) %>%
      mutate(delta = .data[[expanded]] - .data[[reference]])
    point <- points %>% filter(model_id %in% c(expanded, reference)) %>%
      select(model_id, metric, point_estimate) %>%
      pivot_wider(names_from = model_id, values_from = point_estimate) %>%
      mutate(point_delta = .data[[expanded]] - .data[[reference]])
    delta %>% group_by(metric) %>% summarise(
      bootstrap_mean_delta = mean(delta, na.rm = TRUE),
      bootstrap_se_delta = sd(delta, na.rm = TRUE),
      ci_lower = quantile(delta, 0.025, na.rm = TRUE, names = FALSE),
      ci_upper = quantile(delta, 0.975, na.rm = TRUE, names = FALSE),
      probability_lower = mean(delta < 0, na.rm = TRUE), .groups = "drop"
    ) %>% left_join(point, by = "metric") %>% mutate(
      comparison_id = paste(expanded, "vs", reference, sep = "_"),
      expanded_model = expanded, reference_model = reference,
      ci_excludes_zero = ci_lower > 0 | ci_upper < 0,
      .before = 1
    )
  }))
}
count_differences <- paired_difference(
  count_draws, count_point,
  tibble(expanded = "negative_binomial", reference = "poisson")
)
severity_differences <- paired_difference(
  severity_draws, severity_point,
  tribble(
    ~expanded, ~reference,
    "gamma_log", "log1p_duan",
    "tweedie_log", "log1p_duan",
    "tweedie_log", "gamma_log"
  )
)
write_csv(count_differences, file.path(output_dir, "count_paired_differences.csv"))
write_csv(severity_differences, file.path(output_dir, "severity_paired_differences.csv"))

# Save final model objects before predicting the complete test panel in chunks.
xgb.save(poisson_fit$model, file.path(model_dir, "count_poisson_excess.ubj"))
xgb.save(nb_model, file.path(model_dir, "count_nb2_excess.ubj"))
xgb.save(log_fit$model, file.path(model_dir, "minutes_log1p.ubj"))
xgb.save(gamma_fit$model, file.path(model_dir, "minutes_gamma_log.ubj"))
xgb.save(tweedie_fit$model, file.path(model_dir, "minutes_tweedie_log.ubj"))

selected_count_object <- if (selected_count_model == "poisson") poisson_fit$model else nb_model
selected_severity_object <- switch(
  selected_severity_model,
  log1p_duan = log_fit$model, gamma_log = gamma_fit$model, tweedie_log = tweedie_fit$model
)
predict_selected_count <- function(dmat) {
  if (selected_count_model == "poisson") 1 + pmax(predict(selected_count_object, dmat), 0)
  else 1 + exp(predict(selected_count_object, dmat, outputmargin = TRUE))
}
predict_selected_minutes <- function(dmat) {
  if (selected_severity_model == "log1p_duan") {
    pmax(duan_smearing * exp(predict(selected_severity_object, dmat)) - 1, 0)
  } else {
    pmax(predict(selected_severity_object, dmat), 0)
  }
}

test_dat <- dat[split == "test"]
phase2_predictions <- as.data.table(readRDS(phase2_predictions_path))[, .(
  GID_2 = as.character(GID_2), date = as.Date(date), truth = as.integer(truth),
  full_probability = as.numeric(full_83_xgb_calibrated)
)]
setkey(test_dat, GID_2, date)
setkey(phase2_predictions, GID_2, date)
if (nrow(test_dat) != nrow(phase2_predictions) ||
    any(test_dat$outage_3h_or_more != phase2_predictions$truth)) {
  stop("Phase 2 predictions do not match the canonical test panel.")
}
test_dat[, full_probability := phase2_predictions$full_probability]

chunk_size <- 100000L
chunk_starts <- seq.int(1L, nrow(test_dat), by = chunk_size)
burden_chunks <- vector("list", length(chunk_starts))
log_message("Predicting corrected conditional burden for ", nrow(test_dat), " test rows...")
for (i in seq_along(chunk_starts)) {
  lo <- chunk_starts[i]
  hi <- min(lo + chunk_size - 1L, nrow(test_dat))
  dchunk <- xgb.DMatrix(make_matrix(test_dat[lo:hi], headline_features), missing = NA)
  conditional_count <- predict_selected_count(dchunk)
  conditional_minutes <- predict_selected_minutes(dchunk)
  chunk <- test_dat[lo:hi, .(
    GID_2, date, state_name, actual_outage = outage_3h_or_more,
    actual_count = n_outages, actual_minutes = total_length_min,
    full_probability
  )]
  chunk[, `:=`(
    conditional_count = conditional_count,
    conditional_minutes = conditional_minutes,
    expected_count = full_probability * conditional_count,
    expected_minutes = full_probability * conditional_minutes
  )]
  burden_chunks[[i]] <- chunk
}
burden <- rbindlist(burden_chunks)
rm(burden_chunks, test_dat, dat); gc(verbose = FALSE)
saveRDS(burden, file.path(output_dir, "decomposed_burden_test.rds"), compress = "gzip")

make_calibration_bins <- function(df, pred_col, actual_col, outcome) {
  copy(df)[, decile := frank(get(pred_col), ties.method = "average") / .N][,
    decile := pmin(10L, pmax(1L, ceiling(decile * 10)))][, .(
      n = .N, mean_predicted = mean(get(pred_col)), mean_observed = mean(get(actual_col)),
      sum_predicted = sum(get(pred_col)), sum_observed = sum(get(actual_col))
    ), by = decile][, `:=`(
      outcome = outcome,
      observed_to_predicted_ratio = fifelse(sum_predicted > 0, sum_observed / sum_predicted, NA_real_)
    )]
}
burden_calibration <- rbindlist(list(
  make_calibration_bins(burden, "expected_count", "actual_count", "count"),
  make_calibration_bins(burden, "expected_minutes", "actual_minutes", "minutes")
))
write_csv(as_tibble(burden_calibration), file.path(output_dir, "burden_calibration_deciles.csv"))

monthly <- burden[, .(
  expected_count = sum(expected_count), observed_count = sum(actual_count),
  expected_minutes = sum(expected_minutes), observed_minutes = sum(actual_minutes)
), by = .(month = as.Date(format(date, "%Y-%m-01")))]
state <- burden[, .(
  expected_count = sum(expected_count), observed_count = sum(actual_count),
  expected_minutes = sum(expected_minutes), observed_minutes = sum(actual_minutes)
), by = state_name]
write_csv(as_tibble(monthly), file.path(output_dir, "burden_monthly.csv"))
write_csv(as_tibble(state), file.path(output_dir, "burden_state.csv"))

aggregate_metrics <- bind_rows(lapply(list(monthly = monthly, state = state), function(x) {
  bind_rows(lapply(c("count", "minutes"), function(outcome) {
    pred <- x[[paste0("expected_", outcome)]]
    obs <- x[[paste0("observed_", outcome)]]
    tibble(
      outcome = outcome, n_groups = nrow(x),
      pearson = cor(pred, obs, method = "pearson"),
      spearman = cor(pred, obs, method = "spearman"),
      calibration_ratio = sum(pred) / sum(obs),
      mae = mean(abs(pred - obs)), rmse = sqrt(mean((pred - obs)^2))
    )
  }))
}), .id = "aggregation")
write_csv(aggregate_metrics, file.path(output_dir, "burden_aggregate_point_metrics.csv"))

observed_colour <- "#333333"
predicted_colour <- "#0072B2"
monthly_long <- as_tibble(monthly) %>%
  pivot_longer(-month, names_to = c("series", "outcome"), names_sep = "_", values_to = "value") %>%
  mutate(series = recode(series, observed = "Observed", expected = "Expected"))
plot_monthly <- function(outcome, title, y_label) {
  ggplot(filter(monthly_long, outcome == !!outcome),
         aes(month, value, colour = series, group = series)) +
    geom_line(linewidth = 0.75) + geom_point(size = 1.5) +
    scale_colour_manual(values = c(Observed = observed_colour, Expected = predicted_colour),
                        breaks = c("Observed", "Expected")) +
    scale_y_continuous(labels = scales::label_number(big.mark = ","), expand = expansion(mult = c(0, 0.06))) +
    labs(title = title, x = NULL, y = y_label, colour = NULL) +
    theme_minimal(base_size = 10) +
    theme(legend.position = "bottom", panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold"))
}
plot_state <- function(outcome, title, axis_label) {
  d <- as_tibble(state) %>% transmute(
    observed = .data[[paste0("observed_", outcome)]],
    expected = .data[[paste0("expected_", outcome)]]
  )
  upper <- max(c(d$observed, d$expected))
  ggplot(d, aes(expected, observed)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey55", linetype = "dashed") +
    geom_point(colour = predicted_colour, alpha = 0.82, size = 2) +
    coord_equal(xlim = c(0, upper), ylim = c(0, upper)) +
    scale_x_continuous(labels = scales::label_number(big.mark = ",")) +
    scale_y_continuous(labels = scales::label_number(big.mark = ",")) +
    labs(title = title, x = paste("Expected", axis_label), y = paste("Observed", axis_label)) +
    theme_minimal(base_size = 10) +
    theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold"))
}
figure <- (
  plot_monthly("count", "A. Monthly event count", "Events") |
    plot_monthly("minutes", "B. Monthly cumulative outage minutes", "Minutes")
) / (
  plot_state("count", "C. State event count", "events") |
    plot_state("minutes", "D. State cumulative outage minutes", "minutes")
) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
png_path <- file.path(figure_dir, "corrected_two_part_burden.png")
pdf_path <- file.path(figure_dir, "corrected_two_part_burden.pdf")
ggsave(png_path, figure, width = 10.5, height = 7.8, dpi = 320, bg = "white")
ggsave(pdf_path, figure, width = 10.5, height = 7.8, device = grDevices::pdf, bg = "white")

formula_audit <- tibble(
  transform = c("correct", "old_14b", "correct_minus_old"),
  expression = c("S * exp(f) - 1", "S * (exp(f) - 1)", "S - 1"),
  mean_test_prediction = c(
    mean(log_test_minutes), mean(log_test_old_formula),
    mean(log_test_minutes - log_test_old_formula)
  ),
  max_absolute_difference = c(
    0, max(abs(log_test_minutes - log_test_old_formula)),
    max(abs((log_test_minutes - log_test_old_formula) - (duan_smearing - 1)))
  )
)
write_csv(formula_audit, file.path(output_dir, "duan_formula_audit.csv"))

config <- list(
  run_id = run_id, created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
  seed = seed, n_bootstrap = n_boot,
  upstream = list(phase2 = basename(phase2_dir), phase4 = basename(phase4_dir)),
  split = list(
    train = c(as.character(train_start), as.character(train_end)),
    validation = c(as.character(validation_start), as.character(validation_end)),
    test = c(as.character(test_start), as.character(test_end))
  ),
  common_xgboost = list(
    max_rounds = max_rounds, early_stopping_rounds = early_stopping_rounds,
    eta = base_params$eta, max_depth = base_params$max_depth,
    min_child_weight = base_params$min_child_weight, gamma = base_params$gamma,
    subsample = base_params$subsample, colsample_bytree = base_params$colsample_bytree,
    n_features = length(headline_features)
  ),
  count = list(
    target = "excess events n_outages - 1 among positive municipality-nights",
    composition = "E[N|x] = P(Y=1|x) * (1 + E[N-1|x,Y=1])",
    poisson_best_iteration = poisson_fit$best_iteration,
    nb2_best_iteration = nb_best_iteration, nb2_alpha = nb_alpha, nb2_theta = nb_theta,
    validation_mae = as.list(count_validation_mae), selected = selected_count_model
  ),
  severity = list(
    target = "total cumulative outage minutes among positive municipality-nights",
    composition = "E[M|x] = P(Y=1|x) * E[M|x,Y=1]",
    duan_formula = "S * exp(f(x)) - 1", duan_smearing = duan_smearing,
    duan_smearing_source = "2020-01-01 through 2020-06-30 validation residuals",
    gamma_best_iteration = gamma_fit$best_iteration,
    tweedie_variance_power = selected_tweedie_power,
    tweedie_best_iteration = tweedie_fit$best_iteration,
    validation_mae = as.list(severity_validation_mae), selected = selected_severity_model,
    winsor_cap_training_q99 = winsor_cap,
    tail90_training_threshold = tail90, tail95_training_threshold = tail95
  ),
  classifier = list(
    source = "definitive Phase 2 full_83_xgb calibrated test probabilities",
    retrained = FALSE
  ),
  uncertainty = list(
    method = "paired nonparametric ISO-week block bootstrap",
    source_draw_matrix = block_counts_path, n_blocks = ncol(block_counts)
  )
)
write_json(
  config, file.path(output_dir, "run_config.json"),
  pretty = TRUE, auto_unbox = TRUE, digits = 16
)

checks <- list()
add_check <- function(id, condition, observed, expected) {
  checks[[length(checks) + 1L]] <<- tibble(
    check_id = id, status = if (isTRUE(condition)) "PASS" else "FAIL",
    observed = as.character(observed), expected = as.character(expected)
  )
}
add_check("upstream_independent_qa", !any(phase2_qa$status == "FAIL") && !any(phase4_qa$status == "FAIL"),
          "Phase2 PASS; Phase4 PASS", "No upstream FAIL")
add_check("headline_features", length(headline_features) == 83L, length(headline_features), "83")
add_check("positive_support", identical(c(nrow(train_pos), nrow(validation_pos), nrow(test_pos)),
                                        c(15951L, 1695L, 5350L)),
          paste(nrow(train_pos), nrow(validation_pos), nrow(test_pos), sep = "/"),
          "15951/1695/5350")
add_check("target_consistency", all((support$n_positive > 0) & (support$total_events >= support$n_positive)),
          paste(support$n_positive, support$total_events, sep = "/", collapse = ";"),
          "events >= positive municipality-nights")
add_check("duan_formula", max(abs(log_test_minutes -
                                     pmax(duan_smearing * exp(log_test_margin) - 1, 0))) < 1e-10,
          max(abs(log_test_minutes - pmax(duan_smearing * exp(log_test_margin) - 1, 0))), "<1e-10")
add_check("conditional_prediction_support",
          all(is.finite(as.matrix(conditional_test_predictions[, c(
            "poisson_count", "nb2_count", "log1p_duan_minutes", "gamma_minutes", "tweedie_minutes"
          )]))) && min(conditional_test_predictions$poisson_count) >= 1 &&
            min(conditional_test_predictions$nb2_count) >= 1,
          paste(range(conditional_test_predictions$poisson_count), collapse = "/"),
          "finite; count predictions >=1")
add_check("selection_is_validation_only",
          selected_count_model == names(which.min(count_validation_mae))[1] &&
            selected_severity_model == names(which.min(severity_validation_mae))[1],
          paste(selected_count_model, selected_severity_model, sep = "/"), "validation MAE minima")
add_check("bootstrap_dimensions", nrow(count_draws) == 2L * n_boot &&
            nrow(severity_draws) == 3L * n_boot && ncol(block_counts) == 79L,
          paste(nrow(count_draws), nrow(severity_draws), ncol(block_counts), sep = "/"),
          paste(2L * n_boot, 3L * n_boot, 79L, sep = "/"))
add_check("burden_support", nrow(burden) == 1334151L && sum(burden$actual_outage) == 5350L,
          paste(nrow(burden), sum(burden$actual_outage), sep = "/"), "1334151/5350")
add_check("burden_decomposition", max(abs(burden$expected_count -
                                              burden$full_probability * burden$conditional_count)) < 1e-12 &&
            max(abs(burden$expected_minutes -
                      burden$full_probability * burden$conditional_minutes)) < 1e-10,
          "exact within floating-point tolerance", "separate probability products")
add_check("aggregate_support", nrow(monthly) == 18L && nrow(state) == 32L,
          paste(nrow(monthly), nrow(state), sep = "/"), "18/32")
add_check("model_artifacts", length(list.files(model_dir, pattern = "\\.ubj$")) == 5L,
          length(list.files(model_dir, pattern = "\\.ubj$")), "5")
add_check("figure_outputs", file.info(png_path)$size > 10000 && file.info(pdf_path)$size > 5000,
          paste(file.info(png_path)$size, file.info(pdf_path)$size, sep = "/"), ">10000/>5000")
qa <- bind_rows(checks)
write_csv(qa, file.path(output_dir, "qa_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else "PASS"

selected_severity_metrics <- severity_uncertainty %>%
  filter(model_id == selected_severity_model,
         metric %in% c("mae", "rmse", "median_absolute_error", "winsorized_mae_99",
                       "winsorized_rmse_99", "pearson", "spearman", "calibration_ratio"))
report <- c(
  "# Phase 6 Corrected Count And Duration Models", "",
  paste0("- Run ID: `", run_id, "`"), paste0("- Status: **", status, "**"),
  "- Classifier probabilities: definitive Phase 2 full-model calibrated predictions (not retrained)",
  paste0("- Conditional samples: train 15,951; validation 1,695; test 5,350 municipality-nights"),
  paste0("- Selected count model: `", selected_count_model, "` (validation MAE)"),
  paste0("- Selected minutes model: `", selected_severity_model, "` (validation MAE)"),
  paste0("- Duan transform: `S * exp(f(x)) - 1`; S = ", sprintf("%.6f", duan_smearing)),
  paste0("- Bootstrap: ", n_boot, " paired resamples of 79 ISO-week blocks"), "",
  "## Selected Minutes Model Test Metrics", "",
  paste0("- ", selected_severity_metrics$metric, ": ",
         sprintf("%.4f", selected_severity_metrics$point_estimate), " (95% CI ",
         sprintf("%.4f", selected_severity_metrics$ci_lower), "-",
         sprintf("%.4f", selected_severity_metrics$ci_upper), ")"), "",
  "## Interpretation", "",
  "Count and cumulative minutes remain separate conditional outcomes. Their unconditional expectations are formed by multiplying each conditional mean by the same calibrated outage probability; conditional count and conditional minutes are never multiplied together.",
  "All model and transform selection uses the pre-test validation period. The heavy-tailed minute outcome is reported with untrimmed, training-cap winsorized, median, and tail errors.", "",
  "## QA", "",
  paste0("- PASS: ", sum(qa$status == "PASS")),
  paste0("- FAIL: ", sum(qa$status == "FAIL"))
)
writeLines(report, file.path(output_dir, "QA_REPORT.md"))
log_message("Phase 6 status: ", status)
if (status == "FAIL") stop("Phase 6 internal QA failed.")
