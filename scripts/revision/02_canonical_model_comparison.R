# Phase 2: same-sample model comparison. Writes only to data/revision/<run_id>.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(glmnet)
  library(jsonlite)
  library(parsnip)
  library(readr)
  library(recipes)
  library(tibble)
  library(workflows)
  library(xgboost)
  library(yardstick)
})

set.seed(42)

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
run_id <- parse_arg("--run-id", paste0(
  format(Sys.time(), "%Y%m%d_%H%M%S"), "_phase2_canonical_comparison"
))
features_path <- file.path(project_dir, "data", "model_ready", "features_engineered.rds")
splits_path <- file.path(project_dir, "data", "model_ready", "splits_fixed.rds")
phase1_dir <- file.path(project_dir, "data", "revision", "20260813_140718_phase1_freeze_audit")
headline_dir <- file.path(
  project_dir, "data", "baselines", "binary_threshold", "ablation_batches",
  "20260512_111324", "20260512_111341_forecast_strict_bench_rs_history"
)
output_dir <- file.path(project_dir, "data", "revision", run_id)
models_dir <- file.path(output_dir, "models")
feature_sets_dir <- file.path(output_dir, "feature_sets")

required_inputs <- c(
  features_path, splits_path, file.path(phase1_dir, "QA_REPORT.md"),
  file.path(phase1_dir, "qa_checks.csv"),
  file.path(headline_dir, "features_used.txt"),
  file.path(headline_dir, "run_config.json"),
  file.path(headline_dir, "metrics_summary.csv"),
  file.path(headline_dir, "platt_calibration.csv"),
  file.path(headline_dir, "operating_thresholds_table.csv")
)
missing_inputs <- required_inputs[!file.exists(required_inputs)]
if (length(missing_inputs)) {
  stop("Missing required inputs:\n", paste0("  - ", missing_inputs, collapse = "\n"))
}
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to overwrite non-empty run: ", output_dir)
}
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(feature_sets_dir, recursive = TRUE, showWarnings = FALSE)
log_path <- file.path(output_dir, "run.log")
phase1_qa <- read_csv(file.path(phase1_dir, "qa_checks.csv"), show_col_types = FALSE)
if (any(phase1_qa$status == "FAIL")) stop("Phase 1 contains failed QA checks.")
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))

log_message <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", msg, "\n",
      file = log_path, append = TRUE, sep = "")
}

get_nthread <- function() {
  n <- suppressWarnings(parallel::detectCores())
  if (is.na(n) || n < 2L) return(1L)
  max(1L, as.integer(n - 1L))
}

clip_prob <- function(prob, eps = 1e-15) pmin(pmax(as.numeric(prob), eps), 1 - eps)
truth_factor <- function(y) factor(ifelse(y == 1L, "1", "0"), levels = c("1", "0"))
compute_roc_auc <- function(y, p) suppressWarnings(as.numeric(
  roc_auc_vec(truth_factor(y), p, event_level = "first")
))
compute_pr_auc <- function(y, p) {
  # yardstick's trapezoidal PR convention assigns about 0.5 to a fully tied
  # score. For the degenerate no-skill forecast, use the standard prevalence
  # convention; non-degenerate models retain the frozen headline definition.
  if (length(unique(as.numeric(p))) < 2L) return(mean(y))
  suppressWarnings(as.numeric(pr_auc_vec(truth_factor(y), p, event_level = "first")))
}
compute_brier <- function(y, p) mean((as.numeric(p) - y)^2)
compute_logloss <- function(y, p) {
  p <- clip_prob(p)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}
compute_ece_equal <- function(y, p, n_bins = 10L) {
  bins <- cut(p, seq(0, 1, length.out = n_bins + 1L),
              include.lowest = TRUE, labels = FALSE)
  sum(vapply(seq_len(n_bins), function(b) {
    idx <- which(bins == b)
    if (!length(idx)) return(0)
    length(idx) / length(p) * abs(mean(p[idx]) - mean(y[idx]))
  }, numeric(1)))
}

fit_platt <- function(prob, y) {
  # This clipping exactly matches the frozen headline pipeline.
  p <- pmin(pmax(as.numeric(prob), 0.001), 0.999)
  if (length(unique(p)) < 2L) {
    return(list(type = "intercept_only", intercept = qlogis(mean(y)), slope = 0))
  }
  logit_prob <- qlogis(p)
  fit <- glm(y ~ logit_prob, family = binomial(link = "logit"))
  coefs <- as.numeric(coef(fit))
  if (length(coefs) != 2L || any(!is.finite(coefs))) stop("Platt fit failed.")
  list(type = "platt", intercept = coefs[1], slope = coefs[2])
}

apply_platt <- function(prob, calibration) {
  p <- pmin(pmax(as.numeric(prob), 0.001), 0.999)
  plogis(calibration$intercept + calibration$slope * qlogis(p))
}

threshold_grid <- unique(round(c(
  seq(0.0001, 0.01, by = 0.0001), seq(0.01, 0.10, by = 0.001),
  seq(0.10, 0.90, by = 0.01), seq(0.90, 0.99, by = 0.005),
  seq(0.99, 0.999, by = 0.001)
), 6))

threshold_metrics <- function(y, prob, dates, threshold) {
  pred <- prob >= threshold
  tp <- sum(y == 1L & pred); fp <- sum(y == 0L & pred)
  tn <- sum(y == 0L & !pred); fn <- sum(y == 1L & !pred)
  alerts <- tp + fp
  precision <- if (alerts > 0L) tp / alerts else NA_real_
  recall <- if (tp + fn > 0L) tp / (tp + fn) else NA_real_
  f1 <- if (is.finite(precision) && is.finite(recall) && precision + recall > 0) {
    2 * precision * recall / (precision + recall)
  } else NA_real_
  tibble(
    threshold = threshold, tp = tp, fp = fp, tn = tn, fn = fn,
    n_alerts = alerts, alerts_per_night = alerts / n_distinct(dates),
    precision = precision, recall = recall, f1 = f1
  )
}

select_precision_floor <- function(y, prob, dates, target = 0.10) {
  sweep <- bind_rows(lapply(threshold_grid, function(t) threshold_metrics(y, prob, dates, t)))
  candidates <- sweep %>% filter(!is.na(precision), !is.na(recall), precision >= target)
  if (!nrow(candidates)) return(list(selected = NULL, sweep = sweep))
  list(selected = candidates %>% filter(recall == max(recall)) %>% slice(1), sweep = sweep)
}

compute_topk <- function(y, prob, dates, gids, k_values) {
  ranked <- data.table(
    date = as.Date(dates), GID_2 = as.character(gids), truth = as.integer(y),
    score = as.numeric(prob)
  )
  # Stable secondary ordering makes tied-score baselines exactly reproducible.
  setorder(ranked, date, -score, GID_2)
  ranked[, rank := seq_len(.N), by = date]
  prevalence <- mean(y); total_positives <- sum(y); n_days <- uniqueN(ranked$date)
  bind_rows(lapply(k_values, function(k) {
    selected <- ranked[rank <= k]
    hits <- sum(selected$truth); alerts <- nrow(selected); precision <- hits / alerts
    tibble(
      k = as.integer(k), n_days = n_days, n_alerts = alerts, n_hits = hits,
      precision = precision, recall = hits / total_positives,
      lift = precision / prevalence
    )
  }))
}

write_latex_table <- function(df, path) {
  fmt <- df %>% transmute(
    Model = model_label, Features = n_features,
    `ROC-AUC` = sprintf("%.3f", roc_auc), `PR-AUC` = sprintf("%.3f", pr_auc),
    `PR lift` = sprintf("%.1f", pr_lift), Brier = sprintf("%.4f", brier),
    BSS = sprintf("%.3f", brier_skill_score), ECE = sprintf("%.4f", ece_equal_10),
    Precision = ifelse(is.na(test_precision), "--", sprintf("%.3f", test_precision)),
    Recall = ifelse(is.na(test_recall), "--", sprintf("%.3f", test_recall)),
    `Alerts/night` = ifelse(is.na(alerts_per_night), "--", sprintf("%.1f", alerts_per_night)),
    `P@10` = sprintf("%.3f", top10_precision)
  )
  escape <- function(x) gsub("_", "\\\\_", as.character(x), fixed = TRUE)
  lines <- c(
    paste0("\\begin{tabular}{l", paste(rep("r", ncol(fmt) - 1L), collapse = ""), "}"),
    "\\toprule", paste(escape(names(fmt)), collapse = " & "), "\\\\", "\\midrule"
  )
  for (i in seq_len(nrow(fmt))) {
    lines <- c(lines, paste(escape(unlist(fmt[i, ], use.names = FALSE)), collapse = " & "), "\\\\")
  }
  writeLines(c(lines, "\\bottomrule", "\\end{tabular}"), path)
}

log_message("Phase 2 canonical comparison")
log_message("Run ID: ", run_id)
log_message("Loading canonical model-ready RDS...")
features_object <- readRDS(features_path)
splits_fixed <- readRDS(splits_path)
headline_features <- readLines(file.path(headline_dir, "features_used.txt"), warn = FALSE)
headline_features <- headline_features[nzchar(headline_features)]
required_columns <- c("GID_2", "date", "outage_3h_or_more", headline_features)
missing_columns <- setdiff(required_columns, names(features_object))
if (length(missing_columns)) stop("Missing canonical columns: ", paste(missing_columns, collapse = ", "))
dat <- features_object[, required_columns, drop = FALSE]
rm(features_object); gc(verbose = FALSE)
dat$date <- as.Date(dat$date); dat$GID_2 <- as.character(dat$GID_2)
dat$outage_3h_or_more <- as.integer(dat$outage_3h_or_more)
if (anyDuplicated(dat[c("GID_2", "date")])) stop("Duplicate municipality-date keys.")
if (!setequal(unique(dat$outage_3h_or_more), c(0L, 1L))) stop("Outcome is not binary.")
if (length(headline_features) != 83L || anyDuplicated(headline_features)) {
  stop("Frozen headline list is not 83 unique ordered predictors.")
}

train_start <- as.Date(splits_fixed$train_range[1]); train_end <- as.Date(splits_fixed$train_range[2])
val_start <- as.Date(splits_fixed$val_range[1]); val_end <- as.Date(splits_fixed$val_range[2])
test_start <- as.Date(splits_fixed$test_range[1]); test_end <- as.Date(splits_fixed$test_range[2])
train_idx <- which(dat$date >= train_start & dat$date <= train_end)
val_idx <- which(dat$date >= val_start & dat$date <= val_end)
test_idx <- which(dat$date >= test_start & dat$date <= test_end)
calib_idx_in_val <- which(dat$date[val_idx] >= as.Date("2020-01-01") &
                            dat$date[val_idx] <= as.Date("2020-03-31"))
threshold_idx_in_val <- which(dat$date[val_idx] >= as.Date("2020-04-01") &
                                dat$date[val_idx] <= as.Date("2020-06-30"))

split_summary <- tibble(
  split = c("train", "calibration_fit", "threshold_selection", "validation_total", "test"),
  start = as.Date(c(train_start, "2020-01-01", "2020-04-01", val_start, test_start)),
  end = as.Date(c(train_end, "2020-03-31", "2020-06-30", val_end, test_end)),
  n_rows = c(length(train_idx), length(calib_idx_in_val), length(threshold_idx_in_val),
             length(val_idx), length(test_idx)),
  n_positive = c(
    sum(dat$outage_3h_or_more[train_idx]),
    sum(dat$outage_3h_or_more[val_idx][calib_idx_in_val]),
    sum(dat$outage_3h_or_more[val_idx][threshold_idx_in_val]),
    sum(dat$outage_3h_or_more[val_idx]), sum(dat$outage_3h_or_more[test_idx])
  ),
  n_nights = c(
    n_distinct(dat$date[train_idx]), n_distinct(dat$date[val_idx][calib_idx_in_val]),
    n_distinct(dat$date[val_idx][threshold_idx_in_val]), n_distinct(dat$date[val_idx]),
    n_distinct(dat$date[test_idx])
  ),
  n_municipalities = c(
    n_distinct(dat$GID_2[train_idx]), n_distinct(dat$GID_2[val_idx][calib_idx_in_val]),
    n_distinct(dat$GID_2[val_idx][threshold_idx_in_val]), n_distinct(dat$GID_2[val_idx]),
    n_distinct(dat$GID_2[test_idx])
  )
) %>% mutate(prevalence = n_positive / n_rows)
write_csv(split_summary, file.path(output_dir, "split_summary.csv"))
if (!identical(as.integer(split_summary$n_rows), c(2683044L, 223587L, 223587L, 447174L, 1334151L))) {
  stop("Split row counts drifted from Phase 1.")
}

# Mutually exclusive feature families spanning the full headline inventory.
history_features <- headline_features[grepl(
  "^(hist_|cumulative_outage_minutes|days_since_last_outage)", headline_features
)]
ntl_features <- headline_features[grepl("^(ntl_|nb_)", headline_features) |
                                    headline_features == "clim_n_obs"]
weather_features <- headline_features[grepl(
  "^(atm|dew|max_dew|min_dew|lai_high|lai_low|rain|rh|skin_temp|temp|wdr|wind_u|wind_v|wsp|max_temp|min_temp)_lag[0-9]+$|^has_weather$",
  headline_features
)]
static_features <- headline_features[grepl("^built_share_(area|mask)_static$", headline_features)]
calendar_features <- headline_features[grepl(
  "^(dow|doy|month|year|is_holiday|is_public_holiday|is_weekend|is_covid_period)(_|$)|^moon_fraction$",
  headline_features
)]
family_lookup <- tibble(
  feature = headline_features,
  feature_family = case_when(
    headline_features %in% history_features ~ "outage_history",
    headline_features %in% ntl_features ~ "nighttime_lights",
    headline_features %in% weather_features ~ "weather",
    headline_features %in% static_features ~ "static_built_environment",
    headline_features %in% calendar_features ~ "calendar_astronomy",
    TRUE ~ "unassigned"
  )
)
if (any(family_lookup$feature_family == "unassigned")) {
  stop("Unassigned features: ", paste(family_lookup$feature[family_lookup$feature_family == "unassigned"], collapse = ", "))
}
write_csv(family_lookup, file.path(output_dir, "feature_family_dictionary.csv"))

model_features <- list(
  constant_prevalence = character(), municipality_climatology = character(),
  history_only_xgb = history_features,
  weather_calendar_static_xgb = headline_features[headline_features %in%
    c(weather_features, calendar_features, static_features)],
  history_free_xgb = setdiff(headline_features, history_features),
  full_83_xgb = headline_features, ridge_logistic_83 = headline_features
)
expected_feature_counts <- c(0L, 0L, 10L, 50L, 73L, 83L, 83L)
if (!identical(unname(vapply(model_features, length, integer(1))), expected_feature_counts)) {
  stop("Unexpected model feature counts: ", paste(vapply(model_features, length, integer(1)), collapse = ","))
}
for (id in names(model_features)) {
  writeLines(model_features[[id]], file.path(feature_sets_dir, paste0(id, ".txt")))
}
feature_set_manifest <- bind_rows(lapply(names(model_features), function(id) {
  feats <- model_features[[id]]
  if (!length(feats)) return(tibble(
    model_id = id, feature_order = NA_integer_, feature = NA_character_, feature_family = NA_character_
  ))
  tibble(model_id = id, feature_order = seq_along(feats), feature = feats) %>%
    left_join(family_lookup, by = "feature")
}))
write_csv(feature_set_manifest, file.path(output_dir, "feature_set_manifest.csv"))

y_train <- dat$outage_3h_or_more[train_idx]; y_val <- dat$outage_3h_or_more[val_idx]
y_test <- dat$outage_3h_or_more[test_idx]
date_val <- dat$date[val_idx]; date_test <- dat$date[test_idx]
gid_val <- dat$GID_2[val_idx]; gid_test <- dat$GID_2[test_idx]
train_prevalence <- mean(y_train)
pred_val <- data.table(GID_2 = gid_val, date = date_val, truth = y_val)
pred_test <- data.table(GID_2 = gid_test, date = date_test, truth = y_test)
model_manifest_rows <- list(); importance_rows <- list()

# Models 1-2: train-only prevalence and smoothed municipality climatology.
pred_val[, constant_prevalence_raw := train_prevalence]
pred_test[, constant_prevalence_raw := train_prevalence]
model_manifest_rows$constant_prevalence <- tibble(
  model_id = "constant_prevalence", model_type = "constant", n_features = 0L,
  training_seconds = 0, details = "Training-period national prevalence; no calibration"
)
log_message("Constant prevalence: ", sprintf("%.8f", train_prevalence))

clim_stats <- data.table(GID_2 = dat$GID_2[train_idx], truth = y_train)[,
  .(n = .N, positives = sum(truth)), by = GID_2
]
clim_stats[, raw_rate := positives / n]
m <- train_prevalence; observed_var <- var(clim_stats$raw_rate); q <- mean(1 / clim_stats$n)
a_ratio <- observed_var / (m * (1 - m))
prior_strength <- if (is.finite(a_ratio) && a_ratio > q && a_ratio < 1) {
  (1 - a_ratio) / (a_ratio - q)
} else 30
prior_strength <- min(max(prior_strength, 1e-3), 1e7)
prior_alpha <- m * prior_strength; prior_beta <- (1 - m) * prior_strength
clim_stats[, smoothed_rate := (positives + prior_alpha) / (n + prior_alpha + prior_beta)]
clim_rate <- setNames(clim_stats$smoothed_rate, clim_stats$GID_2)
if (any(!gid_val %in% names(clim_rate)) || any(!gid_test %in% names(clim_rate))) {
  stop("A validation/test municipality is absent from the training climatology.")
}
pred_val[, municipality_climatology_raw := unname(clim_rate[GID_2])]
pred_test[, municipality_climatology_raw := unname(clim_rate[GID_2])]
write_csv(as_tibble(clim_stats), file.path(output_dir, "municipality_climatology_training.csv"))
write_csv(tibble(
  method = "beta_binomial_method_of_moments", train_prevalence = m,
  observed_municipality_rate_variance = observed_var, mean_inverse_municipality_n = q,
  prior_strength = prior_strength, prior_alpha = prior_alpha, prior_beta = prior_beta
), file.path(output_dir, "municipality_climatology_prior.csv"))
model_manifest_rows$municipality_climatology <- tibble(
  model_id = "municipality_climatology", model_type = "empirical_bayes_climatology",
  n_features = 0L, training_seconds = 0,
  details = paste0("Training-only beta-binomial smoothing; prior strength=", signif(prior_strength, 6))
)
rm(clim_rate); gc(verbose = FALSE)

xgb_params <- list(
  trees = 500L, tree_depth = 6L, min_n = 10L, loss_reduction = 0,
  sample_size = 0.8, mtry = 0.8, learn_rate = 0.05
)

prepare_model_frame <- function(idx, feature_names) {
  out <- dat[idx, c("outage_3h_or_more", feature_names), drop = FALSE]
  out$outage_3h_or_more <- factor(as.character(out$outage_3h_or_more), levels = c("1", "0"))
  logical_cols <- names(out)[vapply(out, is.logical, logical(1))]
  for (nm in logical_cols) out[[nm]] <- as.integer(out[[nm]])
  out
}

fit_xgb <- function(model_id, feature_names) {
  log_message("Training ", model_id, " with ", length(feature_names), " features...")
  set.seed(42)
  train_frame <- prepare_model_frame(train_idx, feature_names)
  zero_variance <- feature_names[vapply(train_frame[feature_names], function(x) {
    length(unique(x[!is.na(x)])) <= 1L
  }, logical(1))]
  n_pos <- sum(train_frame$outage_3h_or_more == "1")
  n_neg <- sum(train_frame$outage_3h_or_more == "0")
  scale_pos_weight <- n_neg / n_pos
  mtry_count <- max(1L, floor(xgb_params$mtry * length(feature_names)))
  model_recipe <- recipe(outage_3h_or_more ~ ., data = train_frame) %>% step_zv(all_predictors())
  model_spec <- boost_tree(
    trees = xgb_params$trees, tree_depth = xgb_params$tree_depth,
    min_n = xgb_params$min_n, loss_reduction = xgb_params$loss_reduction,
    sample_size = xgb_params$sample_size, mtry = mtry_count,
    learn_rate = xgb_params$learn_rate
  ) %>%
    set_engine("xgboost", scale_pos_weight = scale_pos_weight,
               nthread = get_nthread(), verbosity = 0) %>%
    set_mode("classification")
  workflow_spec <- workflow() %>% add_recipe(model_recipe) %>% add_model(model_spec)
  started <- proc.time()[["elapsed"]]
  fitted <- fit(workflow_spec, data = train_frame)
  elapsed <- proc.time()[["elapsed"]] - started
  rm(train_frame); gc(verbose = FALSE)

  val_frame <- prepare_model_frame(val_idx, feature_names)
  val_prob <- predict(fitted, new_data = val_frame, type = "prob")$.pred_1
  rm(val_frame); gc(verbose = FALSE)
  test_frame <- prepare_model_frame(test_idx, feature_names)
  test_prob <- predict(fitted, new_data = test_frame, type = "prob")$.pred_1
  rm(test_frame); gc(verbose = FALSE)

  engine <- extract_fit_engine(fitted)
  xgb.save(engine, file.path(models_dir, paste0(model_id, ".ubj")))
  importance <- as_tibble(xgb.importance(model = engine)) %>%
    mutate(model_id = model_id, .before = 1)
  rm(fitted, engine); gc(verbose = FALSE)
  log_message("Completed ", model_id, " in ", sprintf("%.1f", elapsed), " seconds")
  list(
    val = as.numeric(val_prob), test = as.numeric(test_prob), importance = importance,
    manifest = tibble(
      model_id = model_id, model_type = "class_weighted_xgboost",
      n_features = length(feature_names), training_seconds = elapsed,
      details = paste0(
        "500 trees; depth 6; min_n 10; subsample 0.8; mtry ", mtry_count,
        "; eta 0.05; scale_pos_weight ", signif(scale_pos_weight, 8),
        "; zero-variance recipe removals: ",
        ifelse(!length(zero_variance), "none", paste(zero_variance, collapse = "|"))
      )
    )
  )
}

xgb_model_ids <- c(
  "history_only_xgb", "weather_calendar_static_xgb", "history_free_xgb", "full_83_xgb"
)
for (model_id in xgb_model_ids) {
  result <- fit_xgb(model_id, model_features[[model_id]])
  pred_val[, (paste0(model_id, "_raw")) := result$val]
  pred_test[, (paste0(model_id, "_raw")) := result$test]
  importance_rows[[model_id]] <- result$importance
  model_manifest_rows[[model_id]] <- result$manifest
  rm(result); gc(verbose = FALSE)
}

# Model 7: class-weighted ridge logistic regression on the same 83 predictors.
# Lambda tuning is chronological and remains entirely inside the training era.
log_message("Preparing ridge logistic regression...")
train_medians <- vapply(headline_features, function(nm) {
  value <- median(dat[[nm]][train_idx], na.rm = TRUE)
  if (!is.finite(value)) stop("No finite training median for ", nm)
  as.numeric(value)
}, numeric(1))
write_csv(tibble(feature = names(train_medians), train_median = train_medians),
          file.path(output_dir, "ridge_training_medians.csv"))

build_imputed_matrix <- function(idx, feature_names, medians) {
  frame <- dat[idx, feature_names, drop = FALSE]
  for (j in seq_along(feature_names)) {
    if (anyNA(frame[[j]])) {
      v <- frame[[j]]; v[is.na(v)] <- medians[[feature_names[j]]]; frame[[j]] <- v
    }
  }
  out <- data.matrix(frame); storage.mode(out) <- "double"; out
}

tune_fit_idx <- which(dat$date >= train_start & dat$date <= as.Date("2019-06-30"))
tune_holdout_idx <- which(dat$date >= as.Date("2019-07-01") & dat$date <= train_end)
x_tune_fit <- build_imputed_matrix(tune_fit_idx, headline_features, train_medians)
y_tune_fit <- dat$outage_3h_or_more[tune_fit_idx]
tune_weight_ratio <- sum(y_tune_fit == 0L) / sum(y_tune_fit == 1L)
w_tune_fit <- ifelse(y_tune_fit == 1L, tune_weight_ratio, 1)
set.seed(42); tune_started <- proc.time()[["elapsed"]]
ridge_path <- glmnet(
  x_tune_fit, y_tune_fit, family = "binomial", alpha = 0, weights = w_tune_fit,
  nlambda = 50L, standardize = TRUE, intercept = TRUE, thresh = 1e-7, maxit = 100000
)
tune_elapsed <- proc.time()[["elapsed"]] - tune_started
rm(x_tune_fit, y_tune_fit, w_tune_fit); gc(verbose = FALSE)

x_tune_holdout <- build_imputed_matrix(tune_holdout_idx, headline_features, train_medians)
y_tune_holdout <- dat$outage_3h_or_more[tune_holdout_idx]
tune_predictions <- as.matrix(predict(ridge_path, newx = x_tune_holdout, type = "response"))
tune_holdout_ratio <- sum(y_tune_holdout == 0L) / sum(y_tune_holdout == 1L)
w_tune_holdout <- ifelse(y_tune_holdout == 1L, tune_holdout_ratio, 1)
weighted_logloss <- vapply(seq_len(ncol(tune_predictions)), function(j) {
  p <- clip_prob(tune_predictions[, j])
  loss <- -(y_tune_holdout * log(p) + (1 - y_tune_holdout) * log(1 - p))
  weighted.mean(loss, w_tune_holdout)
}, numeric(1))
ridge_tuning <- tibble(
  lambda_index = seq_along(ridge_path$lambda), lambda = ridge_path$lambda,
  weighted_logloss = weighted_logloss
)
selected_lambda_index <- which.min(ridge_tuning$weighted_logloss)
selected_lambda <- ridge_tuning$lambda[selected_lambda_index]
ridge_tuning <- ridge_tuning %>% mutate(selected = row_number() == selected_lambda_index)
write_csv(ridge_tuning, file.path(output_dir, "ridge_lambda_tuning.csv"))
rm(x_tune_holdout, y_tune_holdout, w_tune_holdout, tune_predictions, ridge_path)
gc(verbose = FALSE)

log_message("Ridge selected lambda: ", signif(selected_lambda, 8))
x_train_ridge <- build_imputed_matrix(train_idx, headline_features, train_medians)
ridge_weight_ratio <- sum(y_train == 0L) / sum(y_train == 1L)
w_train_ridge <- ifelse(y_train == 1L, ridge_weight_ratio, 1)
set.seed(42); ridge_started <- proc.time()[["elapsed"]]
ridge_fit <- glmnet(
  x_train_ridge, y_train, family = "binomial", alpha = 0, weights = w_train_ridge,
  lambda = selected_lambda, standardize = TRUE, intercept = TRUE,
  thresh = 1e-7, maxit = 100000
)
ridge_elapsed <- proc.time()[["elapsed"]] - ridge_started
rm(x_train_ridge, w_train_ridge); gc(verbose = FALSE)
x_val_ridge <- build_imputed_matrix(val_idx, headline_features, train_medians)
pred_val[, ridge_logistic_83_raw := as.numeric(
  predict(ridge_fit, newx = x_val_ridge, s = selected_lambda, type = "response")
)]
rm(x_val_ridge); gc(verbose = FALSE)
x_test_ridge <- build_imputed_matrix(test_idx, headline_features, train_medians)
pred_test[, ridge_logistic_83_raw := as.numeric(
  predict(ridge_fit, newx = x_test_ridge, s = selected_lambda, type = "response")
)]
rm(x_test_ridge); gc(verbose = FALSE)
saveRDS(ridge_fit, file.path(models_dir, "ridge_logistic_83.rds"), compress = "gzip")
ridge_coefficients <- as.matrix(coef(ridge_fit, s = selected_lambda))
write_csv(tibble(term = rownames(ridge_coefficients), coefficient = as.numeric(ridge_coefficients[, 1])),
          file.path(output_dir, "ridge_coefficients.csv"))
rm(ridge_fit); gc(verbose = FALSE)
model_manifest_rows$ridge_logistic_83 <- tibble(
  model_id = "ridge_logistic_83", model_type = "class_weighted_ridge_logistic",
  n_features = 83L, training_seconds = ridge_elapsed,
  details = paste0(
    "alpha=0; lambda=", signif(selected_lambda, 8),
    "; training-only median imputation; tuning 2017-2019 H1/2019 H2",
    "; tuning_seconds=", signif(tune_elapsed, 6),
    "; scale_pos_weight=", signif(ridge_weight_ratio, 8)
  )
)
log_message("Completed ridge logistic in ", sprintf("%.1f", ridge_elapsed), " seconds")

model_ids <- names(model_features)
model_labels <- c(
  constant_prevalence = "National prevalence",
  municipality_climatology = "Municipality climatology",
  history_only_xgb = "History-only XGBoost",
  weather_calendar_static_xgb = "Weather/calendar/static XGBoost",
  history_free_xgb = "History-free remote-sensing XGBoost",
  full_83_xgb = "Full 83-feature XGBoost",
  ridge_logistic_83 = "Ridge logistic, same 83 features"
)
calibration_rows <- list(); threshold_rows <- list(); topk_rows <- list()
metric_rows <- list(); sweep_rows <- list()
k_values <- c(1L, 2L, 5L, 10L, 20L, 30L, 50L, 100L)
constant_brier <- compute_brier(y_test, pred_test$constant_prevalence_raw)

for (model_id in model_ids) {
  raw_col <- paste0(model_id, "_raw"); calibrated_col <- paste0(model_id, "_calibrated")
  raw_val <- pred_val[[raw_col]]; raw_test <- pred_test[[raw_col]]
  if (model_id == "constant_prevalence") {
    calibration <- list(type = "not_applicable", intercept = qlogis(train_prevalence), slope = 1)
    calibrated_val <- raw_val; calibrated_test <- raw_test
  } else {
    calibration <- fit_platt(raw_val[calib_idx_in_val], y_val[calib_idx_in_val])
    calibrated_val <- apply_platt(raw_val, calibration)
    calibrated_test <- apply_platt(raw_test, calibration)
  }
  if (any(!is.finite(calibrated_val)) || any(!is.finite(calibrated_test))) {
    stop("Non-finite calibrated predictions for ", model_id)
  }
  pred_val[, (calibrated_col) := calibrated_val]
  pred_test[, (calibrated_col) := calibrated_test]
  calibration_rows[[model_id]] <- tibble(
    model_id = model_id, calibration_method = calibration$type,
    calibration_fit_start = as.Date("2020-01-01"), calibration_fit_end = as.Date("2020-03-31"),
    calibration_fit_rows = length(calib_idx_in_val),
    calibration_fit_positives = sum(y_val[calib_idx_in_val]),
    intercept = calibration$intercept, slope = calibration$slope
  )

  selection <- select_precision_floor(
    y_val[threshold_idx_in_val], calibrated_val[threshold_idx_in_val],
    date_val[threshold_idx_in_val], 0.10
  )
  sweep_rows[[model_id]] <- selection$sweep %>% mutate(model_id = model_id, .before = 1)
  if (is.null(selection$selected)) {
    selected_threshold <- NA_real_
    empty_op <- tibble(
      threshold = NA_real_, tp = NA_integer_, fp = NA_integer_, tn = NA_integer_, fn = NA_integer_,
      n_alerts = NA_integer_, alerts_per_night = NA_real_, precision = NA_real_,
      recall = NA_real_, f1 = NA_real_
    )
    val_operating <- empty_op; test_operating <- empty_op
  } else {
    selected_threshold <- selection$selected$threshold[[1]]
    val_operating <- selection$selected
    test_operating <- threshold_metrics(y_test, calibrated_test, date_test, selected_threshold)
  }
  threshold_rows[[model_id]] <- tibble(
    model_id = model_id, policy_type = "precision_floor", policy_target = 0.10,
    calibration_fit_period = "2020-01-01/2020-03-31",
    threshold_selection_period = "2020-04-01/2020-06-30",
    selected_threshold = selected_threshold,
    val_precision = val_operating$precision, val_recall = val_operating$recall,
    val_f1 = val_operating$f1, val_alerts_per_night = val_operating$alerts_per_night,
    test_precision = test_operating$precision, test_recall = test_operating$recall,
    test_f1 = test_operating$f1, test_alerts_per_night = test_operating$alerts_per_night,
    test_tp = test_operating$tp, test_fp = test_operating$fp,
    test_tn = test_operating$tn, test_fn = test_operating$fn
  )

  topk <- compute_topk(y_test, calibrated_test, date_test, gid_test, k_values) %>%
    mutate(model_id = model_id, .before = 1)
  topk_rows[[model_id]] <- topk
  top10 <- topk %>% filter(k == 10L)
  pr_auc <- compute_pr_auc(y_test, calibrated_test); brier <- compute_brier(y_test, calibrated_test)
  metric_rows[[model_id]] <- tibble(
    model_id = model_id, model_label = unname(model_labels[model_id]),
    n_features = length(model_features[[model_id]]), n_test = length(y_test),
    n_positive = sum(y_test), prevalence = mean(y_test),
    roc_auc = compute_roc_auc(y_test, calibrated_test), pr_auc = pr_auc,
    pr_lift = pr_auc / mean(y_test), brier = brier,
    brier_skill_score = 1 - brier / constant_brier,
    logloss = compute_logloss(y_test, calibrated_test),
    ece_equal_10 = compute_ece_equal(y_test, calibrated_test, 10L),
    operating_threshold = selected_threshold,
    test_precision = test_operating$precision, test_recall = test_operating$recall,
    test_f1 = test_operating$f1, alerts_per_night = test_operating$alerts_per_night,
    top10_precision = top10$precision, top10_recall = top10$recall,
    top10_lift = top10$lift, top10_hits = top10$n_hits,
    calibration_method = calibration$type
  )
  log_message(
    "Scored ", model_id, ": ROC=", sprintf("%.4f", metric_rows[[model_id]]$roc_auc),
    ", PR=", sprintf("%.4f", pr_auc), ", Brier=", sprintf("%.6f", brier)
  )
}

canonical_table <- bind_rows(metric_rows)
calibration_table <- bind_rows(calibration_rows)
operating_table <- bind_rows(threshold_rows)
topk_table <- bind_rows(topk_rows)
threshold_sweeps <- bind_rows(sweep_rows)
model_manifest <- bind_rows(model_manifest_rows) %>% slice(match(model_ids, model_id))
write_csv(canonical_table, file.path(output_dir, "canonical_model_table.csv"))
write_latex_table(canonical_table, file.path(output_dir, "canonical_model_table.tex"))
write_csv(calibration_table, file.path(output_dir, "calibration_parameters.csv"))
write_csv(operating_table, file.path(output_dir, "operating_metrics.csv"))
write_csv(topk_table, file.path(output_dir, "topk_metrics.csv"))
write_csv(threshold_sweeps, file.path(output_dir, "threshold_sweeps.csv"))
write_csv(model_manifest, file.path(output_dir, "model_run_manifest.csv"))

all_importance <- bind_rows(importance_rows) %>%
  rename(feature = Feature, gain = Gain, cover = Cover, frequency = Frequency) %>%
  left_join(family_lookup, by = "feature")
if (anyNA(all_importance$feature_family)) stop("Unclassified XGBoost importance feature.")
family_importance <- all_importance %>%
  group_by(model_id, feature_family) %>%
  summarise(
    n_features_used_in_splits = n_distinct(feature), total_gain = sum(gain),
    total_cover = sum(cover), total_frequency = sum(frequency), .groups = "drop"
  ) %>%
  group_by(model_id) %>% mutate(gain_share = total_gain / sum(total_gain)) %>% ungroup()
write_csv(all_importance, file.path(output_dir, "xgb_feature_importance.csv"))
write_csv(family_importance, file.path(output_dir, "xgb_feature_family_importance.csv"))

comparison_specs <- tribble(
  ~comparison_id, ~expanded_model, ~reference_model, ~interpretation,
  "incremental_history", "full_83_xgb", "history_free_xgb",
  "Incremental predictive contribution of the 10 lagged outage-history features",
  "incremental_ntl", "history_free_xgb", "weather_calendar_static_xgb",
  "Incremental predictive contribution of NTL-derived predictors conditional on weather/calendar/static predictors",
  "history_vs_climatology", "history_only_xgb", "municipality_climatology",
  "Nonlinear lagged-history model relative to a smoothed municipality risk baseline",
  "nonlinearity_full_features", "full_83_xgb", "ridge_logistic_83",
  "Class-weighted nonlinear model relative to a class-weighted linear model on the same 83 predictors"
)
comparison_metrics <- c(
  "roc_auc", "pr_auc", "pr_lift", "brier", "brier_skill_score", "ece_equal_10",
  "test_precision", "test_recall", "alerts_per_night", "top10_precision", "top10_recall"
)
paired_comparisons <- bind_rows(lapply(seq_len(nrow(comparison_specs)), function(i) {
  spec <- comparison_specs[i, ]; expanded <- canonical_table %>% filter(model_id == spec$expanded_model)
  reference <- canonical_table %>% filter(model_id == spec$reference_model)
  bind_rows(lapply(comparison_metrics, function(metric) {
    expanded_value <- expanded[[metric]]; reference_value <- reference[[metric]]
    raw_delta <- expanded_value - reference_value
    higher_is_better <- !metric %in% c("brier", "ece_equal_10", "alerts_per_night")
    tibble(
      comparison_id = spec$comparison_id, expanded_model = spec$expanded_model,
      reference_model = spec$reference_model, interpretation = spec$interpretation,
      metric = metric, expanded_value = expanded_value, reference_value = reference_value,
      raw_delta = raw_delta,
      improvement_oriented_delta = ifelse(higher_is_better, raw_delta, -raw_delta),
      same_test_rows = TRUE, n_paired_rows = length(y_test)
    )
  }))
}))
write_csv(paired_comparisons, file.path(output_dir, "paired_model_comparisons.csv"))
write_csv(paired_comparisons %>%
            filter(comparison_id %in% c("incremental_history", "incremental_ntl"),
                   metric %in% c("roc_auc", "pr_auc", "brier_skill_score", "top10_precision")),
          file.path(output_dir, "ntl_history_contribution_summary.csv"))

# One shared table per split makes the paired comparison auditable by construction.
setcolorder(pred_val, c("GID_2", "date", "truth", setdiff(names(pred_val), c("GID_2", "date", "truth"))))
setcolorder(pred_test, c("GID_2", "date", "truth", setdiff(names(pred_test), c("GID_2", "date", "truth"))))
saveRDS(as.data.frame(pred_val), file.path(output_dir, "paired_predictions_validation.rds"), compress = "gzip")
saveRDS(as.data.frame(pred_test), file.path(output_dir, "paired_predictions_test.rds"), compress = "gzip")

# Verify that the full-model refit reproduces the frozen headline run.
headline_metrics <- read_csv(file.path(headline_dir, "metrics_summary.csv"), show_col_types = FALSE) %>%
  filter(split == "test")
headline_platt <- read_csv(file.path(headline_dir, "platt_calibration.csv"), show_col_types = FALSE)
headline_operating <- read_csv(
  file.path(headline_dir, "operating_thresholds_table.csv"), show_col_types = FALSE
) %>% filter(method == "platt", policy_type == "precision_floor", abs(policy_target - 0.10) < 1e-12)
full_row <- canonical_table %>% filter(model_id == "full_83_xgb")
full_cal <- calibration_table %>% filter(model_id == "full_83_xgb")
full_op <- operating_table %>% filter(model_id == "full_83_xgb")
headline_reproduction <- tribble(
  ~quantity, ~frozen_value, ~refit_value, ~tolerance,
  "roc_auc", headline_metrics$roc_auc[[1]], full_row$roc_auc[[1]], 1e-6,
  "pr_auc", headline_metrics$pr_auc[[1]], full_row$pr_auc[[1]], 1e-6,
  "brier_platt", headline_platt$brier_score[[1]], full_row$brier[[1]], 1e-6,
  "ece_equal_platt", headline_platt$ece_equal[[1]], full_row$ece_equal_10[[1]], 1e-6,
  "platt_intercept", headline_platt$platt_intercept[[1]], full_cal$intercept[[1]], 1e-6,
  "platt_slope", headline_platt$platt_slope[[1]], full_cal$slope[[1]], 1e-6,
  "operating_threshold", headline_operating$selected_threshold[[1]], full_op$selected_threshold[[1]], 1e-12,
  "test_precision", headline_operating$test_precision[[1]], full_op$test_precision[[1]], 1e-6,
  "test_recall", headline_operating$test_recall[[1]], full_op$test_recall[[1]], 1e-6,
  "test_alerts_per_day", headline_operating$test_alerts_per_day[[1]], full_op$test_alerts_per_night[[1]], 1e-6
) %>% mutate(
  absolute_difference = abs(refit_value - frozen_value), pass = absolute_difference <= tolerance
)
write_csv(headline_reproduction, file.path(output_dir, "headline_reproduction_check.csv"))

qa_checks <- tribble(
  ~check_id, ~status, ~observed, ~expected,
  "phase1_has_no_failures", "PASS",
  paste0(sum(phase1_qa$status == "PASS"), " PASS; ", sum(phase1_qa$status == "WARN"), " WARN"),
  "No Phase 1 FAIL",
  "canonical_rds_rows", ifelse(nrow(dat) == 4464369L, "PASS", "FAIL"), as.character(nrow(dat)), "4464369",
  "headline_feature_count", ifelse(length(headline_features) == 83L, "PASS", "FAIL"),
  as.character(length(headline_features)), "83",
  "model_count", ifelse(nrow(canonical_table) == 7L, "PASS", "FAIL"), as.character(nrow(canonical_table)), "7",
  "test_row_count", ifelse(nrow(pred_test) == 1334151L, "PASS", "FAIL"), as.character(nrow(pred_test)), "1334151",
  "test_positive_count", ifelse(sum(pred_test$truth) == 5350L, "PASS", "FAIL"),
  as.character(sum(pred_test$truth)), "5350",
  "test_key_uniqueness", ifelse(anyDuplicated(pred_test[, .(GID_2, date)]) == 0L, "PASS", "FAIL"),
  as.character(anyDuplicated(pred_test[, .(GID_2, date)])), "0 duplicates",
  "prediction_completeness", ifelse(all(vapply(pred_test, function(x) !anyNA(x), logical(1))), "PASS", "FAIL"),
  as.character(sum(vapply(pred_test, function(x) sum(is.na(x)), integer(1)))), "0 missing cells",
  "full_features_identical_to_frozen", ifelse(identical(model_features$full_83_xgb, headline_features), "PASS", "FAIL"),
  as.character(identical(model_features$full_83_xgb, headline_features)), "TRUE",
  "history_free_is_full_minus_history", ifelse(
    identical(model_features$history_free_xgb, setdiff(headline_features, history_features)), "PASS", "FAIL"
  ), as.character(length(model_features$history_free_xgb)), "73 ordered features",
  "strict_calibration_rows", ifelse(length(calib_idx_in_val) == 223587L, "PASS", "FAIL"),
  as.character(length(calib_idx_in_val)), "223587",
  "strict_threshold_rows", ifelse(length(threshold_idx_in_val) == 223587L, "PASS", "FAIL"),
  as.character(length(threshold_idx_in_val)), "223587",
  "headline_reproduction", ifelse(all(headline_reproduction$pass), "PASS", "WARN"),
  paste0(sum(headline_reproduction$pass), "/", nrow(headline_reproduction), " within tolerance"),
  "All frozen headline quantities within tolerance"
)
write_csv(qa_checks, file.path(output_dir, "qa_checks.csv"))

run_config <- list(
  run_id = run_id, created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
  git_commit = system2("git", c("rev-parse", "HEAD"), stdout = TRUE), seed = 42L,
  input = list(
    features_path = features_path, splits_path = splits_path, phase1_dir = phase1_dir,
    frozen_headline_dir = headline_dir,
    canonical_rds_sha256 = "a05db6b77dd7b55824c9a7f5cdf90083a20adb14e9ee50b02b3e7f539f3e0ad7"
  ),
  split = list(
    train = c(as.character(train_start), as.character(train_end)),
    calibration_fit = c("2020-01-01", "2020-03-31"),
    threshold_selection = c("2020-04-01", "2020-06-30"),
    test = c(as.character(test_start), as.character(test_end))
  ),
  models = lapply(model_ids, function(id) list(id = id, n_features = length(model_features[[id]]))),
  xgboost = xgb_params,
  logistic = list(
    penalty = "ridge", alpha = 0, selected_lambda = selected_lambda, class_weighted = TRUE,
    imputation = "training median", tuning_fit = c("2017-01-01", "2019-06-30"),
    tuning_holdout = c("2019-07-01", "2019-12-31"),
    tuning_metric = "class-weighted log-loss"
  ),
  calibration = list(method = "Platt", clip = c(0.001, 0.999), bins = 10L),
  metrics = list(
    roc_auc = "yardstick ROC-AUC with event level 1",
    pr_auc = paste0(
      "yardstick trapezoidal PR-AUC with event level 1; the fully tied constant ",
      "forecast is assigned test prevalence by the standard no-skill convention"
    ),
    ece = "10 equal-width probability bins"
  ),
  operating_policy = list(type = "precision_floor", target = 0.10, grid = threshold_grid),
  top_k = k_values, brier_skill_reference = "training-period national prevalence",
  pairing = "All seven methods scored on identical ordered test municipality-night rows"
)
write_json(run_config, file.path(output_dir, "run_config.json"), pretty = TRUE, auto_unbox = TRUE)

status <- if (any(qa_checks$status == "FAIL")) "FAIL" else
  if (any(qa_checks$status == "WARN")) "PASS_WITH_WARNINGS" else "PASS"
report_lines <- c(
  "# Phase 2 Canonical Model Comparison", "",
  paste0("- Run ID: `", run_id, "`"), paste0("- Status: **", status, "**"),
  paste0("- Models: ", nrow(canonical_table)),
  paste0("- Paired test rows: ", format(nrow(pred_test), big.mark = ",")),
  paste0("- Test positives: ", format(sum(pred_test$truth), big.mark = ",")),
  "- Calibration fit: 2020-01-01 through 2020-03-31",
  "- Threshold selection: 2020-04-01 through 2020-06-30",
  "- Test: 2020-07-01 through 2021-12-31", "", "## Canonical Results", "",
  "| Model | Features | ROC-AUC | PR-AUC | PR lift | Brier | BSS | ECE | Precision | Recall | Alerts/night | P@10 |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  vapply(seq_len(nrow(canonical_table)), function(i) {
    r <- canonical_table[i, ]
    paste0(
      "| ", r$model_label, " | ", r$n_features, " | ", sprintf("%.3f", r$roc_auc),
      " | ", sprintf("%.3f", r$pr_auc), " | ", sprintf("%.1f", r$pr_lift),
      " | ", sprintf("%.4f", r$brier), " | ", sprintf("%.3f", r$brier_skill_score),
      " | ", sprintf("%.4f", r$ece_equal_10), " | ",
      ifelse(is.na(r$test_precision), "--", sprintf("%.3f", r$test_precision)), " | ",
      ifelse(is.na(r$test_recall), "--", sprintf("%.3f", r$test_recall)), " | ",
      ifelse(is.na(r$alerts_per_night), "--", sprintf("%.1f", r$alerts_per_night)), " | ",
      sprintf("%.3f", r$top10_precision), " |"
    )
  }, character(1)),
  "", "## Interpretation Guardrail", "",
  paste0(
    "Nested-model deltas quantify incremental predictive contribution on a common held-out sample. ",
    "They are not causal effects or a Shapley decomposition because predictors interact and share information."
  ), "", "## QA", "",
  paste0("- PASS: ", sum(qa_checks$status == "PASS")),
  paste0("- WARN: ", sum(qa_checks$status == "WARN")),
  paste0("- FAIL: ", sum(qa_checks$status == "FAIL")), "",
  "See `qa_checks.csv` and `headline_reproduction_check.csv` for machine-readable checks."
)
writeLines(report_lines, file.path(output_dir, "QA_REPORT.md"))
if (any(qa_checks$status == "FAIL")) {
  stop("Phase 2 completed with failed QA. See ", file.path(output_dir, "QA_REPORT.md"))
}
log_message("Phase 2 complete with status ", status)
log_message("Report: ", file.path(output_dir, "QA_REPORT.md"))
