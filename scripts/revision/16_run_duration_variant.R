# Fit one Phase 5 duration-sensitivity variant without materializing a duplicate
# model-ready panel. Existing files are read-only; outputs are isolated by run ID.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
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
compute_pr_auc <- function(y, p) suppressWarnings(as.numeric(
  pr_auc_vec(truth_factor(y), p, event_level = "first")
))
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

compute_topk <- function(y, prob, dates, gids, k_values = c(5L, 10L, 20L, 50L)) {
  ranked <- data.table(
    date = as.Date(dates), GID_2 = as.character(gids), truth = as.integer(y),
    score = as.numeric(prob)
  )
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

compute_history_in_place <- function(dat) {
  setorder(dat, GID_2, date)
  dat[, `:=`(
    .cs_outage = cumsum(outage_3h_or_more),
    .cs_n_outages = cumsum(n_outages),
    .cs_minutes = cumsum(total_length_min),
    .cs_n_env = cumsum(n_outages_environmental),
    .cs_n_tech = cumsum(n_outages_technical)
  ), by = GID_2]
  dat[, `:=`(
    hist_outage_count_30d = shift(.cs_outage, 1L) - shift(.cs_outage, 31L),
    hist_n_outages_30d = shift(.cs_n_outages, 1L) - shift(.cs_n_outages, 31L),
    hist_outage_count_env_30d = shift(.cs_n_env, 1L) - shift(.cs_n_env, 31L),
    hist_outage_count_tech_30d = shift(.cs_n_tech, 1L) - shift(.cs_n_tech, 31L),
    hist_outage_count_90d = shift(.cs_outage, 1L) - shift(.cs_outage, 91L),
    cumulative_outage_minutes_90d = shift(.cs_minutes, 1L) - shift(.cs_minutes, 91L)
  ), by = GID_2]
  dat[, `:=`(
    hist_outage_rate_30d = hist_outage_count_30d / 30,
    hist_outage_rate_90d = hist_outage_count_90d / 90
  )]

  dat[, hist_median_duration_180d := {
    pos_idx <- which(outage_3h_or_more == 1L)
    pos_dates <- date[pos_idx]
    pos_dur <- total_length_min[pos_idx]
    out <- rep(NA_real_, .N)
    if (length(pos_idx)) {
      lo_ptr <- 1L; hi_ptr <- 0L
      cutoffs_lo <- date - 180L; cutoffs_hi <- date - 1L
      for (i in seq_len(.N)) {
        while (hi_ptr < length(pos_dates) && pos_dates[hi_ptr + 1L] <= cutoffs_hi[i]) {
          hi_ptr <- hi_ptr + 1L
        }
        while (lo_ptr <= length(pos_dates) && pos_dates[lo_ptr] < cutoffs_lo[i]) {
          lo_ptr <- lo_ptr + 1L
        }
        if (hi_ptr >= lo_ptr) out[i] <- median(pos_dur[lo_ptr:hi_ptr], na.rm = TRUE)
      }
    }
    out
  }, by = GID_2]

  dat[, days_since_last_outage := {
    today <- fifelse(outage_3h_or_more == 1L, as.integer(date), NA_integer_)
    last_so_far <- nafill(today, type = "locf")
    as.integer(date) - shift(last_so_far, 1L)
  }, by = GID_2]
  dat[, c(".cs_outage", ".cs_n_outages", ".cs_minutes", ".cs_n_env", ".cs_n_tech") := NULL]
  setorder(dat, .row_id)
  invisible(dat)
}

history_leakage_checks <- function(dat) {
  positive_rows <- which(dat$outage_3h_or_more == 1L)
  if (!length(positive_rows)) return(TRUE)
  sample_rows <- unique(c(head(positive_rows, 5L), tail(positive_rows, 5L)))
  all(vapply(sample_rows, function(i) {
    expected <- dat[
      GID_2 == dat$GID_2[i] & date >= dat$date[i] - 30L & date < dat$date[i],
      sum(outage_3h_or_more)
    ]
    observed <- dat$hist_outage_count_30d[i]
    is.na(observed) || identical(as.numeric(observed), as.numeric(expected))
  }, logical(1)))
}

project_dir <- detect_project_dir()
analysis_run_id <- parse_arg("--analysis-run-id", "20260814_050000_phase5_duration_sensitivity")
label_run_id <- parse_arg("--label-run-id", "20260814_040000_phase5_duration_labels")
variant_id <- parse_arg("--variant")
if (is.null(variant_id)) stop("--variant is required.")

analysis_dir <- file.path(project_dir, "data", "revision", analysis_run_id)
variant_dir <- file.path(analysis_dir, "variants", variant_id)
models_dir <- file.path(variant_dir, "models")
if (dir.exists(variant_dir) && length(list.files(variant_dir, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to overwrite non-empty variant directory: ", variant_dir)
}
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
log_path <- file.path(variant_dir, "run.log")
writeLines(capture.output(sessionInfo()), file.path(variant_dir, "session_info.txt"))
log_message <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", msg, "\n",
      file = log_path, append = TRUE, sep = "")
}

label_dir <- file.path(project_dir, "data", "revision", label_run_id)
labels_path <- file.path(label_dir, "duration_labels_model_keys.rds")
variant_manifest_path <- file.path(label_dir, "duration_variant_manifest.csv")
label_summary_path <- file.path(label_dir, "duration_label_summary.csv")
label_qa_path <- file.path(label_dir, "independent_validation_checks.csv")
base_path <- file.path(project_dir, "data", "model_ready", "features_engineered.rds")
phase2_dir <- file.path(project_dir, "data", "revision", "20260813_144700_phase2_canonical_comparison")
family_path <- file.path(phase2_dir, "feature_family_dictionary.csv")
headline_dir <- file.path(
  project_dir, "data", "baselines", "binary_threshold", "ablation_batches",
  "20260512_111324", "20260512_111341_forecast_strict_bench_rs_history"
)
features_path <- file.path(headline_dir, "features_used.txt")
required <- c(labels_path, variant_manifest_path, label_summary_path, label_qa_path,
              base_path, family_path, features_path)
if (any(!file.exists(required))) stop("Missing input(s): ", paste(required[!file.exists(required)], collapse = ", "))
label_qa <- read_csv(label_qa_path, show_col_types = FALSE)
if (any(label_qa$status == "FAIL")) stop("Duration-label bundle failed independent QA.")

variant_manifest <- read_csv(variant_manifest_path, show_col_types = FALSE)
variant_spec <- variant_manifest %>% filter(variant_id == .env$variant_id)
if (nrow(variant_spec) != 1L) stop("Unknown or duplicated variant: ", variant_id)
expected_positives <- read_csv(label_summary_path, show_col_types = FALSE) %>%
  filter(variant_id == .env$variant_id) %>% pull(model_key_municipality_nights)

headline_features <- readLines(features_path, warn = FALSE)
headline_features <- headline_features[nzchar(headline_features)]
family_lookup <- read_csv(family_path, show_col_types = FALSE)
history_features <- family_lookup %>% filter(feature_family == "outage_history") %>% pull(feature)
nonhistory_features <- setdiff(headline_features, history_features)
weather_calendar_static <- family_lookup %>%
  filter(feature_family %in% c("weather", "calendar_astronomy", "static_built_environment")) %>%
  pull(feature)
model_features <- list(
  weather_calendar_static_xgb = headline_features[headline_features %in% weather_calendar_static],
  history_free_xgb = nonhistory_features,
  full_83_xgb = headline_features
)
if (!identical(unname(vapply(model_features, length, integer(1))), c(50L, 73L, 83L)) ||
    length(history_features) != 10L) stop("Canonical feature inventory drifted.")

log_message("Phase 5 duration sensitivity variant: ", variant_id)
log_message("Loading canonical model-ready data and retaining only non-history predictors...")
base_object <- readRDS(base_path)
needed_base <- c("GID_2", "date", nonhistory_features)
missing_base <- setdiff(needed_base, names(base_object))
if (length(missing_base)) stop("Pre-history data lack: ", paste(missing_base, collapse = ", "))
dat <- as.data.table(base_object[, needed_base, drop = FALSE])
rm(base_object); gc(verbose = FALSE)
dat[, `:=`(GID_2 = as.character(GID_2), date = as.Date(date), .row_id = .I)]
if (nrow(dat) != 4464369L || anyDuplicated(dat[, .(GID_2, date)])) {
  stop("Canonical model support drifted.")
}

labels <- as.data.table(readRDS(labels_path))
target_variant_id <- variant_id
labels <- labels[variant_id == target_variant_id]
if (nrow(labels) != expected_positives || anyDuplicated(labels[, .(GID_2, date)])) {
  stop("Variant label support drifted.")
}
labels[, `:=`(GID_2 = as.character(GID_2), date = as.Date(date))]
label_cols <- c(
  "outage_3h_or_more", "n_outages", "total_length_min",
  "n_outages_environmental", "n_outages_technical"
)
dat[, `:=`(
  outage_3h_or_more = 0L, n_outages = 0L, total_length_min = 0,
  n_outages_environmental = 0L, n_outages_technical = 0L
)]
dat[labels, on = .(GID_2, date), (label_cols) := mget(paste0("i.", label_cols))]
if (sum(dat$outage_3h_or_more) != expected_positives) stop("Label join lost positives.")
rm(labels); gc(verbose = FALSE)

log_message("Computing ten variant-specific strictly-past history features...")
compute_history_in_place(dat)
if (!history_leakage_checks(dat)) stop("Strictly-past history spot check failed.")
if (length(setdiff(headline_features, names(dat)))) stop("A headline feature is unavailable after history construction.")

train_idx <- which(dat$date >= as.Date("2017-01-01") & dat$date <= as.Date("2019-12-31"))
val_idx <- which(dat$date >= as.Date("2020-01-01") & dat$date <= as.Date("2020-06-30"))
test_idx <- which(dat$date >= as.Date("2020-07-01") & dat$date <= as.Date("2021-12-31"))
calib_idx_in_val <- which(dat$date[val_idx] <= as.Date("2020-03-31"))
threshold_idx_in_val <- which(dat$date[val_idx] >= as.Date("2020-04-01"))
if (!identical(c(length(train_idx), length(val_idx), length(test_idx)),
               c(2683044L, 447174L, 1334151L))) stop("Chronological split support drifted.")

y_train <- dat$outage_3h_or_more[train_idx]
y_val <- dat$outage_3h_or_more[val_idx]
y_test <- dat$outage_3h_or_more[test_idx]
date_val <- dat$date[val_idx]; date_test <- dat$date[test_idx]
gid_val <- dat$GID_2[val_idx]; gid_test <- dat$GID_2[test_idx]
split_summary <- tibble(
  variant_id = variant_id,
  split = c("train", "calibration_fit", "threshold_selection", "test"),
  start = as.Date(c("2017-01-01", "2020-01-01", "2020-04-01", "2020-07-01")),
  end = as.Date(c("2019-12-31", "2020-03-31", "2020-06-30", "2021-12-31")),
  n_rows = c(length(train_idx), length(calib_idx_in_val), length(threshold_idx_in_val), length(test_idx)),
  n_positive = c(sum(y_train), sum(y_val[calib_idx_in_val]),
                 sum(y_val[threshold_idx_in_val]), sum(y_test))
) %>% mutate(prevalence = n_positive / n_rows)
write_csv(split_summary, file.path(variant_dir, "split_summary.csv"))

prepare_model_frame <- function(idx, feature_names) {
  out <- as.data.frame(dat[idx, c("outage_3h_or_more", feature_names), with = FALSE])
  out$outage_3h_or_more <- factor(as.character(out$outage_3h_or_more), levels = c("1", "0"))
  logical_cols <- names(out)[vapply(out, is.logical, logical(1))]
  for (nm in logical_cols) out[[nm]] <- as.integer(out[[nm]])
  out
}

xgb_params <- list(
  trees = 500L, tree_depth = 6L, min_n = 10L, loss_reduction = 0,
  sample_size = 0.8, mtry = 0.8, learn_rate = 0.05
)
fit_xgb <- function(model_id, feature_names) {
  log_message("Training ", model_id, " (", length(feature_names), " features)...")
  set.seed(42)
  train_frame <- prepare_model_frame(train_idx, feature_names)
  zero_variance <- feature_names[vapply(train_frame[feature_names], function(x) {
    length(unique(x[!is.na(x)])) <= 1L
  }, logical(1))]
  n_pos <- sum(train_frame$outage_3h_or_more == "1")
  n_neg <- sum(train_frame$outage_3h_or_more == "0")
  scale_pos_weight <- n_neg / n_pos
  mtry_count <- max(1L, floor(0.8 * length(feature_names)))
  model_recipe <- recipe(outage_3h_or_more ~ ., data = train_frame) %>% step_zv(all_predictors())
  model_spec <- boost_tree(
    trees = 500L, tree_depth = 6L, min_n = 10L, loss_reduction = 0,
    sample_size = 0.8, mtry = mtry_count, learn_rate = 0.05
  ) %>% set_engine(
    "xgboost", scale_pos_weight = scale_pos_weight,
    nthread = get_nthread(), verbosity = 0
  ) %>% set_mode("classification")
  workflow_spec <- workflow() %>% add_recipe(model_recipe) %>% add_model(model_spec)
  started <- proc.time()[["elapsed"]]
  fitted <- fit(workflow_spec, data = train_frame)
  elapsed <- proc.time()[["elapsed"]] - started
  rm(train_frame); gc(verbose = FALSE)

  val_prob <- predict(fitted, new_data = prepare_model_frame(val_idx, feature_names), type = "prob")$.pred_1
  gc(verbose = FALSE)
  test_prob <- predict(fitted, new_data = prepare_model_frame(test_idx, feature_names), type = "prob")$.pred_1
  gc(verbose = FALSE)
  engine <- extract_fit_engine(fitted)
  xgb.save(engine, file.path(models_dir, paste0(model_id, ".ubj")))
  importance <- as_tibble(xgb.importance(model = engine)) %>% mutate(model_id = model_id, .before = 1)
  rm(fitted, engine); gc(verbose = FALSE)
  log_message("Completed ", model_id, " in ", sprintf("%.1f", elapsed), " seconds")
  list(
    val = as.numeric(val_prob), test = as.numeric(test_prob), importance = importance,
    manifest = tibble(
      variant_id = variant_id, model_id = model_id, n_features = length(feature_names),
      training_seconds = elapsed, train_positives = n_pos,
      scale_pos_weight = scale_pos_weight, mtry_count = mtry_count,
      zero_variance_removals = ifelse(length(zero_variance), paste(zero_variance, collapse = "|"), "none")
    )
  )
}

pred_val <- data.table(GID_2 = gid_val, date = date_val, truth = y_val)
pred_test <- data.table(GID_2 = gid_test, date = date_test, truth = y_test)
importance_rows <- list(); manifest_rows <- list()
for (model_id in names(model_features)) {
  result <- fit_xgb(model_id, model_features[[model_id]])
  pred_val[, (paste0(model_id, "_raw")) := result$val]
  pred_test[, (paste0(model_id, "_raw")) := result$test]
  importance_rows[[model_id]] <- result$importance
  manifest_rows[[model_id]] <- result$manifest
  rm(result); gc(verbose = FALSE)
}

metric_rows <- list(); calibration_rows <- list(); operating_rows <- list()
topk_rows <- list(); sweep_rows <- list()
constant_test_prob <- rep(mean(y_train), length(y_test))
constant_brier <- compute_brier(y_test, constant_test_prob)
model_labels <- c(
  weather_calendar_static_xgb = "Weather/calendar/static XGBoost",
  history_free_xgb = "History-free remote-sensing XGBoost",
  full_83_xgb = "Full 83-feature XGBoost"
)
for (model_id in names(model_features)) {
  raw_val <- pred_val[[paste0(model_id, "_raw")]]
  raw_test <- pred_test[[paste0(model_id, "_raw")]]
  calibration <- fit_platt(raw_val[calib_idx_in_val], y_val[calib_idx_in_val])
  calibrated_val <- apply_platt(raw_val, calibration)
  calibrated_test <- apply_platt(raw_test, calibration)
  if (any(!is.finite(calibrated_val)) || any(!is.finite(calibrated_test))) {
    stop("Non-finite calibrated probabilities for ", model_id)
  }
  pred_val[, (paste0(model_id, "_calibrated")) := calibrated_val]
  pred_test[, (paste0(model_id, "_calibrated")) := calibrated_test]

  selection <- select_precision_floor(
    y_val[threshold_idx_in_val], calibrated_val[threshold_idx_in_val],
    date_val[threshold_idx_in_val], 0.10
  )
  sweep_rows[[model_id]] <- selection$sweep %>%
    mutate(variant_id = variant_id, model_id = model_id, .before = 1)
  if (is.null(selection$selected)) {
    selected_threshold <- NA_real_
    empty <- tibble(
      threshold = NA_real_, tp = NA_integer_, fp = NA_integer_, tn = NA_integer_, fn = NA_integer_,
      n_alerts = NA_integer_, alerts_per_night = NA_real_, precision = NA_real_, recall = NA_real_, f1 = NA_real_
    )
    val_operating <- empty; test_operating <- empty
  } else {
    selected_threshold <- selection$selected$threshold[[1]]
    val_operating <- selection$selected
    test_operating <- threshold_metrics(y_test, calibrated_test, date_test, selected_threshold)
  }
  topk <- compute_topk(y_test, calibrated_test, date_test, gid_test) %>%
    mutate(variant_id = variant_id, model_id = model_id, .before = 1)
  top10 <- topk %>% filter(k == 10L)
  pr_auc <- compute_pr_auc(y_test, calibrated_test)
  brier <- compute_brier(y_test, calibrated_test)
  metric_rows[[model_id]] <- tibble(
    variant_id = variant_id, model_id = model_id, model_label = unname(model_labels[model_id]),
    n_features = length(model_features[[model_id]]), n_test = length(y_test),
    n_positive = sum(y_test), prevalence = mean(y_test),
    roc_auc = compute_roc_auc(y_test, calibrated_test), pr_auc = pr_auc,
    pr_lift = pr_auc / mean(y_test), brier = brier,
    brier_skill_score = 1 - brier / constant_brier,
    logloss = compute_logloss(y_test, calibrated_test),
    ece_equal_10 = compute_ece_equal(y_test, calibrated_test),
    operating_threshold = selected_threshold,
    test_precision = test_operating$precision, test_recall = test_operating$recall,
    test_f1 = test_operating$f1, alerts_per_night = test_operating$alerts_per_night,
    top10_precision = top10$precision, top10_recall = top10$recall,
    top10_lift = top10$lift, top10_hits = top10$n_hits
  )
  calibration_rows[[model_id]] <- tibble(
    variant_id = variant_id, model_id = model_id, method = calibration$type,
    calibration_fit_start = as.Date("2020-01-01"), calibration_fit_end = as.Date("2020-03-31"),
    n_rows = length(calib_idx_in_val), n_positive = sum(y_val[calib_idx_in_val]),
    intercept = calibration$intercept, slope = calibration$slope
  )
  operating_rows[[model_id]] <- tibble(
    variant_id = variant_id, model_id = model_id, selected_threshold = selected_threshold,
    val_precision = val_operating$precision, val_recall = val_operating$recall,
    val_alerts_per_night = val_operating$alerts_per_night,
    test_precision = test_operating$precision, test_recall = test_operating$recall,
    test_alerts_per_night = test_operating$alerts_per_night,
    test_tp = test_operating$tp, test_fp = test_operating$fp,
    test_tn = test_operating$tn, test_fn = test_operating$fn
  )
  topk_rows[[model_id]] <- topk
  log_message("Scored ", model_id, ": ROC=", sprintf("%.4f", metric_rows[[model_id]]$roc_auc),
              ", PR=", sprintf("%.4f", pr_auc), ", P@10=", sprintf("%.4f", top10$precision))
}

metrics <- bind_rows(metric_rows)
calibration <- bind_rows(calibration_rows)
operating <- bind_rows(operating_rows)
topk <- bind_rows(topk_rows)
sweeps <- bind_rows(sweep_rows)
importance <- bind_rows(importance_rows) %>%
  rename(feature = Feature, gain = Gain, cover = Cover, frequency = Frequency) %>%
  left_join(family_lookup, by = "feature")
if (anyNA(importance$feature_family)) stop("An importance feature lacks a family.")
family_importance <- importance %>% group_by(model_id, feature_family) %>%
  summarise(total_gain = sum(gain), n_features_used = n_distinct(feature), .groups = "drop") %>%
  group_by(model_id) %>% mutate(gain_share = total_gain / sum(total_gain)) %>% ungroup() %>%
  mutate(variant_id = variant_id, .before = 1)

comparison_specs <- tribble(
  ~comparison_id, ~expanded_model, ~reference_model,
  "incremental_ntl", "history_free_xgb", "weather_calendar_static_xgb",
  "incremental_history", "full_83_xgb", "history_free_xgb"
)
comparison_metrics <- c(
  "roc_auc", "pr_auc", "pr_lift", "brier", "brier_skill_score",
  "ece_equal_10", "test_precision", "test_recall", "alerts_per_night",
  "top10_precision", "top10_recall"
)
contributions <- bind_rows(lapply(seq_len(nrow(comparison_specs)), function(i) {
  spec <- comparison_specs[i, ]
  expanded <- metrics %>% filter(model_id == spec$expanded_model)
  reference <- metrics %>% filter(model_id == spec$reference_model)
  bind_rows(lapply(comparison_metrics, function(metric) {
    raw_delta <- expanded[[metric]] - reference[[metric]]
    higher_better <- !metric %in% c("brier", "ece_equal_10", "alerts_per_night")
    tibble(
      variant_id = variant_id, comparison_id = spec$comparison_id,
      expanded_model = spec$expanded_model, reference_model = spec$reference_model,
      metric = metric, expanded_value = expanded[[metric]], reference_value = reference[[metric]],
      raw_delta = raw_delta,
      improvement_oriented_delta = ifelse(higher_better, raw_delta, -raw_delta)
    )
  }))
}))

write_csv(metrics, file.path(variant_dir, "model_metrics.csv"))
write_csv(calibration, file.path(variant_dir, "calibration_parameters.csv"))
write_csv(operating, file.path(variant_dir, "operating_metrics.csv"))
write_csv(topk, file.path(variant_dir, "topk_metrics.csv"))
write_csv(sweeps, file.path(variant_dir, "threshold_sweeps.csv"))
write_csv(bind_rows(manifest_rows), file.path(variant_dir, "model_manifest.csv"))
write_csv(importance, file.path(variant_dir, "feature_importance.csv"))
write_csv(family_importance, file.path(variant_dir, "feature_family_importance.csv"))
write_csv(contributions, file.path(variant_dir, "skill_contributions.csv"))
saveRDS(as.data.frame(pred_val), file.path(variant_dir, "predictions_validation.rds"), compress = "gzip")
saveRDS(as.data.frame(pred_test), file.path(variant_dir, "predictions_test.rds"), compress = "gzip")

phase2_full <- read_csv(file.path(phase2_dir, "canonical_model_table.csv"), show_col_types = FALSE) %>%
  filter(model_id == "full_83_xgb")
current_full <- metrics %>% filter(model_id == "full_83_xgb")
checks <- list()
add_check <- function(id, condition, observed, expected) {
  checks[[length(checks) + 1L]] <<- tibble(
    check_id = id, status = if (isTRUE(condition)) "PASS" else "FAIL",
    observed = as.character(observed), expected = as.character(expected)
  )
}
add_check("row_support", nrow(dat) == 4464369L, nrow(dat), "4464369")
add_check("positive_support", sum(dat$outage_3h_or_more) == expected_positives,
          sum(dat$outage_3h_or_more), expected_positives)
add_check("feature_inventory", identical(unname(vapply(model_features, length, integer(1))), c(50L, 73L, 83L)),
          paste(vapply(model_features, length, integer(1)), collapse = "/"), "50/73/83")
add_check("strict_history", history_leakage_checks(dat), history_leakage_checks(dat), "TRUE")
add_check("prediction_completeness", !anyNA(pred_val) && !anyNA(pred_test),
          sum(is.na(pred_val)) + sum(is.na(pred_test)), "0")
add_check("test_pairing", all(vapply(metric_rows, function(x) x$n_test == 1334151L, logical(1))),
          paste(unique(metrics$n_test), collapse = "/"), "1334151")
add_check("importance_normalization", all(abs(family_importance %>%
  group_by(model_id) %>% summarise(x = sum(gain_share), .groups = "drop") %>% pull(x) - 1) < 1e-10),
  paste(round(family_importance %>% group_by(model_id) %>%
                summarise(x = sum(gain_share), .groups = "drop") %>% pull(x), 10), collapse = "/"), "1/1/1")
add_check("finite_metrics", all(is.finite(unlist(metrics %>%
  select(roc_auc, pr_auc, brier, brier_skill_score, ece_equal_10, top10_precision)))),
  "checked", "all finite")
if (variant_id == "night_gt180") {
  add_check("headline_positive_reproduction", sum(dat$outage_3h_or_more) == 23008L,
            sum(dat$outage_3h_or_more), "23008 (frozen 22996 + 12 mapped additions)")
  add_check("headline_roc_proximity", abs(current_full$roc_auc - phase2_full$roc_auc) <= 0.01,
            sprintf("%.6f vs %.6f", current_full$roc_auc, phase2_full$roc_auc), "absolute delta <=0.01")
}
qa <- bind_rows(checks)
write_csv(qa, file.path(variant_dir, "qa_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else "PASS"

write_json(list(
  analysis_run_id = analysis_run_id, label_run_id = label_run_id,
  variant = as.list(variant_spec), seed = 42L,
  input = list(canonical_features_nonhistory_columns_only = base_path, headline_features = features_path,
               duration_labels = labels_path),
  split = list(train = c("2017-01-01", "2019-12-31"),
               calibration_fit = c("2020-01-01", "2020-03-31"),
               threshold_selection = c("2020-04-01", "2020-06-30"),
               test = c("2020-07-01", "2021-12-31")),
  models = lapply(names(model_features), function(id) list(id = id, n_features = length(model_features[[id]]))),
  xgboost = xgb_params,
  calibration = list(method = "Platt", clip = c(0.001, 0.999)),
  operating_policy = list(type = "precision_floor", target = 0.10),
  history = "ten features rebuilt from each variant using only dates t-1 and earlier"
), file.path(variant_dir, "run_config.json"), pretty = TRUE, auto_unbox = TRUE, digits = 16)
writeLines(c(
  "# Duration Variant QA", "", paste0("- Variant: `", variant_id, "`"),
  paste0("- Status: **", status, "**"), paste0("- PASS: ", sum(qa$status == "PASS")),
  paste0("- FAIL: ", sum(qa$status == "FAIL"))
), file.path(variant_dir, "QA_REPORT.md"))
log_message("Variant status: ", status)
if (status == "FAIL") stop("Duration variant QA failed: ", variant_id)
