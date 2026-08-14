# Phase 3: persistent versus dynamic skill.
# Writes only below data/revision/<run_id>/ and figures/revision/<run_id>/.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(parsnip)
  library(patchwork)
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
  format(Sys.time(), "%Y%m%d_%H%M%S"), "_phase3_dynamic_skill"
))
phase2_dir <- file.path(
  project_dir, "data", "revision", "20260813_144700_phase2_canonical_comparison"
)
features_path <- file.path(project_dir, "data", "model_ready", "features_engineered.rds")
splits_path <- file.path(project_dir, "data", "model_ready", "splits_fixed.rds")
headline_features_path <- file.path(phase2_dir, "feature_sets", "full_83_xgb.txt")
output_dir <- file.path(project_dir, "data", "revision", run_id)
models_dir <- file.path(output_dir, "models")
feature_sets_dir <- file.path(output_dir, "feature_sets")
figure_dir <- file.path(project_dir, "figures", "revision", run_id)

required_inputs <- c(
  features_path, splits_path, headline_features_path,
  file.path(phase2_dir, "paired_predictions_validation.rds"),
  file.path(phase2_dir, "paired_predictions_test.rds"),
  file.path(phase2_dir, "canonical_model_table.csv"),
  file.path(phase2_dir, "independent_qa_checks.csv")
)
missing_inputs <- required_inputs[!file.exists(required_inputs)]
if (length(missing_inputs)) {
  stop("Missing required inputs:\n", paste0("  - ", missing_inputs, collapse = "\n"))
}
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to overwrite non-empty run: ", output_dir)
}
if (dir.exists(figure_dir) && length(list.files(figure_dir, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to overwrite non-empty figure directory: ", figure_dir)
}
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(feature_sets_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
log_path <- file.path(output_dir, "run.log")
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

truth_factor <- function(y) factor(ifelse(y == 1L, "1", "0"), levels = c("1", "0"))
compute_roc_auc <- function(y, score) {
  if (length(unique(y)) < 2L) return(NA_real_)
  if (length(unique(as.numeric(score))) < 2L) return(0.5)
  suppressWarnings(as.numeric(roc_auc_vec(truth_factor(y), score, event_level = "first")))
}
compute_pr_auc <- function(y, score) {
  if (length(unique(y)) < 2L) return(NA_real_)
  if (length(unique(as.numeric(score))) < 2L) return(mean(y))
  suppressWarnings(as.numeric(pr_auc_vec(truth_factor(y), score, event_level = "first")))
}
compute_brier <- function(y, prob) mean((as.numeric(prob) - y)^2)

within_auc <- function(y, score) {
  n_pos <- sum(y == 1L); n_neg <- sum(y == 0L)
  if (!n_pos || !n_neg) return(NA_real_)
  ranks <- rank(as.numeric(score), ties.method = "average")
  (sum(ranks[y == 1L]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
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

log_message("Phase 3 persistent versus dynamic skill")
log_message("Run ID: ", run_id)
phase2_qa <- read_csv(file.path(phase2_dir, "independent_qa_checks.csv"), show_col_types = FALSE)
if (any(phase2_qa$status == "FAIL")) stop("Definitive Phase 2 run has failed QA checks.")

log_message("Loading canonical model-ready RDS...")
features_object <- readRDS(features_path)
splits_fixed <- readRDS(splits_path)
headline_features <- readLines(headline_features_path, warn = FALSE)
headline_features <- headline_features[nzchar(headline_features)]
required_columns <- c("GID_2", "date", "outage_3h_or_more", headline_features)
missing_columns <- setdiff(required_columns, names(features_object))
if (length(missing_columns)) stop("Missing canonical columns: ", paste(missing_columns, collapse = ", "))
dat <- features_object[, required_columns, drop = FALSE]
rm(features_object); gc(verbose = FALSE)
dat$date <- as.Date(dat$date); dat$GID_2 <- as.character(dat$GID_2)
dat$outage_3h_or_more <- as.integer(dat$outage_3h_or_more)
if (length(headline_features) != 83L || anyDuplicated(headline_features)) {
  stop("Frozen headline feature list is not 83 unique ordered predictors.")
}
if (anyDuplicated(dat[c("GID_2", "date")])) stop("Duplicate canonical keys.")

train_start <- as.Date(splits_fixed$train_range[1]); train_end <- as.Date(splits_fixed$train_range[2])
val_start <- as.Date(splits_fixed$val_range[1]); val_end <- as.Date(splits_fixed$val_range[2])
test_start <- as.Date(splits_fixed$test_range[1]); test_end <- as.Date(splits_fixed$test_range[2])
train_idx <- which(dat$date >= train_start & dat$date <= train_end)
val_idx <- which(dat$date >= val_start & dat$date <= val_end)
test_idx <- which(dat$date >= test_start & dat$date <= test_end)
calib_idx_in_val <- which(dat$date[val_idx] >= as.Date("2020-01-01") &
                            dat$date[val_idx] <= as.Date("2020-03-31"))

phase2_val <- as.data.table(readRDS(file.path(phase2_dir, "paired_predictions_validation.rds")))
phase2_test <- as.data.table(readRDS(file.path(phase2_dir, "paired_predictions_test.rds")))
if (nrow(phase2_val) != length(val_idx) || nrow(phase2_test) != length(test_idx)) {
  stop("Phase 2 prediction row counts do not match canonical split rows.")
}
if (!all(phase2_val$GID_2 == dat$GID_2[val_idx]) ||
    !all(as.Date(phase2_val$date) == dat$date[val_idx]) ||
    !all(phase2_val$truth == dat$outage_3h_or_more[val_idx]) ||
    !all(phase2_test$GID_2 == dat$GID_2[test_idx]) ||
    !all(as.Date(phase2_test$date) == dat$date[test_idx]) ||
    !all(phase2_test$truth == dat$outage_3h_or_more[test_idx])) {
  stop("Phase 2 predictions are not key/truth aligned to the canonical RDS.")
}

# Municipality history registry. Ever-outage and first-outage states use the
# complete observed 2017-2021 panel and are explicitly diagnostic, not deployable.
registry <- data.table(
  GID_2 = dat$GID_2,
  date = dat$date,
  truth = dat$outage_3h_or_more
)[, .(
  training_positive_nights = sum(truth == 1L & date >= train_start & date <= train_end),
  validation_positive_nights = sum(truth == 1L & date >= val_start & date <= val_end),
  test_positive_nights = sum(truth == 1L & date >= test_start & date <= test_end),
  all_positive_nights = sum(truth == 1L),
  first_observed_outage = if (any(truth == 1L)) min(date[truth == 1L]) else as.Date(NA)
), by = GID_2]
registry[, `:=`(
  had_training_outage = training_positive_nights > 0L,
  has_both_test_classes = test_positive_nights > 0L & test_positive_nights < length(unique(dat$date[test_idx])),
  ever_observed_outage = all_positive_nights > 0L
)]
write_csv(as_tibble(registry), file.path(output_dir, "municipality_history_registry.csv"))

no_days_features <- setdiff(headline_features, "days_since_last_outage")
if (length(no_days_features) != 82L) stop("No-days-since feature set must contain 82 predictors.")
training_medians <- vapply(headline_features, function(nm) {
  value <- median(dat[[nm]][train_idx], na.rm = TRUE)
  if (!is.finite(value)) stop("No finite training median for ", nm)
  as.numeric(value)
}, numeric(1))
missing_indicator_features <- headline_features[vapply(
  dat[train_idx, headline_features, drop = FALSE], anyNA, logical(1)
)]
indicator_names <- paste0(missing_indicator_features, "__missing")
explicit_features <- c(headline_features, indicator_names)
if (length(missing_indicator_features) != 64L || length(explicit_features) != 147L) {
  stop("Explicit-missing feature inventory drifted: expected 64 indicators and 147 total predictors.")
}
writeLines(no_days_features, file.path(feature_sets_dir, "full_no_days_since_xgb.txt"))
writeLines(explicit_features, file.path(feature_sets_dir, "full_explicit_missing_xgb.txt"))
write_csv(tibble(
  feature = headline_features,
  training_median = training_medians,
  missing_indicator_added = headline_features %in% missing_indicator_features,
  indicator_name = ifelse(
    headline_features %in% missing_indicator_features,
    paste0(headline_features, "__missing"), NA_character_
  ),
  training_missing_n = vapply(
    dat[train_idx, headline_features, drop = FALSE], function(x) sum(is.na(x)), integer(1)
  ),
  validation_missing_n = vapply(
    dat[val_idx, headline_features, drop = FALSE], function(x) sum(is.na(x)), integer(1)
  ),
  test_missing_n = vapply(
    dat[test_idx, headline_features, drop = FALSE], function(x) sum(is.na(x)), integer(1)
  ),
  test_missing_without_training_missing = vapply(headline_features, function(nm) {
    sum(is.na(dat[[nm]][test_idx])) > 0L && !anyNA(dat[[nm]][train_idx])
  }, logical(1))
), file.path(output_dir, "explicit_missing_preprocessing.csv"))

xgb_params <- list(
  trees = 500L, tree_depth = 6L, min_n = 10L, loss_reduction = 0,
  sample_size = 0.8, mtry = 0.8, learn_rate = 0.05
)

prepare_frame <- function(idx, base_features, explicit_missing = FALSE) {
  out <- dat[idx, c("outage_3h_or_more", base_features), drop = FALSE]
  if (explicit_missing) {
    for (nm in missing_indicator_features) {
      out[[paste0(nm, "__missing")]] <- as.integer(is.na(out[[nm]]))
    }
    # Every imputation value is training-derived. Predictors whose missingness
    # first appears after training are imputed but cannot receive a learned
    # indicator without using future availability patterns.
    for (nm in base_features) {
      if (anyNA(out[[nm]])) out[[nm]][is.na(out[[nm]])] <- training_medians[[nm]]
    }
    if (anyNA(out)) stop("Explicit-missing preprocessing left NA values.")
  }
  out$outage_3h_or_more <- factor(as.character(out$outage_3h_or_more), levels = c("1", "0"))
  logical_cols <- names(out)[vapply(out, is.logical, logical(1))]
  for (nm in logical_cols) out[[nm]] <- as.integer(out[[nm]])
  out
}

fit_variant <- function(model_id, base_features, explicit_missing = FALSE) {
  log_message("Training ", model_id, "...")
  set.seed(42)
  train_frame <- prepare_frame(train_idx, base_features, explicit_missing)
  specified_features <- setdiff(names(train_frame), "outage_3h_or_more")
  zero_variance <- specified_features[vapply(train_frame[specified_features], function(x) {
    length(unique(x[!is.na(x)])) <= 1L
  }, logical(1))]
  n_pos <- sum(train_frame$outage_3h_or_more == "1")
  n_neg <- sum(train_frame$outage_3h_or_more == "0")
  scale_pos_weight <- n_neg / n_pos
  mtry_count <- max(1L, floor(xgb_params$mtry * length(specified_features)))
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

  val_frame <- prepare_frame(val_idx, base_features, explicit_missing)
  val_prob <- predict(fitted, new_data = val_frame, type = "prob")$.pred_1
  rm(val_frame); gc(verbose = FALSE)
  test_frame <- prepare_frame(test_idx, base_features, explicit_missing)
  test_prob <- predict(fitted, new_data = test_frame, type = "prob")$.pred_1
  rm(test_frame); gc(verbose = FALSE)

  engine <- extract_fit_engine(fitted)
  xgb.save(engine, file.path(models_dir, paste0(model_id, ".ubj")))
  importance <- as_tibble(xgb.importance(model = engine)) %>%
    mutate(model_id = model_id, .before = 1)
  rm(fitted, engine); gc(verbose = FALSE)
  log_message("Completed ", model_id, " in ", sprintf("%.1f", elapsed), " seconds")
  list(
    val_raw = as.numeric(val_prob), test_raw = as.numeric(test_prob),
    importance = importance,
    manifest = tibble(
      model_id = model_id, n_specified_features = length(specified_features),
      n_zero_variance_removed = length(zero_variance),
      zero_variance_removed = ifelse(length(zero_variance), paste(zero_variance, collapse = "|"), "none"),
      explicit_missing = explicit_missing, n_missing_indicators = ifelse(explicit_missing, 64L, 0L),
      mtry_count = mtry_count, scale_pos_weight = scale_pos_weight,
      training_seconds = elapsed
    )
  )
}

no_days_result <- fit_variant("full_no_days_since_xgb", no_days_features, FALSE)
explicit_result <- fit_variant("full_explicit_missing_xgb", headline_features, TRUE)
variant_manifest <- bind_rows(no_days_result$manifest, explicit_result$manifest)
write_csv(variant_manifest, file.path(output_dir, "variant_model_manifest.csv"))
write_csv(bind_rows(no_days_result$importance, explicit_result$importance),
          file.path(output_dir, "variant_feature_importance.csv"))

y_val <- dat$outage_3h_or_more[val_idx]; y_test <- dat$outage_3h_or_more[test_idx]
variant_calibration <- list()
for (model_id in c("full_no_days_since_xgb", "full_explicit_missing_xgb")) {
  result <- if (model_id == "full_no_days_since_xgb") no_days_result else explicit_result
  calibration <- fit_platt(result$val_raw[calib_idx_in_val], y_val[calib_idx_in_val])
  result$val_calibrated <- apply_platt(result$val_raw, calibration)
  result$test_calibrated <- apply_platt(result$test_raw, calibration)
  if (model_id == "full_no_days_since_xgb") no_days_result <- result else explicit_result <- result
  variant_calibration[[model_id]] <- tibble(
    model_id = model_id, calibration_method = calibration$type,
    calibration_fit_start = as.Date("2020-01-01"), calibration_fit_end = as.Date("2020-03-31"),
    calibration_fit_rows = length(calib_idx_in_val),
    calibration_fit_positives = sum(y_val[calib_idx_in_val]),
    intercept = calibration$intercept, slope = calibration$slope
  )
}
write_csv(bind_rows(variant_calibration), file.path(output_dir, "variant_calibration_parameters.csv"))

variant_val_predictions <- data.frame(
  GID_2 = phase2_val$GID_2, date = as.Date(phase2_val$date), truth = phase2_val$truth,
  full_no_days_since_xgb_raw = no_days_result$val_raw,
  full_no_days_since_xgb_calibrated = no_days_result$val_calibrated,
  full_explicit_missing_xgb_raw = explicit_result$val_raw,
  full_explicit_missing_xgb_calibrated = explicit_result$val_calibrated
)
saveRDS(variant_val_predictions, file.path(output_dir, "variant_predictions_validation.rds"), compress = "gzip")

model_ids <- c(
  "municipality_climatology", "history_only_xgb", "history_free_xgb", "full_83_xgb",
  "full_no_days_since_xgb", "full_explicit_missing_xgb"
)
model_labels <- c(
  municipality_climatology = "Municipality climatology",
  history_only_xgb = "History-only XGBoost",
  history_free_xgb = "History-free remote-sensing",
  full_83_xgb = "Full 83-feature XGBoost",
  full_no_days_since_xgb = "Full model without days-since",
  full_explicit_missing_xgb = "Full model with missing indicators"
)

test_predictions <- data.table(
  GID_2 = phase2_test$GID_2,
  date = as.Date(phase2_test$date),
  truth = as.integer(phase2_test$truth),
  municipality_climatology = phase2_test$municipality_climatology_calibrated,
  history_only_xgb = phase2_test$history_only_xgb_calibrated,
  history_free_xgb = phase2_test$history_free_xgb_calibrated,
  full_83_xgb = phase2_test$full_83_xgb_calibrated,
  full_no_days_since_xgb = no_days_result$test_calibrated,
  full_explicit_missing_xgb = explicit_result$test_calibrated,
  full_no_days_since_xgb_raw = no_days_result$test_raw,
  full_explicit_missing_xgb_raw = explicit_result$test_raw
)
test_predictions <- merge(test_predictions, registry, by = "GID_2", all.x = TRUE, sort = FALSE)
# Restore canonical key ordering after the registry join.
canonical_order <- data.table(GID_2 = phase2_test$GID_2, date = as.Date(phase2_test$date), row_id = seq_len(nrow(phase2_test)))
test_predictions <- merge(canonical_order, test_predictions, by = c("GID_2", "date"), sort = FALSE)
setorder(test_predictions, row_id)
test_predictions[, row_id := NULL]
if (!all(test_predictions$GID_2 == phase2_test$GID_2) ||
    !all(test_predictions$date == as.Date(phase2_test$date)) ||
    !all(test_predictions$truth == phase2_test$truth)) {
  stop("Registry join changed canonical test-key alignment.")
}
test_predictions[, first_outage_state := fcase(
  is.na(first_observed_outage), "never_observed_outage",
  date <= first_observed_outage, "pre_or_first_observed_outage",
  default = "post_first_observed_outage"
)]

# Test-period mean centering uses scores but not labels. It is a diagnostic for
# pooled discrimination after removing persistent municipality score levels.
centered_cols <- character()
for (model_id in model_ids) {
  centered_col <- paste0(model_id, "__centered")
  test_predictions[, (centered_col) := get(model_id) - mean(get(model_id)), by = GID_2]
  centered_cols <- c(centered_cols, centered_col)
}

scope_masks <- list(
  all_test = rep(TRUE, nrow(test_predictions)),
  both_classes_in_test = test_predictions$has_both_test_classes,
  training_outage_municipalities = test_predictions$had_training_outage,
  no_training_outage_municipalities = !test_predictions$had_training_outage,
  ever_outage_municipalities = test_predictions$ever_observed_outage,
  pre_or_first_observed_outage = test_predictions$first_outage_state == "pre_or_first_observed_outage",
  post_first_observed_outage = test_predictions$first_outage_state == "post_first_observed_outage"
)
scope_labels <- c(
  all_test = "All test municipality-nights",
  both_classes_in_test = "Municipalities with both test classes",
  training_outage_municipalities = "Municipalities with a training outage",
  no_training_outage_municipalities = "No outage observed during training",
  ever_outage_municipalities = "Ever-outage municipalities only",
  pre_or_first_observed_outage = "Up to and including first observed outage",
  post_first_observed_outage = "Strictly after first observed outage"
)

evaluate_model_scope <- function(model_id, scope_id, mask) {
  d <- test_predictions[mask, .(
    GID_2, date, truth, score = get(model_id),
    centered_score = get(paste0(model_id, "__centered"))
  )]
  within <- d[, {
    n_pos <- sum(truth == 1L); n_neg <- sum(truth == 0L)
    .(n_rows = .N, n_positive = n_pos, n_negative = n_neg,
      comparable_pairs = as.double(n_pos) * as.double(n_neg),
      within_roc_auc = within_auc(truth, score))
  }, by = GID_2]
  eligible <- within[!is.na(within_roc_auc)]
  macro_auc <- if (nrow(eligible)) mean(eligible$within_roc_auc) else NA_real_
  pair_auc <- if (nrow(eligible) && sum(eligible$comparable_pairs) > 0) {
    weighted.mean(eligible$within_roc_auc, eligible$comparable_pairs)
  } else NA_real_
  within_summary <- if (nrow(eligible)) {
    quantile(eligible$within_roc_auc, c(0.25, 0.5, 0.75), names = FALSE)
  } else rep(NA_real_, 3)
  list(
    summary = tibble(
      model_id = model_id, model_label = unname(model_labels[model_id]),
      evaluation_scope = scope_id, scope_label = unname(scope_labels[scope_id]),
      n_rows = nrow(d), n_nights = n_distinct(d$date), n_municipalities = n_distinct(d$GID_2),
      n_positive = sum(d$truth), prevalence = mean(d$truth),
      pooled_roc_auc = compute_roc_auc(d$truth, d$score),
      pooled_pr_auc = compute_pr_auc(d$truth, d$score),
      brier = compute_brier(d$truth, d$score),
      centered_pooled_roc_auc = compute_roc_auc(d$truth, d$centered_score),
      centered_pooled_pr_auc = compute_pr_auc(d$truth, d$centered_score),
      within_macro_roc_auc = macro_auc,
      within_pair_weighted_roc_auc = pair_auc,
      within_q25_roc_auc = within_summary[1], within_median_roc_auc = within_summary[2],
      within_q75_roc_auc = within_summary[3],
      n_within_auc_municipalities = nrow(eligible),
      n_comparable_pairs = sum(eligible$comparable_pairs)
    ),
    within = within %>% mutate(
      model_id = model_id, evaluation_scope = scope_id, .before = 1
    )
  )
}

dynamic_rows <- list(); within_rows <- list(); counter <- 1L
for (scope_id in names(scope_masks)) {
  log_message("Evaluating scope: ", scope_id)
  for (model_id in model_ids) {
    result <- evaluate_model_scope(model_id, scope_id, scope_masks[[scope_id]])
    dynamic_rows[[counter]] <- result$summary
    within_rows[[counter]] <- result$within
    counter <- counter + 1L
  }
}
dynamic_table <- bind_rows(dynamic_rows)
within_table <- bind_rows(within_rows)
write_csv(dynamic_table, file.path(output_dir, "dynamic_skill_table.csv"))
write_csv(within_table, file.path(output_dir, "within_municipality_auc_by_scope.csv"))

scope_support <- bind_rows(lapply(names(scope_masks), function(scope_id) {
  mask <- scope_masks[[scope_id]]
  tibble(
    evaluation_scope = scope_id, scope_label = unname(scope_labels[scope_id]),
    n_rows = sum(mask), n_nights = n_distinct(test_predictions$date[mask]),
    n_municipalities = n_distinct(test_predictions$GID_2[mask]),
    n_positive = sum(test_predictions$truth[mask]), prevalence = mean(test_predictions$truth[mask])
  )
}))
write_csv(scope_support, file.path(output_dir, "dynamic_scope_support.csv"))

# Publication-ready supplementary figure.
model_order <- rev(unname(model_labels[model_ids]))
all_test_plot <- dynamic_table %>%
  filter(evaluation_scope == "all_test") %>%
  select(model_label, pooled_roc_auc, centered_pooled_roc_auc,
         within_macro_roc_auc, within_pair_weighted_roc_auc) %>%
  tidyr::pivot_longer(-model_label, names_to = "measure", values_to = "auc") %>%
  mutate(
    model_label = factor(model_label, levels = model_order),
    measure = recode(
      measure,
      pooled_roc_auc = "Pooled ROC-AUC",
      centered_pooled_roc_auc = "Municipality-centered pooled",
      within_macro_roc_auc = "Within-municipality macro",
      within_pair_weighted_roc_auc = "Within-municipality pair-weighted"
    )
  )

scope_order <- c(
  "All test municipality-nights", "Municipalities with both test classes",
  "Municipalities with a training outage", "No outage observed during training",
  "Ever-outage municipalities only", "Up to and including first observed outage",
  "Strictly after first observed outage"
)
history_scope_plot <- dynamic_table %>%
  filter(model_id == "full_83_xgb") %>%
  select(scope_label, pooled_roc_auc, centered_pooled_roc_auc) %>%
  tidyr::pivot_longer(-scope_label, names_to = "measure", values_to = "auc") %>%
  mutate(
    scope_label = factor(scope_label, levels = rev(scope_order)),
    measure = recode(
      measure,
      pooled_roc_auc = "Pooled ROC-AUC",
      centered_pooled_roc_auc = "Municipality-centered pooled"
    )
  )
write_csv(bind_rows(
  all_test_plot %>% transmute(panel = "A", item = as.character(model_label), measure, auc),
  history_scope_plot %>% transmute(panel = "B", item = as.character(scope_label), measure, auc)
), file.path(output_dir, "dynamic_skill_figure_data.csv"))

palette_a <- c(
  "Pooled ROC-AUC" = "#0072B2", "Municipality-centered pooled" = "#D55E00",
  "Within-municipality macro" = "#009E73", "Within-municipality pair-weighted" = "#CC79A7"
)
plot_a <- ggplot(all_test_plot, aes(x = auc, y = model_label, colour = measure, shape = measure)) +
  geom_vline(xintercept = 0.5, colour = "grey65", linewidth = 0.45, linetype = "dashed") +
  geom_point(size = 2.7, position = position_dodge(width = 0.55)) +
  scale_colour_manual(values = palette_a) +
  scale_x_continuous(limits = c(0.30, 0.95), breaks = seq(0.3, 0.9, 0.1)) +
  labs(title = "A. Persistent versus within-municipality discrimination", x = "ROC-AUC", y = NULL,
       colour = NULL, shape = NULL) +
  theme_minimal(base_size = 10.5) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"), axis.text.y = element_text(colour = "black"))

palette_b <- c("Pooled ROC-AUC" = "#0072B2", "Municipality-centered pooled" = "#D55E00")
plot_b <- ggplot(history_scope_plot, aes(x = auc, y = scope_label, colour = measure, shape = measure)) +
  geom_vline(xintercept = 0.5, colour = "grey65", linewidth = 0.45, linetype = "dashed") +
  geom_point(size = 2.9, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = palette_b) +
  scale_x_continuous(limits = c(0.30, 0.95), breaks = seq(0.3, 0.9, 0.1)) +
  labs(title = "B. Headline model by outage-history status", x = "ROC-AUC", y = NULL,
       colour = NULL, shape = NULL) +
  theme_minimal(base_size = 10.5) +
  theme(legend.position = "none", panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"), axis.text.y = element_text(colour = "black"))

combined_plot <- plot_a / plot_b + plot_layout(heights = c(1, 1.05), guides = "keep")
png_path <- file.path(figure_dir, "dynamic_skill_figure.png")
pdf_path <- file.path(figure_dir, "dynamic_skill_figure.pdf")
ggsave(png_path, combined_plot, width = 10.5, height = 8.2, dpi = 320, bg = "white")
ggsave(pdf_path, combined_plot, width = 10.5, height = 8.2, device = grDevices::pdf, bg = "white")

saveRDS(as.data.frame(test_predictions), file.path(output_dir, "phase3_predictions_test.rds"), compress = "gzip")

compact_table <- dynamic_table %>%
  filter(evaluation_scope == "all_test") %>%
  select(model_label, n_positive, pooled_roc_auc, pooled_pr_auc,
         centered_pooled_roc_auc, within_macro_roc_auc,
         within_pair_weighted_roc_auc, n_within_auc_municipalities)
write_csv(compact_table, file.path(output_dir, "dynamic_skill_table_all_test.csv"))
latex_lines <- c(
  "\\begin{tabular}{lrrrrrrr}", "\\toprule",
  "Model & Positives & Pooled ROC & PR-AUC & Centered ROC & Within macro & Within pair-weighted & Eligible municipalities \\\\",
  "\\midrule"
)
for (i in seq_len(nrow(compact_table))) {
  r <- compact_table[i, ]
  label <- gsub("_", "\\\\_", r$model_label, fixed = TRUE)
  latex_lines <- c(latex_lines, sprintf(
    "%s & %d & %.3f & %.3f & %.3f & %.3f & %.3f & %d \\\\",
    label, r$n_positive, r$pooled_roc_auc, r$pooled_pr_auc,
    r$centered_pooled_roc_auc, r$within_macro_roc_auc,
    r$within_pair_weighted_roc_auc, r$n_within_auc_municipalities
  ))
}
writeLines(c(latex_lines, "\\bottomrule", "\\end{tabular}"),
           file.path(output_dir, "dynamic_skill_table_all_test.tex"))

phase2_canonical <- read_csv(file.path(phase2_dir, "canonical_model_table.csv"), show_col_types = FALSE)
phase2_full <- phase2_canonical %>% filter(model_id == "full_83_xgb")
full_all <- dynamic_table %>% filter(model_id == "full_83_xgb", evaluation_scope == "all_test")
clim_all <- dynamic_table %>% filter(
  model_id == "municipality_climatology", evaluation_scope == "all_test"
)
centered_means <- test_predictions[, lapply(.SD, mean), by = GID_2, .SDcols = centered_cols]
centered_mean_max <- max(abs(as.matrix(centered_means[, ..centered_cols])))

qa_checks <- tribble(
  ~check_id, ~status, ~observed, ~expected,
  "phase2_has_no_failures", "PASS",
  paste0(sum(phase2_qa$status == "PASS"), " PASS; ", sum(phase2_qa$status == "WARN"), " WARN"),
  "No Phase 2 FAIL",
  "canonical_test_rows", ifelse(nrow(test_predictions) == 1334151L, "PASS", "FAIL"),
  as.character(nrow(test_predictions)), "1334151",
  "canonical_test_positives", ifelse(sum(test_predictions$truth) == 5350L, "PASS", "FAIL"),
  as.character(sum(test_predictions$truth)), "5350",
  "canonical_key_alignment", ifelse(
    all(test_predictions$GID_2 == phase2_test$GID_2) &
      all(test_predictions$date == as.Date(phase2_test$date)) &
      all(test_predictions$truth == phase2_test$truth), "PASS", "FAIL"
  ), "Phase 3 versus Phase 2 paired keys/truth", "Exact",
  "phase2_headline_prediction_reuse", ifelse(
    max(abs(test_predictions$full_83_xgb - phase2_test$full_83_xgb_calibrated)) < 1e-15,
    "PASS", "FAIL"
  ), as.character(max(abs(test_predictions$full_83_xgb - phase2_test$full_83_xgb_calibrated))), "<1e-15",
  "headline_pooled_roc_reproduction", ifelse(
    abs(full_all$pooled_roc_auc - phase2_full$roc_auc) < 1e-12, "PASS", "FAIL"
  ), as.character(abs(full_all$pooled_roc_auc - phase2_full$roc_auc)), "<1e-12",
  "headline_pooled_pr_reproduction", ifelse(
    abs(full_all$pooled_pr_auc - phase2_full$pr_auc) < 1e-12, "PASS", "FAIL"
  ), as.character(abs(full_all$pooled_pr_auc - phase2_full$pr_auc)), "<1e-12",
  "no_days_feature_count", ifelse(length(no_days_features) == 82L, "PASS", "FAIL"),
  as.character(length(no_days_features)), "82",
  "explicit_missing_inventory", ifelse(
    length(missing_indicator_features) == 64L && length(explicit_features) == 147L,
    "PASS", "FAIL"
  ), paste(length(missing_indicator_features), length(explicit_features), sep = "/"), "64/147",
  "temporally_new_missingness", ifelse(
    sum(vapply(headline_features, function(nm) {
      sum(is.na(dat[[nm]][test_idx])) > 0L && !anyNA(dat[[nm]][train_idx])
    }, logical(1))) == 1L,
    "PASS", "FAIL"
  ), paste0("1 predictor; ", sum(vapply(headline_features, function(nm) {
    if (!anyNA(dat[[nm]][train_idx])) sum(is.na(dat[[nm]][test_idx])) else 0L
  }, integer(1))), " cells"), "1 predictor; 2457 cells",
  "variant_prediction_completeness", ifelse(
    all(is.finite(no_days_result$test_calibrated)) && all(is.finite(explicit_result$test_calibrated)),
    "PASS", "FAIL"
  ), paste(sum(!is.finite(no_days_result$test_calibrated)),
           sum(!is.finite(explicit_result$test_calibrated)), sep = "/"), "0/0",
  "dynamic_table_dimensions", ifelse(nrow(dynamic_table) == 42L, "PASS", "FAIL"),
  as.character(nrow(dynamic_table)), "42 model-scope rows",
  "both_class_municipalities", ifelse(sum(registry$has_both_test_classes) == 802L, "PASS", "FAIL"),
  as.character(sum(registry$has_both_test_classes)), "802",
  "training_outage_municipalities", ifelse(sum(registry$had_training_outage) == 1045L, "PASS", "FAIL"),
  as.character(sum(registry$had_training_outage)), "1045",
  "ever_and_never_outage_municipalities", ifelse(
    sum(registry$ever_observed_outage) == 1129L && sum(!registry$ever_observed_outage) == 1328L,
    "PASS", "FAIL"
  ), paste(sum(registry$ever_observed_outage), sum(!registry$ever_observed_outage), sep = "/"), "1129/1328",
  "first_outage_partition", ifelse(
    sum(test_predictions$truth[test_predictions$first_outage_state == "pre_or_first_observed_outage"]) == 65L &&
      sum(test_predictions$truth[test_predictions$first_outage_state == "post_first_observed_outage"]) == 5285L,
    "PASS", "FAIL"
  ), paste(
    sum(test_predictions$truth[test_predictions$first_outage_state == "pre_or_first_observed_outage"]),
    sum(test_predictions$truth[test_predictions$first_outage_state == "post_first_observed_outage"]), sep = "/"
  ), "65/5285",
  "centering_group_means", ifelse(centered_mean_max < 1e-12, "PASS", "FAIL"),
  as.character(centered_mean_max), "<1e-12",
  "climatology_dynamic_null", ifelse(
    abs(clim_all$centered_pooled_roc_auc - 0.5) < 1e-12 &&
      abs(clim_all$within_macro_roc_auc - 0.5) < 1e-12 &&
      abs(clim_all$within_pair_weighted_roc_auc - 0.5) < 1e-12,
    "PASS", "FAIL"
  ), paste(clim_all$centered_pooled_roc_auc, clim_all$within_macro_roc_auc,
           clim_all$within_pair_weighted_roc_auc, sep = "/"), "0.5/0.5/0.5",
  "figure_outputs", ifelse(
    file.exists(png_path) && file.info(png_path)$size > 10000L &&
      file.exists(pdf_path) && file.info(pdf_path)$size > 5000L,
    "PASS", "FAIL"
  ), paste(file.info(png_path)$size, file.info(pdf_path)$size, sep = "/"),
  "PNG >10000; vector PDF >5000"
)
write_csv(qa_checks, file.path(output_dir, "qa_checks.csv"))

run_config <- list(
  run_id = run_id, created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
  git_commit = system2("git", c("rev-parse", "HEAD"), stdout = TRUE), seed = 42L,
  input = list(
    canonical_features_path = features_path,
    canonical_rds_sha256 = "a05db6b77dd7b55824c9a7f5cdf90083a20adb14e9ee50b02b3e7f539f3e0ad7",
    definitive_phase2_dir = phase2_dir
  ),
  splits = list(
    train = c(as.character(train_start), as.character(train_end)),
    calibration_fit = c("2020-01-01", "2020-03-31"),
    test = c(as.character(test_start), as.character(test_end))
  ),
  variants = list(
    no_days_since = list(n_features = 82L, removed = "days_since_last_outage"),
    explicit_missing = list(
      n_original_features = 83L, n_indicators = 64L, n_specified_features = 147L,
      imputation = "training median for every missing value",
      indicator_rule = "indicator added only when predictor has missing values in training",
      unseen_test_missingness = "clim_n_obs has 2457 test missing cells and is imputed without an indicator"
    )
  ),
  xgboost = xgb_params,
  dynamic_evaluation = list(
    within_macro = "Equal-weight mean across municipalities with both classes in the evaluated scope",
    within_pair_weighted = "Weighted by n_positive * n_negative within municipality",
    centered = "Subtract municipality mean score computed over the complete test period; labels are not used",
    pre_first = "Rows dated before or on the municipality first observed positive night",
    post_first = "Rows strictly after the municipality first observed positive night",
    ever_outage = "Uses complete 2017-2021 outcomes and is diagnostic, not deployable"
  )
)
write_json(run_config, file.path(output_dir, "run_config.json"), pretty = TRUE, auto_unbox = TRUE)

status <- if (any(qa_checks$status == "FAIL")) "FAIL" else
  if (any(qa_checks$status == "WARN")) "PASS_WITH_WARNINGS" else "PASS"
full_rows <- dynamic_table %>% filter(model_id == "full_83_xgb")
all_models <- dynamic_table %>% filter(evaluation_scope == "all_test")
report <- c(
  "# Phase 3 Persistent Versus Dynamic Skill", "",
  paste0("- Run ID: `", run_id, "`"), paste0("- Status: **", status, "**"),
  paste0("- Paired test rows: ", format(nrow(test_predictions), big.mark = ",")),
  paste0("- Test positives: ", format(sum(test_predictions$truth), big.mark = ",")),
  paste0("- Municipalities with estimable within-municipality AUC: ",
         full_all$n_within_auc_municipalities), "",
  "## All-Test Dynamic Metrics", "",
  "| Model | Pooled ROC | PR-AUC | Centered ROC | Within macro | Within pair-weighted |",
  "|---|---:|---:|---:|---:|---:|",
  vapply(seq_len(nrow(all_models)), function(i) {
    r <- all_models[i, ]
    sprintf("| %s | %.3f | %.3f | %.3f | %.3f | %.3f |",
            r$model_label, r$pooled_roc_auc, r$pooled_pr_auc,
            r$centered_pooled_roc_auc, r$within_macro_roc_auc,
            r$within_pair_weighted_roc_auc)
  }, character(1)), "", "## Headline History Scopes", "",
  "| Scope | Rows | Positives | Pooled ROC | Centered ROC |",
  "|---|---:|---:|---:|---:|",
  vapply(seq_len(nrow(full_rows)), function(i) {
    r <- full_rows[i, ]
    sprintf("| %s | %s | %s | %.3f | %.3f |", r$scope_label,
            format(r$n_rows, big.mark = ","), format(r$n_positive, big.mark = ","),
            r$pooled_roc_auc, r$centered_pooled_roc_auc)
  }, character(1)), "", "## Interpretation Guardrail", "",
  paste0(
    "Within-municipality and centered metrics are diagnostics of dynamic discrimination, not causal effects. ",
    "The first-outage and ever-outage restrictions use observed outcome histories and must not be described as deployable sampling rules."
  ), "", "## QA", "",
  paste0("- PASS: ", sum(qa_checks$status == "PASS")),
  paste0("- WARN: ", sum(qa_checks$status == "WARN")),
  paste0("- FAIL: ", sum(qa_checks$status == "FAIL"))
)
writeLines(report, file.path(output_dir, "QA_REPORT.md"))
if (any(qa_checks$status == "FAIL")) stop("Phase 3 failed QA. See ", file.path(output_dir, "QA_REPORT.md"))
log_message("Phase 3 complete with status ", status)
log_message("Report: ", file.path(output_dir, "QA_REPORT.md"))
