# Independent validation of the definitive Phase 5 duration-sensitivity bundle.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(readr)
  library(tibble)
  library(tidyr)
  library(yardstick)
})

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

truth_factor <- function(y) factor(ifelse(y == 1L, "1", "0"), levels = c("1", "0"))
roc_value <- function(y, p) as.numeric(roc_auc_vec(truth_factor(y), p, event_level = "first"))
pr_value <- function(y, p) as.numeric(pr_auc_vec(truth_factor(y), p, event_level = "first"))

top10_precision <- function(y, p, dates, gids) {
  ranked <- data.table(
    date = as.Date(dates), GID_2 = as.character(gids), truth = as.integer(y),
    score = as.numeric(p)
  )
  setorder(ranked, date, -score, GID_2)
  ranked[, rank := seq_len(.N), by = date]
  ranked[rank <= 10L, mean(truth)]
}

project_dir <- detect_project_dir()
analysis_run_id <- parse_arg("--analysis-run-id", "20260814_063000_phase5_duration_sensitivity")
label_run_id <- parse_arg("--label-run-id", "20260814_040000_phase5_duration_labels")
analysis_dir <- file.path(project_dir, "data", "revision", analysis_run_id)
label_dir <- file.path(project_dir, "data", "revision", label_run_id)
figure_path <- file.path(project_dir, "figures", "revision", analysis_run_id, "duration_sensitivity.pdf")

manifest <- read_csv(file.path(label_dir, "duration_variant_manifest.csv"), show_col_types = FALSE)
labels <- as_tibble(readRDS(file.path(label_dir, "duration_labels_model_keys.rds")))
metrics <- read_csv(file.path(analysis_dir, "all_model_metrics.csv"), show_col_types = FALSE)
uncertainty <- read_csv(file.path(analysis_dir, "duration_classifier_uncertainty.csv"), show_col_types = FALSE)
draws <- read_csv(file.path(analysis_dir, "duration_classifier_bootstrap_draws.csv"), show_col_types = FALSE)
contribution_uncertainty <- read_csv(
  file.path(analysis_dir, "duration_contribution_uncertainty.csv"), show_col_types = FALSE
)
contribution_draws <- read_csv(
  file.path(analysis_dir, "duration_contribution_bootstrap_draws.csv"), show_col_types = FALSE
)
publication <- read_csv(
  file.path(analysis_dir, "duration_sensitivity_publication_table.csv"), show_col_types = FALSE
)
summary_qa <- read_csv(file.path(analysis_dir, "summary_qa_checks.csv"), show_col_types = FALSE)

variant_ids <- manifest$variant_id
model_ids <- c("weather_calendar_static_xgb", "history_free_xgb", "full_83_xgb")
metric_rechecks <- list()
prediction_key_reference <- NULL
for (variant_id in variant_ids) {
  pred <- as.data.table(readRDS(file.path(
    analysis_dir, "variants", variant_id, "predictions_test.rds"
  )))
  pred[, date := as.Date(date)]
  key <- pred[, .(GID_2, date)]
  if (is.null(prediction_key_reference)) prediction_key_reference <- key
  if (!identical(key, prediction_key_reference)) stop("Test key ordering differs for ", variant_id)
  stored_variant <- metrics %>% filter(variant_id == .env$variant_id)
  for (model_id in model_ids) {
    score <- pred[[paste0(model_id, "_calibrated")]]
    stored <- stored_variant %>% filter(model_id == .env$model_id)
    threshold <- stored$operating_threshold
    if (is.finite(threshold)) {
      selected <- score >= threshold
      tp <- sum(pred$truth == 1L & selected); fp <- sum(pred$truth == 0L & selected)
      precision <- tp / (tp + fp); recall <- tp / sum(pred$truth)
      alerts <- (tp + fp) / uniqueN(pred$date)
    } else {
      precision <- recall <- alerts <- NA_real_
    }
    metric_rechecks[[paste(variant_id, model_id)]] <- tibble(
      variant_id = variant_id, model_id = model_id,
      stored_n_positive = stored$n_positive, recomputed_n_positive = sum(pred$truth),
      stored_roc_auc = stored$roc_auc, recomputed_roc_auc = roc_value(pred$truth, score),
      stored_pr_auc = stored$pr_auc, recomputed_pr_auc = pr_value(pred$truth, score),
      stored_brier = stored$brier, recomputed_brier = mean((score - pred$truth)^2),
      stored_top10_precision = stored$top10_precision,
      recomputed_top10_precision = top10_precision(pred$truth, score, pred$date, pred$GID_2),
      stored_precision = stored$test_precision, recomputed_precision = precision,
      stored_recall = stored$test_recall, recomputed_recall = recall,
      stored_alerts_per_night = stored$alerts_per_night, recomputed_alerts_per_night = alerts
    )
  }
  rm(pred); gc(verbose = FALSE)
}
metric_rechecks <- bind_rows(metric_rechecks) %>% mutate(
  max_absolute_difference = pmax(
    abs(stored_roc_auc - recomputed_roc_auc),
    abs(stored_pr_auc - recomputed_pr_auc),
    abs(stored_brier - recomputed_brier),
    abs(stored_top10_precision - recomputed_top10_precision),
    abs(stored_precision - recomputed_precision),
    abs(stored_recall - recomputed_recall),
    abs(stored_alerts_per_night - recomputed_alerts_per_night),
    na.rm = TRUE
  )
)
write_csv(metric_rechecks, file.path(analysis_dir, "independent_metric_rechecks.csv"))

recomputed_uncertainty <- draws %>%
  pivot_longer(
    c(roc_auc, pr_auc, pr_lift, brier, brier_skill_score, precision, recall, alerts_per_night),
    names_to = "metric", values_to = "value"
  ) %>% group_by(variant_id, model_id, metric) %>% summarise(
    lower_recomputed = unname(quantile(value, 0.025, na.rm = TRUE)),
    upper_recomputed = unname(quantile(value, 0.975, na.rm = TRUE)), .groups = "drop"
  )
uncertainty_recheck <- uncertainty %>%
  left_join(recomputed_uncertainty, by = c("variant_id", "model_id", "metric")) %>%
  mutate(
    lower_difference = abs(lower - lower_recomputed),
    upper_difference = abs(upper - upper_recomputed)
  )
write_csv(uncertainty_recheck, file.path(analysis_dir, "independent_uncertainty_rechecks.csv"))

recomputed_contribution <- contribution_draws %>%
  group_by(variant_id, comparison_id, metric) %>% summarise(
    lower_recomputed = unname(quantile(raw_delta, 0.025, na.rm = TRUE)),
    upper_recomputed = unname(quantile(raw_delta, 0.975, na.rm = TRUE)), .groups = "drop"
  )
contribution_recheck <- contribution_uncertainty %>%
  left_join(recomputed_contribution, by = c("variant_id", "comparison_id", "metric")) %>%
  mutate(
    lower_difference = abs(lower - lower_recomputed),
    upper_difference = abs(upper - upper_recomputed)
  )
write_csv(contribution_recheck, file.path(analysis_dir, "independent_contribution_rechecks.csv"))

is_nested <- function(ids) {
  ids <- manifest %>% filter(variant_id %in% ids) %>% arrange(duration_minutes) %>% pull(variant_id)
  all(vapply(seq_len(length(ids) - 1L), function(i) {
    broader <- labels %>% filter(variant_id == ids[i]) %>% select(GID_2, date)
    narrower <- labels %>% filter(variant_id == ids[i + 1L]) %>% select(GID_2, date)
    nrow(anti_join(narrower, broader, by = c("GID_2", "date"))) == 0L
  }, logical(1)))
}
night_ids <- manifest %>% filter(variant_family == "night_duration_threshold") %>% pull(variant_id)
active_ids <- manifest %>% filter(variant_family == "active_at_overpass_minimum_duration") %>% pull(variant_id)
labels15 <- labels %>% filter(variant_id == "active0130_min015") %>%
  select(-variant_id) %>% arrange(GID_2, date)
labels30 <- labels %>% filter(variant_id == "active0130_min030") %>%
  select(-variant_id) %>% arrange(GID_2, date)

full_metrics <- metrics %>% filter(model_id == "full_83_xgb") %>%
  select(variant_id, roc_auc, pr_auc, pr_lift, brier, brier_skill_score,
         top10_precision, top10_recall)
publication_metric_check <- publication %>%
  select(variant_id, roc_auc, pr_auc, pr_lift, brier, brier_skill_score,
         top10_precision, top10_recall) %>%
  inner_join(full_metrics, by = "variant_id", suffix = c("_publication", "_metrics"))
publication_max_difference <- max(abs(unlist(publication_metric_check %>%
  select(ends_with("_publication"))) - unlist(publication_metric_check %>%
  select(ends_with("_metrics")))))

checks <- list()
add_check <- function(id, condition, observed, expected) {
  checks[[length(checks) + 1L]] <<- tibble(
    check_id = id, status = if (isTRUE(condition)) "PASS" else "FAIL",
    observed = as.character(observed), expected = as.character(expected)
  )
}
add_check("summary_internal_qa", all(summary_qa$status == "PASS"),
          sum(summary_qa$status == "FAIL"), "0 failures")
add_check("prediction_metric_reproduction", max(metric_rechecks$max_absolute_difference, na.rm = TRUE) < 1e-12,
          format(max(metric_rechecks$max_absolute_difference, na.rm = TRUE), scientific = TRUE), "<1e-12")
add_check("prediction_positive_reproduction", all(
  metric_rechecks$stored_n_positive == metric_rechecks$recomputed_n_positive
), paste(unique(metric_rechecks$stored_n_positive - metric_rechecks$recomputed_n_positive), collapse = "/"), "0")
add_check("uncertainty_quantile_reproduction", max(
  c(uncertainty_recheck$lower_difference, uncertainty_recheck$upper_difference), na.rm = TRUE
) < 1e-12, format(max(c(uncertainty_recheck$lower_difference,
                       uncertainty_recheck$upper_difference), na.rm = TRUE), scientific = TRUE), "<1e-12")
add_check("contribution_quantile_reproduction", max(
  c(contribution_recheck$lower_difference, contribution_recheck$upper_difference), na.rm = TRUE
) < 1e-12, format(max(c(contribution_recheck$lower_difference,
                       contribution_recheck$upper_difference), na.rm = TRUE), scientific = TRUE), "<1e-12")
add_check("nested_night_labels", is_nested(night_ids), is_nested(night_ids), "TRUE")
add_check("nested_active_labels", is_nested(active_ids), is_nested(active_ids), "TRUE")
add_check("active_15_30_label_identity", identical(labels15, labels30),
          nrow(anti_join(labels15, labels30, by = names(labels15))) +
            nrow(anti_join(labels30, labels15, by = names(labels15))), "0 differences")
add_check("publication_metric_reproduction", publication_max_difference < 1e-15,
          format(publication_max_difference, scientific = TRUE), "<1e-15")
add_check("headline_control", abs(
  full_metrics$roc_auc[full_metrics$variant_id == "night_gt180"] - 0.918657
) < 0.001, full_metrics$roc_auc[full_metrics$variant_id == "night_gt180"], "within 0.001")
add_check("figure_pdf", file.exists(figure_path) && file.info(figure_path)$size > 5000,
          ifelse(file.exists(figure_path), file.info(figure_path)$size, 0), ">5000 bytes")
qa <- bind_rows(checks)
write_csv(qa, file.path(analysis_dir, "independent_qa_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else "PASS"
writeLines(c(
  "# Independent Phase 5 Validation", "",
  paste0("- Status: **", status, "**"),
  paste0("- PASS: ", sum(qa$status == "PASS")),
  paste0("- FAIL: ", sum(qa$status == "FAIL")), "",
  "All 27 point estimates were recomputed from saved predictions. Bootstrap",
  "percentiles, paired contribution intervals, label nesting, publication-table",
  "values, and the headline control were checked independently."
), file.path(analysis_dir, "INDEPENDENT_VALIDATION_REPORT.md"))
cat("Independent Phase 5 duration validation: ", status, "\n", sep = "")
if (status == "FAIL") stop("Independent Phase 5 validation failed.")
