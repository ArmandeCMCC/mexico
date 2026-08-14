# Independent artifact-level validation for a completed Phase 2 run.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(readr)
  library(tibble)
  library(yardstick)
})

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))
  stop("Run from the project root or scripts directory.")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: Rscript scripts/revision/03_validate_canonical_comparison.R <run-id-or-directory>")
}
project_dir <- detect_project_dir()
run_dir <- if (dir.exists(args[[1]])) normalizePath(args[[1]]) else
  file.path(project_dir, "data", "revision", args[[1]])
if (!dir.exists(run_dir)) stop("Run directory not found: ", run_dir)

required_files <- c(
  "canonical_model_table.csv", "paired_predictions_test.rds", "paired_predictions_validation.rds",
  "split_summary.csv",
  "feature_set_manifest.csv", "feature_family_dictionary.csv", "calibration_parameters.csv",
  "operating_metrics.csv", "topk_metrics.csv", "paired_model_comparisons.csv",
  "headline_reproduction_check.csv", "model_run_manifest.csv", "run_config.json", "QA_REPORT.md"
)
missing_files <- required_files[!file.exists(file.path(run_dir, required_files))]
if (length(missing_files)) stop("Missing artifacts: ", paste(missing_files, collapse = ", "))

table <- read_csv(file.path(run_dir, "canonical_model_table.csv"), show_col_types = FALSE)
test <- as.data.table(readRDS(file.path(run_dir, "paired_predictions_test.rds")))
val <- as.data.table(readRDS(file.path(run_dir, "paired_predictions_validation.rds")))
families <- read_csv(file.path(run_dir, "feature_family_dictionary.csv"), show_col_types = FALSE)
topk <- read_csv(file.path(run_dir, "topk_metrics.csv"), show_col_types = FALSE)
paired <- read_csv(file.path(run_dir, "paired_model_comparisons.csv"), show_col_types = FALSE)
reproduction <- read_csv(file.path(run_dir, "headline_reproduction_check.csv"), show_col_types = FALSE)
split_summary <- read_csv(file.path(run_dir, "split_summary.csv"), show_col_types = FALSE)
expected_test_nights <- split_summary$n_nights[split_summary$split == "test"]
if (length(expected_test_nights) != 1L) stop("Canonical test-night count is missing or duplicated.")

model_ids <- c(
  "constant_prevalence", "municipality_climatology", "history_only_xgb",
  "weather_calendar_static_xgb", "history_free_xgb", "full_83_xgb", "ridge_logistic_83"
)
raw_cols <- paste0(model_ids, "_raw"); calibrated_cols <- paste0(model_ids, "_calibrated")
truth_factor <- function(y) factor(ifelse(y == 1L, "1", "0"), levels = c("1", "0"))
roc <- function(y, p) suppressWarnings(as.numeric(roc_auc_vec(truth_factor(y), p, event_level = "first")))
pr <- function(y, p) {
  if (length(unique(as.numeric(p))) < 2L) return(mean(y))
  suppressWarnings(as.numeric(pr_auc_vec(truth_factor(y), p, event_level = "first")))
}
ece <- function(y, p, n_bins = 10L) {
  bins <- cut(p, seq(0, 1, length.out = n_bins + 1L), include.lowest = TRUE, labels = FALSE)
  sum(vapply(seq_len(n_bins), function(b) {
    idx <- which(bins == b)
    if (!length(idx)) return(0)
    length(idx) / length(p) * abs(mean(p[idx]) - mean(y[idx]))
  }, numeric(1)))
}

checks <- list()
add_check <- function(id, condition, observed, expected, severity = "FAIL") {
  checks[[length(checks) + 1L]] <<- tibble(
    check_id = id, status = if (isTRUE(condition)) "PASS" else severity,
    observed = as.character(observed), expected = as.character(expected)
  )
}
add_check("seven_models", identical(table$model_id, model_ids), paste(table$model_id, collapse = "|"),
          paste(model_ids, collapse = "|"))
add_check("test_rows", nrow(test) == 1334151L, nrow(test), 1334151L)
add_check("validation_rows", nrow(val) == 447174L, nrow(val), 447174L)
add_check("test_positives", sum(test$truth) == 5350L, sum(test$truth), 5350L)
add_check("test_nights", uniqueN(test$date) == expected_test_nights,
          uniqueN(test$date), expected_test_nights)
add_check("test_municipalities", uniqueN(test$GID_2) == 2457L, uniqueN(test$GID_2), 2457L)
add_check("test_unique_keys", anyDuplicated(test[, .(GID_2, date)]) == 0L,
          anyDuplicated(test[, .(GID_2, date)]), 0L)
add_check("prediction_columns", all(c(raw_cols, calibrated_cols) %in% names(test)),
          paste(setdiff(c(raw_cols, calibrated_cols), names(test)), collapse = "|"), "none missing")
add_check("test_predictions_complete", !anyNA(test[, ..calibrated_cols]),
          sum(is.na(test[, ..calibrated_cols])), 0L)
add_check("validation_predictions_complete", !anyNA(val[, ..calibrated_cols]),
          sum(is.na(val[, ..calibrated_cols])), 0L)
add_check("feature_dictionary_83", nrow(families) == 83L && n_distinct(families$feature) == 83L,
          paste(nrow(families), n_distinct(families$feature), sep = "/"), "83/83")
add_check("feature_families_complete", !anyNA(families$feature_family),
          sum(is.na(families$feature_family)), 0L)
expected_counts <- c(0L, 0L, 10L, 50L, 73L, 83L, 83L)
add_check("feature_counts", identical(as.integer(table$n_features), expected_counts),
          paste(table$n_features, collapse = ","), paste(expected_counts, collapse = ","))

metric_rechecks <- bind_rows(lapply(model_ids, function(id) {
  p <- test[[paste0(id, "_calibrated")]]
  tibble(
    model_id = id, roc_auc_recomputed = roc(test$truth, p),
    pr_auc_recomputed = pr(test$truth, p), brier_recomputed = mean((p - test$truth)^2),
    ece_recomputed = ece(test$truth, p)
  )
})) %>% left_join(table, by = "model_id") %>% mutate(
  roc_abs_diff = abs(roc_auc_recomputed - roc_auc),
  pr_abs_diff = abs(pr_auc_recomputed - pr_auc),
  brier_abs_diff = abs(brier_recomputed - brier),
  ece_abs_diff = abs(ece_recomputed - ece_equal_10)
)
write_csv(metric_rechecks, file.path(run_dir, "independent_metric_rechecks.csv"))
max_metric_diff <- max(c(
  metric_rechecks$roc_abs_diff, metric_rechecks$pr_abs_diff,
  metric_rechecks$brier_abs_diff, metric_rechecks$ece_abs_diff
))
add_check("metric_recomputation", max_metric_diff < 1e-12, max_metric_diff, "<1e-12")
add_check("topk_row_count", nrow(topk) == 56L, nrow(topk), 56L)
add_check("topk_alert_counts", all(topk$n_alerts == topk$k * topk$n_days),
          sum(topk$n_alerts != topk$k * topk$n_days), 0L)
add_check("paired_comparison_rows", nrow(paired) == 44L && all(paired$same_test_rows),
          paste(nrow(paired), all(paired$same_test_rows), sep = "/"), "44/TRUE")
add_check("headline_reproduction", all(reproduction$pass),
          paste0(sum(reproduction$pass), "/", nrow(reproduction)),
          paste0(nrow(reproduction), "/", nrow(reproduction)), severity = "WARN")

qa <- bind_rows(checks)
write_csv(qa, file.path(run_dir, "independent_qa_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else
  if (any(qa$status == "WARN")) "PASS_WITH_WARNINGS" else "PASS"
writeLines(c(
  "# Independent Phase 2 Validation", "", paste0("- Run: `", basename(run_dir), "`"),
  paste0("- Status: **", status, "**"), paste0("- PASS: ", sum(qa$status == "PASS")),
  paste0("- WARN: ", sum(qa$status == "WARN")), paste0("- FAIL: ", sum(qa$status == "FAIL")),
  "", "Metrics were independently recomputed from the saved paired test predictions."
), file.path(run_dir, "INDEPENDENT_VALIDATION_REPORT.md"))
cat("Independent validation status:", status, "\n")
cat("Report:", file.path(run_dir, "INDEPENDENT_VALIDATION_REPORT.md"), "\n")
if (status == "FAIL") quit(status = 1L)
