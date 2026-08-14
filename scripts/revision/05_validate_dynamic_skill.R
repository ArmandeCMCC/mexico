# Independent artifact-level validation for a completed Phase 3 run.

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
  stop("Usage: Rscript scripts/revision/05_validate_dynamic_skill.R <run-id-or-directory>")
}
project_dir <- detect_project_dir()
run_dir <- if (dir.exists(args[[1]])) normalizePath(args[[1]]) else
  file.path(project_dir, "data", "revision", args[[1]])
if (!dir.exists(run_dir)) stop("Run directory not found: ", run_dir)
run_id <- basename(run_dir)
phase2_dir <- file.path(
  project_dir, "data", "revision", "20260813_144700_phase2_canonical_comparison"
)
figure_dir <- file.path(project_dir, "figures", "revision", run_id)

required_files <- c(
  "dynamic_skill_table.csv", "dynamic_skill_table_all_test.csv",
  "within_municipality_auc_by_scope.csv", "dynamic_scope_support.csv",
  "municipality_history_registry.csv", "phase3_predictions_test.rds",
  "variant_predictions_validation.rds", "variant_model_manifest.csv",
  "explicit_missing_preprocessing.csv", "variant_calibration_parameters.csv",
  "dynamic_skill_figure_data.csv", "qa_checks.csv", "run_config.json", "QA_REPORT.md"
)
missing_files <- required_files[!file.exists(file.path(run_dir, required_files))]
if (length(missing_files)) stop("Missing Phase 3 artifacts: ", paste(missing_files, collapse = ", "))

dynamic <- read_csv(file.path(run_dir, "dynamic_skill_table.csv"), show_col_types = FALSE)
within_saved <- read_csv(file.path(run_dir, "within_municipality_auc_by_scope.csv"), show_col_types = FALSE)
support <- read_csv(file.path(run_dir, "dynamic_scope_support.csv"), show_col_types = FALSE)
registry <- read_csv(file.path(run_dir, "municipality_history_registry.csv"), show_col_types = FALSE)
manifest <- read_csv(file.path(run_dir, "variant_model_manifest.csv"), show_col_types = FALSE)
preprocessing <- read_csv(file.path(run_dir, "explicit_missing_preprocessing.csv"), show_col_types = FALSE)
pred <- as.data.table(readRDS(file.path(run_dir, "phase3_predictions_test.rds")))
phase2 <- as.data.table(readRDS(file.path(phase2_dir, "paired_predictions_test.rds")))

model_ids <- c(
  "municipality_climatology", "history_only_xgb", "history_free_xgb", "full_83_xgb",
  "full_no_days_since_xgb", "full_explicit_missing_xgb"
)
scope_ids <- c(
  "all_test", "both_classes_in_test", "training_outage_municipalities",
  "no_training_outage_municipalities", "ever_outage_municipalities",
  "pre_or_first_observed_outage", "post_first_observed_outage"
)
centered_cols <- paste0(model_ids, "__centered")

truth_factor <- function(y) factor(ifelse(y == 1L, "1", "0"), levels = c("1", "0"))
roc <- function(y, score) {
  if (length(unique(y)) < 2L) return(NA_real_)
  if (length(unique(as.numeric(score))) < 2L) return(0.5)
  suppressWarnings(as.numeric(roc_auc_vec(truth_factor(y), score, event_level = "first")))
}
pr <- function(y, score) {
  if (length(unique(y)) < 2L) return(NA_real_)
  if (length(unique(as.numeric(score))) < 2L) return(mean(y))
  suppressWarnings(as.numeric(pr_auc_vec(truth_factor(y), score, event_level = "first")))
}
within_auc <- function(y, score) {
  n_pos <- sum(y == 1L); n_neg <- sum(y == 0L)
  if (!n_pos || !n_neg) return(NA_real_)
  ranks <- rank(as.numeric(score), ties.method = "average")
  (sum(ranks[y == 1L]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

checks <- list()
add_check <- function(id, condition, observed, expected, severity = "FAIL") {
  checks[[length(checks) + 1L]] <<- tibble(
    check_id = id, status = if (isTRUE(condition)) "PASS" else severity,
    observed = as.character(observed), expected = as.character(expected)
  )
}

add_check("test_rows", nrow(pred) == 1334151L, nrow(pred), 1334151L)
add_check("test_positives", sum(pred$truth) == 5350L, sum(pred$truth), 5350L)
add_check("test_nights", uniqueN(pred$date) == 543L, uniqueN(pred$date), 543L)
add_check("test_municipalities", uniqueN(pred$GID_2) == 2457L, uniqueN(pred$GID_2), 2457L)
add_check("unique_keys", anyDuplicated(pred[, .(GID_2, date)]) == 0L,
          anyDuplicated(pred[, .(GID_2, date)]), 0L)
add_check("phase2_key_truth_alignment", nrow(pred) == nrow(phase2) &&
            all(pred$GID_2 == phase2$GID_2) && all(as.Date(pred$date) == as.Date(phase2$date)) &&
            all(pred$truth == phase2$truth),
          "Phase 3 versus definitive Phase 2", "Exact")
add_check("phase2_headline_score_alignment",
          max(abs(pred$full_83_xgb - phase2$full_83_xgb_calibrated)) < 1e-15,
          max(abs(pred$full_83_xgb - phase2$full_83_xgb_calibrated)), "<1e-15")
add_check("prediction_columns", all(c(model_ids, centered_cols) %in% names(pred)),
          paste(setdiff(c(model_ids, centered_cols), names(pred)), collapse = "|"), "none missing")
add_check("prediction_completeness", !anyNA(pred[, c(model_ids, centered_cols), with = FALSE]),
          sum(is.na(pred[, c(model_ids, centered_cols), with = FALSE])), 0L)
add_check("dynamic_dimensions", nrow(dynamic) == 42L &&
            setequal(dynamic$model_id, model_ids) && setequal(dynamic$evaluation_scope, scope_ids),
          paste(nrow(dynamic), n_distinct(dynamic$model_id), n_distinct(dynamic$evaluation_scope), sep = "/"),
          "42/6/7")
add_check("scope_support_rows", nrow(support) == 7L && setequal(support$evaluation_scope, scope_ids),
          nrow(support), 7L)
add_check("registry_rows", nrow(registry) == 2457L && n_distinct(registry$GID_2) == 2457L,
          paste(nrow(registry), n_distinct(registry$GID_2), sep = "/"), "2457/2457")
add_check("registry_group_counts",
          sum(registry$has_both_test_classes) == 802L &&
            sum(registry$had_training_outage) == 1045L &&
            sum(registry$ever_observed_outage) == 1129L,
          paste(sum(registry$has_both_test_classes), sum(registry$had_training_outage),
                sum(registry$ever_observed_outage), sep = "/"), "802/1045/1129")
add_check("first_outage_partition",
          sum(pred$truth[pred$first_outage_state == "pre_or_first_observed_outage"]) == 65L &&
            sum(pred$truth[pred$first_outage_state == "post_first_observed_outage"]) == 5285L,
          paste(sum(pred$truth[pred$first_outage_state == "pre_or_first_observed_outage"]),
                sum(pred$truth[pred$first_outage_state == "post_first_observed_outage"]), sep = "/"),
          "65/5285")
add_check("variant_feature_counts",
          manifest$n_specified_features[manifest$model_id == "full_no_days_since_xgb"] == 82L &&
            manifest$n_specified_features[manifest$model_id == "full_explicit_missing_xgb"] == 147L,
          paste(manifest$n_specified_features, collapse = "/"), "82/147")
add_check("explicit_indicator_count", sum(preprocessing$missing_indicator_added) == 64L,
          sum(preprocessing$missing_indicator_added), 64L)
temporally_new <- preprocessing %>% filter(test_missing_without_training_missing)
add_check("temporally_new_missingness",
          nrow(temporally_new) == 1L &&
            temporally_new$feature[[1]] == "clim_n_obs" &&
            temporally_new$training_missing_n[[1]] == 0L &&
            temporally_new$test_missing_n[[1]] == 2457L &&
            !temporally_new$missing_indicator_added[[1]],
          if (nrow(temporally_new)) {
            paste(temporally_new$feature, temporally_new$training_missing_n,
                  temporally_new$test_missing_n, temporally_new$missing_indicator_added,
                  sep = "/", collapse = "|")
          } else {
            "none"
          },
          "clim_n_obs/0/2457/FALSE")

center_means <- pred[, lapply(.SD, mean), by = GID_2, .SDcols = centered_cols]
max_center_mean <- max(abs(as.matrix(center_means[, ..centered_cols])))
add_check("centered_group_means", max_center_mean < 1e-12, max_center_mean, "<1e-12")

recomputed <- bind_rows(lapply(model_ids, function(model_id) {
  d <- pred[, .(GID_2, truth, score = get(model_id), centered = get(paste0(model_id, "__centered")))]
  within <- d[, {
    n_pos <- sum(truth == 1L); n_neg <- sum(truth == 0L)
    .(pairs = as.double(n_pos) * as.double(n_neg), auc = within_auc(truth, score))
  }, by = GID_2][!is.na(auc)]
  tibble(
    model_id = model_id,
    pooled_roc_recomputed = roc(d$truth, d$score),
    pooled_pr_recomputed = pr(d$truth, d$score),
    centered_roc_recomputed = roc(d$truth, d$centered),
    within_macro_recomputed = mean(within$auc),
    within_pair_recomputed = weighted.mean(within$auc, within$pairs),
    eligible_recomputed = nrow(within)
  )
}))
saved_all <- dynamic %>% filter(evaluation_scope == "all_test") %>%
  select(model_id, pooled_roc_auc, pooled_pr_auc, centered_pooled_roc_auc,
         within_macro_roc_auc, within_pair_weighted_roc_auc, n_within_auc_municipalities)
metric_rechecks <- recomputed %>% left_join(saved_all, by = "model_id") %>% mutate(
  pooled_roc_diff = abs(pooled_roc_recomputed - pooled_roc_auc),
  pooled_pr_diff = abs(pooled_pr_recomputed - pooled_pr_auc),
  centered_roc_diff = abs(centered_roc_recomputed - centered_pooled_roc_auc),
  within_macro_diff = abs(within_macro_recomputed - within_macro_roc_auc),
  within_pair_diff = abs(within_pair_recomputed - within_pair_weighted_roc_auc),
  eligible_diff = abs(eligible_recomputed - n_within_auc_municipalities)
)
write_csv(metric_rechecks, file.path(run_dir, "independent_metric_rechecks.csv"))
max_metric_diff <- max(c(
  metric_rechecks$pooled_roc_diff, metric_rechecks$pooled_pr_diff,
  metric_rechecks$centered_roc_diff, metric_rechecks$within_macro_diff,
  metric_rechecks$within_pair_diff
))
add_check("all_test_metric_recomputation", max_metric_diff < 1e-12,
          max_metric_diff, "<1e-12")
add_check("within_eligible_recomputation", all(metric_rechecks$eligible_diff == 0),
          max(metric_rechecks$eligible_diff), 0L)

clim <- saved_all %>% filter(model_id == "municipality_climatology")
add_check("climatology_dynamic_null",
          abs(clim$centered_pooled_roc_auc - 0.5) < 1e-12 &&
            abs(clim$within_macro_roc_auc - 0.5) < 1e-12 &&
            abs(clim$within_pair_weighted_roc_auc - 0.5) < 1e-12,
          paste(clim$centered_pooled_roc_auc, clim$within_macro_roc_auc,
                clim$within_pair_weighted_roc_auc, sep = "/"), "0.5/0.5/0.5")
add_check("within_saved_all_test_rows",
          nrow(within_saved %>% filter(evaluation_scope == "all_test")) == 6L * 2457L,
          nrow(within_saved %>% filter(evaluation_scope == "all_test")), 14742L)

png_path <- file.path(figure_dir, "dynamic_skill_figure.png")
pdf_path <- file.path(figure_dir, "dynamic_skill_figure.pdf")
add_check("figure_outputs", file.exists(png_path) && file.info(png_path)$size > 10000L &&
            file.exists(pdf_path) && file.info(pdf_path)$size > 5000L,
          paste(ifelse(file.exists(png_path), file.info(png_path)$size, 0),
                ifelse(file.exists(pdf_path), file.info(pdf_path)$size, 0), sep = "/"),
          "PNG >10000; vector PDF >5000")

qa <- bind_rows(checks)
write_csv(qa, file.path(run_dir, "independent_qa_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else
  if (any(qa$status == "WARN")) "PASS_WITH_WARNINGS" else "PASS"
writeLines(c(
  "# Independent Phase 3 Validation", "", paste0("- Run: `", run_id, "`"),
  paste0("- Status: **", status, "**"), paste0("- PASS: ", sum(qa$status == "PASS")),
  paste0("- WARN: ", sum(qa$status == "WARN")), paste0("- FAIL: ", sum(qa$status == "FAIL")),
  "", "All-test pooled, centered, macro-within, and pair-weighted within-municipality metrics were independently recomputed from saved predictions."
), file.path(run_dir, "INDEPENDENT_VALIDATION_REPORT.md"))
cat("Independent Phase 3 validation status:", status, "\n")
cat("Report:", file.path(run_dir, "INDEPENDENT_VALIDATION_REPORT.md"), "\n")
if (status == "FAIL") quit(status = 1L)
