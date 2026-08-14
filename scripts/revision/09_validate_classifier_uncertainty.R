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
  stop("Usage: Rscript scripts/revision/09_validate_classifier_uncertainty.R <run-id>")
}
project_dir <- detect_project_dir()
run_id <- args[[1]]
run_dir <- file.path(project_dir, "data", "revision", run_id)
figure_dir <- file.path(project_dir, "figures", "revision", run_id)
phase2_dir <- file.path(
  project_dir, "data", "revision", "20260813_144700_phase2_canonical_comparison"
)

required_files <- c(
  "iso_week_block_manifest.csv", "bootstrap_block_counts.rds",
  "classifier_point_estimates.csv", "classifier_bootstrap_draws.csv",
  "classifier_uncertainty_table.csv", "paired_difference_uncertainty.csv",
  "topk_point_estimates.csv", "topk_bootstrap_draws.csv", "topk_uncertainty_table.csv",
  "topk_difference_uncertainty.csv", "cost_loss_curve.csv", "optimal_cost_loss_policy.csv",
  "qa_checks.csv", "run_config.json", "QA_REPORT.md"
)
missing_files <- required_files[!file.exists(file.path(run_dir, required_files))]
if (length(missing_files)) stop("Missing Phase 4 artifacts: ", paste(missing_files, collapse = ", "))

pred <- as.data.table(readRDS(file.path(phase2_dir, "paired_predictions_test.rds")))
pred[, date := as.Date(date)]
weeks <- sort(unique(format(pred$date, "%G-W%V")))
pred[, block_id := match(format(date, "%G-W%V"), weeks)]
counts <- readRDS(file.path(run_dir, "bootstrap_block_counts.rds"))
points <- read_csv(file.path(run_dir, "classifier_point_estimates.csv"), show_col_types = FALSE)
draws <- read_csv(file.path(run_dir, "classifier_bootstrap_draws.csv"), show_col_types = FALSE)
topk_points <- read_csv(file.path(run_dir, "topk_point_estimates.csv"), show_col_types = FALSE)
topk_draws <- read_csv(file.path(run_dir, "topk_bootstrap_draws.csv"), show_col_types = FALSE)
cost_curve <- read_csv(file.path(run_dir, "cost_loss_curve.csv"), show_col_types = FALSE)
optimal <- read_csv(file.path(run_dir, "optimal_cost_loss_policy.csv"), show_col_types = FALSE)

model_ids <- c(
  "constant_prevalence", "municipality_climatology", "history_only_xgb",
  "weather_calendar_static_xgb", "history_free_xgb", "full_83_xgb", "ridge_logistic_83"
)
score_columns <- setNames(paste0(model_ids, "_calibrated"), model_ids)
selected_replicates <- unique(c(1L, as.integer(nrow(counts) / 2L), nrow(counts)))

checks <- list()
add_check <- function(id, condition, observed, expected) {
  checks[[length(checks) + 1L]] <<- tibble(
    check_id = id, status = if (isTRUE(condition)) "PASS" else "FAIL",
    observed = as.character(observed), expected = as.character(expected)
  )
}

add_check("test_support", nrow(pred) == 1334151L && sum(pred$truth) == 5350L &&
            uniqueN(pred$date) == 543L && uniqueN(pred$GID_2) == 2457L,
          paste(nrow(pred), sum(pred$truth), uniqueN(pred$date), uniqueN(pred$GID_2), sep = "/"),
          "1334151/5350/543/2457")
add_check("bootstrap_dimensions", nrow(counts) >= 1000L && ncol(counts) == 79L,
          paste(dim(counts), collapse = "/"), ">=1000/79")
add_check("bootstrap_block_totals", all(rowSums(counts) == 79L),
          paste(range(rowSums(counts)), collapse = "/"), "79/79")
add_check("draw_dimensions", nrow(draws) == length(model_ids) * nrow(counts),
          nrow(draws), length(model_ids) * nrow(counts))

truth_factor <- factor(ifelse(pred$truth == 1L, "1", "0"), levels = c("1", "0"))
point_rechecks <- bind_rows(lapply(model_ids, function(model_id) {
  score <- pred[[score_columns[[model_id]]]]
  if (model_id == "constant_prevalence") {
    roc <- 0.5; pr <- mean(pred$truth)
  } else {
    roc <- as.numeric(roc_auc_vec(truth_factor, score, event_level = "first"))
    pr <- as.numeric(pr_auc_vec(truth_factor, score, event_level = "first"))
  }
  tibble(model_id = model_id, roc_recomputed = roc, pr_recomputed = pr,
         brier_recomputed = mean((score - pred$truth)^2))
})) %>%
  left_join(points %>% select(model_id, roc_auc, pr_auc, brier), by = "model_id") %>%
  mutate(
    roc_diff = abs(roc_recomputed - roc_auc), pr_diff = abs(pr_recomputed - pr_auc),
    brier_diff = abs(brier_recomputed - brier)
  )
write_csv(point_rechecks, file.path(run_dir, "independent_point_rechecks.csv"))
add_check("point_metric_recomputation",
          max(point_rechecks$roc_diff) < 1e-10 && max(point_rechecks$pr_diff) < 1e-12 &&
            max(point_rechecks$brier_diff) < 1e-12,
          paste(signif(max(point_rechecks$roc_diff), 3), signif(max(point_rechecks$pr_diff), 3),
                signif(max(point_rechecks$brier_diff), 3), sep = "/"),
          "ROC <1e-10; PR/Brier <1e-12")

bootstrap_rechecks <- bind_rows(lapply(selected_replicates, function(b) {
  weights <- counts[b, pred$block_id]
  yf <- truth_factor
  bind_rows(lapply(model_ids, function(model_id) {
    score <- pred[[score_columns[[model_id]]]]
    if (model_id == "constant_prevalence") {
      roc <- 0.5; pr <- weighted.mean(pred$truth, weights)
    } else {
      roc <- as.numeric(roc_auc_vec(
        yf, score, case_weights = hardhat::importance_weights(weights), event_level = "first"
      ))
      pr <- as.numeric(pr_auc_vec(
        yf, score, case_weights = hardhat::importance_weights(weights), event_level = "first"
      ))
    }
    saved <- draws %>% filter(bootstrap_id == b, model_id == !!model_id)
    tibble(
      bootstrap_id = b, model_id = model_id, roc_recomputed = roc, pr_recomputed = pr,
      roc_saved = saved$roc_auc, pr_saved = saved$pr_auc,
      roc_diff = abs(roc - saved$roc_auc), pr_diff = abs(pr - saved$pr_auc)
    )
  }))
}))
write_csv(bootstrap_rechecks, file.path(run_dir, "independent_bootstrap_rank_rechecks.csv"))
add_check("weighted_rank_recomputation",
          max(bootstrap_rechecks$roc_diff) < 1e-10 && max(bootstrap_rechecks$pr_diff) < 1e-12,
          paste(signif(max(bootstrap_rechecks$roc_diff), 3),
                signif(max(bootstrap_rechecks$pr_diff), 3), sep = "/"),
          "ROC <1e-10; PR <1e-12")

full_k10_point <- topk_points %>% filter(model_id == "full_83_xgb", k == 10L)
ranked <- pred[, .(date, GID_2, truth, score = full_83_xgb_calibrated)]
setorder(ranked, date, -score, GID_2)
ranked[, rank := seq_len(.N), by = date]
hits <- sum(ranked[rank <= 10L]$truth)
add_check("top10_point_recomputation",
          hits == full_k10_point$n_hits &&
            abs(hits / (10 * uniqueN(pred$date)) - full_k10_point$precision) < 1e-12,
          paste(hits, full_k10_point$n_hits, sep = "/"), "682/682")

b <- selected_replicates[[1]]
daily <- ranked[, .(n_positive_day = sum(truth), hits = sum(truth[rank <= 10L])), by = date]
daily[, block_id := match(format(date, "%G-W%V"), weeks)]
weekly <- daily[, .(hits = sum(hits), positives = sum(n_positive_day), nights = .N), by = block_id][order(block_id)]
boot_hits <- sum(counts[b, ] * weekly$hits)
boot_pos <- sum(counts[b, ] * weekly$positives)
boot_nights <- sum(counts[b, ] * weekly$nights)
saved_k10 <- topk_draws %>% filter(model_id == "full_83_xgb", k == 10L, bootstrap_id == b)
add_check("top10_bootstrap_recomputation",
          abs(boot_hits / (10 * boot_nights) - saved_k10$precision) < 1e-12 &&
            abs(boot_hits / boot_pos - saved_k10$recall) < 1e-12,
          paste(signif(boot_hits / (10 * boot_nights) - saved_k10$precision, 3),
                signif(boot_hits / boot_pos - saved_k10$recall, 3), sep = "/"), "0/0")

optimal_recheck <- cost_curve %>%
  group_by(model_id, cost_ratio) %>%
  summarise(recomputed = max(net_value_per_night), .groups = "drop") %>%
  inner_join(optimal %>% select(model_id, cost_ratio, optimal_net_value_per_night),
             by = c("model_id", "cost_ratio")) %>%
  mutate(diff = abs(recomputed - optimal_net_value_per_night))
add_check("cost_loss_optimization", max(optimal_recheck$diff) < 1e-12,
          max(optimal_recheck$diff), "<1e-12")

png_path <- file.path(figure_dir, "classifier_decision_value.png")
pdf_path <- file.path(figure_dir, "classifier_decision_value.pdf")
add_check("figure_outputs", file.exists(png_path) && file.info(png_path)$size > 10000L &&
            file.exists(pdf_path) && file.info(pdf_path)$size > 5000L,
          paste(ifelse(file.exists(png_path), file.info(png_path)$size, 0),
                ifelse(file.exists(pdf_path), file.info(pdf_path)$size, 0), sep = "/"),
          "PNG >10000; vector PDF >5000")

qa <- bind_rows(checks)
write_csv(qa, file.path(run_dir, "independent_qa_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else "PASS"
writeLines(c(
  "# Independent Phase 4 Classifier Validation", "",
  paste0("- Run: `", run_id, "`"), paste0("- Status: **", status, "**"),
  paste0("- PASS: ", sum(qa$status == "PASS")),
  paste0("- FAIL: ", sum(qa$status == "FAIL")), "",
  paste0(
    "Point estimates, selected exact weighted bootstrap rank metrics, Top-10 allocation, ",
    "and cost-loss optimization were independently recomputed from definitive Phase 2 predictions."
  )
), file.path(run_dir, "INDEPENDENT_VALIDATION_REPORT.md"))
cat("Independent Phase 4 classifier validation status:", status, "\n")
if (status == "FAIL") quit(status = 1L)
