# Combine the nine Phase 5 duration variants and add paired ISO-week uncertainty.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(patchwork)
  library(Rcpp)
  library(readr)
  library(tibble)
  library(tidyr)
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

interval <- function(x) {
  tibble(
    estimate = mean(x, na.rm = TRUE),
    lower = unname(quantile(x, 0.025, na.rm = TRUE)),
    upper = unname(quantile(x, 0.975, na.rm = TRUE))
  )
}

project_dir <- detect_project_dir()
analysis_run_id <- parse_arg("--analysis-run-id", "20260814_063000_phase5_duration_sensitivity")
label_run_id <- parse_arg("--label-run-id", "20260814_040000_phase5_duration_labels")
analysis_dir <- file.path(project_dir, "data", "revision", analysis_run_id)
figure_dir <- file.path(project_dir, "figures", "revision", analysis_run_id)
label_dir <- file.path(project_dir, "data", "revision", label_run_id)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
log_path <- file.path(analysis_dir, "summary.log")
log_message <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", msg, "\n",
      file = log_path, append = TRUE, sep = "")
}

manifest <- read_csv(file.path(label_dir, "duration_variant_manifest.csv"), show_col_types = FALSE)
label_summary <- read_csv(file.path(label_dir, "duration_label_summary.csv"), show_col_types = FALSE)
variant_ids <- manifest$variant_id
model_ids <- c("weather_calendar_static_xgb", "history_free_xgb", "full_83_xgb")

required_variant_files <- c(
  "qa_checks.csv", "model_metrics.csv", "calibration_parameters.csv",
  "operating_metrics.csv", "topk_metrics.csv", "model_manifest.csv",
  "feature_importance.csv", "feature_family_importance.csv",
  "skill_contributions.csv", "predictions_test.rds", "predictions_validation.rds"
)
for (variant_id in variant_ids) {
  vdir <- file.path(analysis_dir, "variants", variant_id)
  missing <- required_variant_files[!file.exists(file.path(vdir, required_variant_files))]
  if (length(missing)) stop("Incomplete variant ", variant_id, ": ", paste(missing, collapse = ", "))
  qa <- read_csv(file.path(vdir, "qa_checks.csv"), show_col_types = FALSE)
  if (any(qa$status == "FAIL")) stop("Variant failed QA: ", variant_id)
}

bind_variant_csv <- function(filename) bind_rows(lapply(variant_ids, function(id) {
  read_csv(file.path(analysis_dir, "variants", id, filename), show_col_types = FALSE)
}))
all_metrics <- bind_variant_csv("model_metrics.csv")
all_calibration <- bind_variant_csv("calibration_parameters.csv")
all_operating <- bind_variant_csv("operating_metrics.csv")
all_topk <- bind_variant_csv("topk_metrics.csv")
all_model_manifest <- bind_variant_csv("model_manifest.csv")
all_importance <- bind_variant_csv("feature_importance.csv")
all_family_importance <- bind_variant_csv("feature_family_importance.csv")
all_contributions <- bind_variant_csv("skill_contributions.csv")

write_csv(all_metrics, file.path(analysis_dir, "all_model_metrics.csv"))
write_csv(all_calibration, file.path(analysis_dir, "all_calibration_parameters.csv"))
write_csv(all_operating, file.path(analysis_dir, "all_operating_metrics.csv"))
write_csv(all_topk, file.path(analysis_dir, "all_topk_metrics.csv"))
write_csv(all_model_manifest, file.path(analysis_dir, "all_model_manifests.csv"))
write_csv(all_importance, file.path(analysis_dir, "all_feature_importance.csv"))
write_csv(all_family_importance, file.path(analysis_dir, "all_feature_family_importance.csv"))
write_csv(all_contributions, file.path(analysis_dir, "all_skill_contributions.csv"))

block_counts_path <- file.path(
  project_dir, "data", "revision", "20260813_214000_phase4_classifier_uncertainty",
  "bootstrap_block_counts.rds"
)
block_counts <- readRDS(block_counts_path)
if (!identical(dim(block_counts), c(1000L, 79L)) || any(rowSums(block_counts) != 79L)) {
  stop("Definitive Phase 4 bootstrap draws drifted.")
}
n_boot <- nrow(block_counts)

Rcpp::cppFunction(code = '
Rcpp::NumericMatrix duration_weighted_rank_metrics_cpp(
    Rcpp::IntegerVector truth,
    Rcpp::NumericVector score,
    Rcpp::IntegerVector block_id,
    Rcpp::IntegerMatrix block_counts) {
  const int n = truth.size();
  const int B = block_counts.nrow();
  Rcpp::NumericMatrix out(B, 2);
  for (int b = 0; b < B; ++b) {
    double total_pos = 0.0, total_neg = 0.0;
    for (int i = 0; i < n; ++i) {
      const double w = block_counts(b, block_id[i] - 1);
      if (truth[i] == 1) total_pos += w; else total_neg += w;
    }
    if (total_pos <= 0.0 || total_neg <= 0.0) {
      out(b, 0) = NA_REAL; out(b, 1) = NA_REAL; continue;
    }
    double cum_tp = 0.0, cum_fp = 0.0;
    double prev_recall = 0.0, prev_precision = 1.0;
    double pr_area = 0.0, roc_num = 0.0;
    int i = 0;
    while (i < n) {
      const double s = score[i];
      double group_pos = 0.0, group_neg = 0.0;
      int j = i;
      while (j < n && score[j] == s) {
        const double w = block_counts(b, block_id[j] - 1);
        if (truth[j] == 1) group_pos += w; else group_neg += w;
        ++j;
      }
      if (group_pos + group_neg > 0.0) {
        roc_num += group_pos * (total_neg - cum_fp - 0.5 * group_neg);
        cum_tp += group_pos; cum_fp += group_neg;
        const double recall = cum_tp / total_pos;
        const double precision = cum_tp / (cum_tp + cum_fp);
        pr_area += (recall - prev_recall) * (prev_precision + precision) * 0.5;
        prev_recall = recall; prev_precision = precision;
      }
      i = j;
    }
    out(b, 0) = roc_num / (total_pos * total_neg);
    out(b, 1) = pr_area;
  }
  Rcpp::colnames(out) = Rcpp::CharacterVector::create("roc_auc", "pr_auc");
  return out;
}')

draw_rows <- list()
for (variant_id in variant_ids) {
  log_message("Bootstrapping variant: ", variant_id)
  pred <- as.data.table(readRDS(file.path(
    analysis_dir, "variants", variant_id, "predictions_test.rds"
  )))
  pred[, `:=`(date = as.Date(date), iso_week = format(as.Date(date), "%G-W%V"))]
  pred[, block_id := match(iso_week, colnames(block_counts))]
  if (anyNA(pred$block_id) || nrow(pred) != 1334151L || uniqueN(pred$date) != 543L) {
    stop("Bootstrap support drifted for ", variant_id)
  }
  weekly_base <- pred[, .(
    n_rows = .N, n_nights = uniqueN(date), n_positive = sum(truth)
  ), by = block_id][order(block_id)]
  if (!identical(weekly_base$block_id, seq_len(79L))) stop("Weekly ordering drifted.")
  boot_rows <- as.numeric(block_counts %*% weekly_base$n_rows)
  boot_nights <- as.numeric(block_counts %*% weekly_base$n_nights)
  boot_positives <- as.numeric(block_counts %*% weekly_base$n_positive)
  boot_prevalence <- boot_positives / boot_rows
  train_prevalence <- read_csv(file.path(
    analysis_dir, "variants", variant_id, "split_summary.csv"
  ), show_col_types = FALSE) %>% filter(split == "train") %>% pull(prevalence)
  constant_weekly <- pred[, .(
    squared_error = sum((train_prevalence - truth)^2)
  ), by = block_id][order(block_id)]
  boot_constant_brier <- as.numeric(block_counts %*% constant_weekly$squared_error) / boot_rows

  for (model_id in model_ids) {
    log_message("  rank uncertainty: ", model_id)
    score_col <- paste0(model_id, "_calibrated")
    score <- pred[[score_col]]
    ord <- order(score, decreasing = TRUE)
    rank_draws <- duration_weighted_rank_metrics_cpp(
      pred$truth[ord], score[ord], pred$block_id[ord], block_counts
    )
    weekly_model <- pred[, .(
      squared_error = sum((get(score_col) - truth)^2)
    ), by = block_id][order(block_id)]
    boot_brier <- as.numeric(block_counts %*% weekly_model$squared_error) / boot_rows
    threshold <- all_metrics %>%
      filter(variant_id == .env$variant_id, model_id == .env$model_id) %>%
      pull(operating_threshold)
    weekly_operating <- pred[, {
      selected <- get(score_col) >= threshold
      .(tp = sum(truth == 1L & selected), fp = sum(truth == 0L & selected))
    }, by = block_id][order(block_id)]
    boot_tp <- as.numeric(block_counts %*% weekly_operating$tp)
    boot_fp <- as.numeric(block_counts %*% weekly_operating$fp)
    draw_rows[[paste(variant_id, model_id)]] <- tibble(
      bootstrap_id = seq_len(n_boot), variant_id = variant_id, model_id = model_id,
      roc_auc = rank_draws[, "roc_auc"], pr_auc = rank_draws[, "pr_auc"],
      prevalence = boot_prevalence,
      pr_lift = rank_draws[, "pr_auc"] / boot_prevalence,
      brier = boot_brier, brier_skill_score = 1 - boot_brier / boot_constant_brier,
      precision = boot_tp / (boot_tp + boot_fp), recall = boot_tp / boot_positives,
      alerts_per_night = (boot_tp + boot_fp) / boot_nights
    )
  }
  rm(pred); gc(verbose = FALSE)
}
classifier_draws <- bind_rows(draw_rows)
write_csv(classifier_draws, file.path(analysis_dir, "duration_classifier_bootstrap_draws.csv"))

uncertainty_metrics <- c(
  "roc_auc", "pr_auc", "pr_lift", "brier", "brier_skill_score",
  "precision", "recall", "alerts_per_night"
)
classifier_uncertainty <- classifier_draws %>%
  pivot_longer(all_of(uncertainty_metrics), names_to = "metric", values_to = "value") %>%
  group_by(variant_id, model_id, metric) %>%
  summarise(
    bootstrap_mean = mean(value, na.rm = TRUE),
    lower = unname(quantile(value, 0.025, na.rm = TRUE)),
    upper = unname(quantile(value, 0.975, na.rm = TRUE)), .groups = "drop"
  )
write_csv(classifier_uncertainty, file.path(analysis_dir, "duration_classifier_uncertainty.csv"))

contribution_specs <- tribble(
  ~comparison_id, ~expanded_model, ~reference_model,
  "incremental_ntl", "history_free_xgb", "weather_calendar_static_xgb",
  "incremental_history", "full_83_xgb", "history_free_xgb"
)
contribution_metrics <- c("roc_auc", "pr_auc", "pr_lift", "brier", "brier_skill_score")
contribution_draws <- bind_rows(lapply(variant_ids, function(variant_id) {
  bind_rows(lapply(seq_len(nrow(contribution_specs)), function(i) {
    spec <- contribution_specs[i, ]
    expanded <- classifier_draws %>%
      filter(variant_id == .env$variant_id, model_id == spec$expanded_model) %>%
      arrange(bootstrap_id)
    reference <- classifier_draws %>%
      filter(variant_id == .env$variant_id, model_id == spec$reference_model) %>%
      arrange(bootstrap_id)
    bind_rows(lapply(contribution_metrics, function(metric) {
      raw_delta <- expanded[[metric]] - reference[[metric]]
      higher_better <- metric != "brier"
      tibble(
        bootstrap_id = expanded$bootstrap_id, variant_id = variant_id,
        comparison_id = spec$comparison_id, metric = metric,
        raw_delta = raw_delta,
        improvement_oriented_delta = ifelse(higher_better, raw_delta, -raw_delta)
      )
    }))
  }))
}))
write_csv(contribution_draws, file.path(analysis_dir, "duration_contribution_bootstrap_draws.csv"))
contribution_uncertainty <- contribution_draws %>%
  group_by(variant_id, comparison_id, metric) %>%
  summarise(
    estimate = mean(raw_delta, na.rm = TRUE),
    lower = unname(quantile(raw_delta, 0.025, na.rm = TRUE)),
    upper = unname(quantile(raw_delta, 0.975, na.rm = TRUE)),
    support_direction = if_else(lower > 0 | upper < 0, "supported", "not_supported"),
    .groups = "drop"
  )
write_csv(contribution_uncertainty, file.path(analysis_dir, "duration_contribution_uncertainty.csv"))

full_points <- all_metrics %>% filter(model_id == "full_83_xgb") %>%
  select(variant_id, n_positive, prevalence, roc_auc, pr_auc, pr_lift, brier,
         brier_skill_score, ece_equal_10, operating_threshold, test_precision,
         test_recall, alerts_per_night, top10_precision, top10_recall)
wide_ci <- classifier_uncertainty %>% filter(model_id == "full_83_xgb") %>%
  select(variant_id, metric, lower, upper) %>%
  pivot_wider(names_from = metric, values_from = c(lower, upper), names_glue = "{metric}_{.value}")
wide_contrib <- contribution_uncertainty %>%
  filter(metric %in% c("roc_auc", "pr_auc", "brier_skill_score")) %>%
  select(variant_id, comparison_id, metric, estimate, lower, upper, support_direction) %>%
  pivot_wider(
    names_from = c(comparison_id, metric),
    values_from = c(estimate, lower, upper, support_direction),
    names_glue = "{comparison_id}_{metric}_{.value}"
  )
gain_shares <- all_family_importance %>% filter(model_id == "full_83_xgb") %>%
  select(variant_id, feature_family, gain_share) %>%
  pivot_wider(names_from = feature_family, values_from = gain_share, names_prefix = "gain_share_")

publication_table <- manifest %>%
  left_join(label_summary %>% select(variant_id, model_key_municipality_nights), by = "variant_id") %>%
  left_join(full_points, by = "variant_id") %>%
  left_join(wide_ci, by = "variant_id") %>%
  left_join(wide_contrib, by = "variant_id") %>%
  left_join(gain_shares, by = "variant_id") %>%
  mutate(
    family_label = recode(
      variant_family,
      night_duration_threshold = "Night-overlap duration",
      active_at_overpass_minimum_duration = "Active at 01:30"
    )
  )
write_csv(publication_table, file.path(analysis_dir, "duration_sensitivity_publication_table.csv"))

latex_rows <- publication_table %>% transmute(
  Definition = if_else(variant_family == "night_duration_threshold",
                       paste0("Night $>$", duration_minutes, " min"),
                       paste0("Active 01:30, $\\geq$", duration_minutes, " min")),
  Positives = model_key_municipality_nights,
  `ROC-AUC` = sprintf("%.3f [%.3f, %.3f]", roc_auc, roc_auc_lower, roc_auc_upper),
  `PR-AUC` = sprintf("%.3f [%.3f, %.3f]", pr_auc, pr_auc_lower, pr_auc_upper),
  `PR lift` = sprintf("%.1f", pr_lift),
  `P@10` = sprintf("%.3f", top10_precision),
  `Delta ROC NTL` = sprintf("%+.3f", incremental_ntl_roc_auc_estimate),
  `Delta ROC history` = sprintf("%+.3f", incremental_history_roc_auc_estimate)
)
latex_lines <- c(
  "\\begin{tabular}{lrrrrrrr}", "\\toprule",
  "Definition & Positives & ROC-AUC & PR-AUC & PR lift & P@10 & $\\Delta$ ROC NTL & $\\Delta$ ROC history \\\\",
  "\\midrule"
)
for (i in seq_len(nrow(latex_rows))) {
  latex_lines <- c(latex_lines, paste(unlist(latex_rows[i, ], use.names = FALSE), collapse = " & "), "\\\\")
}
writeLines(c(latex_lines, "\\bottomrule", "\\end{tabular}"),
           file.path(analysis_dir, "duration_sensitivity_publication_table.tex"))

plot_data <- publication_table %>% mutate(
  family_label = factor(family_label, levels = c("Night-overlap duration", "Active at 01:30"))
)
family_colors <- c("Night-overlap duration" = "#007C91", "Active at 01:30" = "#C65D32")
p_roc <- ggplot(plot_data, aes(duration_minutes, roc_auc, color = family_label)) +
  geom_hline(yintercept = 0.5, color = "grey75", linewidth = 0.4) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.2) +
  geom_errorbar(aes(ymin = roc_auc_lower, ymax = roc_auc_upper), width = 0, linewidth = 0.55) +
  scale_x_log10(breaks = c(15, 30, 60, 120, 180, 360, 720)) +
  scale_color_manual(values = family_colors) +
  labs(x = "Duration threshold (minutes, log scale)", y = "Full-model ROC-AUC", color = NULL,
       title = "A. Ranking skill")
p_lift <- ggplot(plot_data, aes(duration_minutes, pr_lift, color = family_label)) +
  geom_hline(yintercept = 1, color = "grey75", linewidth = 0.4) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.2) +
  geom_errorbar(aes(ymin = pr_lift_lower, ymax = pr_lift_upper), width = 0, linewidth = 0.55) +
  scale_x_log10(breaks = c(15, 30, 60, 120, 180, 360, 720)) +
  scale_color_manual(values = family_colors) +
  labs(x = "Duration threshold (minutes, log scale)", y = "Full-model PR lift", color = NULL,
       title = "B. Precision-recall lift")
contrib_plot <- contribution_uncertainty %>% filter(metric == "roc_auc") %>%
  left_join(manifest %>% select(variant_id, variant_family, duration_minutes), by = "variant_id") %>%
  mutate(
    contribution = recode(comparison_id,
                          incremental_ntl = "NTL contribution",
                          incremental_history = "History contribution"),
    family_label = recode(variant_family,
                          night_duration_threshold = "Night-overlap duration",
                          active_at_overpass_minimum_duration = "Active at 01:30")
  )
contribution_colors <- c("NTL contribution" = "#007C91", "History contribution" = "#C65D32")
p_contrib <- ggplot(contrib_plot, aes(duration_minutes, estimate, color = contribution)) +
  geom_hline(yintercept = 0, color = "grey55", linewidth = 0.45) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.1) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0, linewidth = 0.5) +
  facet_wrap(~family_label, nrow = 1) +
  scale_x_log10(breaks = c(15, 30, 60, 120, 180, 360, 720)) +
  scale_color_manual(values = contribution_colors) +
  labs(x = "Duration threshold (minutes, log scale)", y = "Incremental ROC-AUC", color = NULL,
       title = "C. Incremental NTL and outage-history skill")
common_theme <- theme_minimal(base_size = 10) + theme(
  panel.grid.minor = element_blank(), legend.position = "bottom",
  plot.title = element_text(face = "bold", size = 10),
  axis.text.x = element_text(size = 8)
)
figure <- (p_roc + common_theme | p_lift + common_theme) /
  (p_contrib + common_theme) +
  plot_annotation(
    title = "Duration and overpass-timing sensitivity",
    subtitle = "Points use the fixed chronological protocol; bars are 95% paired ISO-week block intervals"
  )
ggsave(file.path(figure_dir, "duration_sensitivity.pdf"), figure, width = 10.5, height = 7.2, device = "pdf")
ggsave(file.path(figure_dir, "duration_sensitivity.png"), figure, width = 10.5, height = 7.2, dpi = 300)

# Exact duplicated-label reproducibility check for the independently fitted 15/30-minute variants.
pred15 <- readRDS(file.path(analysis_dir, "variants", "active0130_min015", "predictions_test.rds"))
pred30 <- readRDS(file.path(analysis_dir, "variants", "active0130_min030", "predictions_test.rds"))
identical_15_30_predictions <- identical(pred15, pred30)
rm(pred15, pred30); gc(verbose = FALSE)

checks <- list()
add_check <- function(id, condition, observed, expected) {
  checks[[length(checks) + 1L]] <<- tibble(
    check_id = id, status = if (isTRUE(condition)) "PASS" else "FAIL",
    observed = as.character(observed), expected = as.character(expected)
  )
}
add_check("variant_completeness", n_distinct(all_metrics$variant_id) == 9L,
          n_distinct(all_metrics$variant_id), "9")
add_check("model_completeness", nrow(all_metrics) == 27L,
          nrow(all_metrics), "27")
add_check("shared_test_support", all(all_metrics$n_test == 1334151L),
          paste(unique(all_metrics$n_test), collapse = "/"), "1334151")
add_check("bootstrap_draw_support", nrow(classifier_draws) == 27000L,
          nrow(classifier_draws), "27000")
add_check("contribution_draw_support", nrow(contribution_draws) == 90000L,
          nrow(contribution_draws), "90000")
rank_calibration_uncertainty <- classifier_uncertainty %>%
  filter(metric %in% c("roc_auc", "pr_auc", "pr_lift", "brier", "brier_skill_score"))
add_check("finite_rank_calibration_uncertainty",
          all(is.finite(rank_calibration_uncertainty$lower)) &&
            all(is.finite(rank_calibration_uncertainty$upper)),
          sum(!is.finite(rank_calibration_uncertainty$lower)) +
            sum(!is.finite(rank_calibration_uncertainty$upper)), "0")
missing_operating_models <- all_metrics %>%
  filter(!is.finite(operating_threshold)) %>% select(variant_id, model_id)
missing_operating_uncertainty <- classifier_uncertainty %>%
  filter(metric %in% c("precision", "recall", "alerts_per_night"),
         !is.finite(lower) | !is.finite(upper)) %>%
  distinct(variant_id, model_id)
add_check("operating_na_matches_unmet_precision_floor", identical(
  missing_operating_models %>% arrange(variant_id, model_id),
  missing_operating_uncertainty %>% arrange(variant_id, model_id)
), nrow(missing_operating_uncertainty), "4 models with no eligible validation threshold")
add_check("importance_normalization", all(abs(all_family_importance %>%
  group_by(variant_id, model_id) %>% summarise(x = sum(gain_share), .groups = "drop") %>%
  pull(x) - 1) < 1e-9), "checked 27 models", "all 1")
add_check("active_15_30_metric_identity", identical(
  all_metrics %>% filter(variant_id == "active0130_min015") %>% select(-variant_id),
  all_metrics %>% filter(variant_id == "active0130_min030") %>% select(-variant_id)
), "independent fits", "exactly identical")
add_check("active_15_30_prediction_identity", identical_15_30_predictions,
          identical_15_30_predictions, "TRUE")
add_check("headline_control_proximity", abs(
  publication_table$roc_auc[publication_table$variant_id == "night_gt180"] - 0.918657
) < 0.001, publication_table$roc_auc[publication_table$variant_id == "night_gt180"],
"within 0.001 of frozen headline")
qa <- bind_rows(checks)
write_csv(qa, file.path(analysis_dir, "summary_qa_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else "PASS"

write_json(list(
  analysis_run_id = analysis_run_id, label_run_id = label_run_id,
  variants = variant_ids, models = model_ids,
  bootstrap = list(n = 1000L, unit = "ISO week", blocks = 79L,
                   draw_source = block_counts_path, interval = "percentile 95%"),
  figure = file.path(figure_dir, "duration_sensitivity.pdf")
), file.path(analysis_dir, "summary_config.json"), pretty = TRUE, auto_unbox = TRUE)
writeLines(c(
  "# Phase 5 Duration Sensitivity", "", paste0("- Status: **", status, "**"),
  paste0("- Variants: ", n_distinct(all_metrics$variant_id)),
  paste0("- Models: ", nrow(all_metrics)),
  "- Uncertainty: 1,000 paired resamples of 79 ISO-week blocks",
  paste0("- Summary QA PASS: ", sum(qa$status == "PASS")),
  paste0("- Summary QA FAIL: ", sum(qa$status == "FAIL"))
), file.path(analysis_dir, "SUMMARY_QA_REPORT.md"))
log_message("Duration sensitivity summary status: ", status)
if (status == "FAIL") stop("Duration sensitivity summary QA failed.")
