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
  library(yardstick)
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
  "--run-id", paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_phase4_classifier_uncertainty")
)
n_boot <- as.integer(parse_arg("--n-bootstrap", "1000"))
bootstrap_seed <- as.integer(parse_arg("--seed", "20260813"))
if (!is.finite(n_boot) || n_boot < 1000L) stop("Phase 4 requires at least 1,000 bootstrap resamples.")

phase2_dir <- file.path(
  project_dir, "data", "revision", "20260813_144700_phase2_canonical_comparison"
)
predictions_path <- file.path(phase2_dir, "paired_predictions_test.rds")
canonical_path <- file.path(phase2_dir, "canonical_model_table.csv")
topk_phase2_path <- file.path(phase2_dir, "topk_metrics.csv")
phase2_qa_path <- file.path(phase2_dir, "independent_qa_checks.csv")
required_inputs <- c(predictions_path, canonical_path, topk_phase2_path, phase2_qa_path)
missing_inputs <- required_inputs[!file.exists(required_inputs)]
if (length(missing_inputs)) stop("Missing required input(s): ", paste(missing_inputs, collapse = ", "))

output_dir <- file.path(project_dir, "data", "revision", run_id)
figure_dir <- file.path(project_dir, "figures", "revision", run_id)
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to overwrite non-empty run directory: ", output_dir)
}
if (dir.exists(figure_dir) && length(list.files(figure_dir, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to overwrite non-empty figure directory: ", figure_dir)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
log_path <- file.path(output_dir, "run.log")
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
log_message <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", msg, "\n",
      file = log_path, append = TRUE, sep = "")
}

model_ids <- c(
  "constant_prevalence", "municipality_climatology", "history_only_xgb",
  "weather_calendar_static_xgb", "history_free_xgb", "full_83_xgb",
  "ridge_logistic_83"
)
ranked_model_ids <- setdiff(model_ids, "constant_prevalence")
decision_model_ids <- c(
  "municipality_climatology", "history_only_xgb", "history_free_xgb", "full_83_xgb"
)
model_labels <- c(
  constant_prevalence = "National prevalence",
  municipality_climatology = "Municipality climatology",
  history_only_xgb = "History-only XGBoost",
  weather_calendar_static_xgb = "Weather/calendar/static XGBoost",
  history_free_xgb = "History-free remote-sensing XGBoost",
  full_83_xgb = "Full 83-feature XGBoost",
  ridge_logistic_83 = "Ridge logistic, same 83 features",
  random_allocation = "Random allocation (expected)"
)
score_columns <- setNames(paste0(model_ids, "_calibrated"), model_ids)
topk_values <- c(5L, 10L, 20L, 50L)
decision_k_values <- 1:100
cost_ratios <- seq(0, 0.20, by = 0.005)

log_message("Phase 4 classifier uncertainty and decision value")
log_message("Run ID: ", run_id)
log_message("Loading definitive Phase 2 paired predictions...")
pred <- as.data.table(readRDS(predictions_path))
pred[, date := as.Date(date)]
canonical <- read_csv(canonical_path, show_col_types = FALSE)
phase2_topk <- read_csv(topk_phase2_path, show_col_types = FALSE)
phase2_qa <- read_csv(phase2_qa_path, show_col_types = FALSE)

required_columns <- c("GID_2", "date", "truth", unname(score_columns))
if (length(setdiff(required_columns, names(pred)))) {
  stop("Paired predictions lack required columns: ", paste(setdiff(required_columns, names(pred)), collapse = ", "))
}
if (anyNA(pred[, ..required_columns])) stop("Paired predictions contain missing required values.")
if (anyDuplicated(pred[, .(GID_2, date)])) stop("Paired predictions contain duplicate keys.")
if (!identical(sort(unique(pred$truth)), c(0L, 1L))) stop("Truth is not binary integer 0/1.")
if (nrow(pred) != 1334151L || sum(pred$truth) != 5350L ||
    uniqueN(pred$date) != 543L || uniqueN(pred$GID_2) != 2457L) {
  stop("Definitive Phase 2 test support has drifted.")
}
if (any(phase2_qa$status == "FAIL")) stop("Definitive Phase 2 has failed independent QA checks.")

iso_weeks <- sort(unique(format(pred$date, "%G-W%V")))
pred[, iso_week := format(date, "%G-W%V")]
pred[, block_id := match(iso_week, iso_weeks)]
n_blocks <- length(iso_weeks)
if (n_blocks != 79L) stop("Expected 79 ISO-week blocks; found ", n_blocks)

block_manifest <- pred[, .(
  start_date = min(date), end_date = max(date), n_nights = uniqueN(date),
  n_rows = .N, n_positive = sum(truth), prevalence = mean(truth)
), by = .(block_id, iso_week)][order(block_id)]
write_csv(as_tibble(block_manifest), file.path(output_dir, "iso_week_block_manifest.csv"))

set.seed(bootstrap_seed)
block_counts <- matrix(0L, nrow = n_boot, ncol = n_blocks)
for (b in seq_len(n_boot)) {
  block_counts[b, ] <- tabulate(sample.int(n_blocks, n_blocks, replace = TRUE), n_blocks)
}
colnames(block_counts) <- iso_weeks
if (any(rowSums(block_counts) != n_blocks)) stop("Bootstrap draw matrix is malformed.")
saveRDS(block_counts, file.path(output_dir, "bootstrap_block_counts.rds"), compress = "gzip")
write_csv(
  as_tibble(block_counts) %>% mutate(bootstrap_id = row_number(), .before = 1),
  file.path(output_dir, "bootstrap_block_counts.csv")
)

Rcpp::cppFunction(code = '
Rcpp::NumericMatrix weighted_rank_metrics_cpp(
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
        cum_tp += group_pos;
        cum_fp += group_neg;
        const double recall = cum_tp / total_pos;
        const double precision = cum_tp / (cum_tp + cum_fp);
        pr_area += (recall - prev_recall) * (prev_precision + precision) * 0.5;
        prev_recall = recall;
        prev_precision = precision;
      }
      i = j;
    }
    out(b, 0) = roc_num / (total_pos * total_neg);
    out(b, 1) = pr_area;
  }
  Rcpp::colnames(out) = Rcpp::CharacterVector::create("roc_auc", "pr_auc");
  return out;
}')

weekly_base <- pred[, .(
  n_rows = .N, n_nights = uniqueN(date), n_positive = sum(truth)
), by = block_id][order(block_id)]
if (!identical(weekly_base$block_id, seq_len(n_blocks))) stop("Weekly block ordering drifted.")
boot_rows <- as.numeric(block_counts %*% weekly_base$n_rows)
boot_nights <- as.numeric(block_counts %*% weekly_base$n_nights)
boot_positives <- as.numeric(block_counts %*% weekly_base$n_positive)
boot_prevalence <- boot_positives / boot_rows

thresholds <- setNames(canonical$operating_threshold, canonical$model_id)
classifier_draws <- vector("list", length(model_ids))
point_rows <- vector("list", length(model_ids))

for (model_id in model_ids) {
  log_message("Bootstrapping classifier metrics: ", model_id)
  score_col <- score_columns[[model_id]]
  score <- pred[[score_col]]

  weekly_model <- pred[, .(
    squared_error = sum((get(score_col) - truth)^2)
  ), by = block_id][order(block_id)]
  boot_brier <- as.numeric(block_counts %*% weekly_model$squared_error) / boot_rows

  if (model_id == "constant_prevalence") {
    boot_roc <- rep(0.5, n_boot)
    boot_pr <- boot_prevalence
  } else {
    ord <- order(score, decreasing = TRUE)
    rank_boot <- weighted_rank_metrics_cpp(
      pred$truth[ord], score[ord], pred$block_id[ord], block_counts
    )
    boot_roc <- rank_boot[, "roc_auc"]
    boot_pr <- rank_boot[, "pr_auc"]
  }

  threshold <- thresholds[[model_id]]
  if (is.finite(threshold)) {
    weekly_operating <- pred[, {
      selected <- get(score_col) >= threshold
      .(tp = sum(truth == 1L & selected), fp = sum(truth == 0L & selected))
    }, by = block_id][order(block_id)]
    boot_tp <- as.numeric(block_counts %*% weekly_operating$tp)
    boot_fp <- as.numeric(block_counts %*% weekly_operating$fp)
    boot_precision <- boot_tp / (boot_tp + boot_fp)
    boot_recall <- boot_tp / boot_positives
    boot_alerts_per_night <- (boot_tp + boot_fp) / boot_nights
  } else {
    boot_precision <- boot_recall <- boot_alerts_per_night <- rep(NA_real_, n_boot)
  }

  classifier_draws[[model_id]] <- tibble(
    bootstrap_id = seq_len(n_boot), model_id = model_id,
    roc_auc = boot_roc, pr_auc = boot_pr, pr_lift = boot_pr / boot_prevalence,
    brier = boot_brier, precision = boot_precision, recall = boot_recall,
    alerts_per_night = boot_alerts_per_night,
    n_rows = boot_rows, n_nights = boot_nights, n_positive = boot_positives,
    prevalence = boot_prevalence
  )

  if (model_id == "constant_prevalence") {
    point_roc <- 0.5
    point_pr <- mean(pred$truth)
  } else {
    truth_factor <- factor(ifelse(pred$truth == 1L, "1", "0"), levels = c("1", "0"))
    point_roc <- as.numeric(roc_auc_vec(truth_factor, score, event_level = "first"))
    point_pr <- as.numeric(pr_auc_vec(truth_factor, score, event_level = "first"))
  }
  selected <- if (is.finite(threshold)) score >= threshold else rep(FALSE, nrow(pred))
  point_tp <- if (is.finite(threshold)) sum(pred$truth == 1L & selected) else NA_real_
  point_fp <- if (is.finite(threshold)) sum(pred$truth == 0L & selected) else NA_real_
  point_rows[[model_id]] <- tibble(
    model_id = model_id, model_label = unname(model_labels[model_id]),
    n_rows = nrow(pred), n_nights = uniqueN(pred$date), n_positive = sum(pred$truth),
    prevalence = mean(pred$truth), operating_threshold = threshold,
    roc_auc = point_roc, pr_auc = point_pr, pr_lift = point_pr / mean(pred$truth),
    brier = mean((score - pred$truth)^2),
    precision = if (is.finite(threshold)) point_tp / (point_tp + point_fp) else NA_real_,
    recall = if (is.finite(threshold)) point_tp / sum(pred$truth) else NA_real_,
    alerts_per_night = if (is.finite(threshold))
      (point_tp + point_fp) / uniqueN(pred$date) else NA_real_
  )
}

classifier_draws <- bind_rows(classifier_draws)
constant_brier <- classifier_draws %>%
  filter(model_id == "constant_prevalence") %>%
  select(bootstrap_id, constant_brier = brier)
classifier_draws <- classifier_draws %>%
  left_join(constant_brier, by = "bootstrap_id") %>%
  mutate(brier_skill_score = 1 - brier / constant_brier) %>%
  select(-constant_brier)
classifier_points <- bind_rows(point_rows) %>%
  mutate(
    brier_skill_score = 1 - brier /
      brier[model_id == "constant_prevalence"]
  )
write_csv(classifier_points, file.path(output_dir, "classifier_point_estimates.csv"))
write_csv(classifier_draws, file.path(output_dir, "classifier_bootstrap_draws.csv"))

ci_from_draws <- function(draws, points, id_cols, metric_cols) {
  long_draws <- draws %>%
    pivot_longer(all_of(metric_cols), names_to = "metric", values_to = "estimate")
  long_points <- points %>%
    pivot_longer(all_of(metric_cols), names_to = "metric", values_to = "point_estimate") %>%
    select(all_of(id_cols), metric, point_estimate)
  long_draws %>%
    group_by(across(all_of(c(id_cols, "metric")))) %>%
    summarise(
      bootstrap_mean = mean(estimate, na.rm = TRUE),
      bootstrap_se = sd(estimate, na.rm = TRUE),
      ci_lower = unname(quantile(estimate, 0.025, na.rm = TRUE)),
      ci_upper = unname(quantile(estimate, 0.975, na.rm = TRUE)),
      n_valid = sum(is.finite(estimate)), .groups = "drop"
    ) %>%
    left_join(long_points, by = c(setNames(id_cols, id_cols), "metric")) %>%
    relocate(point_estimate, .after = metric)
}

classifier_metrics <- c(
  "roc_auc", "pr_auc", "pr_lift", "brier", "brier_skill_score",
  "precision", "recall", "alerts_per_night"
)
classifier_ci <- ci_from_draws(
  classifier_draws, classifier_points, "model_id", classifier_metrics
) %>% mutate(model_label = unname(model_labels[model_id]), .after = model_id)
write_csv(classifier_ci, file.path(output_dir, "classifier_uncertainty_table.csv"))

comparison_pairs <- tribble(
  ~comparison_id, ~expanded_model, ~reference_model,
  "full_vs_climatology", "full_83_xgb", "municipality_climatology",
  "full_vs_history_only", "full_83_xgb", "history_only_xgb",
  "full_vs_weather", "full_83_xgb", "weather_calendar_static_xgb",
  "full_vs_history_free", "full_83_xgb", "history_free_xgb",
  "full_vs_ridge", "full_83_xgb", "ridge_logistic_83",
  "history_free_vs_weather", "history_free_xgb", "weather_calendar_static_xgb",
  "history_only_vs_climatology", "history_only_xgb", "municipality_climatology"
)
difference_metrics <- c(
  "roc_auc", "pr_auc", "pr_lift", "brier", "brier_skill_score",
  "precision", "recall"
)
paired_difference_draws <- bind_rows(lapply(seq_len(nrow(comparison_pairs)), function(i) {
  pair <- comparison_pairs[i, ]
  expanded <- classifier_draws %>%
    filter(model_id == pair$expanded_model) %>%
    select(bootstrap_id, all_of(difference_metrics))
  reference <- classifier_draws %>%
    filter(model_id == pair$reference_model) %>%
    select(bootstrap_id, all_of(difference_metrics))
  expanded %>%
    inner_join(reference, by = "bootstrap_id", suffix = c("_expanded", "_reference")) %>%
    pivot_longer(-bootstrap_id, names_to = c("metric", ".value"),
                 names_pattern = "^(.*)_(expanded|reference)$") %>%
    mutate(
      comparison_id = pair$comparison_id,
      expanded_model = pair$expanded_model,
      reference_model = pair$reference_model,
      raw_delta = expanded - reference,
      improvement_oriented_delta = if_else(metric == "brier", -raw_delta, raw_delta)
    ) %>%
    select(comparison_id, expanded_model, reference_model, bootstrap_id, metric,
           expanded, reference, raw_delta, improvement_oriented_delta)
}))
write_csv(paired_difference_draws, file.path(output_dir, "paired_difference_draws.csv"))

point_long <- classifier_points %>%
  select(model_id, all_of(difference_metrics)) %>%
  pivot_longer(-model_id, names_to = "metric", values_to = "estimate")
paired_difference_ci <- paired_difference_draws %>%
  group_by(comparison_id, expanded_model, reference_model, metric) %>%
  summarise(
    bootstrap_mean_delta = mean(raw_delta, na.rm = TRUE),
    bootstrap_se_delta = sd(raw_delta, na.rm = TRUE),
    ci_lower = unname(quantile(raw_delta, 0.025, na.rm = TRUE)),
    ci_upper = unname(quantile(raw_delta, 0.975, na.rm = TRUE)),
    probability_improvement = mean(improvement_oriented_delta > 0, na.rm = TRUE),
    ci_excludes_zero = ci_lower > 0 | ci_upper < 0,
    n_valid = sum(is.finite(raw_delta)), .groups = "drop"
  ) %>%
  left_join(
    comparison_pairs %>%
      crossing(metric = difference_metrics) %>%
      left_join(point_long, by = c("expanded_model" = "model_id", "metric")) %>%
      rename(expanded_point = estimate) %>%
      left_join(point_long, by = c("reference_model" = "model_id", "metric")) %>%
      rename(reference_point = estimate) %>%
      mutate(point_delta = expanded_point - reference_point) %>%
      select(comparison_id, metric, expanded_point, reference_point, point_delta),
    by = c("comparison_id", "metric")
  ) %>%
  relocate(expanded_point, reference_point, point_delta, .after = metric)
write_csv(paired_difference_ci, file.path(output_dir, "paired_difference_uncertainty.csv"))

log_message("Precomputing daily national Top-K allocations...")
daily_support <- pred[, .(
  n_obs_day = .N, n_positive_day = sum(truth), block_id = unique(block_id),
  iso_week = unique(iso_week)
), by = date]
if (any(daily_support$n_obs_day != 2457L)) stop("Top-K requires 2,457 observations on every test night.")

topk_daily_list <- vector("list", length(ranked_model_ids))
names(topk_daily_list) <- ranked_model_ids
for (model_id in ranked_model_ids) {
  score_col <- score_columns[[model_id]]
  ranked <- pred[, .(date, GID_2, truth, score = get(score_col))]
  setorder(ranked, date, -score, GID_2)
  ranked[, `:=`(rank = seq_len(.N), cumulative_hits = cumsum(truth)), by = date]
  topk_daily_list[[model_id]] <- ranked[rank <= max(decision_k_values), .(
    date, k = rank, hits = cumulative_hits
  )] %>%
    as_tibble() %>%
    left_join(as_tibble(daily_support), by = "date") %>%
    mutate(model_id = model_id, alerts = k)
}
topk_daily <- bind_rows(topk_daily_list)
random_daily <- as_tibble(daily_support) %>%
  crossing(k = decision_k_values) %>%
  mutate(
    model_id = "random_allocation", alerts = k,
    hits = k * n_positive_day / n_obs_day
  )
topk_daily <- bind_rows(topk_daily, random_daily)
write_csv(
  topk_daily %>% filter(k %in% topk_values),
  file.path(output_dir, "daily_topk_support_selected.csv")
)

topk_weekly <- topk_daily %>%
  group_by(model_id, k, block_id, iso_week) %>%
  summarise(
    hits = sum(hits), alerts = sum(alerts), n_positive = sum(n_positive_day),
    n_rows = sum(n_obs_day), n_nights = n_distinct(date), .groups = "drop"
  )

topk_points_all <- topk_weekly %>%
  group_by(model_id, k) %>%
  summarise(
    n_hits = sum(hits), n_alerts = sum(alerts), n_positive = sum(n_positive),
    n_rows = sum(n_rows), n_nights = sum(n_nights), .groups = "drop"
  ) %>%
  mutate(
    precision = n_hits / n_alerts, recall = n_hits / n_positive,
    lift = precision / (n_positive / n_rows), hits_per_night = n_hits / n_nights,
    model_label = unname(model_labels[model_id]), .after = model_id
  )
topk_points <- topk_points_all %>% filter(k %in% topk_values)
write_csv(topk_points, file.path(output_dir, "topk_point_estimates.csv"))

topk_draws <- bind_rows(lapply(split(topk_weekly %>% filter(k %in% topk_values),
                                     interaction(topk_weekly$model_id[topk_weekly$k %in% topk_values],
                                                 topk_weekly$k[topk_weekly$k %in% topk_values], drop = TRUE)),
                               function(wk) {
  wk <- wk %>% arrange(block_id)
  if (!identical(wk$block_id, seq_len(n_blocks))) stop("Top-K weekly support lost a block.")
  hits <- as.numeric(block_counts %*% wk$hits)
  alerts <- as.numeric(block_counts %*% wk$alerts)
  positives <- as.numeric(block_counts %*% wk$n_positive)
  rows <- as.numeric(block_counts %*% wk$n_rows)
  nights <- as.numeric(block_counts %*% wk$n_nights)
  tibble(
    bootstrap_id = seq_len(n_boot), model_id = wk$model_id[[1]], k = wk$k[[1]],
    n_hits = hits, n_alerts = alerts, n_positive = positives, n_rows = rows,
    n_nights = nights, precision = hits / alerts, recall = hits / positives,
    lift = (hits / alerts) / (positives / rows), hits_per_night = hits / nights
  )
}))
write_csv(topk_draws, file.path(output_dir, "topk_bootstrap_draws.csv"))
topk_ci <- ci_from_draws(
  topk_draws, topk_points, c("model_id", "k"),
  c("precision", "recall", "lift", "hits_per_night")
) %>% mutate(model_label = unname(model_labels[model_id]), .after = model_id)
write_csv(topk_ci, file.path(output_dir, "topk_uncertainty_table.csv"))

topk_comparators <- c(
  "random_allocation", "municipality_climatology", "history_only_xgb", "history_free_xgb"
)
topk_difference_draws <- bind_rows(lapply(topk_comparators, function(reference_model) {
  expanded <- topk_draws %>%
    filter(model_id == "full_83_xgb") %>%
    select(bootstrap_id, k, precision, recall, hits_per_night)
  reference <- topk_draws %>%
    filter(model_id == reference_model) %>%
    select(bootstrap_id, k, precision, recall, hits_per_night)
  expanded %>%
    inner_join(reference, by = c("bootstrap_id", "k"), suffix = c("_full", "_reference")) %>%
    pivot_longer(-c(bootstrap_id, k), names_to = c("metric", ".value"),
                 names_pattern = "^(.*)_(full|reference)$") %>%
    mutate(
      comparison_id = paste0("full_vs_", reference_model),
      reference_model = reference_model, delta = full - reference
    ) %>%
    select(comparison_id, reference_model, bootstrap_id, k, metric, full, reference, delta)
}))
write_csv(topk_difference_draws, file.path(output_dir, "topk_difference_draws.csv"))
topk_difference_ci <- topk_difference_draws %>%
  group_by(comparison_id, reference_model, k, metric) %>%
  summarise(
    bootstrap_mean_delta = mean(delta), bootstrap_se_delta = sd(delta),
    ci_lower = unname(quantile(delta, 0.025)),
    ci_upper = unname(quantile(delta, 0.975)),
    probability_full_better = mean(delta > 0),
    ci_excludes_zero = ci_lower > 0 | ci_upper < 0,
    n_valid = sum(is.finite(delta)), .groups = "drop"
  )
write_csv(topk_difference_ci, file.path(output_dir, "topk_difference_uncertainty.csv"))

log_message("Computing cost-loss policy curves...")
cost_loss_curve <- topk_points_all %>%
  filter(model_id %in% c(decision_model_ids, "random_allocation")) %>%
  select(model_id, model_label, k, n_hits, n_alerts, n_nights, hits_per_night, precision) %>%
  bind_rows(tibble(
    model_id = c(decision_model_ids, "random_allocation"),
    model_label = unname(model_labels[c(decision_model_ids, "random_allocation")]),
    k = 0L, n_hits = 0, n_alerts = 0, n_nights = uniqueN(pred$date),
    hits_per_night = 0, precision = NA_real_
  )) %>%
  crossing(cost_ratio = cost_ratios) %>%
  mutate(
    net_value_per_night = hits_per_night - cost_ratio * k,
    value_definition = "captured outages per night minus cost_ratio times alerts per night"
  )
optimal_cost_loss <- cost_loss_curve %>%
  group_by(model_id, model_label, cost_ratio) %>%
  arrange(desc(net_value_per_night), k, .by_group = TRUE) %>%
  slice(1) %>%
  ungroup() %>%
  rename(optimal_k = k, optimal_net_value_per_night = net_value_per_night)
write_csv(cost_loss_curve, file.path(output_dir, "cost_loss_curve.csv"))
write_csv(optimal_cost_loss, file.path(output_dir, "optimal_cost_loss_policy.csv"))

figure_models <- c(
  "random_allocation", "municipality_climatology", "history_only_xgb",
  "history_free_xgb", "full_83_xgb"
)
palette <- c(
  random_allocation = "#666666", municipality_climatology = "#CC79A7",
  history_only_xgb = "#D55E00", history_free_xgb = "#009E73",
  full_83_xgb = "#0072B2"
)
plot_topk <- topk_ci %>%
  filter(model_id %in% figure_models, metric == "precision") %>%
  mutate(model_label = factor(model_label, levels = unname(model_labels[figure_models])))
plot_a <- ggplot(
  plot_topk,
  aes(x = k, y = point_estimate, colour = model_id, group = model_id)
) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 2.1) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 1.3, linewidth = 0.45) +
  scale_colour_manual(
    values = palette, breaks = names(palette),
    labels = unname(model_labels[names(palette)])
  ) +
  scale_x_continuous(breaks = topk_values) +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1)) +
  labs(
    title = "A. Precision under fixed national alert budgets",
    x = "Alerts per night (K)", y = "Precision", colour = NULL
  ) +
  theme_minimal(base_size = 10.5) +
  theme(
    legend.position = "bottom", panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold")
  )

plot_cost <- optimal_cost_loss %>%
  filter(model_id %in% figure_models) %>%
  mutate(model_label = factor(model_label, levels = unname(model_labels[figure_models])))
plot_b <- ggplot(
  plot_cost,
  aes(x = cost_ratio, y = optimal_net_value_per_night, colour = model_id)
) +
  geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.45, linetype = "dashed") +
  geom_line(linewidth = 0.85) +
  scale_colour_manual(
    values = palette, breaks = names(palette),
    labels = unname(model_labels[names(palette)])
  ) +
  scale_x_continuous(labels = scales::label_percent(accuracy = 1)) +
  labs(
    title = "B. Best achievable net value across alert budgets",
    x = "Inspection cost relative to one captured outage",
    y = "Net value per night", colour = NULL
  ) +
  theme_minimal(base_size = 10.5) +
  theme(legend.position = "none", panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"))

combined_plot <- plot_a / plot_b + plot_layout(heights = c(1, 1), guides = "keep")
png_path <- file.path(figure_dir, "classifier_decision_value.png")
pdf_path <- file.path(figure_dir, "classifier_decision_value.pdf")
ggsave(png_path, combined_plot, width = 10.5, height = 7.7, dpi = 320, bg = "white")
ggsave(pdf_path, combined_plot, width = 10.5, height = 7.7,
       device = grDevices::pdf, bg = "white")

canonical_check <- classifier_points %>%
  select(model_id, roc_auc, pr_auc, brier, brier_skill_score, precision, recall,
         alerts_per_night) %>%
  left_join(
    canonical %>% select(
      model_id, canonical_roc = roc_auc, canonical_pr = pr_auc,
      canonical_brier = brier, canonical_bss = brier_skill_score,
      canonical_precision = test_precision, canonical_recall = test_recall,
      canonical_alerts = alerts_per_night
    ), by = "model_id"
  ) %>%
  mutate(
    roc_diff = abs(roc_auc - canonical_roc), pr_diff = abs(pr_auc - canonical_pr),
    brier_diff = abs(brier - canonical_brier), bss_diff = abs(brier_skill_score - canonical_bss),
    precision_diff = abs(precision - canonical_precision),
    recall_diff = abs(recall - canonical_recall),
    alerts_diff = abs(alerts_per_night - canonical_alerts)
  )
write_csv(canonical_check, file.path(output_dir, "canonical_reproduction_check.csv"))

topk_check <- topk_points %>%
  filter(model_id %in% ranked_model_ids) %>%
  select(model_id, k, n_hits, precision, recall, lift) %>%
  left_join(
    phase2_topk %>% select(
      model_id, k, canonical_hits = n_hits, canonical_precision = precision,
      canonical_recall = recall, canonical_lift = lift
    ), by = c("model_id", "k")
  ) %>%
  mutate(
    hits_diff = abs(n_hits - canonical_hits),
    precision_diff = abs(precision - canonical_precision),
    recall_diff = abs(recall - canonical_recall),
    lift_diff = abs(lift - canonical_lift)
  )
write_csv(topk_check, file.path(output_dir, "topk_reproduction_check.csv"))

qa_checks <- tribble(
  ~check_id, ~status, ~observed, ~expected,
  "phase2_independent_qa", ifelse(any(phase2_qa$status == "FAIL"), "FAIL", "PASS"),
  paste(table(phase2_qa$status), collapse = "; "), "No FAIL",
  "test_support", ifelse(
    nrow(pred) == 1334151L && sum(pred$truth) == 5350L && uniqueN(pred$date) == 543L &&
      uniqueN(pred$GID_2) == 2457L, "PASS", "FAIL"
  ), paste(nrow(pred), sum(pred$truth), uniqueN(pred$date), uniqueN(pred$GID_2), sep = "/"),
  "1334151/5350/543/2457",
  "iso_week_blocks", ifelse(n_blocks == 79L, "PASS", "FAIL"), as.character(n_blocks), "79",
  "bootstrap_count", ifelse(n_boot >= 1000L, "PASS", "FAIL"), as.character(n_boot), ">=1000",
  "paired_draw_matrix", ifelse(all(rowSums(block_counts) == n_blocks), "PASS", "FAIL"),
  paste(range(rowSums(block_counts)), collapse = "/"), "79/79",
  "classifier_draw_completeness", ifelse(
    nrow(classifier_draws) == length(model_ids) * n_boot &&
      !anyNA(classifier_draws[, c("roc_auc", "pr_auc", "pr_lift", "brier", "brier_skill_score")]),
    "PASS", "FAIL"
  ), as.character(nrow(classifier_draws)), as.character(length(model_ids) * n_boot),
  "canonical_metric_reproduction", ifelse(
    max(canonical_check$roc_diff, na.rm = TRUE) < 1e-10 &&
      max(canonical_check$pr_diff, na.rm = TRUE) < 1e-12 &&
      max(canonical_check$brier_diff, na.rm = TRUE) < 1e-12 &&
      max(canonical_check$bss_diff, na.rm = TRUE) < 1e-12 &&
      max(canonical_check$precision_diff, na.rm = TRUE) < 1e-12 &&
      max(canonical_check$recall_diff, na.rm = TRUE) < 1e-12 &&
      max(canonical_check$alerts_diff, na.rm = TRUE) < 1e-12,
    "PASS", "FAIL"
  ), paste(
    signif(max(canonical_check$roc_diff, na.rm = TRUE), 3),
    signif(max(canonical_check$pr_diff, na.rm = TRUE), 3),
    signif(max(canonical_check$brier_diff, na.rm = TRUE), 3), sep = "/"
  ), "ROC <1e-10; others <1e-12",
  "topk_reproduction", ifelse(
    max(topk_check$hits_diff) == 0 && max(topk_check$precision_diff) < 1e-12 &&
      max(topk_check$recall_diff) < 1e-12 && max(topk_check$lift_diff) < 1e-12,
    "PASS", "FAIL"
  ), paste(max(topk_check$hits_diff), signif(max(topk_check$precision_diff), 3), sep = "/"),
  "0/<1e-12",
  "topk_draw_completeness", ifelse(
    nrow(topk_draws) == (length(ranked_model_ids) + 1L) * length(topk_values) * n_boot &&
      !anyNA(topk_draws[, c("precision", "recall", "lift", "hits_per_night")]),
    "PASS", "FAIL"
  ), as.character(nrow(topk_draws)),
  as.character((length(ranked_model_ids) + 1L) * length(topk_values) * n_boot),
  "cost_loss_zero_option", ifelse(
    all(optimal_cost_loss$optimal_net_value_per_night >= -1e-15), "PASS", "FAIL"
  ), as.character(min(optimal_cost_loss$optimal_net_value_per_night)), ">=0",
  "figure_outputs", ifelse(
    file.exists(png_path) && file.info(png_path)$size > 10000L &&
      file.exists(pdf_path) && file.info(pdf_path)$size > 5000L, "PASS", "FAIL"
  ), paste(file.info(png_path)$size, file.info(pdf_path)$size, sep = "/"),
  "PNG >10000; vector PDF >5000"
)
write_csv(qa_checks, file.path(output_dir, "qa_checks.csv"))

run_config <- list(
  run_id = run_id, created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
  git_commit = system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
  definitive_phase2_dir = phase2_dir,
  bootstrap = list(
    type = "paired nonparametric ISO-week block bootstrap",
    n_resamples = n_boot, n_blocks = n_blocks, seed = bootstrap_seed,
    ci = "95% percentile interval using R quantile type 7",
    partial_blocks = "Observed test nights retained; five ISO weeks contain five observed nights and 74 contain seven"
  ),
  metrics = list(
    rank = "Exact weighted ROC-AUC with half-credit for ties and exact trapezoidal PR-AUC",
    brier_skill_reference = "Training-national-prevalence predictor on the same resampled observations",
    operating_thresholds = "Model-specific precision-floor thresholds selected on 2020-Q2 in definitive Phase 2",
    topk = c(5L, 10L, 20L, 50L)
  ),
  cost_loss = list(
    formula = "net value per night = captured outages per night - cost_ratio * alerts per night",
    benefit_normalization = "One captured outage has unit benefit",
    cost_ratio_grid = c(min(cost_ratios), max(cost_ratios), 0.005),
    alert_budget_grid = c(min(decision_k_values), max(decision_k_values), 1L),
    zero_alert_option = TRUE
  )
)
write_json(run_config, file.path(output_dir, "run_config.json"), pretty = TRUE, auto_unbox = TRUE)

status <- if (any(qa_checks$status == "FAIL")) "FAIL" else
  if (any(qa_checks$status == "WARN")) "PASS_WITH_WARNINGS" else "PASS"
headline_ci <- classifier_ci %>% filter(model_id == "full_83_xgb")
top10 <- topk_ci %>% filter(model_id == "full_83_xgb", k == 10L)
report <- c(
  "# Phase 4 Classifier Uncertainty And Decision Value", "",
  paste0("- Run ID: `", run_id, "`"), paste0("- Status: **", status, "**"),
  paste0("- Bootstrap: ", n_boot, " paired resamples of ", n_blocks, " ISO-week blocks"),
  paste0("- Test support: ", format(nrow(pred), big.mark = ","), " municipality-nights; ",
         format(sum(pred$truth), big.mark = ","), " positives"), "",
  "## Headline Model", "",
  vapply(seq_len(nrow(headline_ci)), function(i) {
    r <- headline_ci[i, ]
    sprintf("- %s: %.4f (95%% CI %.4f-%.4f)", r$metric, r$point_estimate, r$ci_lower, r$ci_upper)
  }, character(1)), "", "## Top-10 Policy", "",
  vapply(seq_len(nrow(top10)), function(i) {
    r <- top10[i, ]
    sprintf("- %s: %.4f (95%% CI %.4f-%.4f)", r$metric, r$point_estimate, r$ci_lower, r$ci_upper)
  }, character(1)), "", "## Interpretation", "",
  paste0(
    "All intervals preserve storm-scale temporal dependence by resampling complete ISO-week blocks. ",
    "Model differences are paired because every model is evaluated on the same resampled observations."
  ),
  paste0(
    "The cost-loss analysis is a normalized operational comparison, not a monetary valuation: ",
    "one captured outage is assigned unit benefit and each alert incurs the stated relative cost."
  ), "", "## QA", "",
  paste0("- PASS: ", sum(qa_checks$status == "PASS")),
  paste0("- WARN: ", sum(qa_checks$status == "WARN")),
  paste0("- FAIL: ", sum(qa_checks$status == "FAIL"))
)
writeLines(report, file.path(output_dir, "QA_REPORT.md"))
log_message("Phase 4 classifier status: ", status)
if (status == "FAIL") stop("Phase 4 classifier failed QA. See ", file.path(output_dir, "QA_REPORT.md"))
