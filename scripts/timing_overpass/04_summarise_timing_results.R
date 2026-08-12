# Summarise timing-overpass robustness runs.
#
# Input: a timing batch directory produced by 03_run_timing_ablations.R.
# Output: compact CSV summaries under <batch_dir>/summary/.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(jsonlite)
  library(tidyr)
library(tibble)
})

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) return(y)
  if (length(x) == 1L && is.na(x)) return(y)
  x
}

col_or_null <- function(df, nm) {
  if (is.null(df) || nrow(df) == 0L || !nm %in% names(df)) return(NULL)
  df[[nm]]
}

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (basename(dirname(getwd())) == "scripts") return(normalizePath(file.path(getwd(), "..", "..")))
  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))
  normalizePath(getwd())
}

classify_family <- function(feature) {
  case_when(
    grepl("^hist_|^cumulative_outage_minutes|^days_since_last_outage", feature) ~ "outage_history",
    grepl("^ntl_", feature) ~ "ntl",
    grepl("^nb_.*_lag[0-9]+$", feature) ~ "ntl_spatial",
    grepl("^(atm|dew|max_dew|min_dew|lai_high|lai_low|rain|rh|skin_temp|temp|wdr|wind_u|wind_v|wsp|max_temp|min_temp)_lag[0-9]+$", feature) ~ "weather_lagged",
    grepl("^has_weather$", feature) ~ "weather_availability",
    grepl("^built_share_(area|mask)_static$", feature) ~ "built_infrastructure",
    grepl("^moon_fraction$", feature) ~ "moon_phase",
    grepl("^cenace_", feature) ~ "cenace_prices",
    grepl("^(dow|doy|month|year|is_holiday|is_public_holiday|is_weekend|is_covid_period)", feature) ~ "time_calendar",
    TRUE ~ "other"
  )
}

infer_run_dir <- function(output_root, variant_id, ablation_name) {
  if (is.na(output_root) || !dir.exists(output_root)) return(NA_character_)
  ablation_run_name <- paste(variant_id, ablation_name, sep = "_")
  dirs <- list.dirs(output_root, full.names = TRUE, recursive = FALSE)
  if (length(dirs) == 0L) return(NA_character_)
  matched <- dirs[grepl(paste0("_", ablation_run_name, "$"), basename(dirs))]
  if (length(matched) == 0L) return(NA_character_)
  info <- file.info(matched)
  matched[order(info$mtime, decreasing = TRUE)][[1]]
}

args <- commandArgs(trailingOnly = TRUE)
project_dir <- detect_project_dir()

if (length(args) < 1L) {
  stop("Usage: Rscript scripts/timing_overpass/04_summarise_timing_results.R <timing_batch_dir>")
}

batch_dir <- normalizePath(args[[1]], mustWork = TRUE)
manifest_path <- file.path(batch_dir, "timing_ablation_manifest.csv")
if (!file.exists(manifest_path)) {
  stop("Cannot find timing manifest: ", manifest_path)
}

summary_dir <- file.path(batch_dir, "summary")
dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)

label_summary_path <- file.path(project_dir, "data", "model_ready", "timing_overpass", "labels", "overpass_label_summary.csv")
label_summary <- if (file.exists(label_summary_path)) {
  read_csv(label_summary_path, show_col_types = FALSE)
} else {
  tibble()
}

manifest <- read_csv(manifest_path, show_col_types = FALSE)
runs_manifest <- manifest %>%
  rowwise() %>%
  mutate(
    output_dir = if_else(
      is.na(output_dir),
      infer_run_dir(output_root, variant_id, ablation_name),
      output_dir
    ),
    run_id = if_else(is.na(run_id) & !is.na(output_dir), basename(output_dir), run_id)
  ) %>%
  ungroup()
runs <- runs_manifest %>% filter(!is.na(output_dir), dir.exists(output_dir))

if (nrow(runs) == 0L) {
  stop("No run folders found in manifest.")
}

run_summary <- runs %>%
  rowwise() %>%
  do({
    row <- .
    metrics_file <- file.path(row$output_dir, "metrics_summary.csv")
    threshold_file <- file.path(row$output_dir, "operating_thresholds_table.csv")
    config_file <- file.path(row$output_dir, "run_config.json")
    benchmark_file <- file.path(row$output_dir, "benchmark_table_same_run.csv")
    features_used_file <- file.path(row$output_dir, "features_used.txt")

    metrics_test <- if (file.exists(metrics_file)) {
      read_csv(metrics_file, show_col_types = FALSE) %>%
        filter(split == "test") %>%
        slice(1)
    } else {
      tibble()
    }

    benchmark_platt <- if (file.exists(benchmark_file)) {
      read_csv(benchmark_file, show_col_types = FALSE) %>%
        filter(model_type == "binary", method == "platt") %>%
        slice(1)
    } else {
      tibble()
    }

    operating <- if (file.exists(threshold_file)) {
      read_csv(threshold_file, show_col_types = FALSE) %>%
        filter(
          policy_type == row$reference_policy_type,
          abs(policy_target - row$reference_policy_target) < 1e-8
        ) %>%
        slice(1)
    } else {
      tibble()
    }

    cfg <- if (file.exists(config_file)) jsonlite::read_json(config_file) else list()
    n_features <- cfg$model$n_features %||% NA_integer_
    if (is.na(n_features) && file.exists(features_used_file)) {
      n_features <- length(readLines(features_used_file, warn = FALSE))
    }

    tibble(
      batch_id = row$batch_id,
      variant_id = row$variant_id,
      ablation_name = row$ablation_name,
      status = row$status,
      exit_code = row$exit_code,
      run_id = row$run_id,
      output_dir = row$output_dir,
      n_features = n_features,
      roc_auc_test = col_or_null(metrics_test, "roc_auc") %||% col_or_null(benchmark_platt, "roc_auc_test") %||% NA_real_,
      pr_auc_test = col_or_null(metrics_test, "pr_auc") %||% col_or_null(benchmark_platt, "pr_auc_test") %||% NA_real_,
      brier_test = col_or_null(metrics_test, "brier_platt") %||% col_or_null(metrics_test, "brier_score") %||% col_or_null(benchmark_platt, "brier_test") %||% NA_real_,
      logloss_test = col_or_null(metrics_test, "logloss_platt") %||% col_or_null(metrics_test, "logloss") %||% col_or_null(benchmark_platt, "logloss_test") %||% NA_real_,
      policy_precision_test = col_or_null(operating, "test_precision") %||% NA_real_,
      policy_recall_test = col_or_null(operating, "test_recall") %||% NA_real_,
      policy_alerts_per_day_test = col_or_null(operating, "test_alerts_per_day") %||% NA_real_
    )
  }) %>%
  ungroup()

if (nrow(label_summary) > 0L) {
  run_summary <- run_summary %>%
    left_join(
      label_summary %>%
        select(
          variant_id, buffer_minutes, analysis_role,
          model_key_muni_night_labels,
          share_of_current_panel_positives
        ),
      by = "variant_id"
    )
}

family_gain <- runs %>%
  rowwise() %>%
  do({
    row <- .
    imp_file <- file.path(row$output_dir, "feature_importance.csv")
    if (!file.exists(imp_file)) {
      tibble()
    } else {
      imp <- read_csv(imp_file, show_col_types = FALSE)
      if (!all(c("Feature", "Gain") %in% names(imp))) {
        tibble()
      } else {
        imp %>%
          mutate(family = classify_family(Feature)) %>%
          group_by(family) %>%
          summarise(gain = sum(Gain, na.rm = TRUE), .groups = "drop") %>%
          mutate(
            gain_share = gain / sum(gain, na.rm = TRUE),
            batch_id = row$batch_id,
            variant_id = row$variant_id,
            ablation_name = row$ablation_name,
            run_id = row$run_id
          )
      }
    }
  }) %>%
  ungroup() %>%
  select(batch_id, variant_id, ablation_name, run_id, family, gain, gain_share)

channel_comparison <- run_summary %>%
  select(
    variant_id, ablation_name, roc_auc_test, pr_auc_test,
    policy_recall_test, policy_precision_test, policy_alerts_per_day_test
  ) %>%
  pivot_wider(
    names_from = ablation_name,
    values_from = c(
      roc_auc_test, pr_auc_test, policy_recall_test,
      policy_precision_test, policy_alerts_per_day_test
    )
  ) %>%
  mutate(
    delta_roc_auc_history_minus_rs =
      roc_auc_test_bench_rs_history - roc_auc_test_bench_rs_only,
    delta_pr_auc_history_minus_rs =
      pr_auc_test_bench_rs_history - pr_auc_test_bench_rs_only,
    delta_recall_history_minus_rs =
      policy_recall_test_bench_rs_history - policy_recall_test_bench_rs_only
  )

family_wide <- family_gain %>%
  filter(ablation_name == "bench_rs_history") %>%
  select(variant_id, family, gain_share) %>%
  pivot_wider(names_from = family, values_from = gain_share, names_prefix = "gain_share_")

channel_comparison <- channel_comparison %>%
  left_join(family_wide, by = "variant_id")

write_csv(run_summary, file.path(summary_dir, "timing_model_summary.csv"))
write_csv(family_gain, file.path(summary_dir, "timing_feature_family_gain.csv"))
write_csv(channel_comparison, file.path(summary_dir, "timing_channel_comparison.csv"))

message("Wrote summaries to: ", summary_dir)
