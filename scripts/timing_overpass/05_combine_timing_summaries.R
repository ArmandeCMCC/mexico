# Combine timing-overpass robustness summaries across variant batches.
#
# Usage:
#   Rscript scripts/timing_overpass/05_combine_timing_summaries.R \
#     data/baselines/binary_threshold/timing_overpass/20260701_143905 \
#     data/baselines/binary_threshold/timing_overpass/20260702_152021 \
#     data/baselines/binary_threshold/timing_overpass/20260702_154208 \
#     data/baselines/binary_threshold/timing_overpass/20260702_160418

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (basename(dirname(getwd())) == "scripts") return(normalizePath(file.path(getwd(), "..", "..")))
  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))
  normalizePath(getwd())
}

read_required_csv <- function(path) {
  if (!file.exists(path)) stop("Missing required summary file: ", path)
  read_csv(path, show_col_types = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
project_dir <- detect_project_dir()

if (length(args) < 1L) {
  stop("Usage: Rscript scripts/timing_overpass/05_combine_timing_summaries.R <batch_dir> [<batch_dir> ...]")
}

batch_dirs <- normalizePath(args, mustWork = TRUE)
batch_ids <- basename(batch_dirs)

label_summary_path <- file.path(
  project_dir, "data", "model_ready", "timing_overpass", "labels",
  "overpass_label_summary.csv"
)
label_summary <- read_required_csv(label_summary_path) %>%
  select(
    variant_id, buffer_minutes, analysis_role, event_rows_selected,
    model_key_muni_night_labels, current_panel_positive_labels,
    share_of_current_panel_positives
  )

model_all <- bind_rows(lapply(seq_along(batch_dirs), function(i) {
  read_required_csv(file.path(batch_dirs[[i]], "summary", "timing_model_summary.csv")) %>%
    mutate(source_batch_dir = batch_dirs[[i]], source_batch_id = batch_ids[[i]])
})) %>%
  left_join(
    label_summary %>% select(-buffer_minutes, -analysis_role, -model_key_muni_night_labels, -share_of_current_panel_positives),
    by = "variant_id"
  ) %>%
  arrange(buffer_minutes, ablation_name)

channel_all <- bind_rows(lapply(seq_along(batch_dirs), function(i) {
  read_required_csv(file.path(batch_dirs[[i]], "summary", "timing_channel_comparison.csv")) %>%
    mutate(source_batch_dir = batch_dirs[[i]], source_batch_id = batch_ids[[i]])
})) %>%
  left_join(label_summary, by = "variant_id") %>%
  arrange(buffer_minutes)

family_all <- bind_rows(lapply(seq_along(batch_dirs), function(i) {
  read_required_csv(file.path(batch_dirs[[i]], "summary", "timing_feature_family_gain.csv")) %>%
    mutate(source_batch_dir = batch_dirs[[i]], source_batch_id = batch_ids[[i]])
})) %>%
  left_join(label_summary %>% select(variant_id, buffer_minutes), by = "variant_id") %>%
  arrange(buffer_minutes, ablation_name, family)

model_wide <- model_all %>%
  select(
    variant_id, ablation_name, status, n_features, roc_auc_test, pr_auc_test,
    brier_test, logloss_test, policy_precision_test, policy_recall_test,
    policy_alerts_per_day_test
  ) %>%
  pivot_wider(
    names_from = ablation_name,
    values_from = c(
      status, n_features, roc_auc_test, pr_auc_test, brier_test, logloss_test,
      policy_precision_test, policy_recall_test, policy_alerts_per_day_test
    )
  )

family_wide <- family_all %>%
  filter(ablation_name == "bench_rs_history") %>%
  select(variant_id, family, gain_share) %>%
  pivot_wider(names_from = family, values_from = gain_share, names_prefix = "gain_share_history_")

publication_table <- label_summary %>%
  filter(variant_id %in% model_all$variant_id) %>%
  select(
    variant_id, buffer_minutes, model_key_muni_night_labels,
    current_panel_positive_labels, share_of_current_panel_positives
  ) %>%
  left_join(model_wide, by = "variant_id") %>%
  left_join(
    channel_all %>%
      select(
        variant_id,
        delta_roc_auc_history_minus_rs,
        delta_pr_auc_history_minus_rs,
        delta_recall_history_minus_rs
      ),
    by = "variant_id"
  ) %>%
  left_join(family_wide, by = "variant_id") %>%
  arrange(buffer_minutes)

output_root <- file.path(
  project_dir, "data", "baselines", "binary_threshold", "timing_overpass",
  paste0("full_timing_table_", format(Sys.time(), "%Y%m%d_%H%M%S"))
)
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

write_csv(model_all, file.path(output_root, "timing_model_summary_all.csv"))
write_csv(channel_all, file.path(output_root, "timing_channel_comparison_all.csv"))
write_csv(family_all, file.path(output_root, "timing_feature_family_gain_all.csv"))
write_csv(publication_table, file.path(output_root, "timing_table_publication_ready.csv"))
write_csv(
  tibble(source_batch_id = batch_ids, source_batch_dir = batch_dirs),
  file.path(output_root, "source_batches.csv")
)

message("Wrote combined timing tables to: ", output_root)
