# 08b_probcal_ablation_validation.R
# Validate and aggregate probability-calibrated ablation runs
#
# Purpose:
#   - Validate that all 07b ablation runs are comparable
#   - Check required 05b outputs exist for each successful run
#   - Extract a standardized comparison row per ablation
#   - Build a leaderboard / summary table for reporting
#
# Usage:
#   Rscript scripts/08b_probcal_ablation_validation.R [batch_dir]
#
# Examples:
#   Rscript scripts/08b_probcal_ablation_validation.R
#   Rscript scripts/08b_probcal_ablation_validation.R data/baselines/binary_threshold/ablation_batches/20260301_201347
#
# Outputs (written into batch_dir):
#   - validation_required_files.csv      Per-run check of required 05b outputs
#   - validation_config_checks.csv       Batch-level comparability checks
#   - validation_feature_guardrail.csv   Per-run n_features_used <= n_features_eligible
#   - ablation_summary_table.csv         Full metrics for all valid runs
#   - ablation_leaderboard.csv           Combined view with both ranks (sorted by recall)
#   - ablation_leaderboard_by_recall.csv Headline operational comparison
#   - ablation_leaderboard_by_pr_auc.csv Secondary discrimination comparison
#   - validation_report.txt              Human-readable summary

suppressPackageStartupMessages({
  library(jsonlite)
  library(readr)
  library(dplyr)
  library(tibble)
  library(purrr)
  library(stringr)
})

# PATH DETECTION

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (dir.exists("data") || dir.exists("scripts")) return(normalizePath(getwd()))
  warning("Could not determine project directory. Using current directory.")
  normalizePath(getwd())
}

project_dir <- detect_project_dir()
batch_root <- file.path(project_dir, "data", "baselines", "binary_threshold", "ablation_batches")

# ARG PARSING

args <- commandArgs(trailingOnly = TRUE)

resolve_batch_dir <- function(arg_batch_dir = NULL) {
  if (!is.null(arg_batch_dir) && nzchar(arg_batch_dir)) {
    if (!dir.exists(arg_batch_dir)) {
      stop("Batch directory does not exist: ", arg_batch_dir)
    }
    return(normalizePath(arg_batch_dir))
  }
  
  if (!dir.exists(batch_root)) {
    stop("No batch root found: ", batch_root)
  }
  
  batches <- list.dirs(batch_root, recursive = FALSE, full.names = TRUE)
  if (length(batches) == 0) {
    stop("No ablation batch directories found under: ", batch_root)
  }
  
  batches <- batches[order(basename(batches), decreasing = TRUE)]
  normalizePath(batches[1])
}

batch_dir <- resolve_batch_dir(if (length(args) >= 1) args[1] else NULL)
manifest_path <- file.path(batch_dir, "ablation_manifest.csv")

if (!file.exists(manifest_path)) {
  stop("Manifest not found: ", manifest_path)
}

# LOAD MANIFEST

manifest <- read_csv(manifest_path, show_col_types = FALSE)

cat("\n")
cat("=======================================================================\n")
cat("08b_probcal_ablation_validation.R - Ablation Validation & Aggregation\n")
cat("=======================================================================\n\n")
cat("Project directory: ", project_dir, "\n", sep = "")
cat("Batch directory:   ", batch_dir, "\n", sep = "")
cat("Manifest:          ", manifest_path, "\n", sep = "")
cat("Runs in manifest:  ", nrow(manifest), "\n\n", sep = "")

successful_runs <- manifest %>%
  filter(status == "success", !is.na(output_dir), dir.exists(output_dir))

if (nrow(successful_runs) == 0) {
  stop("No successful runs found in manifest.")
}

cat("Successful runs detected: ", nrow(successful_runs), "\n\n", sep = "")

# REQUIRED OUTPUT FILES

required_files <- c(
  "metrics_summary.csv",
  "calibration_comparison_test.csv",
  "operating_thresholds_table.csv",
  "benchmark_table_same_run.csv",
  "benchmark_uncertainty_bootstrap.csv",
  "temporal_stability_summary.csv",
  "deployment_decision_table.csv",
  "run_config.json",
  "features_used.txt"
)

check_required_files <- function(run_row) {
  outdir <- run_row$output_dir
  tibble(
    ablation_name = run_row$ablation_name,
    run_id = run_row$run_id,
    output_dir = outdir,
    file = required_files,
    exists = file.exists(file.path(outdir, required_files))
  )
}

required_files_check <- bind_rows(
  lapply(seq_len(nrow(successful_runs)), function(i) check_required_files(successful_runs[i, ]))
)

write_csv(required_files_check, file.path(batch_dir, "validation_required_files.csv"))

# SAFE READERS

safe_read_csv <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(read_csv(path, show_col_types = FALSE), error = function(e) NULL)
}

safe_read_json <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(fromJSON(path, simplifyVector = TRUE), error = function(e) NULL)
}

safe_read_lines <- function(path) {
  if (!file.exists(path)) return(character(0))
  tryCatch(readLines(path, warn = FALSE), error = function(e) character(0))
}

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x

extract_scalar <- function(x, default = NA) {
  if (is.null(x) || length(x) == 0) return(default)
  x[[1]]
}

# LOAD RUN ARTIFACTS

load_run_bundle <- function(run_row) {
  outdir <- run_row$output_dir
  
  list(
    manifest = run_row,
    config = safe_read_json(file.path(outdir, "run_config.json")),
    metrics = safe_read_csv(file.path(outdir, "metrics_summary.csv")),
    calibration = safe_read_csv(file.path(outdir, "calibration_comparison_test.csv")),
    operating = safe_read_csv(file.path(outdir, "operating_thresholds_table.csv")),
    benchmark = safe_read_csv(file.path(outdir, "benchmark_table_same_run.csv")),
    bootstrap = safe_read_csv(file.path(outdir, "benchmark_uncertainty_bootstrap.csv")),
    temporal = safe_read_csv(file.path(outdir, "temporal_stability_summary.csv")),
    deployment = safe_read_csv(file.path(outdir, "deployment_decision_table.csv")),
    features_used = safe_read_lines(file.path(outdir, "features_used.txt"))
  )
}

run_bundles <- lapply(seq_len(nrow(successful_runs)), function(i) load_run_bundle(successful_runs[i, ]))
names(run_bundles) <- successful_runs$ablation_name

# CONFIG COMPARABILITY CHECKS

config_rows <- bind_rows(lapply(run_bundles, function(b) {
  cfg <- b$config
  tibble(
    ablation_name = b$manifest$ablation_name,
    run_id = b$manifest$run_id,
    
    task_mode = cfg$task_mode %||% NA_character_,
    strict_enabled = cfg$calibration$strict_separation$enabled %||% NA,
    strict_ratio = cfg$calibration$strict_separation$ratio %||% NA_real_,
    raw_use_threshold_select_only = cfg$calibration$raw_use_threshold_select_only %||% NA,
    
    default_calibrated_method = cfg$calibration$default_calibrated_method %||% NA_character_,
    secondary_calibrated_method = cfg$calibration$secondary_calibrated_method %||% NA_character_,
    benchmark_binary_methods = paste(cfg$calibration$benchmark_binary_methods %||% character(0), collapse = ","),
    
    train_start = cfg$splits$train$start %||% NA_character_,
    train_end = cfg$splits$train$end %||% NA_character_,
    val_start = cfg$splits$val$start %||% NA_character_,
    val_end = cfg$splits$val$end %||% NA_character_,
    test_start = cfg$splits$test$start %||% NA_character_,
    test_end = cfg$splits$test$end %||% NA_character_,
    
    train_rows = cfg$data$train_rows %||% NA_integer_,
    val_rows = cfg$data$val_rows %||% NA_integer_,
    test_rows = cfg$data$test_rows %||% NA_integer_,
    val_days = cfg$data$val_days %||% NA_integer_,
    test_days = cfg$data$test_days %||% NA_integer_,
    
    reference_policy_type = cfg$ablation$reference_policy_type %||% NA_character_,
    reference_policy_target = cfg$ablation$reference_policy_target %||% NA_real_,
    
    n_features = cfg$model$n_features %||% NA_integer_
  )
}))

check_constant <- function(df, col) {
  vals <- unique(df[[col]])
  vals <- vals[!is.na(vals)]
  tibble(
    check = col,
    n_unique_non_na = length(vals),
    passes = length(vals) <= 1,
    values = paste(vals, collapse = " | ")
  )
}

config_checks <- bind_rows(lapply(
  c(
    "task_mode",
    "strict_enabled",
    "strict_ratio",
    "raw_use_threshold_select_only",
    "default_calibrated_method",
    "secondary_calibrated_method",
    "benchmark_binary_methods",
    "train_start", "train_end",
    "val_start", "val_end",
    "test_start", "test_end",
    "train_rows", "val_rows", "test_rows",
    "val_days", "test_days",
    "reference_policy_type",
    "reference_policy_target"
  ),
  function(col) check_constant(config_rows, col)
))

write_csv(config_checks, file.path(batch_dir, "validation_config_checks.csv"))

# STANDARDISED EXTRACTION

get_test_row <- function(metrics_df) {
  if (is.null(metrics_df)) return(NULL)
  out <- metrics_df %>% filter(split == "test")
  if (nrow(out) == 0) return(NULL)
  out[1, ]
}

get_calibration_row <- function(cal_df, method) {
  if (is.null(cal_df)) return(NULL)
  out <- cal_df %>% filter(method == !!method)
  if (nrow(out) == 0) return(NULL)
  out[1, ]
}

get_operating_row <- function(op_df, method, policy_type, policy_target) {
  if (is.null(op_df)) return(NULL)
  
  out <- op_df %>%
    filter(method == !!method, policy_type == !!policy_type)
  
  if (!is.na(policy_target)) {
    out <- out %>% filter(abs(policy_target - !!policy_target) < 1e-8)
  } else {
    out <- out %>% filter(is.na(policy_target))
  }
  
  if (nrow(out) == 0) return(NULL)
  out[1, ]
}

get_temporal_row <- function(temp_df, method) {
  if (is.null(temp_df)) return(NULL)
  out <- temp_df %>% filter(method == !!method)
  if (nrow(out) == 0) return(NULL)
  out[1, ]
}

get_boot_metric <- function(boot_df, method, metric_name, policy_type, policy_target) {
  if (is.null(boot_df)) return(NULL)
  
  out <- boot_df %>%
    filter(method == !!method, metric == !!metric_name, policy_type == !!policy_type)
  
  if (!is.na(policy_target)) {
    out <- out %>% filter(abs(policy_target - !!policy_target) < 1e-8)
  } else {
    out <- out %>% filter(is.na(policy_target))
  }
  
  if (nrow(out) == 0) return(NULL)
  out[1, ]
}

get_deployment_row <- function(dep_df, method) {
  if (is.null(dep_df)) return(NULL)
  out <- dep_df %>% filter(method == !!method)
  if (nrow(out) == 0) return(NULL)
  out[1, ]
}

extract_summary_row <- function(bundle) {
  cfg <- bundle$config
  metrics_test <- get_test_row(bundle$metrics)
  
  default_method <- cfg$calibration$default_calibrated_method %||% "platt"
  secondary_method <- cfg$calibration$secondary_calibrated_method %||% "beta"
  ref_policy_type <- cfg$ablation$reference_policy_type %||% NA_character_
  ref_policy_target <- cfg$ablation$reference_policy_target %||% NA_real_
  
  cal_default <- get_calibration_row(bundle$calibration, default_method)
  cal_secondary <- get_calibration_row(bundle$calibration, secondary_method)
  op_default <- get_operating_row(bundle$operating, default_method, ref_policy_type, ref_policy_target)
  temp_default <- get_temporal_row(bundle$temporal, default_method)
  dep_default <- get_deployment_row(bundle$deployment, default_method)
  
  boot_precision <- get_boot_metric(bundle$bootstrap, default_method, "precision", ref_policy_type, ref_policy_target)
  boot_recall <- get_boot_metric(bundle$bootstrap, default_method, "recall", ref_policy_type, ref_policy_target)
  boot_f1 <- get_boot_metric(bundle$bootstrap, default_method, "f1", ref_policy_type, ref_policy_target)
  boot_alerts <- get_boot_metric(bundle$bootstrap, default_method, "alerts_per_day", ref_policy_type, ref_policy_target)
  
  tibble(
    ablation_name = bundle$manifest$ablation_name,
    run_id = bundle$manifest$run_id,
    output_dir = bundle$manifest$output_dir,
    
    task_mode = cfg$task_mode %||% NA_character_,
    strict_enabled = cfg$calibration$strict_separation$enabled %||% NA,
    default_method = default_method,
    secondary_method = secondary_method,
    reference_policy_type = ref_policy_type,
    reference_policy_target = ref_policy_target,
    
    families = bundle$manifest$families %||% NA_character_,
    include_regex = bundle$manifest$include_regex %||% NA_character_,
    exclude_regex = bundle$manifest$exclude_regex %||% NA_character_,
    
    n_features_raw = bundle$manifest$n_features_raw %||% NA_integer_,
    n_features_eligible = bundle$manifest$n_features_eligible %||% NA_integer_,
    n_features_used = cfg$model$n_features %||% NA_integer_,
    
    test_rows = cfg$data$test_rows %||% NA_integer_,
    test_days = cfg$data$test_days %||% NA_integer_,
    test_prevalence = cfg$data$test_prevalence %||% NA_real_,
    
    roc_auc_test = metrics_test$roc_auc %||% NA_real_,
    pr_auc_test = metrics_test$pr_auc %||% NA_real_,
    
    raw_logloss_test = metrics_test$logloss %||% NA_real_,
    raw_brier_test = metrics_test$brier_score %||% NA_real_,
    raw_ece_equal_test = metrics_test$ece_equal %||% NA_real_,
    raw_ece_quantile_test = metrics_test$ece_quantile %||% NA_real_,
    
    default_logloss_test = cal_default$logloss %||% NA_real_,
    default_brier_test = cal_default$brier %||% NA_real_,
    default_ece_equal_test = cal_default$ece_equal %||% NA_real_,
    default_ece_quantile_test = cal_default$ece_quantile %||% NA_real_,
    default_cal_intercept = cal_default$cal_intercept %||% NA_real_,
    default_cal_slope = cal_default$cal_slope %||% NA_real_,
    
    secondary_logloss_test = cal_secondary$logloss %||% NA_real_,
    secondary_brier_test = cal_secondary$brier %||% NA_real_,
    secondary_ece_equal_test = cal_secondary$ece_equal %||% NA_real_,
    secondary_ece_quantile_test = cal_secondary$ece_quantile %||% NA_real_,
    
    operating_threshold = op_default$selected_threshold %||% NA_real_,
    test_precision_at_policy = op_default$test_precision %||% NA_real_,
    test_recall_at_policy = op_default$test_recall %||% NA_real_,
    test_f1_at_policy = op_default$test_f1 %||% NA_real_,
    test_alerts_per_day_at_policy = op_default$test_alerts_per_day %||% NA_real_,
    test_kappa_at_policy = op_default$test_kappa %||% NA_real_,
    
    precision_ci_lower = boot_precision$ci_lower %||% NA_real_,
    precision_ci_upper = boot_precision$ci_upper %||% NA_real_,
    recall_ci_lower = boot_recall$ci_lower %||% NA_real_,
    recall_ci_upper = boot_recall$ci_upper %||% NA_real_,
    f1_ci_lower = boot_f1$ci_lower %||% NA_real_,
    f1_ci_upper = boot_f1$ci_upper %||% NA_real_,
    alerts_per_day_ci_lower = boot_alerts$ci_lower %||% NA_real_,
    alerts_per_day_ci_upper = boot_alerts$ci_upper %||% NA_real_,
    
    temporal_precision_sd = temp_default$precision_sd %||% NA_real_,
    temporal_recall_sd = temp_default$recall_sd %||% NA_real_,
    temporal_f1_sd = temp_default$f1_sd %||% NA_real_,
    temporal_f1_cv = temp_default$f1_cv %||% NA_real_,
    temporal_alerts_per_day_sd = temp_default$alerts_per_day_sd %||% NA_real_,
    
    deployment_ready = dep_default$deployment_ready %||% NA,
    deployment_precision_floor = dep_default$deployment_precision_floor %||% NA_real_,
    
    n_required_files_present = sum(required_files_check$exists[required_files_check$run_id == bundle$manifest$run_id], na.rm = TRUE),
    n_required_files_total = length(required_files),
    required_files_complete = sum(required_files_check$exists[required_files_check$run_id == bundle$manifest$run_id], na.rm = TRUE) == length(required_files)
  )
}

summary_table <- bind_rows(lapply(run_bundles, extract_summary_row))

# FEATURE COUNT GUARDRAIL
# Ensures n_features_used <= n_features_eligible for every run.
# Violations indicate a bug in feature filtering logic.

feature_guardrail <- summary_table %>%
  mutate(
    feature_guardrail_status = case_when(
      is.na(n_features_used) | is.na(n_features_eligible) ~ "missing_data",
      n_features_used <= n_features_eligible ~ "pass",
      TRUE ~ "fail"
    ),
    feature_guardrail_pass = (feature_guardrail_status == "pass"),
    feature_guardrail_msg = case_when(
      feature_guardrail_status == "missing_data" ~ "missing n_features_used or n_features_eligible",
      feature_guardrail_status == "pass" ~ "ok",
      TRUE ~ paste0("VIOLATION: used=", n_features_used, " > eligible=", n_features_eligible)
    )
  ) %>%
  select(ablation_name, run_id, n_features_raw, n_features_eligible, n_features_used,
         feature_guardrail_status, feature_guardrail_pass, feature_guardrail_msg)

write_csv(feature_guardrail, file.path(batch_dir, "validation_feature_guardrail.csv"))

# Batch-level guardrail summary (for report/console only, not stored per-row)
n_guardrail_pass <- sum(feature_guardrail$feature_guardrail_status == "pass", na.rm = TRUE)
n_guardrail_fail <- sum(feature_guardrail$feature_guardrail_status == "fail", na.rm = TRUE)
n_guardrail_missing <- sum(feature_guardrail$feature_guardrail_status == "missing_data", na.rm = TRUE)
feature_guardrail_all_pass <- (n_guardrail_fail == 0) && (n_guardrail_missing == 0)

if (n_guardrail_fail > 0) {
  violations <- feature_guardrail %>% filter(feature_guardrail_status == "fail")
  warning("Feature guardrail violations detected:\n",
          paste0("  - ", violations$ablation_name, ": ", violations$feature_guardrail_msg, collapse = "\n"))
}
if (n_guardrail_missing > 0) {
  missing_runs <- feature_guardrail %>% filter(feature_guardrail_status == "missing_data")
  warning("Feature guardrail has missing data for:\n",
          paste0("  - ", missing_runs$ablation_name, collapse = "\n"))
}

# Join per-run guardrail status into summary_table
summary_table <- summary_table %>%
  left_join(
    feature_guardrail %>%
      select(run_id, feature_guardrail_status, feature_guardrail_pass, feature_guardrail_msg),
    by = "run_id"
  )

# LEADERBOARDS
# Three leaderboard outputs:
#   ablation_leaderboard_by_recall.csv  - Headline operational comparison
#                                         Sorted by recall at policy (higher = better)
#   ablation_leaderboard_by_pr_auc.csv  - Secondary discrimination comparison
#                                         Sorted by PR-AUC (higher = better)
#   ablation_leaderboard.csv            - Combined view with both ranks,
#                                         sorted by recall (primary)

# Primary leaderboard: sorted by recall at policy (operational focus)
leaderboard_recall <- summary_table %>%
  arrange(desc(test_recall_at_policy), desc(pr_auc_test), default_logloss_test) %>%
  mutate(rank_by_recall = row_number())

# Secondary leaderboard: sorted by PR-AUC (scientific focus)
leaderboard_pr_auc <- summary_table %>%
  arrange(desc(pr_auc_test), desc(test_recall_at_policy), default_logloss_test) %>%
  mutate(rank_by_pr_auc = row_number())

# Combined leaderboard with both rankings (join on both ablation_name and run_id for safety)
leaderboard <- leaderboard_recall %>%
  left_join(
    leaderboard_pr_auc %>% select(ablation_name, run_id, rank_by_pr_auc),
    by = c("ablation_name", "run_id")
  )

write_csv(summary_table, file.path(batch_dir, "ablation_summary_table.csv"))
write_csv(leaderboard, file.path(batch_dir, "ablation_leaderboard.csv"))
write_csv(leaderboard_recall, file.path(batch_dir, "ablation_leaderboard_by_recall.csv"))
write_csv(leaderboard_pr_auc, file.path(batch_dir, "ablation_leaderboard_by_pr_auc.csv"))

# TEXT REPORT

report_path <- file.path(batch_dir, "validation_report.txt")

all_required_pass <- all(required_files_check$exists)
all_config_pass <- all(config_checks$passes)

# Top row taken from recall-ranked leaderboard because recall-at-policy
# is the headline operational criterion for deployment decisions.
top_row <- if (nrow(leaderboard) > 0) leaderboard[1, ] else NULL

report_lines <- c(
  "Ablation Validation Report",
  "=========================",
  "",
  paste0("Batch directory: ", batch_dir),
  paste0("Manifest: ", manifest_path),
  paste0("Successful runs: ", nrow(successful_runs)),
  "",
  "Required files check",
  "--------------------",
  paste0("All required files present for all successful runs: ", all_required_pass),
  "",
  "Config consistency checks",
  "-------------------------",
  paste0("All comparability checks pass: ", all_config_pass),
  "",
  "Feature guardrail check",
  "-----------------------",
  paste0("All guardrail checks pass: ", feature_guardrail_all_pass),
  paste0("  Pass: ", n_guardrail_pass),
  paste0("  Fail: ", n_guardrail_fail),
  paste0("  Missing data: ", n_guardrail_missing),
  ""
)

if (nrow(config_checks) > 0) {
  report_lines <- c(
    report_lines,
    "Detailed config checks:",
    paste0(
      "  - ",
      config_checks$check,
      ": passes=",
      config_checks$passes,
      ", values=",
      config_checks$values
    ),
    ""
  )
}

if (!is.null(top_row)) {
  report_lines <- c(
    report_lines,
    "Top ablation by recall at headline policy",
    "----------------------------------------",
    paste0("Ablation: ", top_row$ablation_name),
    paste0("Features used: ", top_row$n_features_used),
    paste0("PR AUC (test): ", round(top_row$pr_auc_test, 4)),
    paste0("ROC AUC (test): ", round(top_row$roc_auc_test, 4)),
    paste0("Default method: ", top_row$default_method),
    paste0("Recall @ policy: ", round(top_row$test_recall_at_policy, 4)),
    paste0("Precision @ policy: ", round(top_row$test_precision_at_policy, 4)),
    paste0("F1 @ policy: ", round(top_row$test_f1_at_policy, 4)),
    paste0("Alerts/day @ policy: ", round(top_row$test_alerts_per_day_at_policy, 2)),
    paste0("Deployment ready: ", top_row$deployment_ready),
    ""
  )
}

report_lines <- c(
  report_lines,
  "Files written",
  "-------------",
  "validation_required_files.csv",
  "validation_config_checks.csv",
  "validation_feature_guardrail.csv",
  "ablation_summary_table.csv",
  "ablation_leaderboard.csv           (combined, sorted by recall)",
  "ablation_leaderboard_by_recall.csv (headline operational comparison)",
  "ablation_leaderboard_by_pr_auc.csv (secondary discrimination comparison)",
  "validation_report.txt"
)

writeLines(report_lines, report_path)

# CONSOLE SUMMARY

cat("Required files complete for all runs: ", all_required_pass, "\n", sep = "")
cat("All config consistency checks pass:   ", all_config_pass, "\n", sep = "")
cat("Feature guardrail all pass:           ", feature_guardrail_all_pass,
    " (pass=", n_guardrail_pass, ", fail=", n_guardrail_fail, ", missing=", n_guardrail_missing, ")\n\n", sep = "")

cat("Key outputs written:\n")
cat("  - ", file.path(batch_dir, "validation_required_files.csv"), "\n", sep = "")
cat("  - ", file.path(batch_dir, "validation_config_checks.csv"), "\n", sep = "")
cat("  - ", file.path(batch_dir, "validation_feature_guardrail.csv"), "\n", sep = "")
cat("  - ", file.path(batch_dir, "ablation_summary_table.csv"), "\n", sep = "")
cat("  - ", file.path(batch_dir, "ablation_leaderboard.csv"), " (combined)\n", sep = "")
cat("  - ", file.path(batch_dir, "ablation_leaderboard_by_recall.csv"), " (operational)\n", sep = "")
cat("  - ", file.path(batch_dir, "ablation_leaderboard_by_pr_auc.csv"), " (scientific)\n", sep = "")
cat("  - ", report_path, "\n\n", sep = "")

if (nrow(leaderboard) > 0) {
  cat("Leaderboard preview:\n")
  print(
    leaderboard %>%
      select(
        ablation_name,
        n_features_used,
        pr_auc_test,
        roc_auc_test,
        default_method,
        test_precision_at_policy,
        test_recall_at_policy,
        test_f1_at_policy,
        test_alerts_per_day_at_policy,
        deployment_ready
      ),
    n = min(10, nrow(leaderboard))
  )
}

cat("\nDone.\n")