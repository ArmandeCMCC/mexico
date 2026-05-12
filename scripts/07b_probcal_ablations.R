# 07b_probcal_ablations.R
# Orchestrator for probability-calibrated ablation studies
#
# Runs 05b_binary_threshold_eval_ablation.R once per ablation spec,
# passing environment variables for configuration.
# Collects run IDs and paths into a manifest CSV.
#
# Usage:
#   Rscript scripts/07b_probcal_ablations.R [ablation_names] [--dry-run] [--bootstrap-n N]
#
# Examples:
#   Rscript scripts/07b_probcal_ablations.R                      # Run all forecast ablations
#   Rscript scripts/07b_probcal_ablations.R full_model ntl_only  # Run specific ablations
#   Rscript scripts/07b_probcal_ablations.R --dry-run            # Preview without running
#   Rscript scripts/07b_probcal_ablations.R --bootstrap-n 100    # Override bootstrap iterations
#
# Dependencies: _ablation_helpers.R, 05b_binary_threshold_eval_ablation.R

# PATH DETECTION

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (dir.exists("data") || dir.exists("scripts")) return(normalizePath(getwd()))
  warning("Could not determine project directory. Using current directory.")
  normalizePath(getwd())
}

project_dir <- detect_project_dir()
SCRIPT_DIR <- file.path(project_dir, "scripts")

ABLATION_SCRIPT <- file.path(SCRIPT_DIR, "05b_binary_threshold_eval_ablation.R")
HELPERS_SCRIPT <- file.path(SCRIPT_DIR, "_ablation_helpers.R")

DATA_DIR <- file.path(project_dir, "data")
BATCH_ROOT <- file.path(DATA_DIR, "baselines", "binary_threshold", "ablation_batches")
FEATURES_PATH <- file.path(DATA_DIR, "model_ready", "features_engineered.rds")

# Default bootstrap iterations
DEFAULT_BOOTSTRAP_N <- 100

# PARSE COMMAND LINE ARGUMENTS

args <- commandArgs(trailingOnly = TRUE)

dry_run <- "--dry-run" %in% args
args <- args[args != "--dry-run"]

bootstrap_n <- DEFAULT_BOOTSTRAP_N
if ("--bootstrap-n" %in% args) {
  idx <- which(args == "--bootstrap-n")
  if (idx < length(args)) {
    bootstrap_n_raw <- args[idx + 1]
    bootstrap_n <- suppressWarnings(as.integer(bootstrap_n_raw))
    if (is.na(bootstrap_n) || bootstrap_n <= 0) {
      stop("Invalid --bootstrap-n value: '", bootstrap_n_raw, "'. Must be a positive integer.")
    }
    args <- args[-c(idx, idx + 1)]
  }
}

# Remaining args are ablation names (if any)
requested_ablations <- if (length(args) > 0) args else NULL

# LOAD HELPERS

if (!file.exists(HELPERS_SCRIPT)) {
  stop("Cannot find helpers script: ", HELPERS_SCRIPT)
}
source(HELPERS_SCRIPT)

if (!file.exists(ABLATION_SCRIPT)) {
  stop("Cannot find ablation script: ", ABLATION_SCRIPT)
}

if (!file.exists(FEATURES_PATH)) {
  stop("Cannot find features file: ", FEATURES_PATH)
}

# Use headline policy from helpers
REFERENCE_POLICY_TYPE <- HEADLINE_POLICY$type
REFERENCE_POLICY_TARGET <- HEADLINE_POLICY$target

# SELECT ABLATIONS TO RUN

all_ablation_names <- names(ABLATION_SPECS)

if (is.null(requested_ablations)) {
  # Default: run all forecast-mode ablations
  ablations_to_run <- all_ablation_names[
    vapply(ABLATION_SPECS, function(s) s$task_mode == "forecast", logical(1))
  ]
  cat("No ablations specified. Running all forecast-mode ablations.\n")
} else {
  # Validate requested ablations
  invalid <- setdiff(requested_ablations, all_ablation_names)
  if (length(invalid) > 0) {
    stop("Unknown ablation(s): ", paste(invalid, collapse = ", "),
         "\nAvailable: ", paste(all_ablation_names, collapse = ", "))
  }
  ablations_to_run <- requested_ablations
}

# CREATE BATCH FOLDER

batch_timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
batch_dir <- file.path(BATCH_ROOT, batch_timestamp)
dir.create(batch_dir, showWarnings = FALSE, recursive = TRUE)

manifest_path <- file.path(batch_dir, "ablation_manifest.csv")

cat("\n")
cat("=======================================================================\n")
cat("07b_probcal_ablations.R - Probability Calibration Ablation Orchestrator\n")
cat("=======================================================================\n\n")

cat("Configuration:\n")
cat("  Project directory: ", project_dir, "\n")
cat("  Ablation script: ", ABLATION_SCRIPT, "\n")
cat("  Features file: ", FEATURES_PATH, "\n")
cat("  Reference policy: ", REFERENCE_POLICY_TYPE, " = ", REFERENCE_POLICY_TARGET, "\n")
cat("  Bootstrap N: ", bootstrap_n, "\n")
cat("  Dry run: ", dry_run, "\n")
cat("  Batch directory: ", batch_dir, "\n")
cat("  Manifest path: ", manifest_path, "\n")
cat("  Ablations to run: ", paste(ablations_to_run, collapse = ", "), "\n\n")

# PREVIEW ABLATIONS

cat("Ablation preview:\n")
cat(sprintf("  %-25s %-10s %-10s %s\n", "Name", "Mode", "Strict", "Families"))
cat(sprintf("  %-25s %-10s %-10s %s\n", "----", "----", "------", "--------"))

for (abl_name in ablations_to_run) {
  spec <- ABLATION_SPECS[[abl_name]]
  strict_arg <- if (spec$task_mode == "forecast") "strict" else "standard"
  families_str <- paste(spec$include_families, collapse = ", ")
  if (nchar(families_str) > 40) {
    families_str <- paste0(substr(families_str, 1, 37), "...")
  }
  cat(sprintf("  %-25s %-10s %-10s %s\n", abl_name, spec$task_mode, strict_arg, families_str))
}
cat("\n")

if (dry_run) {
  cat("DRY RUN MODE - showing commands that would be executed:\n\n")
}

# RUN ABLATIONS

manifest <- data.frame(
  ablation_name = character(),
  task_mode = character(),
  strict_mode = character(),
  families = character(),
  include_regex = character(),
  exclude_regex = character(),
  reference_policy_type = character(),
  reference_policy_target = numeric(),
  run_id = character(),
  output_dir = character(),
  status = character(),
  start_time = character(),
  end_time = character(),
  duration_min = numeric(),
  n_features_raw = integer(),
  n_features_eligible = integer(),
  n_features_used = integer(),
  roc_auc_test = numeric(),
  pr_auc_test = numeric(),
  recall_at_policy = numeric(),
  stringsAsFactors = FALSE
)

for (i in seq_along(ablations_to_run)) {
  abl_name <- ablations_to_run[i]
  spec <- get_ablation_spec(abl_name)

  # Determine strict mode based on task mode
  strict_arg <- if (spec$task_mode == "forecast") "strict" else "standard"
  families_str <- paste(spec$families, collapse = ", ")

  cat("-----------------------------------------------------------------------\n")
  cat(sprintf("[%d/%d] Ablation: %s\n", i, length(ablations_to_run), abl_name))
  cat("-----------------------------------------------------------------------\n")
  cat("  Description: ", spec$description, "\n")
  cat("  Task mode: ", spec$task_mode, "\n")
  cat("  Strict mode: ", strict_arg, "\n")
  cat("  Families: ", families_str, "\n")
  cat("  Include regex: ", substr(spec$include_regex, 1, 70), if (nchar(spec$include_regex) > 70) "..." else "", "\n")
  if (spec$exclude_regex != "") {
    cat("  Exclude regex: ", spec$exclude_regex, "\n")
  }
  cat("\n")

  # Preflight audit: verify ablation will have features after filtering
  audit <- audit_ablation(abl_name, FEATURES_PATH)
  cat("  Preflight audit:\n")
  cat("    Raw matched features: ", audit$n_raw_matched, "\n")
  cat("    Preliminary eligible matched: ", audit$n_prelim_eligible_matched, "\n")

  if (audit$n_prelim_eligible_matched == 0) {
    cat("  ERROR: Ablation '", abl_name, "' has 0 preliminary eligible features after strict filtering.\n")
    cat("  Skipping this ablation.\n\n")
    manifest <- rbind(manifest, data.frame(
      ablation_name = abl_name,
      task_mode = spec$task_mode,
      strict_mode = strict_arg,
      families = families_str,
      include_regex = spec$include_regex,
      exclude_regex = spec$exclude_regex,
      reference_policy_type = REFERENCE_POLICY_TYPE,
      reference_policy_target = REFERENCE_POLICY_TARGET,
      run_id = NA_character_,
      output_dir = NA_character_,
      status = "failed_preflight",
      start_time = NA_character_,
      end_time = NA_character_,
      duration_min = NA_real_,
      n_features_raw = audit$n_raw_matched,
      n_features_eligible = audit$n_prelim_eligible_matched,
      n_features_used = NA_integer_,
      roc_auc_test = NA_real_,
      pr_auc_test = NA_real_,
      recall_at_policy = NA_real_,
      stringsAsFactors = FALSE
    ))
    # Save manifest immediately on preflight failure
    write.csv(manifest, manifest_path, row.names = FALSE)
    next
  }
  cat("\n")

  # Build environment variable list - ALWAYS set all vars to prevent leakage
  env_vars <- list(
    ABLATION_NAME = abl_name,
    REFERENCE_POLICY_TYPE = REFERENCE_POLICY_TYPE,
    REFERENCE_POLICY_TARGET = as.character(REFERENCE_POLICY_TARGET),
    BOOTSTRAP_N = as.character(bootstrap_n),
    FEATURE_INCLUDE_REGEX = spec$include_regex,
    FEATURE_EXCLUDE_REGEX = "",
    OUTPUT_ROOT_OVERRIDE = batch_dir,
    FEATURES_PATH_OVERRIDE = ""
  )

  # Override exclude regex if spec has one
  if (spec$exclude_regex != "") {
    env_vars$FEATURE_EXCLUDE_REGEX <- spec$exclude_regex
  }

  # Print command preview
  cat("  Environment variables:\n")
  for (nm in names(env_vars)) {
    val <- env_vars[[nm]]
    if (nchar(val) > 60) val <- paste0(substr(val, 1, 57), "...")
    cat(sprintf("    %s=%s\n", nm, val))
  }
  cat("\n  Command:\n")
  cat(sprintf("    Rscript %s %s %s %d\n\n", ABLATION_SCRIPT, spec$task_mode, strict_arg, bootstrap_n))

  if (dry_run) {
    manifest <- rbind(manifest, data.frame(
      ablation_name = abl_name,
      task_mode = spec$task_mode,
      strict_mode = strict_arg,
      families = families_str,
      include_regex = spec$include_regex,
      exclude_regex = spec$exclude_regex,
      reference_policy_type = REFERENCE_POLICY_TYPE,
      reference_policy_target = REFERENCE_POLICY_TARGET,
      run_id = "(dry-run)",
      output_dir = "(dry-run)",
      status = "skipped",
      start_time = NA_character_,
      end_time = NA_character_,
      duration_min = NA_real_,
      n_features_raw = audit$n_raw_matched,
      n_features_eligible = audit$n_prelim_eligible_matched,
      n_features_used = NA_integer_,
      roc_auc_test = NA_real_,
      pr_auc_test = NA_real_,
      recall_at_policy = NA_real_,
      stringsAsFactors = FALSE
    ))
    # Save manifest in dry-run mode
    write.csv(manifest, manifest_path, row.names = FALSE)
    cat("  [DRY RUN] Skipping execution\n\n")
    next
  }

  # Execute: save old env, set new env, run, restore old env
  start_time <- Sys.time()
  cat("  Starting at: ", format(start_time), "\n")

  # Save old environment values
  env_names <- names(env_vars)
  old_env <- Sys.getenv(env_names, unset = NA)

  result <- tryCatch({
    # Set environment variables
    do.call(Sys.setenv, env_vars)

    # Run script via system2
    output <- system2(
      "Rscript",
      args = c(ABLATION_SCRIPT, spec$task_mode, strict_arg, as.character(bootstrap_n)),
      stdout = TRUE,
      stderr = TRUE
    )
    exit_code <- attr(output, "status")
    if (is.null(exit_code)) exit_code <- 0

    list(success = (exit_code == 0), output = output, exit_code = exit_code)
  }, error = function(e) {
    list(success = FALSE, output = as.character(e), exit_code = -1)
  })

  # Restore old environment
  for (j in seq_along(env_names)) {
    if (is.na(old_env[j])) {
      Sys.unsetenv(env_names[j])
    } else {
      do.call(Sys.setenv, setNames(list(old_env[j]), env_names[j]))
    }
  }

  end_time <- Sys.time()
  duration_min <- as.numeric(difftime(end_time, start_time, units = "mins"))

  if (result$success) {
    cat("  Completed at: ", format(end_time), "\n")
    cat("  Duration: ", round(duration_min, 2), " minutes\n")

    # Extract run_id from output
    run_id_line <- grep("^Run ID:", result$output, value = TRUE)
    run_id <- if (length(run_id_line) > 0) {
      trimws(gsub("^Run ID:", "", run_id_line[1]))
    } else {
      # Fallback: construct from convention
      paste0(format(start_time, "%Y%m%d_%H%M%S"), "_", spec$task_mode, "_", strict_arg, "_", abl_name)
    }

    output_dir <- file.path(batch_dir, run_id)

    # Read metrics from output files
    n_features_used <- NA_integer_
    roc_auc_test <- NA_real_
    pr_auc_test <- NA_real_
    recall_at_policy <- NA_real_
    default_method <- "platt"  # fallback

    # Read n_features and default_calibrated_method from run_config.json
    config_file <- file.path(output_dir, "run_config.json")
    if (file.exists(config_file)) {
      config <- jsonlite::fromJSON(config_file)
      n_features_used <- config$model$n_features
      if (!is.null(config$calibration$default_calibrated_method)) {
        default_method <- config$calibration$default_calibrated_method
      }
    }

    # Read ROC/PR AUC from metrics_summary.csv (test row)
    metrics_file <- file.path(output_dir, "metrics_summary.csv")
    if (file.exists(metrics_file)) {
      metrics <- read.csv(metrics_file, stringsAsFactors = FALSE)
      test_row <- metrics[metrics$split == "test", ]
      if (nrow(test_row) > 0) {
        roc_auc_test <- test_row$roc_auc[1]
        pr_auc_test <- test_row$pr_auc[1]
      }
    }

    # Read recall at policy from operating_thresholds_table.csv
    ot_file <- file.path(output_dir, "operating_thresholds_table.csv")
    if (file.exists(ot_file)) {
      ot <- read.csv(ot_file, stringsAsFactors = FALSE)

      # Try default method first
      policy_row <- ot[ot$method == default_method &
                       ot$policy_type == REFERENCE_POLICY_TYPE &
                       abs(ot$policy_target - REFERENCE_POLICY_TARGET) < 0.001, ]

      # Fallback: any method matching the policy if default method not found
      if (nrow(policy_row) == 0) {
        policy_row <- ot[ot$policy_type == REFERENCE_POLICY_TYPE &
                         abs(ot$policy_target - REFERENCE_POLICY_TARGET) < 0.001, ]
      }

      if (nrow(policy_row) > 0) {
        recall_at_policy <- policy_row$test_recall[1]
      }
    }

    manifest <- rbind(manifest, data.frame(
      ablation_name = abl_name,
      task_mode = spec$task_mode,
      strict_mode = strict_arg,
      families = families_str,
      include_regex = spec$include_regex,
      exclude_regex = spec$exclude_regex,
      reference_policy_type = REFERENCE_POLICY_TYPE,
      reference_policy_target = REFERENCE_POLICY_TARGET,
      run_id = run_id,
      output_dir = output_dir,
      status = "success",
      start_time = format(start_time),
      end_time = format(end_time),
      duration_min = round(duration_min, 2),
      n_features_raw = audit$n_raw_matched,
      n_features_eligible = audit$n_prelim_eligible_matched,
      n_features_used = n_features_used,
      roc_auc_test = roc_auc_test,
      pr_auc_test = pr_auc_test,
      recall_at_policy = recall_at_policy,
      stringsAsFactors = FALSE
    ))

    cat("  Status: SUCCESS\n")
    cat("  Run ID: ", run_id, "\n")
    cat("  Features used: ", n_features_used, "\n")
    cat("  Output dir: ", output_dir, "\n\n")

  } else {
    cat("  FAILED after ", round(duration_min, 2), " minutes\n")
    cat("  Exit code: ", result$exit_code, "\n")
    cat("  Last 20 lines of output:\n")
    last_lines <- tail(result$output, 20)
    cat(paste("    ", last_lines, collapse = "\n"), "\n\n")

    manifest <- rbind(manifest, data.frame(
      ablation_name = abl_name,
      task_mode = spec$task_mode,
      strict_mode = strict_arg,
      families = families_str,
      include_regex = spec$include_regex,
      exclude_regex = spec$exclude_regex,
      reference_policy_type = REFERENCE_POLICY_TYPE,
      reference_policy_target = REFERENCE_POLICY_TARGET,
      run_id = NA_character_,
      output_dir = NA_character_,
      status = "failed",
      start_time = format(start_time),
      end_time = format(end_time),
      duration_min = round(duration_min, 2),
      n_features_raw = audit$n_raw_matched,
      n_features_eligible = audit$n_prelim_eligible_matched,
      n_features_used = NA_integer_,
      roc_auc_test = NA_real_,
      pr_auc_test = NA_real_,
      recall_at_policy = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  # Save manifest after each run (in case of interruption)
  write.csv(manifest, manifest_path, row.names = FALSE)
}

# SUMMARY

cat("\n=======================================================================\n")
cat("ABLATION ORCHESTRATION COMPLETE\n")
cat("=======================================================================\n\n")

cat("Summary:\n")
cat("  Total ablations: ", nrow(manifest), "\n")
cat("  Successful: ", sum(manifest$status == "success"), "\n")
cat("  Failed: ", sum(manifest$status == "failed"), "\n")
cat("  Failed preflight: ", sum(manifest$status == "failed_preflight"), "\n")
cat("  Skipped (dry-run): ", sum(manifest$status == "skipped"), "\n")
cat("  Batch directory: ", batch_dir, "\n")

if (any(manifest$status == "success")) {
  cat("\nSuccessful runs:\n")
  success_df <- manifest[manifest$status == "success",
                         c("ablation_name", "n_features_used", "roc_auc_test", "recall_at_policy", "duration_min")]
  print(success_df, row.names = FALSE)
}

if (any(manifest$status %in% c("failed", "failed_preflight"))) {
  cat("\nFailed runs:\n")
  failed_df <- manifest[manifest$status %in% c("failed", "failed_preflight"),
                        c("ablation_name", "status", "n_features_eligible")]
  print(failed_df, row.names = FALSE)
}

cat("\nManifest saved to: ", manifest_path, "\n")
cat("\nNext step: Run 08b_probcal_ablation_validation.R to validate and aggregate results.\n")
cat("  Rscript scripts/08b_probcal_ablation_validation.R ", batch_dir, "\n")
