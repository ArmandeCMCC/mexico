# run_full_pipeline.R
# Orchestrator for Approach 2: Anomaly Detection Pipeline
#
# Runs all steps in sequence:
#   01_train_forecaster.R   - Train seasonal + AR(1) model on TRAIN only
#   02_score_anomalies.R    - Score all periods using TRAIN parameters
#   03_integrate_features.R - Join anomaly features to model-ready data
#   04_train_combined_model.R - Train baseline, anomaly-only, combined models
#   05_evaluate_combined.R  - Comprehensive evaluation and comparison
#
# Extended evaluation modes:
#   07_learning_to_rank_refactor.R - LTR with reusable functions
#   08_loso_eval.R          - Leave-One-State-Out cross-validation
#   09_temporal_windows_eval.R - Temporal robustness testing
#
# Usage:
#   Rscript run_full_pipeline.R                    # Main pipeline only
#   Rscript run_full_pipeline.R --skip-training    # Skip steps 1-3
#   Rscript run_full_pipeline.R --mode=main        # Main pipeline (default)
#   Rscript run_full_pipeline.R --mode=ltr         # Main + LTR
#   Rscript run_full_pipeline.R --mode=loso        # LOSO evaluation only
#   Rscript run_full_pipeline.R --mode=temporal    # Temporal windows only
#   Rscript run_full_pipeline.R --mode=extended    # Main + LTR + LOSO + temporal
#   Rscript run_full_pipeline.R --mode=all         # Everything
#

library(tidyverse)

# CONFIGURATION

# Parse command line args
args <- commandArgs(trailingOnly = TRUE)
SKIP_TRAINING <- "--skip-training" %in% args

# Parse mode flag
mode_arg <- grep("^--mode=", args, value = TRUE)
if (length(mode_arg) > 0) {
  RUN_MODE <- sub("^--mode=", "", mode_arg[1])
} else {
  RUN_MODE <- "main"  # Default: main pipeline only
}

# Validate mode
valid_modes <- c("main", "ltr", "loso", "temporal", "extended", "all")
if (!RUN_MODE %in% valid_modes) {
  stop(sprintf("Invalid mode '%s'. Valid modes: %s", RUN_MODE, paste(valid_modes, collapse = ", ")))
}

# Set working directory to project root
# Handle both interactive and Rscript modes
tryCatch({
  script_dir <- dirname(sys.frame(1)$ofile)
  if (!is.null(script_dir) && nchar(script_dir) > 0) {
    project_root <- normalizePath(file.path(script_dir, "../.."))
    setwd(project_root)
  }
}, error = function(e) {
  # If running via Rscript, try commandArgs
  args <- commandArgs(trailingOnly = FALSE)
  script_idx <- grep("--file=", args)
  if (length(script_idx) > 0) {
    script_path <- sub("--file=", "", args[script_idx])
    script_dir <- dirname(script_path)
    project_root <- normalizePath(file.path(script_dir, "../.."))
    setwd(project_root)
  }
})

cat("================================================================================\n")
cat("APPROACH 2: ANOMALY DETECTION PIPELINE\n")
cat("================================================================================\n")
cat(sprintf("Working directory: %s\n", getwd()))
cat(sprintf("Started: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(sprintf("Mode: %s\n", RUN_MODE))
cat(sprintf("Skip training steps: %s\n", SKIP_TRAINING))
cat("================================================================================\n\n")

# Determine which steps to run based on mode
RUN_MAIN <- RUN_MODE %in% c("main", "ltr", "extended", "all")
RUN_LTR <- RUN_MODE %in% c("ltr", "extended", "all")
RUN_LOSO <- RUN_MODE %in% c("loso", "extended", "all")
RUN_TEMPORAL <- RUN_MODE %in% c("temporal", "extended", "all")

# Track timing
timings <- list()
overall_start <- Sys.time()

# HELPER: RUN SCRIPT WITH TIMING

run_script <- function(script_name, description) {
  script_path <- file.path("scripts/approach2_anomaly", script_name)

  if (!file.exists(script_path)) {
    stop(sprintf("Script not found: %s", script_path))
  }

  cat(sprintf("\n%s\n", strrep("=", 60)))
  cat(sprintf("STEP: %s\n", description))
  cat(sprintf("Script: %s\n", script_name))
  cat(sprintf("Sourcing: %s\n", normalizePath(script_path)))
  cat(sprintf("%s\n\n", strrep("=", 60)))

  start_time <- Sys.time()

  # Source the script
  tryCatch({
    source(script_path, local = new.env())
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    cat(sprintf("\n[OK] %s completed in %.1f seconds\n", script_name, elapsed))
    return(elapsed)
  }, error = function(e) {
    cat(sprintf("\n[ERROR] %s failed: %s\n", script_name, e$message))
    stop(e)
  })
}

# MAIN PIPELINE EXECUTION (steps 1-5)

# Run main pipeline for modes: main, ltr, extended, all
if (RUN_MAIN) {

  if (!SKIP_TRAINING) {
    # Step 1: Train forecaster
    timings$train_forecaster <- run_script(
      "01_train_forecaster.R",
      "Train forecasting model (seasonal + AR(1)) on TRAIN period"
    )

    # Step 2: Score anomalies
    timings$score_anomalies <- run_script(
      "02_score_anomalies.R",
      "Generate anomaly scores for all periods"
    )

    # Step 3: Integrate features
    timings$integrate_features <- run_script(
      "03_integrate_features.R",
      "Join anomaly features to model-ready dataset"
    )
  } else {
    cat("\n[SKIPPED] Steps 1-3 (--skip-training flag set)\n")
    cat("Using existing anomaly features from data/processed/anomaly_features/\n")
  }

  # Step 4: Train combined model
  timings$train_model <- run_script(
    "04_train_combined_model.R",
    "Train baseline, anomaly-only, and combined models"
  )

  # Step 5: Evaluate
  timings$evaluate <- run_script(
    "05_evaluate_combined.R",
    "Comprehensive evaluation and comparison"
  )

} else {
  cat(sprintf("\n[SKIPPED] Main pipeline steps 1-5 (mode=%s)\n", RUN_MODE))
  cat("Extended evaluation modes use existing model-ready data.\n")
}

# EXTENDED EVALUATION: LTR (step 7)

if (RUN_LTR) {
  cat("\n")
  cat(strrep("=", 60), "\n")
  cat("EXTENDED: Learning-to-Rank Evaluation\n")
  cat(strrep("=", 60), "\n")

  timings$ltr <- run_script(
    "07_learning_to_rank_refactor.R",
    "Learning-to-Rank model training and evaluation"
  )
}

# EXTENDED EVALUATION: LOSO (step 8)

if (RUN_LOSO) {
  cat("\n")
  cat(strrep("=", 60), "\n")
  cat("EXTENDED: Leave-One-State-Out Cross-Validation\n")
  cat(strrep("=", 60), "\n")

  timings$loso <- run_script(
    "08_loso_eval.R",
    "Leave-One-State-Out cross-validation for spatial generalization"
  )
}

# EXTENDED EVALUATION: TEMPORAL WINDOWS (step 9)

if (RUN_TEMPORAL) {
  cat("\n")
  cat(strrep("=", 60), "\n")
  cat("EXTENDED: Temporal Windows Evaluation\n")
  cat(strrep("=", 60), "\n")

  timings$temporal <- run_script(
    "09_temporal_windows_eval.R",
    "Temporal robustness across COVID, seasonal, and yearly windows"
  )
}

# SUMMARY

overall_elapsed <- as.numeric(difftime(Sys.time(), overall_start, units = "secs"))

cat("\n")
cat(strrep("=", 60), "\n")
cat("PIPELINE COMPLETE\n")
cat(strrep("=", 60), "\n")
cat(sprintf("Mode: %s\n", RUN_MODE))
cat(sprintf("Total time: %.1f seconds (%.1f minutes)\n",
            overall_elapsed, overall_elapsed / 60))
cat("\nStep timings:\n")
for (step in names(timings)) {
  cat(sprintf("  %-25s %.1f s\n", step, timings[[step]]))
}

# Print location of outputs
cat("\nOutputs:\n")
cat("  Anomaly features: data/processed/anomaly_features/\n")
cat("  Combined model-ready: data/processed/model_ready_anomaly/\n")

# Get latest run tags
if (file.exists("runs/anomaly/latest_run_tag.txt")) {
  anomaly_tag <- readLines("runs/anomaly/latest_run_tag.txt", warn = FALSE)[1]
  cat(sprintf("  Forecaster artifacts: runs/anomaly/%s/\n", anomaly_tag))
}

if (file.exists("runs/combined/latest_run_tag.txt")) {
  combined_tag <- readLines("runs/combined/latest_run_tag.txt", warn = FALSE)[1]
  cat(sprintf("  Model artifacts: runs/combined/%s/\n", combined_tag))
  cat(sprintf("  Evaluation: runs/combined/%s/evaluation/\n", combined_tag))
  cat(sprintf("\nTo view the evaluation report:\n"))
  cat(sprintf("  cat runs/combined/%s/evaluation/evaluation_report.txt\n", combined_tag))
}

# Extended evaluation outputs (in runs/ltr/)
if (file.exists("runs/ltr/latest_run_tag.txt")) {
  ltr_tag <- readLines("runs/ltr/latest_run_tag.txt", warn = FALSE)[1]

  if (RUN_LTR) {
    cat(sprintf("\n  LTR results: runs/ltr/%s/\n", ltr_tag))
    cat(sprintf("    cat runs/ltr/%s/evaluation_report.txt\n", ltr_tag))
  }
  if (RUN_LOSO && dir.exists(file.path("runs/ltr", ltr_tag, "loso"))) {
    cat(sprintf("  LOSO results: runs/ltr/%s/loso/\n", ltr_tag))
  }
  if (RUN_TEMPORAL && dir.exists(file.path("runs/ltr", ltr_tag, "time_windows"))) {
    cat(sprintf("  Temporal results: runs/ltr/%s/time_windows/\n", ltr_tag))
  }
}

# Show available modes
cat("\nAvailable modes:\n")
cat("  --mode=main       Main pipeline only (steps 1-5)\n")
cat("  --mode=ltr        Main + Learning-to-Rank\n")
cat("  --mode=loso       LOSO cross-validation only\n")
cat("  --mode=temporal   Temporal windows only\n")
cat("  --mode=extended   Main + LTR + LOSO + Temporal\n")
cat("  --mode=all        Everything\n")

cat("\n")
cat(strrep("=", 60), "\n")
cat(sprintf("Finished: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(strrep("=", 60), "\n")
