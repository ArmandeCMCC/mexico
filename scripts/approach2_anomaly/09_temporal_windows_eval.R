# 09_temporal_windows_eval.R
# Multiple Temporal Test Windows Evaluation
#
# Tests model robustness across different time periods to assess
# temporal stability and detect potential concept drift.
#
# WINDOW DEFINITIONS (per user specification):
#   Window1: TEST 2020-03-01 to 2020-06-30 (COVID shock period)
#   Window2: TEST 2020-07-01 to 2020-12-31 (Post-COVID recovery)
#   Window3: TEST 2021-01-01 to 2021-12-31 (Full year after shock)
#
# SPLIT LOGIC:
#   For each window:
#     TRAIN: all data strictly before test_start
#     VAL:   6 months immediately before test_start (if available),
#            else smallest available pre-window block; fallback documented
#     TEST:  window dates
#
# Outputs are written to runs/ltr/<RUN_TAG>/time_windows/

# Source utilities only (no execution)
source("scripts/approach2_anomaly/ltr_utils.R")

# CONFIGURATION

args <- commandArgs(trailingOnly = TRUE)
args <- args[!grepl("^--", args)]

MODEL_READY_RUN_TAG <- if (length(args) > 0) args[1] else NULL
RUN_TAG <- format(Sys.time(), "%Y%m%d_%H%M%S")

# Training sample fraction - reduce for memory constraints
TRAIN_SAMPLE_FRAC <- 0.4

# Output directory
OUTPUT_DIR <- file.path("runs/ltr", RUN_TAG, "time_windows")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("=== 09_temporal_windows_eval.R ===\n")
cat("Multiple Temporal Test Windows Evaluation\n")
cat(sprintf("Run tag: %s\n", RUN_TAG))
cat(sprintf("Output directory: %s\n", OUTPUT_DIR))

# 1. LOAD DATA

cat("\n=== Loading Data ===\n")

data <- load_model_ready_data(MODEL_READY_RUN_TAG)

# Combine all data
all_data <- bind_rows(
  data$train,
  data$val,
  data$test
)

cat(sprintf("Model-ready data: %s\n", data$data_path))
cat(sprintf("Total data: %s rows\n", format(nrow(all_data), big.mark = ",")))
cat(sprintf("Date range: %s to %s\n", min(all_data$date), max(all_data$date)))

# Data availability check
data_min_date <- min(all_data$date)
data_max_date <- max(all_data$date)

# 2. DEFINE TEMPORAL WINDOWS

cat("\n=== Defining Temporal Windows ===\n")

# Window definitions per specification
windows <- list(
  window1_covid_shock = list(
    name = "COVID Shock (2020-Q2)",
    test_start = as.Date("2020-03-01"),
    test_end = as.Date("2020-06-30"),
    description = "First wave lockdown period - structural break"
  ),
  window2_post_covid = list(
    name = "Post-COVID (2020-H2)",
    test_start = as.Date("2020-07-01"),
    test_end = as.Date("2020-12-31"),
    description = "Recovery and adaptation period"
  ),
  window3_year_2021 = list(
    name = "Year 2021",
    test_start = as.Date("2021-01-01"),
    test_end = as.Date("2021-12-31"),
    description = "Full year after initial COVID shock"
  )
)

# Compute VAL period for each window
# VAL = 6 months before test_start, or fallback to available data
compute_val_period <- function(test_start, data_min, ideal_months = 6) {
  # Ideal VAL: 6 months before test
  ideal_val_start <- test_start %m-% months(ideal_months)
  ideal_val_end <- test_start - 1

  # Actual VAL: constrained by data availability and train requirement
  # Need at least some data before VAL for training
  min_train_date <- data_min + 365  # At least 1 year for training

  if (ideal_val_start >= min_train_date) {
    # Full 6-month VAL available
    return(list(
      val_start = ideal_val_start,
      val_end = ideal_val_end,
      fallback = FALSE,
      fallback_reason = NA
    ))
  } else if (ideal_val_end >= min_train_date) {
    # Partial VAL: use whatever is available
    return(list(
      val_start = min_train_date,
      val_end = ideal_val_end,
      fallback = TRUE,
      fallback_reason = sprintf("Shortened VAL: only %d days available",
                                 as.integer(ideal_val_end - min_train_date + 1))
    ))
  } else {
    # No VAL possible
    return(list(
      val_start = NA,
      val_end = NA,
      fallback = TRUE,
      fallback_reason = "No VAL period available - using NULL for early stopping"
    ))
  }
}

# Print window summary
cat("\nTemporal windows:\n")
for (w_name in names(windows)) {
  w <- windows[[w_name]]

  # Check if window is within data range
  if (w$test_end > data_max_date) {
    cat(sprintf("  %s: SKIPPED (test_end %s > data_max %s)\n",
                w$name, w$test_end, data_max_date))
    windows[[w_name]]$skip <- TRUE
    next
  }

  # Compute VAL period
  val_info <- compute_val_period(w$test_start, data_min_date)
  windows[[w_name]]$val_start <- val_info$val_start
  windows[[w_name]]$val_end <- val_info$val_end
  windows[[w_name]]$val_fallback <- val_info$fallback
  windows[[w_name]]$val_fallback_reason <- val_info$fallback_reason
  windows[[w_name]]$skip <- FALSE

  # TRAIN ends before VAL (or before test if no VAL)
  train_end <- if (!is.na(val_info$val_start)) val_info$val_start - 1 else w$test_start - 1
  windows[[w_name]]$train_end <- train_end

  cat(sprintf("  %s:\n", w$name))
  cat(sprintf("    TEST:  %s to %s\n", w$test_start, w$test_end))
  if (!is.na(val_info$val_start)) {
    cat(sprintf("    VAL:   %s to %s%s\n",
                val_info$val_start, val_info$val_end,
                if (val_info$fallback) " (fallback)" else ""))
  } else {
    cat(sprintf("    VAL:   None (%s)\n", val_info$fallback_reason))
  }
  cat(sprintf("    TRAIN: %s to %s\n", data_min_date, train_end))
}

# 3. GET FEATURES

cat("\n=== Defining Features ===\n")

feature_cols <- get_ltr_features(all_data, mode = "operational")
cat(sprintf("Features: %d\n", length(feature_cols)))

# 4. EVALUATE EACH WINDOW

cat("\n=== Evaluating Temporal Windows ===\n")

K_VALUES <- c(1, 2, 5, 10, 20, 50)

window_results <- list()
window_start_time <- Sys.time()

for (w_name in names(windows)) {
  w <- windows[[w_name]]

  if (isTRUE(w$skip)) {
    cat(sprintf("\n--- %s: SKIPPED ---\n", w$name))
    next
  }

  cat(sprintf("\n--- %s ---\n", w$name))
  cat(sprintf("    %s\n", w$description))

  # Create splits with sampling for memory efficiency
  set.seed(42)
  train_data <- all_data %>%
    filter(date <= w$train_end) %>%
    sample_frac(TRAIN_SAMPLE_FRAC)

  val_data <- if (!is.na(w$val_start)) {
    all_data %>%
      filter(date >= w$val_start, date <= w$val_end) %>%
      sample_frac(TRAIN_SAMPLE_FRAC)
  } else {
    NULL
  }
  gc(verbose = FALSE)

  test_data <- all_data %>% filter(date >= w$test_start, date <= w$test_end)

  n_train <- nrow(train_data)
  n_val <- if (!is.null(val_data)) nrow(val_data) else 0
  n_test <- nrow(test_data)
  n_test_days <- n_distinct(test_data$date)

  cat(sprintf("    TRAIN: %s rows (%.0f%% sampled, %d days)\n",
              format(n_train, big.mark = ","), TRAIN_SAMPLE_FRAC * 100, n_distinct(train_data$date)))
  if (!is.null(val_data)) {
    cat(sprintf("    VAL:   %s rows (%d days)\n",
                format(n_val, big.mark = ","), n_distinct(val_data$date)))
  }
  cat(sprintf("    TEST:  %s rows (%d days)\n",
              format(n_test, big.mark = ","), n_test_days))

  # Skip if insufficient data
  if (n_train < 1000 || n_test < 100 || n_test_days < 5) {
    cat("    SKIPPED: insufficient data\n")
    next
  }

  # Base rates
  base_rate_train <- mean(train_data$outage_3h_or_more)
  base_rate_test <- mean(test_data$outage_3h_or_more)
  cat(sprintf("    Base rate: train=%.4f, test=%.4f (ratio=%.2f)\n",
              base_rate_train, base_rate_test, base_rate_test / base_rate_train))

  # Garbage collection before training
  gc(verbose = FALSE)

  # Train LTR model
  ltr_result <- tryCatch({
    train_ltr(
      train_data = train_data,
      val_data = val_data,
      feature_cols = feature_cols,
      nrounds = 400,
      early_stopping = if (!is.null(val_data)) 40 else NULL,
      seed = 42,
      verbose = 0
    )
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", e$message))
    return(NULL)
  })

  # Clean up training data to free memory
  rm(train_data, val_data)
  gc(verbose = FALSE)

  if (is.null(ltr_result)) next

  cat(sprintf("    Training: %.1f sec, best_iter=%d\n",
              ltr_result$train_time_sec, ltr_result$best_iteration))

  # Predict and evaluate
  test_preds <- predict_ltr(ltr_result, test_data)
  eval_result <- evaluate_topk(test_preds, K_VALUES, allocation = "national", n_boot = 1000)

  # Store results
  window_results[[w_name]] <- list(
    topk = eval_result$topk_metrics %>%
      mutate(
        window = w_name,
        window_name = w$name,
        test_start = as.character(w$test_start),
        test_end = as.character(w$test_end),
        n_train = n_train,
        n_val = n_val,
        n_test = n_test,
        n_test_days = n_test_days,
        base_rate_train = base_rate_train,
        base_rate_test = base_rate_test,
        train_time_sec = ltr_result$train_time_sec,
        best_iteration = ltr_result$best_iteration,
        val_fallback = w$val_fallback
      ),
    bootstrap = eval_result$bootstrap_ci %>%
      mutate(window = w_name, window_name = w$name)
  )

  # Print key results
  k10 <- eval_result$topk_metrics %>% filter(k == 10)
  cat(sprintf("    Results@10: Hits=%d, Precision=%.4f, Lift=%.2f\n",
              k10$n_hits, k10$precision, k10$lift))

  # Save individual window results
  window_dir <- file.path(OUTPUT_DIR, paste0("window=", w_name))
  dir.create(window_dir, recursive = TRUE, showWarnings = FALSE)

  test_preds %>%
    select(GID_2, date, outage_3h_or_more, rank_score) %>%
    write_csv(gzfile(file.path(window_dir, "predictions.csv.gz")))

  write_csv(eval_result$topk_metrics, file.path(window_dir, "topk_metrics.csv"))
  write_csv(eval_result$bootstrap_ci, file.path(window_dir, "bootstrap_ci.csv"))

  # Save split info for reproducibility
  split_info <- tibble(
    split = c("train", "val", "test"),
    start_date = c(as.character(data_min_date),
                   as.character(w$val_start),
                   as.character(w$test_start)),
    end_date = c(as.character(w$train_end),
                 as.character(w$val_end),
                 as.character(w$test_end)),
    n_rows = c(n_train, n_val, n_test),
    n_days = c(NA, NA, n_test_days),  # train/val already cleaned up
    fallback = c(FALSE, w$val_fallback, FALSE)
  )
  write_csv(split_info, file.path(window_dir, "split_info.csv"))

  # Clean up to free memory
  rm(test_data, test_preds, ltr_result, eval_result)
  gc(verbose = FALSE)
}

window_elapsed <- as.numeric(difftime(Sys.time(), window_start_time, units = "secs"))
cat(sprintf("\n\nTemporal evaluation completed in %.1f seconds\n", window_elapsed))

# 5. AGGREGATE AND COMPARE

cat("\n=== Comparing Windows ===\n")

# Combine results
all_topk <- bind_rows(lapply(window_results, function(x) x$topk))
all_bootstrap <- bind_rows(lapply(window_results, function(x) x$bootstrap))

# Summary at k=10
window_summary <- all_topk %>%
  filter(k == 10) %>%
  select(window, window_name, precision, lift, n_hits, n_test_days,
         base_rate_train, base_rate_test, val_fallback) %>%
  arrange(window)  # window names are in chronological order

cat("\n--- Window Performance at k=10 ---\n")
print(window_summary %>%
        select(window_name, lift, n_hits, precision, base_rate_test) %>%
        mutate(across(where(is.numeric), ~round(., 4))))

# Stability analysis
if (nrow(window_summary) > 1) {
  lift_stats <- tibble(
    metric = "lift_at_10",
    mean = mean(window_summary$lift),
    sd = sd(window_summary$lift),
    cv = sd(window_summary$lift) / mean(window_summary$lift),
    min = min(window_summary$lift),
    max = max(window_summary$lift),
    range = max(window_summary$lift) - min(window_summary$lift)
  )

  cat("\n--- Stability Analysis ---\n")
  cat(sprintf("Lift@10 across windows:\n"))
  cat(sprintf("  Mean: %.2f\n", lift_stats$mean))
  cat(sprintf("  SD: %.2f\n", lift_stats$sd))
  cat(sprintf("  CV (SD/Mean): %.1f%%\n", lift_stats$cv * 100))
  cat(sprintf("  Range: %.2f to %.2f\n", lift_stats$min, lift_stats$max))

  stability_label <- case_when(
    lift_stats$cv < 0.10 ~ "STABLE",
    lift_stats$cv < 0.20 ~ "MODERATE",
    TRUE ~ "VARIABLE"
  )
  cat(sprintf("  Assessment: %s\n", stability_label))
}

# 6. SAVE AGGREGATED RESULTS

cat("\n=== Saving Aggregated Results ===\n")

# All window results
write_csv(all_topk, file.path(OUTPUT_DIR, "temporal_all_windows.csv"))
cat("  Saved: temporal_all_windows.csv\n")

# Summary at k=10
write_csv(window_summary, file.path(OUTPUT_DIR, "temporal_summary_k10.csv"))
cat("  Saved: temporal_summary_k10.csv\n")

# Bootstrap CIs
write_csv(all_bootstrap, file.path(OUTPUT_DIR, "temporal_bootstrap_ci.csv"))
cat("  Saved: temporal_bootstrap_ci.csv\n")

# Stability stats
if (exists("lift_stats")) {
  write_csv(lift_stats, file.path(OUTPUT_DIR, "temporal_stability.csv"))
  cat("  Saved: temporal_stability.csv\n")
}

# Metadata
metadata <- tibble(
  run_tag = RUN_TAG,
  model_ready_run = data$run_tag,
  n_windows_evaluated = length(window_results),
  n_features = length(feature_cols),
  elapsed_sec = window_elapsed,
  data_min_date = as.character(data_min_date),
  data_max_date = as.character(data_max_date),
  lift_k10_mean = if (exists("lift_stats")) lift_stats$mean else NA,
  lift_k10_sd = if (exists("lift_stats")) lift_stats$sd else NA,
  lift_k10_cv = if (exists("lift_stats")) lift_stats$cv else NA,
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)
write_csv(metadata, file.path(OUTPUT_DIR, "metadata.csv"))
cat("  Saved: metadata.csv\n")

# 7. GENERATE REPORT

report_lines <- c(
  strrep("=", 72),
  "TEMPORAL WINDOWS EVALUATION REPORT",
  strrep("=", 72),
  sprintf("Run tag: %s", RUN_TAG),
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "WINDOW DEFINITIONS",
  strrep("-", 40),
  "Window1 (COVID Shock):  TEST 2020-03-01 to 2020-06-30",
"Window2 (Post-COVID):   TEST 2020-07-01 to 2020-12-31",
  "Window3 (Year 2021):    TEST 2021-01-01 to 2021-12-31",
  "",
  "SPLIT LOGIC",
  strrep("-", 40),
  "For each window:",
  "  TRAIN: all data strictly before test_start (minus VAL period)",
  "  VAL:   6 months immediately before test_start (if available)",
  "  TEST:  window dates",
  "",
  "This tests whether the model maintains performance across different",
  "time periods, including the COVID shock and recovery phases.",
  "",
  "RESULTS",
  strrep("-", 40),
  sprintf("Windows evaluated: %d", length(window_results)),
  sprintf("Total time: %.1f seconds", window_elapsed),
  ""
)

# Add per-window results
for (w_name in names(window_results)) {
  w <- windows[[w_name]]
  r <- window_summary %>% filter(window == w_name)
  if (nrow(r) > 0) {
    report_lines <- c(report_lines,
                      sprintf("%s:", w$name),
                      sprintf("  TEST: %s to %s", w$test_start, w$test_end),
                      sprintf("  Lift@10: %.2f, Hits: %d, Precision: %.4f",
                              r$lift, r$n_hits, r$precision),
                      sprintf("  Base rate shift: train=%.4f, test=%.4f",
                              r$base_rate_train, r$base_rate_test),
                      "")
  }
}

# Add stability assessment
if (exists("lift_stats")) {
  report_lines <- c(report_lines,
                    "STABILITY ASSESSMENT",
                    strrep("-", 40),
                    sprintf("Lift@10 CV: %.1f%% (%s)",
                            lift_stats$cv * 100, stability_label),
                    sprintf("Range: %.2f to %.2f",
                            lift_stats$min, lift_stats$max),
                    "")
}

report_lines <- c(report_lines,
                  "FILES GENERATED",
                  strrep("-", 40),
                  "temporal_all_windows.csv - All K results for all windows",
                  "temporal_summary_k10.csv - Summary at k=10",
                  "temporal_bootstrap_ci.csv - Bootstrap CIs",
                  "temporal_stability.csv - Stability statistics",
                  "window=<name>/predictions.csv.gz - Predictions per window",
                  "window=<name>/topk_metrics.csv - Metrics per window",
                  "window=<name>/split_info.csv - Split definitions",
                  "",
                  strrep("=", 72))

writeLines(report_lines, file.path(OUTPUT_DIR, "evaluation_report.txt"))
cat("  Saved: evaluation_report.txt\n")

# Update pointer
writeLines(RUN_TAG, file.path(OUTPUT_DIR, "run_tag.txt"))

# 8. SUMMARY

cat("\n")
cat(strrep("=", 60), "\n")
cat("TEMPORAL WINDOWS EVALUATION COMPLETE\n")
cat(strrep("=", 60), "\n")
cat(sprintf("Windows evaluated: %d\n", length(window_results)))
cat(sprintf("Total time: %.1f seconds\n", window_elapsed))
cat(sprintf("\nResults at k=10:\n"))
for (i in 1:nrow(window_summary)) {
  cat(sprintf("  %-25s Lift: %.2f, Hits: %d\n",
              window_summary$window_name[i],
              window_summary$lift[i],
              window_summary$n_hits[i]))
}
if (exists("lift_stats")) {
  cat(sprintf("\nStability (CV): %.1f%% - %s\n", lift_stats$cv * 100, stability_label))
}
cat(sprintf("\nOutput: %s\n", OUTPUT_DIR))
cat(strrep("=", 60), "\n")
