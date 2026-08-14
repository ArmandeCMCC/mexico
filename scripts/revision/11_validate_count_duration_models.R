# Independent validation of the Phase 6 corrected two-part model run.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(jsonlite)
  library(readr)
  library(tibble)
})

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))
  stop("Run from the project root or scripts directory.")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: Rscript scripts/revision/11_validate_count_duration_models.R <run-id>")
}
project_dir <- detect_project_dir()
run_id <- args[[1]]
run_dir <- file.path(project_dir, "data", "revision", run_id)
figure_dir <- file.path(project_dir, "figures", "revision", run_id)
phase2_dir <- file.path(
  project_dir, "data", "revision", "20260813_144700_phase2_canonical_comparison"
)
phase4_dir <- file.path(
  project_dir, "data", "revision", "20260813_214000_phase4_classifier_uncertainty"
)

required_files <- c(
  "split_target_support.csv", "count_validation_predictions.csv",
  "severity_validation_predictions.csv", "conditional_test_predictions.csv",
  "count_point_metrics.csv", "severity_point_metrics.csv",
  "count_bootstrap_draws.csv", "severity_bootstrap_draws.csv",
  "count_uncertainty_table.csv", "severity_uncertainty_table.csv",
  "count_paired_differences.csv", "severity_paired_differences.csv",
  "duan_formula_audit.csv", "decomposed_burden_test.rds",
  "burden_calibration_deciles.csv", "burden_monthly.csv", "burden_state.csv",
  "burden_aggregate_point_metrics.csv", "run_config.json", "qa_checks.csv", "QA_REPORT.md"
)
missing_files <- required_files[!file.exists(file.path(run_dir, required_files))]
if (length(missing_files)) stop("Missing Phase 6 artifacts: ", paste(missing_files, collapse = ", "))

config <- read_json(file.path(run_dir, "run_config.json"), simplifyVector = TRUE)
support <- read_csv(file.path(run_dir, "split_target_support.csv"), show_col_types = FALSE)
count_val <- read_csv(file.path(run_dir, "count_validation_predictions.csv"), show_col_types = FALSE)
severity_val <- read_csv(file.path(run_dir, "severity_validation_predictions.csv"), show_col_types = FALSE)
conditional <- read_csv(file.path(run_dir, "conditional_test_predictions.csv"), show_col_types = FALSE)
count_points <- read_csv(file.path(run_dir, "count_point_metrics.csv"), show_col_types = FALSE)
severity_points <- read_csv(file.path(run_dir, "severity_point_metrics.csv"), show_col_types = FALSE)
count_draws <- read_csv(file.path(run_dir, "count_bootstrap_draws.csv"), show_col_types = FALSE)
severity_draws <- read_csv(file.path(run_dir, "severity_bootstrap_draws.csv"), show_col_types = FALSE)
burden <- as.data.table(readRDS(file.path(run_dir, "decomposed_burden_test.rds")))
monthly_saved <- read_csv(file.path(run_dir, "burden_monthly.csv"), show_col_types = FALSE)
state_saved <- read_csv(file.path(run_dir, "burden_state.csv"), show_col_types = FALSE)
phase2_pred <- as.data.table(readRDS(file.path(phase2_dir, "paired_predictions_test.rds")))
block_counts <- readRDS(file.path(phase4_dir, "bootstrap_block_counts.rds"))

checks <- list()
add_check <- function(id, condition, observed, expected) {
  checks[[length(checks) + 1L]] <<- tibble(
    check_id = id, status = if (isTRUE(condition)) "PASS" else "FAIL",
    observed = as.character(observed), expected = as.character(expected)
  )
}

metric_set_independent <- function(actual, predicted, type, winsor_cap = NA_real_,
                                   tail90 = NA_real_, tail95 = NA_real_) {
  error <- predicted - actual
  result <- c(
    mae = sum(abs(error)) / length(error),
    rmse = sqrt(sum(error^2) / length(error)),
    median_absolute_error = stats::median(abs(error)),
    mean_bias = sum(error) / length(error),
    pearson = stats::cor(predicted, actual, method = "pearson"),
    spearman = stats::cor(predicted, actual, method = "spearman"),
    calibration_ratio = sum(predicted) / sum(actual)
  )
  if (type == "severity") {
    observed_w <- pmin(actual, winsor_cap)
    predicted_w <- pmin(predicted, winsor_cap)
    upper90 <- actual >= tail90
    upper95 <- actual >= tail95
    result <- c(
      result,
      winsorized_mae_99 = sum(abs(predicted_w - observed_w)) / length(actual),
      winsorized_rmse_99 = sqrt(sum((predicted_w - observed_w)^2) / length(actual)),
      tail90_mae = mean(abs(error[upper90])), tail90_bias = mean(error[upper90]),
      tail95_mae = mean(abs(error[upper95])), tail95_bias = mean(error[upper95])
    )
  }
  result
}

add_check(
  "split_support",
  identical(as.integer(support$n_rows), c(2683044L, 447174L, 1334151L)) &&
    identical(as.integer(support$n_positive), c(15951L, 1695L, 5350L)),
  paste(support$n_rows, support$n_positive, sep = "/", collapse = ";"),
  "2683044/15951;447174/1695;1334151/5350"
)
add_check(
  "conditional_support",
  nrow(conditional) == 5350L && !anyDuplicated(conditional[c("GID_2", "date")]) &&
    all(conditional$actual_count >= 1) && all(conditional$actual_minutes > 0),
  paste(nrow(conditional), min(conditional$actual_count), min(conditional$actual_minutes), sep = "/"),
  "5350/count>=1/minutes>0"
)

smear <- as.numeric(config$severity$duan_smearing)
duan_recomputed <- pmax(smear * exp(conditional$log1p_margin) - 1, 0)
old_recomputed <- pmax(smear * (exp(conditional$log1p_margin) - 1), 0)
add_check(
  "duan_transform_recomputation",
  max(abs(duan_recomputed - conditional$log1p_duan_minutes)) < 1e-9,
  format(max(abs(duan_recomputed - conditional$log1p_duan_minutes)), scientific = TRUE), "<1e-9"
)
add_check(
  "old_formula_diagnostic",
  max(abs(old_recomputed - conditional$log1p_old_formula_minutes)) < 1e-9,
  format(max(abs(old_recomputed - conditional$log1p_old_formula_minutes)), scientific = TRUE), "<1e-9"
)

count_columns <- c(poisson = "poisson_count", negative_binomial = "nb2_count")
severity_columns <- c(
  log1p_duan = "log1p_duan_minutes", gamma_log = "gamma_minutes",
  tweedie_log = "tweedie_minutes"
)
count_recheck <- bind_rows(lapply(names(count_columns), function(model_id) {
  result <- metric_set_independent(
    conditional$actual_count, conditional[[count_columns[[model_id]]]], "count"
  )
  tibble(model_id = model_id, metric = names(result), independent = as.numeric(result))
})) %>% left_join(count_points, by = c("model_id", "metric")) %>%
  mutate(absolute_difference = abs(independent - point_estimate))
severity_recheck <- bind_rows(lapply(names(severity_columns), function(model_id) {
  result <- metric_set_independent(
    conditional$actual_minutes, conditional[[severity_columns[[model_id]]]], "severity",
    as.numeric(config$severity$winsor_cap_training_q99),
    as.numeric(config$severity$tail90_training_threshold),
    as.numeric(config$severity$tail95_training_threshold)
  )
  tibble(model_id = model_id, metric = names(result), independent = as.numeric(result))
})) %>% left_join(severity_points, by = c("model_id", "metric")) %>%
  mutate(absolute_difference = abs(independent - point_estimate))
write_csv(count_recheck, file.path(run_dir, "independent_count_metric_rechecks.csv"))
write_csv(severity_recheck, file.path(run_dir, "independent_severity_metric_rechecks.csv"))
add_check(
  "point_metric_recomputation",
  max(count_recheck$absolute_difference, severity_recheck$absolute_difference, na.rm = TRUE) < 1e-8,
  format(max(count_recheck$absolute_difference, severity_recheck$absolute_difference,
             na.rm = TRUE), scientific = TRUE), "<1e-8"
)

count_val_mae <- c(
  poisson = mean(abs(count_val$poisson_count - count_val$actual_count)),
  negative_binomial = mean(abs(count_val$nb2_count - count_val$actual_count))
)
severity_val_mae <- c(
  log1p_duan = mean(abs(severity_val$log1p_duan_minutes - severity_val$actual_minutes)),
  gamma_log = mean(abs(severity_val$gamma_minutes - severity_val$actual_minutes)),
  tweedie_log = mean(abs(severity_val$tweedie_minutes - severity_val$actual_minutes))
)
add_check(
  "validation_only_selection",
  config$count$selected == names(which.min(count_val_mae))[1] &&
    config$severity$selected == names(which.min(severity_val_mae))[1],
  paste(config$count$selected, config$severity$selected, sep = "/"),
  paste(names(which.min(count_val_mae))[1], names(which.min(severity_val_mae))[1], sep = "/")
)

selected_replicates <- unique(c(1L, as.integer(nrow(block_counts) / 2L), nrow(block_counts)))
bootstrap_rechecks <- list()
for (replicate_id in selected_replicates) {
  times <- block_counts[replicate_id, match(conditional$iso_week, colnames(block_counts))]
  index <- rep(seq_len(nrow(conditional)), times = times)
  for (model_id in names(count_columns)) {
    result <- metric_set_independent(
      conditional$actual_count[index], conditional[[count_columns[[model_id]]]][index], "count"
    )
    archived <- count_draws %>% filter(
      bootstrap_id == replicate_id, model_id == !!model_id
    )
    bootstrap_rechecks[[length(bootstrap_rechecks) + 1L]] <- tibble(
      bootstrap_id = replicate_id, model_id = model_id, outcome = "count",
      metric = names(result), independent = as.numeric(result),
      archived = as.numeric(archived[1, names(result)]),
      absolute_difference = abs(independent - archived)
    )
  }
  for (model_id in names(severity_columns)) {
    result <- metric_set_independent(
      conditional$actual_minutes[index], conditional[[severity_columns[[model_id]]]][index], "severity",
      as.numeric(config$severity$winsor_cap_training_q99),
      as.numeric(config$severity$tail90_training_threshold),
      as.numeric(config$severity$tail95_training_threshold)
    )
    archived <- severity_draws %>% filter(
      bootstrap_id == replicate_id, model_id == !!model_id
    )
    bootstrap_rechecks[[length(bootstrap_rechecks) + 1L]] <- tibble(
      bootstrap_id = replicate_id, model_id = model_id, outcome = "minutes",
      metric = names(result), independent = as.numeric(result),
      archived = as.numeric(archived[1, names(result)]),
      absolute_difference = abs(independent - archived)
    )
  }
}
bootstrap_rechecks <- bind_rows(bootstrap_rechecks)
write_csv(bootstrap_rechecks, file.path(run_dir, "independent_bootstrap_metric_rechecks.csv"))
add_check(
  "bootstrap_metric_recomputation",
  max(bootstrap_rechecks$absolute_difference, na.rm = TRUE) < 1e-8,
  format(max(bootstrap_rechecks$absolute_difference, na.rm = TRUE), scientific = TRUE), "<1e-8"
)

setkey(burden, GID_2, date)
phase2_key <- phase2_pred[, .(
  GID_2 = as.character(GID_2), date = as.Date(date),
  truth = as.integer(truth), full_probability_phase2 = as.numeric(full_83_xgb_calibrated)
)]
setkey(phase2_key, GID_2, date)
matched_probability <- phase2_key[burden]$full_probability_phase2
matched_truth <- phase2_key[burden]$truth
add_check(
  "phase2_classifier_reuse",
  max(abs(matched_probability - burden$full_probability)) < 1e-15 &&
    all(matched_truth == burden$actual_outage),
  format(max(abs(matched_probability - burden$full_probability)), scientific = TRUE), "<1e-15"
)
add_check(
  "separate_burden_decomposition",
  max(abs(burden$expected_count - burden$full_probability * burden$conditional_count)) < 1e-12 &&
    max(abs(burden$expected_minutes - burden$full_probability * burden$conditional_minutes)) < 1e-8,
  paste(
    format(max(abs(burden$expected_count - burden$full_probability * burden$conditional_count)),
           scientific = TRUE),
    format(max(abs(burden$expected_minutes - burden$full_probability * burden$conditional_minutes)),
           scientific = TRUE), sep = "/"
  ), "<1e-12/<1e-8"
)

monthly_recomputed <- burden[, .(
  expected_count = sum(expected_count), observed_count = sum(actual_count),
  expected_minutes = sum(expected_minutes), observed_minutes = sum(actual_minutes)
), by = .(month = as.Date(format(date, "%Y-%m-01")))]
state_recomputed <- burden[, .(
  expected_count = sum(expected_count), observed_count = sum(actual_count),
  expected_minutes = sum(expected_minutes), observed_minutes = sum(actual_minutes)
), by = state_name]
setorder(monthly_recomputed, month)
monthly_saved$month <- as.Date(monthly_saved$month)
monthly_saved <- as.data.table(monthly_saved); setorder(monthly_saved, month)
setorder(state_recomputed, state_name)
state_saved <- as.data.table(state_saved); setorder(state_saved, state_name)
aggregate_diff <- max(
  abs(monthly_recomputed$expected_count - monthly_saved$expected_count),
  abs(monthly_recomputed$observed_count - monthly_saved$observed_count),
  abs(monthly_recomputed$expected_minutes - monthly_saved$expected_minutes),
  abs(monthly_recomputed$observed_minutes - monthly_saved$observed_minutes),
  abs(state_recomputed$expected_count - state_saved$expected_count),
  abs(state_recomputed$observed_count - state_saved$observed_count),
  abs(state_recomputed$expected_minutes - state_saved$expected_minutes),
  abs(state_recomputed$observed_minutes - state_saved$observed_minutes)
)
add_check("aggregate_recomputation", aggregate_diff < 1e-8,
          format(aggregate_diff, scientific = TRUE), "<1e-8")

model_files <- file.path(run_dir, "models", c(
  "count_poisson_excess.ubj", "count_nb2_excess.ubj", "minutes_log1p.ubj",
  "minutes_gamma_log.ubj", "minutes_tweedie_log.ubj"
))
add_check("saved_models", all(file.exists(model_files)) && all(file.info(model_files)$size > 1000),
          paste(file.info(model_files)$size, collapse = "/"), "five nonempty UBJ models")
png_path <- file.path(figure_dir, "corrected_two_part_burden.png")
pdf_path <- file.path(figure_dir, "corrected_two_part_burden.pdf")
add_check("figure_outputs", file.exists(png_path) && file.exists(pdf_path) &&
            file.info(png_path)$size > 10000 && file.info(pdf_path)$size > 5000,
          paste(file.info(png_path)$size, file.info(pdf_path)$size, sep = "/"), ">10000/>5000")

qa <- bind_rows(checks)
write_csv(qa, file.path(run_dir, "independent_qa_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else "PASS"
report <- c(
  "# Independent Phase 6 Validation", "",
  paste0("- Run: `", run_id, "`"), paste0("- Status: **", status, "**"),
  paste0("- PASS: ", sum(qa$status == "PASS")),
  paste0("- FAIL: ", sum(qa$status == "FAIL")), "",
  "Split support, validation-only model selection, the corrected Duan transform, point metrics, selected paired ISO-week bootstrap draws, exact reuse of Phase 2 classifier probabilities, separate burden composition, aggregate tables, model files, and figure artifacts were independently recomputed or checked."
)
writeLines(report, file.path(run_dir, "INDEPENDENT_VALIDATION_REPORT.md"))
cat("Independent Phase 6 validation status:", status, "\n")
if (status == "FAIL") stop("Independent Phase 6 validation failed.")
