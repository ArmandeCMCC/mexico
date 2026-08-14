# Independent validation of Phase 4 grouped burden uncertainty.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
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
  stop("Usage: Rscript scripts/revision/13_validate_burden_uncertainty.R <run-id>")
}
project_dir <- detect_project_dir()
run_id <- args[[1]]
run_dir <- file.path(project_dir, "data", "revision", run_id)
figure_dir <- file.path(project_dir, "figures", "revision", run_id)
phase6_dir <- file.path(
  project_dir, "data", "revision", "20260813_234500_phase6_count_duration"
)
phase4_classifier_dir <- file.path(
  project_dir, "data", "revision", "20260813_214000_phase4_classifier_uncertainty"
)

required <- c(
  "burden_correlation_point_estimates.csv", "burden_correlation_bootstrap_draws.csv",
  "burden_correlation_bootstrap_summary.csv", "burden_correlation_leave_one_state_out.csv",
  "burden_correlation_loo_summary.csv", "burden_correlation_uncertainty_table.csv",
  "qa_checks.csv", "run_config.json", "QA_REPORT.md"
)
missing <- required[!file.exists(file.path(run_dir, required))]
if (length(missing)) stop("Missing burden uncertainty artifact(s): ", paste(missing, collapse = ", "))

points <- read_csv(file.path(run_dir, "burden_correlation_point_estimates.csv"), show_col_types = FALSE)
draws <- read_csv(file.path(run_dir, "burden_correlation_bootstrap_draws.csv"), show_col_types = FALSE)
loo <- read_csv(file.path(run_dir, "burden_correlation_leave_one_state_out.csv"), show_col_types = FALSE)
burden <- as.data.table(readRDS(file.path(phase6_dir, "decomposed_burden_test.rds")))
block_counts <- readRDS(file.path(phase4_classifier_dir, "bootstrap_block_counts.rds"))
burden[, `:=`(
  date = as.Date(date), iso_week = format(as.Date(date), "%G-W%V"),
  month = format(as.Date(date), "%Y-%m")
)]

cor_independent <- function(x, y, method) stats::cor(x, y, method = method)
checks <- list()
add_check <- function(id, condition, observed, expected) {
  checks[[length(checks) + 1L]] <<- tibble(
    check_id = id, status = if (isTRUE(condition)) "PASS" else "FAIL",
    observed = as.character(observed), expected = as.character(expected)
  )
}

monthly <- burden[, .(
  expected_count = sum(expected_count), observed_count = sum(actual_count),
  expected_minutes = sum(expected_minutes), observed_minutes = sum(actual_minutes)
), by = month]
state <- burden[, .(
  expected_count = sum(expected_count), observed_count = sum(actual_count),
  expected_minutes = sum(expected_minutes), observed_minutes = sum(actual_minutes)
), by = state_name]
point_rechecks <- bind_rows(lapply(list(monthly = monthly, state = state), function(grouped) {
  aggregation <- if (nrow(grouped) == 18L) "monthly" else "state"
  bind_rows(lapply(c("count", "minutes"), function(outcome) {
    bind_rows(lapply(c("pearson", "spearman"), function(metric) {
      tibble(
        aggregation = aggregation, outcome = outcome, metric = metric,
        independent = cor_independent(
          grouped[[paste0("expected_", outcome)]],
          grouped[[paste0("observed_", outcome)]], metric
        )
      )
    }))
  }))
}), .id = NULL) %>% left_join(points, by = c("aggregation", "outcome", "metric")) %>%
  mutate(absolute_difference = abs(independent - point_estimate))
write_csv(point_rechecks, file.path(run_dir, "independent_point_rechecks.csv"))
add_check("point_recomputation", max(point_rechecks$absolute_difference) < 1e-12,
          format(max(point_rechecks$absolute_difference), scientific = TRUE), "<1e-12")

cube <- burden[, .(
  expected_count = sum(expected_count), observed_count = sum(actual_count),
  expected_minutes = sum(expected_minutes), observed_minutes = sum(actual_minutes)
), by = .(iso_week, month, state_name)]
selected <- unique(c(1L, as.integer(nrow(block_counts) / 2L), nrow(block_counts)))
bootstrap_rechecks <- list()
for (replicate_id in selected) {
  week_weight <- setNames(block_counts[replicate_id, ], colnames(block_counts))
  weighted <- copy(cube)
  weighted[, weight := week_weight[iso_week]]
  for (aggregation in c("monthly", "state")) {
    group_col <- if (aggregation == "monthly") "month" else "state_name"
    grouped <- weighted[, .(
      expected_count = sum(weight * expected_count), observed_count = sum(weight * observed_count),
      expected_minutes = sum(weight * expected_minutes), observed_minutes = sum(weight * observed_minutes)
    ), by = group_col]
    for (outcome in c("count", "minutes")) for (metric in c("pearson", "spearman")) {
      independent <- cor_independent(
        grouped[[paste0("expected_", outcome)]],
        grouped[[paste0("observed_", outcome)]], metric
      )
      archived <- draws %>% filter(
        bootstrap_id == replicate_id, aggregation == !!aggregation,
        outcome == !!outcome, metric == !!metric
      ) %>% pull(value)
      bootstrap_rechecks[[length(bootstrap_rechecks) + 1L]] <- tibble(
        bootstrap_id = replicate_id, aggregation = aggregation, outcome = outcome,
        metric = metric, independent = independent, archived = archived,
        absolute_difference = abs(independent - archived)
      )
    }
  }
}
bootstrap_rechecks <- bind_rows(bootstrap_rechecks)
write_csv(bootstrap_rechecks, file.path(run_dir, "independent_bootstrap_rechecks.csv"))
add_check("bootstrap_recomputation", max(bootstrap_rechecks$absolute_difference) < 1e-12,
          format(max(bootstrap_rechecks$absolute_difference), scientific = TRUE), "<1e-12")

state_month <- burden[, .(
  expected_count = sum(expected_count), observed_count = sum(actual_count),
  expected_minutes = sum(expected_minutes), observed_minutes = sum(actual_minutes)
), by = .(state_name, month)]
states <- sort(unique(state_month$state_name))
loo_rechecks <- bind_rows(lapply(states, function(held_out) {
  reduced <- state_month[state_name != held_out]
  monthly_reduced <- reduced[, lapply(.SD, sum), by = month,
                             .SDcols = c("expected_count", "observed_count",
                                         "expected_minutes", "observed_minutes")]
  state_reduced <- reduced[, lapply(.SD, sum), by = state_name,
                           .SDcols = c("expected_count", "observed_count",
                                       "expected_minutes", "observed_minutes")]
  bind_rows(lapply(list(monthly = monthly_reduced, state = state_reduced), function(grouped) {
    aggregation <- if (nrow(grouped) == 18L) "monthly" else "state"
    bind_rows(lapply(c("count", "minutes"), function(outcome) {
      bind_rows(lapply(c("pearson", "spearman"), function(metric) {
        tibble(
          held_out_state = held_out, aggregation = aggregation, outcome = outcome,
          metric = metric, independent = cor_independent(
            grouped[[paste0("expected_", outcome)]],
            grouped[[paste0("observed_", outcome)]], metric
          )
        )
      }))
    }))
  }))
})) %>% left_join(loo, by = c("held_out_state", "aggregation", "outcome", "metric")) %>%
  mutate(absolute_difference = abs(independent - value))
write_csv(loo_rechecks, file.path(run_dir, "independent_loo_rechecks.csv"))
add_check("loo_recomputation", max(loo_rechecks$absolute_difference) < 1e-12,
          format(max(loo_rechecks$absolute_difference), scientific = TRUE), "<1e-12")

add_check("draw_support", nrow(draws) == 8000L && nrow(loo) == 256L,
          paste(nrow(draws), nrow(loo), sep = "/"), "8000/256")
add_check("burden_support", nrow(burden) == 1334151L && sum(burden$actual_outage) == 5350L,
          paste(nrow(burden), sum(burden$actual_outage), sep = "/"), "1334151/5350")
png_path <- file.path(figure_dir, "burden_correlation_uncertainty.png")
pdf_path <- file.path(figure_dir, "burden_correlation_uncertainty.pdf")
add_check("figure_outputs", file.exists(png_path) && file.exists(pdf_path) &&
            file.info(png_path)$size > 10000 && file.info(pdf_path)$size > 5000,
          paste(file.info(png_path)$size, file.info(pdf_path)$size, sep = "/"), ">10000/>5000")

qa <- bind_rows(checks)
write_csv(qa, file.path(run_dir, "independent_qa_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else "PASS"
writeLines(c(
  "# Independent Phase 4 Burden-Uncertainty Validation", "",
  paste0("- Run: `", run_id, "`"), paste0("- Status: **", status, "**"),
  paste0("- PASS: ", sum(qa$status == "PASS")),
  paste0("- FAIL: ", sum(qa$status == "FAIL")), "",
  "Point correlations, selected ISO-week bootstrap replicates, every leave-one-state-out correlation, support counts, and figure artifacts were independently recomputed or checked."
), file.path(run_dir, "INDEPENDENT_VALIDATION_REPORT.md"))
cat("Independent Phase 4 burden validation status:", status, "\n")
if (status == "FAIL") stop("Independent Phase 4 burden validation failed.")
