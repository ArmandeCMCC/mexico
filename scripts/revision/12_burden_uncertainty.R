# Phase 4 completion: uncertainty for corrected aggregated burden correlations.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
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

project_dir <- detect_project_dir()
run_id <- parse_arg(
  "--run-id", paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_phase4_burden_uncertainty")
)
n_boot <- as.integer(parse_arg("--n-bootstrap", "1000"))
phase6_id <- "20260813_234500_phase6_count_duration"
phase6_dir <- file.path(project_dir, "data", "revision", phase6_id)
phase4_classifier_dir <- file.path(
  project_dir, "data", "revision", "20260813_214000_phase4_classifier_uncertainty"
)
burden_path <- file.path(phase6_dir, "decomposed_burden_test.rds")
phase6_qa_path <- file.path(phase6_dir, "independent_qa_checks.csv")
block_counts_path <- file.path(phase4_classifier_dir, "bootstrap_block_counts.rds")
required <- c(burden_path, phase6_qa_path, block_counts_path)
if (any(!file.exists(required))) stop("Missing required input(s): ", paste(required[!file.exists(required)], collapse = ", "))

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

cor_safe <- function(x, y, method) {
  if (length(x) < 3L || sd(x) == 0 || sd(y) == 0) return(NA_real_)
  suppressWarnings(cor(x, y, method = method))
}

matrix_by_group <- function(dt, block_levels, group_levels, value_col) {
  out <- matrix(0, nrow = length(block_levels), ncol = length(group_levels),
                dimnames = list(block_levels, group_levels))
  row_id <- match(dt$iso_week, block_levels)
  col_id <- match(dt$group, group_levels)
  out[cbind(row_id, col_id)] <- dt[[value_col]]
  out
}

summarize_bootstrap <- function(draws, points) {
  draws %>% group_by(aggregation, outcome, metric) %>% summarise(
    bootstrap_mean = mean(value, na.rm = TRUE),
    bootstrap_se = sd(value, na.rm = TRUE),
    ci_lower = quantile(value, 0.025, na.rm = TRUE, names = FALSE),
    ci_upper = quantile(value, 0.975, na.rm = TRUE, names = FALSE),
    n_valid = sum(is.finite(value)), .groups = "drop"
  ) %>% left_join(points, by = c("aggregation", "outcome", "metric")) %>%
    select(aggregation, outcome, metric, point_estimate, everything())
}

jackknife_summary <- function(full_value, loo_values) {
  n <- length(loo_values)
  clipped_full <- pmin(pmax(full_value, -0.999999), 0.999999)
  clipped_loo <- pmin(pmax(loo_values, -0.999999), 0.999999)
  full_z <- atanh(clipped_full)
  loo_z <- atanh(clipped_loo)
  pseudo_z <- n * full_z - (n - 1) * loo_z
  estimate_z <- mean(pseudo_z)
  se_z <- sd(pseudo_z) / sqrt(n)
  c(
    loo_min = min(loo_values), loo_max = max(loo_values),
    jackknife_estimate = tanh(estimate_z), jackknife_se_fisher = se_z,
    jackknife_ci_lower = tanh(estimate_z - qnorm(0.975) * se_z),
    jackknife_ci_upper = tanh(estimate_z + qnorm(0.975) * se_z)
  )
}

phase6_qa <- read_csv(phase6_qa_path, show_col_types = FALSE)
if (any(phase6_qa$status == "FAIL")) stop("Definitive Phase 6 validation contains a failure.")
block_counts <- readRDS(block_counts_path)
if (is.null(colnames(block_counts)) || nrow(block_counts) < n_boot) {
  stop("Definitive ISO-week bootstrap matrix is unavailable or unlabeled.")
}
block_counts <- block_counts[seq_len(n_boot), , drop = FALSE]
iso_weeks <- colnames(block_counts)

log_message("Phase 4 grouped burden uncertainty")
log_message("Run ID: ", run_id)
burden <- as.data.table(readRDS(burden_path))
burden[, `:=`(
  date = as.Date(date), iso_week = format(as.Date(date), "%G-W%V"),
  month = format(as.Date(date), "%Y-%m")
)]
if (nrow(burden) != 1334151L || sum(burden$actual_outage) != 5350L) {
  stop("Corrected burden support drifted from the definitive test sample.")
}
if (!setequal(unique(burden$iso_week), iso_weeks)) stop("ISO-week support mismatch.")

point_groups <- list(
  monthly = burden[, .(
    expected_count = sum(expected_count), observed_count = sum(actual_count),
    expected_minutes = sum(expected_minutes), observed_minutes = sum(actual_minutes)
  ), by = .(group = month)],
  state = burden[, .(
    expected_count = sum(expected_count), observed_count = sum(actual_count),
    expected_minutes = sum(expected_minutes), observed_minutes = sum(actual_minutes)
  ), by = .(group = state_name)]
)

point_estimates <- bind_rows(lapply(names(point_groups), function(aggregation) {
  grouped <- point_groups[[aggregation]]
  bind_rows(lapply(c("count", "minutes"), function(outcome) {
    bind_rows(lapply(c("pearson", "spearman"), function(metric) {
      tibble(
        aggregation = aggregation, outcome = outcome, metric = metric,
        point_estimate = cor_safe(
          grouped[[paste0("expected_", outcome)]],
          grouped[[paste0("observed_", outcome)]], metric
        ), n_groups = nrow(grouped)
      )
    }))
  }))
}))
write_csv(point_estimates, file.path(output_dir, "burden_correlation_point_estimates.csv"))

cube <- burden[, .(
  expected_count = sum(expected_count), observed_count = sum(actual_count),
  expected_minutes = sum(expected_minutes), observed_minutes = sum(actual_minutes)
), by = .(iso_week, month, state_name)]
rm(burden); gc(verbose = FALSE)

bootstrap_draws <- list()
for (aggregation in c("monthly", "state")) {
  group_col <- if (aggregation == "monthly") "month" else "state_name"
  weekly_group <- cube[, lapply(.SD, sum), by = c("iso_week", group_col),
                       .SDcols = c("expected_count", "observed_count",
                                   "expected_minutes", "observed_minutes")]
  setnames(weekly_group, group_col, "group")
  group_levels <- sort(unique(weekly_group$group))
  matrices <- lapply(
    c("expected_count", "observed_count", "expected_minutes", "observed_minutes"),
    function(col) matrix_by_group(weekly_group, iso_weeks, group_levels, col)
  )
  names(matrices) <- c("expected_count", "observed_count", "expected_minutes", "observed_minutes")
  weighted <- lapply(matrices, function(m) block_counts %*% m)
  for (outcome in c("count", "minutes")) {
    predicted <- weighted[[paste0("expected_", outcome)]]
    observed <- weighted[[paste0("observed_", outcome)]]
    for (metric in c("pearson", "spearman")) {
      values <- vapply(seq_len(n_boot), function(b) {
        cor_safe(predicted[b, ], observed[b, ], metric)
      }, numeric(1))
      bootstrap_draws[[length(bootstrap_draws) + 1L]] <- tibble(
        bootstrap_id = seq_len(n_boot), aggregation = aggregation,
        outcome = outcome, metric = metric, value = values
      )
    }
  }
}
bootstrap_draws <- bind_rows(bootstrap_draws)
bootstrap_summary <- summarize_bootstrap(bootstrap_draws, point_estimates)
write_csv(bootstrap_draws, file.path(output_dir, "burden_correlation_bootstrap_draws.csv"))
write_csv(bootstrap_summary, file.path(output_dir, "burden_correlation_bootstrap_summary.csv"))

# Leave-one-state-out sensitivity for both aggregation scales.
state_month <- cube[, lapply(.SD, sum), by = .(state_name, month),
                    .SDcols = c("expected_count", "observed_count",
                                "expected_minutes", "observed_minutes")]
state_totals <- state_month[, lapply(.SD, sum), by = state_name,
                            .SDcols = c("expected_count", "observed_count",
                                        "expected_minutes", "observed_minutes")]
national_month <- state_month[, lapply(.SD, sum), by = month,
                              .SDcols = c("expected_count", "observed_count",
                                          "expected_minutes", "observed_minutes")]
states <- sort(unique(state_month$state_name))
if (length(states) != 32L) stop("Expected 32 states for leave-one-state-out analysis.")

loo_draws <- bind_rows(lapply(states, function(held_out_state) {
  held_month <- state_month[state_name == held_out_state]
  setkey(national_month, month)
  setkey(held_month, month)
  monthly_loo <- copy(national_month)
  monthly_loo[held_month, `:=`(
    expected_count = expected_count - i.expected_count,
    observed_count = observed_count - i.observed_count,
    expected_minutes = expected_minutes - i.expected_minutes,
    observed_minutes = observed_minutes - i.observed_minutes
  )]
  state_loo <- state_totals[state_name != held_out_state]
  bind_rows(lapply(list(monthly = monthly_loo, state = state_loo), function(grouped) {
    aggregation <- if (nrow(grouped) == 18L) "monthly" else "state"
    bind_rows(lapply(c("count", "minutes"), function(outcome) {
      bind_rows(lapply(c("pearson", "spearman"), function(metric) {
        tibble(
          held_out_state = held_out_state, aggregation = aggregation,
          outcome = outcome, metric = metric,
          value = cor_safe(
            grouped[[paste0("expected_", outcome)]],
            grouped[[paste0("observed_", outcome)]], metric
          )
        )
      }))
    }))
  }))
}))
loo_summary <- loo_draws %>%
  left_join(point_estimates %>% select(aggregation, outcome, metric, point_estimate),
            by = c("aggregation", "outcome", "metric")) %>%
  group_by(aggregation, outcome, metric, point_estimate) %>%
  summarise(
    values = list(value), n_states = n(), .groups = "drop"
  ) %>% rowwise() %>% mutate(
    summary = list(as.list(jackknife_summary(point_estimate, unlist(values))))
  ) %>% ungroup() %>% unnest_wider(summary) %>% select(-values)
write_csv(loo_draws, file.path(output_dir, "burden_correlation_leave_one_state_out.csv"))
write_csv(loo_summary, file.path(output_dir, "burden_correlation_loo_summary.csv"))

combined <- bootstrap_summary %>%
  left_join(loo_summary, by = c("aggregation", "outcome", "metric", "point_estimate"))
write_csv(combined, file.path(output_dir, "burden_correlation_uncertainty_table.csv"))

plot_data <- bootstrap_summary %>% mutate(
  aggregation_label = recode(aggregation, monthly = "Monthly national totals", state = "State totals"),
  outcome_label = recode(outcome, count = "Event count", minutes = "Cumulative minutes"),
  row_label = factor(
    paste(aggregation_label, outcome_label, sep = ": "),
    levels = rev(c(
      "Monthly national totals: Event count",
      "Monthly national totals: Cumulative minutes",
      "State totals: Event count", "State totals: Cumulative minutes"
    ))
  ),
  metric_label = recode(metric, pearson = "Pearson correlation", spearman = "Spearman correlation")
)
palette <- c("Event count" = "#0072B2", "Cumulative minutes" = "#D55E00")
correlation_plot <- ggplot(
  plot_data,
  aes(x = point_estimate, y = row_label, colour = outcome_label)
) +
  geom_vline(xintercept = 0, colour = "grey70", linewidth = 0.45) +
  geom_errorbar(
    aes(xmin = ci_lower, xmax = ci_upper),
    orientation = "y", width = 0.16, linewidth = 0.65
  ) +
  geom_point(size = 2.4) +
  facet_wrap(~metric_label, ncol = 1) +
  scale_colour_manual(values = palette, breaks = names(palette)) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  labs(
    title = "Aggregated burden correlations",
    subtitle = "Points are held-out test estimates; bars are 95% ISO-week bootstrap intervals",
    x = "Correlation", y = NULL, colour = NULL
  ) +
  theme_minimal(base_size = 10.5) +
  theme(
    legend.position = "bottom", panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(), plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold")
  )
png_path <- file.path(figure_dir, "burden_correlation_uncertainty.png")
pdf_path <- file.path(figure_dir, "burden_correlation_uncertainty.pdf")
ggsave(png_path, correlation_plot, width = 8.4, height = 6.4, dpi = 320, bg = "white")
ggsave(pdf_path, correlation_plot, width = 8.4, height = 6.4,
       device = grDevices::pdf, bg = "white")

config <- list(
  run_id = run_id, created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
  corrected_burden_source = phase6_id, n_bootstrap = n_boot,
  bootstrap = list(
    method = "paired nonparametric ISO-week block bootstrap",
    source = block_counts_path, n_blocks = ncol(block_counts),
    interval = "2.5th and 97.5th percentiles"
  ),
  leave_one_state_out = list(
    n_states = length(states),
    interval = "Fisher-z jackknife pseudo-value interval",
    sensitivity_range = "minimum and maximum leave-one-state-out correlation"
  ),
  outcomes = c("separate expected event count", "separate expected cumulative outage minutes"),
  aggregations = c("18 monthly national totals", "32 state totals"),
  metrics = c("Pearson", "Spearman")
)
write_json(config, file.path(output_dir, "run_config.json"),
           pretty = TRUE, auto_unbox = TRUE, digits = 16)

checks <- list()
add_check <- function(id, condition, observed, expected) {
  checks[[length(checks) + 1L]] <<- tibble(
    check_id = id, status = if (isTRUE(condition)) "PASS" else "FAIL",
    observed = as.character(observed), expected = as.character(expected)
  )
}
add_check("phase6_independent_qa", !any(phase6_qa$status == "FAIL"),
          sum(phase6_qa$status == "PASS"), "No FAIL")
add_check("test_support", sum(cube$observed_count) == 7084L &&
            sum(cube$observed_minutes) == 6462333,
          paste(sum(cube$observed_count), sum(cube$observed_minutes), sep = "/"), "7084/6462333")
add_check("group_support", nrow(point_groups$monthly) == 18L && nrow(point_groups$state) == 32L,
          paste(nrow(point_groups$monthly), nrow(point_groups$state), sep = "/"), "18/32")
add_check("bootstrap_support", nrow(bootstrap_draws) == 8L * n_boot &&
            all(bootstrap_summary$n_valid == n_boot),
          paste(nrow(bootstrap_draws), min(bootstrap_summary$n_valid), sep = "/"),
          paste(8L * n_boot, n_boot, sep = "/"))
add_check("bootstrap_bounds", all(bootstrap_draws$value >= -1 & bootstrap_draws$value <= 1),
          paste(range(bootstrap_draws$value), collapse = "/"), "[-1,1]")
add_check("loo_support", nrow(loo_draws) == 32L * 8L && all(loo_summary$n_states == 32L),
          paste(nrow(loo_draws), min(loo_summary$n_states), sep = "/"), "256/32")
add_check("loo_bounds", all(loo_draws$value >= -1 & loo_draws$value <= 1) &&
            all(loo_summary$jackknife_ci_lower >= -1) && all(loo_summary$jackknife_ci_upper <= 1),
          paste(range(loo_draws$value), collapse = "/"), "[-1,1]")
add_check("figure_outputs", file.info(png_path)$size > 10000 && file.info(pdf_path)$size > 5000,
          paste(file.info(png_path)$size, file.info(pdf_path)$size, sep = "/"), ">10000/>5000")
qa <- bind_rows(checks)
write_csv(qa, file.path(output_dir, "qa_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else "PASS"

report_rows <- combined %>% filter(metric == "pearson") %>% arrange(aggregation, outcome)
report <- c(
  "# Phase 4 Grouped Burden Uncertainty", "",
  paste0("- Run ID: `", run_id, "`"), paste0("- Status: **", status, "**"),
  paste0("- Corrected burden source: `", phase6_id, "`"),
  paste0("- Bootstrap: ", n_boot, " paired resamples of 79 ISO-week blocks"),
  "- Geographic sensitivity: 32 leave-one-state-out evaluations", "",
  "## Pearson Results", "",
  paste0(
    "- ", report_rows$aggregation, " / ", report_rows$outcome, ": ",
    sprintf("%.3f", report_rows$point_estimate), " (week-bootstrap 95% CI ",
    sprintf("%.3f", report_rows$ci_lower), "-", sprintf("%.3f", report_rows$ci_upper),
    "; leave-one-state-out range ", sprintf("%.3f", report_rows$loo_min), "-",
    sprintf("%.3f", report_rows$loo_max), ")"
  ), "",
  "## Interpretation", "",
  "The intervals quantify uncertainty in broad spatial and temporal burden-pattern correlations. They do not repair the substantial underprediction of total minutes and must not be interpreted as municipality-night severity accuracy.", "",
  "## QA", "", paste0("- PASS: ", sum(qa$status == "PASS")),
  paste0("- FAIL: ", sum(qa$status == "FAIL"))
)
writeLines(report, file.path(output_dir, "QA_REPORT.md"))
log_message("Phase 4 burden uncertainty status: ", status)
if (status == "FAIL") stop("Phase 4 grouped burden uncertainty QA failed.")
