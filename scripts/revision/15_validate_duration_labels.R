# Independent validation for Phase 5 duration-label preparation.

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

parse_arg <- function(flag, default = NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  hit <- which(args == flag)
  if (!length(hit)) return(default)
  if (hit[1] == length(args)) stop("Missing value after ", flag)
  args[hit[1] + 1L]
}

same_keys <- function(x, y, keys) {
  nrow(anti_join(x, y, by = keys)) == 0L && nrow(anti_join(y, x, by = keys)) == 0L
}

project_dir <- detect_project_dir()
run_id <- parse_arg("--run-id", "20260814_040000_phase5_duration_labels")
run_dir <- file.path(project_dir, "data", "revision", run_id)
if (!dir.exists(run_dir)) stop("Run directory not found: ", run_dir)

required <- c(
  "raw_event_mapping_timing_flags.rds", "duration_labels_raw_mapped.rds",
  "duration_labels_model_keys.rds", "duration_variant_manifest.csv",
  "duration_label_summary.csv", "duration_label_counts_by_split.csv",
  "headline_label_reconciliation.csv", "active_overpass_label_reconciliation.csv",
  "qa_checks.csv"
)
missing <- required[!file.exists(file.path(run_dir, required))]
if (length(missing)) stop("Missing validation inputs: ", paste(missing, collapse = ", "))

events <- as_tibble(readRDS(file.path(run_dir, "raw_event_mapping_timing_flags.rds")))
labels_raw <- as_tibble(readRDS(file.path(run_dir, "duration_labels_raw_mapped.rds")))
labels_model <- as_tibble(readRDS(file.path(run_dir, "duration_labels_model_keys.rds")))
manifest <- read_csv(file.path(run_dir, "duration_variant_manifest.csv"), show_col_types = FALSE)
summary_table <- read_csv(file.path(run_dir, "duration_label_summary.csv"), show_col_types = FALSE)
split_table <- read_csv(file.path(run_dir, "duration_label_counts_by_split.csv"), show_col_types = FALSE)
internal_qa <- read_csv(file.path(run_dir, "qa_checks.csv"), show_col_types = FALSE)

panel <- as.data.table(readRDS(file.path(project_dir, "data", "panel_mex_2017_2021_ntl_ghs.rds")))
panel_keys <- unique(panel[, .(
  GID_2 = as.character(GID_2), date = as.Date(date),
  current_label = as.integer(outage_3h_or_more)
)], by = c("GID_2", "date"))
rm(panel); gc(verbose = FALSE)

# Reparse all source clocks independently and compare the stored UTC instants.
normalized_time <- sub("^24:", "00:", events$time)
clock_date <- events$outage_date + events$clock_rollover_days
clock_string <- paste(clock_date, normalized_time)
rebuilt_start <- rep(as.POSIXct(NA, tz = "UTC"), nrow(events))
for (zone in unique(events$tz_name[!is.na(events$tz_name)])) {
  idx <- which(events$tz_name == zone)
  parsed <- as.POSIXct(
    clock_string[idx], format = "%Y-%m-%d %H:%M:%S", tz = zone
  )
  rebuilt_start[idx] <- as.POSIXct(as.numeric(parsed), origin = "1970-01-01", tz = "UTC")
}

event_rule <- function(family, cutoff) {
  if (family == "night_duration_threshold") {
    events$overlaps_night & events$length_min > cutoff
  } else {
    events$active_at_0130 & events$length_min >= cutoff
  }
}

rebuilt_labels <- bind_rows(lapply(seq_len(nrow(manifest)), function(i) {
  keep <- event_rule(manifest$variant_family[i], manifest$duration_minutes[i])
  events[keep & !is.na(events$GID_2), ] %>%
    count(GID_2, date = outage_date, name = "n_outages") %>%
    mutate(variant_id = manifest$variant_id[i], .before = 1L)
}))

raw_key_counts_match <- labels_raw %>%
  select(variant_id, GID_2, date, n_outages) %>%
  full_join(rebuilt_labels, by = c("variant_id", "GID_2", "date"),
            suffix = c("_saved", "_rebuilt")) %>%
  summarise(ok = all(!is.na(n_outages_saved), !is.na(n_outages_rebuilt),
                     n_outages_saved == n_outages_rebuilt)) %>% pull(ok)

expected_model <- semi_join(
  labels_raw, as_tibble(panel_keys), by = c("GID_2", "date")
)
model_keys_match <- same_keys(
  labels_model, expected_model, c("variant_id", "GID_2", "date")
)

is_nested <- function(ids) {
  pairs <- Map(c, ids[-length(ids)], ids[-1L])
  all(vapply(pairs, function(pair) {
    lo <- labels_model %>% filter(variant_id == pair[1]) %>% select(GID_2, date)
    hi <- labels_model %>% filter(variant_id == pair[2]) %>% select(GID_2, date)
    nrow(anti_join(hi, lo, by = c("GID_2", "date"))) == 0L
  }, logical(1)))
}
night_ids <- manifest %>%
  filter(variant_family == "night_duration_threshold") %>%
  arrange(duration_minutes) %>% pull(variant_id)
active_ids <- manifest %>%
  filter(variant_family == "active_at_overpass_minimum_duration") %>%
  arrange(duration_minutes) %>% pull(variant_id)

current_keys <- as_tibble(panel_keys) %>%
  filter(current_label == 1L) %>% select(GID_2, date)
rebuilt_headline <- labels_model %>%
  filter(variant_id == "night_gt180") %>% select(GID_2, date)

old_timing <- readRDS(file.path(
  project_dir, "data", "model_ready", "timing_overpass", "labels",
  "overpass_muni_night_labels_model_keys.rds"
)) %>% as_tibble() %>% filter(variant_id == "op0130_b000") %>%
  select(GID_2, date)
rebuilt_active_strict <- events %>%
  filter(active_at_0130, length_min > 180, !is.na(GID_2)) %>%
  distinct(GID_2, date = outage_date) %>%
  semi_join(as_tibble(panel_keys), by = c("GID_2", "date"))

observed_event_counts <- manifest %>% rowwise() %>% mutate(
  rebuilt_selected_events = sum(event_rule(variant_family, duration_minutes)),
  rebuilt_mapped_events = sum(event_rule(variant_family, duration_minutes) & !is.na(events$GID_2))
) %>% ungroup() %>% select(variant_id, rebuilt_selected_events, rebuilt_mapped_events)
summary_counts_match <- summary_table %>%
  select(variant_id, selected_event_rows, selected_mapped_event_rows) %>%
  left_join(observed_event_counts, by = "variant_id") %>%
  summarise(ok = all(selected_event_rows == rebuilt_selected_events,
                     selected_mapped_event_rows == rebuilt_mapped_events)) %>% pull(ok)

checks <- list()
add_check <- function(id, condition, observed, expected) {
  checks[[length(checks) + 1L]] <<- tibble(
    check_id = id, status = if (isTRUE(condition)) "PASS" else "FAIL",
    observed = as.character(observed), expected = as.character(expected)
  )
}
add_check("internal_qa", all(internal_qa$status == "PASS"),
          sum(internal_qa$status == "FAIL"), "0 failures")
add_check("raw_event_identity", nrow(events) == 576717L && !anyDuplicated(events$event_row_id),
          paste(nrow(events), anyDuplicated(events$event_row_id), sep = "/"), "576717/0")
add_check("mapping_coverage", mean(!is.na(events$GID_2)) >= 0.999,
          sprintf("%.9f", mean(!is.na(events$GID_2))), ">=0.999")
add_check("clock_rollover", sum(events$clock_rollover_days) == 23L,
          sum(events$clock_rollover_days), "23")
add_check("utc_start_rebuild", all(
  as.numeric(rebuilt_start[!is.na(events$GID_2)]) ==
    as.numeric(events$datetime_start_utc[!is.na(events$GID_2)])
), sum(as.numeric(rebuilt_start) != as.numeric(events$datetime_start_utc), na.rm = TRUE), "0 mismatches")
add_check("duration_arithmetic", all(
  as.numeric(events$datetime_end_utc - events$datetime_start_utc, units = "mins") == events$length_min,
  na.rm = TRUE
), "checked all parseable rows", "exact")
add_check("raw_label_rebuild", raw_key_counts_match, raw_key_counts_match, "TRUE")
add_check("panel_restriction", model_keys_match, model_keys_match, "TRUE")
add_check("nested_threshold_sets", is_nested(night_ids) && is_nested(active_ids),
          paste(is_nested(night_ids), is_nested(active_ids), sep = "/"), "TRUE/TRUE")
add_check("summary_event_counts", summary_counts_match, summary_counts_match, "TRUE")
add_check("split_accounting", all(split_table$positives >= 0) && nrow(split_table) == 36L,
          nrow(split_table), "36")
add_check("headline_reference_subset", nrow(anti_join(current_keys, rebuilt_headline,
                                                       by = c("GID_2", "date"))) == 0L,
          nrow(anti_join(current_keys, rebuilt_headline, by = c("GID_2", "date"))), "0 missing")
add_check("headline_documented_additions", nrow(anti_join(rebuilt_headline, current_keys,
                                                           by = c("GID_2", "date"))) == 12L,
          nrow(anti_join(rebuilt_headline, current_keys, by = c("GID_2", "date"))), "12")
add_check("timing_reference_subset", nrow(anti_join(old_timing, rebuilt_active_strict,
                                                     by = c("GID_2", "date"))) == 0L,
          nrow(anti_join(old_timing, rebuilt_active_strict, by = c("GID_2", "date"))), "0 missing")
add_check("timing_documented_additions", nrow(anti_join(rebuilt_active_strict, old_timing,
                                                         by = c("GID_2", "date"))) == 2L,
          nrow(anti_join(rebuilt_active_strict, old_timing, by = c("GID_2", "date"))), "2")

qa <- bind_rows(checks)
write_csv(qa, file.path(run_dir, "independent_validation_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else "PASS"
writeLines(c(
  "# Independent Duration-Label Validation", "",
  paste0("- Run ID: `", run_id, "`"),
  paste0("- Status: **", status, "**"),
  paste0("- PASS: ", sum(qa$status == "PASS")),
  paste0("- FAIL: ", sum(qa$status == "FAIL")), "",
  "The validator independently reparses every local source clock, reconstructs",
  "all nine event-selection rules and municipality-date key sets, checks exact",
  "panel restriction and nesting, and reconciles both frozen reference labels."
), file.path(run_dir, "INDEPENDENT_VALIDATION_REPORT.md"))
cat("Independent Phase 5 duration-label validation: ", status, "\n", sep = "")
if (status == "FAIL") stop("Independent duration-label validation failed.")
