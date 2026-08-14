# Phase 5 feasible duration labels from the raw 2017-2021 outage file.
# Existing ground truth, panels, timing outputs, and scripts remain unchanged.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(jsonlite)
  library(readr)
  library(stringi)
  library(stringr)
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

normalize_name <- function(x) {
  x <- stri_trans_general(toupper(as.character(x)), "Latin-ASCII")
  x <- str_replace_all(x, "GRAL[.]", "GENERAL")
  x <- str_replace_all(x, "FCO[.]", "FRANCISCO")
  x <- str_replace_all(x, "R[.]", "RIO")
  x <- str_replace_all(x, "[|-]", "")
  str_squish(x)
}

state_id_from_gid <- function(x) {
  vapply(strsplit(as.character(x), ".", fixed = TRUE), function(parts) {
    paste(parts[1:2], collapse = ".")
  }, character(1))
}

local_clock_to_utc <- function(clock_string, timezone) {
  out <- rep(as.POSIXct(NA, tz = "UTC"), length(clock_string))
  for (zone in unique(timezone[!is.na(timezone)])) {
    idx <- which(timezone == zone)
    parsed <- as.POSIXct(
      clock_string[idx], format = "%Y-%m-%d %H:%M:%S", tz = zone
    )
    out[idx] <- as.POSIXct(as.numeric(parsed), origin = "1970-01-01", tz = "UTC")
  }
  out
}

most_frequent <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[1]
}

project_dir <- detect_project_dir()
run_id <- parse_arg(
  "--run-id", paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_phase5_duration_labels")
)
output_dir <- file.path(project_dir, "data", "revision", run_id)
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to overwrite non-empty run directory: ", output_dir)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
log_path <- file.path(output_dir, "run.log")
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
log_message <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", msg, "\n",
      file = log_path, append = TRUE, sep = "")
}

raw_path <- file.path(project_dir, "data", "dist_outage_2017_2021_causa_simple.csv")
ground_truth_path <- file.path(project_dir, "data", "night_outages_3hrs_with_locations.csv")
cause_path <- file.path(project_dir, "data", "night_outages_3hrs_with_locations_clean_by_reason.csv")
panel_path <- file.path(project_dir, "data", "panel_mex_2017_2021_ntl_ghs.rds")
timezone_path <- file.path(
  project_dir, "data", "model_ready", "timing_overpass", "labels",
  "overpass_municipality_timezones.csv"
)
existing_timing_labels_path <- file.path(
  project_dir, "data", "model_ready", "timing_overpass", "labels",
  "overpass_muni_night_labels_model_keys.rds"
)
required <- c(raw_path, ground_truth_path, cause_path, panel_path, timezone_path,
              existing_timing_labels_path)
if (any(!file.exists(required))) stop("Missing required input(s): ", paste(required[!file.exists(required)], collapse = ", "))

log_message("Phase 5 raw duration and active-at-overpass label construction")
log_message("Run ID: ", run_id)

panel <- as.data.table(readRDS(panel_path))
panel[, `:=`(GID_2 = as.character(GID_2), date = as.Date(date))]
panel_keys <- unique(panel[, .(GID_2, date, current_label = as.integer(outage_3h_or_more))],
                     by = c("GID_2", "date"))
panel_municipalities <- unique(panel[, .(
  GID_2, state_clean = as.character(state_clean),
  municipality_clean = as.character(municipality_clean),
  NAME_1 = as.character(NAME_1), NAME_2 = as.character(NAME_2)
)], by = "GID_2")
panel_municipalities[, `:=`(
  state_id = state_id_from_gid(GID_2),
  municipality_key = normalize_name(municipality_clean)
)]
rm(panel); gc(verbose = FALSE)

raw <- read_csv(raw_path, show_col_types = FALSE)
names(raw) <- c(
  "year", "date", "time", "length_min", "state", "municipality",
  "substation", "node", "reason", "network", "reason_simple"
)
raw <- raw %>% mutate(
  event_row_id = row_number(), outage_date = as.Date(date),
  raw_state_key = normalize_name(state),
  raw_municipality_key = normalize_name(municipality),
  length_min = as.numeric(length_min)
)
if (nrow(raw) != 576717L || min(raw$length_min) != 6 || anyNA(raw$length_min)) {
  stop("Raw outage support or duration range drifted from Phase 1.")
}

ground_truth <- read_csv(ground_truth_path, show_col_types = FALSE) %>% mutate(
  raw_state_key = normalize_name(state),
  raw_municipality_key = normalize_name(municipality),
  GID_2 = as.character(GID_2)
)
direct_crosswalk <- ground_truth %>%
  filter(!is.na(GID_2)) %>%
  distinct(raw_state_key, raw_municipality_key, GID_2) %>%
  mutate(mapping_method = "existing_ground_truth")
if (nrow(direct_crosswalk %>% count(raw_state_key, raw_municipality_key) %>% filter(n > 1L))) {
  stop("Existing ground truth contains an ambiguous raw municipality mapping.")
}

state_crosswalk <- direct_crosswalk %>%
  transmute(raw_state_key, state_id = state_id_from_gid(GID_2)) %>% distinct()
if (nrow(state_crosswalk %>% count(raw_state_key) %>% filter(n > 1L)) ||
    n_distinct(state_crosswalk$raw_state_key) != 33L) {
  stop("Raw-to-panel state mapping is incomplete or ambiguous.")
}

raw_pairs <- raw %>% count(
  raw_state_key, raw_municipality_key, name = "raw_event_rows"
)
panel_candidates <- as_tibble(panel_municipalities) %>%
  select(state_id, municipality_key, GID_2) %>% distinct() %>%
  add_count(state_id, municipality_key, name = "n_candidates") %>%
  filter(n_candidates == 1L) %>% select(-n_candidates)

crosswalk <- raw_pairs %>%
  left_join(direct_crosswalk, by = c("raw_state_key", "raw_municipality_key"))
remaining <- crosswalk %>% filter(is.na(GID_2)) %>%
  select(raw_state_key, raw_municipality_key, raw_event_rows) %>%
  left_join(state_crosswalk, by = "raw_state_key") %>%
  left_join(panel_candidates,
            by = c("state_id", "raw_municipality_key" = "municipality_key")) %>%
  mutate(mapping_method = if_else(!is.na(GID_2), "canonical_unique_within_state", NA_character_)) %>%
  select(raw_state_key, raw_municipality_key, raw_event_rows, GID_2, mapping_method)
crosswalk <- bind_rows(
  crosswalk %>% filter(!is.na(GID_2)), remaining
)

manual_aliases <- tribble(
  ~raw_state_key, ~raw_municipality_key, ~GID_2,
  "GUERRERO", "JOSE AZUETA", "MEX.12.79_2",
  "GUERRERO", "LA UNION ISIDORO MONTES DE OCA", "MEX.12.45_2",
  "GUERRERO", "LA UNION DE ISIDORO MONTES DE OCA", "MEX.12.45_2",
  "ESTADO DE MEXICO", "SAN JOSE DEL RINCON(POB S/DAP)", "MEX.15.76_2",
  "CHIHUAHUA", "GENERAL TRIAS", "MEX.6.61_2",
  "HIDALGO", "SANTIAGO TULANTEPEC DE LUGO GUERRERO", "MEX.13.56_2",
  "MICHOACAN", "YURICUARO", "MEX.16.107_2",
  "ESTADO DE MEXICO", "ORO, EL", "MEX.15.36_2",
  "JALISCO", "ARENAL", "MEX.14.32_2",
  "JALISCO", "YAHUALICA DE GLEZ GALLO", "MEX.14.118_2",
  "ESTADO DE MEXICO", "TEJUPILCO (POB SIN DAP)", "MEX.15.84_2",
  "SONORA", "PLUTARCO ELIAS CALLES", "MEX.26.28_2",
  "ESTADO DE MEXICO", "ACAMBAY", "MEX.15.1_2",
  "TLAXCALA", "ZITLALTEPEC DE TRINIDAD SANCHEZ SANTOS", "MEX.29.60_2",
  "YUCATAN", "VILLA BACA", "MEX.31.4_2",
  "ESTADO DE MEXICO", "LUVIANOS (POB SIN DAP)", "MEX.15.54_2",
  "JALISCO", "BOLA¥OS", "MEX.14.16_2",
  "NUEVO LEON", "DRIO COSS", "MEX.19.13_2",
  "NAYARIT", "EL NAYAR", "MEX.18.6_2"
)
if (!all(manual_aliases$GID_2 %in% panel_municipalities$GID_2)) {
  stop("A manual alias target is absent from the frozen panel.")
}
crosswalk <- crosswalk %>%
  left_join(manual_aliases %>% rename(manual_GID_2 = GID_2),
            by = c("raw_state_key", "raw_municipality_key")) %>%
  mutate(
    mapping_method = if_else(is.na(GID_2) & !is.na(manual_GID_2),
                             "documented_manual_alias", mapping_method),
    GID_2 = coalesce(GID_2, manual_GID_2)
  ) %>% select(-manual_GID_2) %>% arrange(raw_state_key, raw_municipality_key)
if (nrow(crosswalk) != 1417L || anyDuplicated(crosswalk[c("raw_state_key", "raw_municipality_key")])) {
  stop("Final crosswalk does not have one row per raw municipality pair.")
}

raw <- raw %>% left_join(
  crosswalk %>% select(raw_state_key, raw_municipality_key, GID_2, mapping_method),
  by = c("raw_state_key", "raw_municipality_key")
)
mapping_coverage <- raw %>% summarise(
  raw_event_rows = n(), mapped_event_rows = sum(!is.na(GID_2)),
  unmatched_event_rows = sum(is.na(GID_2)), mapped_share = mean(!is.na(GID_2))
)
if (mapping_coverage$mapped_share < 0.999) stop("Raw event mapping coverage is below 99.9%.")

timezone_crosswalk <- read_csv(timezone_path, show_col_types = FALSE) %>%
  transmute(GID_2 = as.character(GID_2), tz_name = as.character(tz_name))
raw <- raw %>% left_join(timezone_crosswalk, by = "GID_2")
if (any(is.na(raw$tz_name) & !is.na(raw$GID_2))) stop("A mapped raw event lacks an IANA timezone.")

cause_detail <- read_csv(cause_path, show_col_types = FALSE) %>%
  filter(!is.na(classification_general)) %>%
  distinct(reason, classification_general)
if (nrow(cause_detail %>% count(reason) %>% filter(n > 1L))) {
  stop("Detailed reason-to-general-cause mapping is ambiguous.")
}
raw <- raw %>% left_join(cause_detail, by = "reason") %>% mutate(
  cause_mapping_method = if_else(!is.na(classification_general), "existing_detailed_reason", "simple_fallback"),
  classification_general = case_when(
    !is.na(classification_general) ~ classification_general,
    reason_simple %in% c("FRIO", "RAMA", "DESCARGAS", "TROMBA", "TORMENTA", "VIENTOS") ~ "Environmental",
    reason_simple == "LIBRANZA_PROGRAMADA" ~ "Planned",
    reason_simple %in% c(
      "FALLA", "SOBRECARGA", "LIBRANZA_NON_PROGRAMADA", "FALLA_TRANSMISSION",
      "FALLA_GENERATION", "FALLA_DISTRIBUTION_OTHER_AREA"
    ) ~ "Technical",
    TRUE ~ "Other"
  )
)

log_message("Converting local event clocks with historical IANA rules...")
time_string <- format(raw$time)
raw$clock_rollover_days <- as.integer(str_detect(time_string, "^24:"))
normalized_time_string <- str_replace(time_string, "^24:", "00:")
start_clock <- paste(
  raw$outage_date + raw$clock_rollover_days, normalized_time_string
)
night_start_clock <- paste(raw$outage_date, "22:00:00")
night_end_clock <- paste(raw$outage_date + 1L, "06:00:00")
overpass_clock <- paste(raw$outage_date + 1L, "01:30:00")
raw$datetime_start_utc <- local_clock_to_utc(start_clock, raw$tz_name)
raw$datetime_end_utc <- raw$datetime_start_utc + raw$length_min * 60
raw$night_start_utc <- local_clock_to_utc(night_start_clock, raw$tz_name)
raw$night_end_utc <- local_clock_to_utc(night_end_clock, raw$tz_name)
raw$overpass_datetime_utc <- local_clock_to_utc(overpass_clock, raw$tz_name)
raw <- raw %>% mutate(
  overlaps_night = !is.na(GID_2) & datetime_start_utc < night_end_utc &
    datetime_end_utc > night_start_utc,
  active_at_0130 = !is.na(GID_2) & datetime_start_utc <= overpass_datetime_utc &
    datetime_end_utc >= overpass_datetime_utc
)
if (any(is.na(raw$datetime_start_utc) & !is.na(raw$GID_2))) {
  stop("A mapped event has an unparseable local start time.")
}

variant_manifest <- tribble(
  ~variant_id, ~variant_family, ~duration_minutes, ~duration_operator, ~definition,
  "night_gt060", "night_duration_threshold", 60L, ">", "Overlaps 22:00-06:00 local night and lasts more than 60 minutes.",
  "night_gt120", "night_duration_threshold", 120L, ">", "Overlaps 22:00-06:00 local night and lasts more than 120 minutes.",
  "night_gt180", "night_duration_threshold", 180L, ">", "Overlaps 22:00-06:00 local night and lasts more than 180 minutes.",
  "night_gt360", "night_duration_threshold", 360L, ">", "Overlaps 22:00-06:00 local night and lasts more than 360 minutes.",
  "night_gt720", "night_duration_threshold", 720L, ">", "Overlaps 22:00-06:00 local night and lasts more than 720 minutes.",
  "active0130_min015", "active_at_overpass_minimum_duration", 15L, ">=", "Active at nominal 01:30 local overpass and lasts at least 15 minutes.",
  "active0130_min030", "active_at_overpass_minimum_duration", 30L, ">=", "Active at nominal 01:30 local overpass and lasts at least 30 minutes.",
  "active0130_min060", "active_at_overpass_minimum_duration", 60L, ">=", "Active at nominal 01:30 local overpass and lasts at least 60 minutes.",
  "active0130_min180", "active_at_overpass_minimum_duration", 180L, ">=", "Active at nominal 01:30 local overpass and lasts at least 180 minutes."
)

event_selected <- function(variant_id, family, cutoff) {
  if (family == "night_duration_threshold") {
    raw$overlaps_night & raw$length_min > cutoff
  } else {
    raw$active_at_0130 & raw$length_min >= cutoff
  }
}

labels_raw <- bind_rows(lapply(seq_len(nrow(variant_manifest)), function(i) {
  variant <- variant_manifest[i, ]
  selected <- event_selected(variant$variant_id, variant$variant_family, variant$duration_minutes)
  raw[selected, ] %>% group_by(GID_2, outage_date) %>% summarise(
    n_outages = n(), min_length_min = min(length_min), max_length_min = max(length_min),
    total_length_min = sum(length_min), mean_length_min = mean(length_min),
    median_length_min = median(length_min),
    n_outages_environmental = sum(classification_general == "Environmental"),
    n_outages_technical = sum(classification_general == "Technical"),
    n_outages_planned = sum(classification_general == "Planned"),
    n_outages_other = sum(classification_general == "Other"),
    classification_general = most_frequent(classification_general), .groups = "drop"
  ) %>% transmute(
    variant_id = variant$variant_id, GID_2 = as.character(GID_2),
    date = as.Date(outage_date), outage_3h_or_more = 1L,
    n_outages, min_length_min, max_length_min, total_length_min,
    mean_length_min, median_length_min, n_outages_environmental,
    n_outages_technical, n_outages_planned, n_outages_other,
    classification_general
  )
}))
if (anyDuplicated(labels_raw[c("variant_id", "GID_2", "date")])) {
  stop("Duration labels contain duplicate variant-municipality-date keys.")
}
labels_model <- semi_join(labels_raw, as_tibble(panel_keys), by = c("GID_2", "date"))

split_for_date <- function(x) case_when(
  x >= as.Date("2017-01-01") & x <= as.Date("2019-12-31") ~ "train",
  x >= as.Date("2020-01-01") & x <= as.Date("2020-03-31") ~ "calibration_fit",
  x >= as.Date("2020-04-01") & x <= as.Date("2020-06-30") ~ "threshold_selection",
  x >= as.Date("2020-07-01") & x <= as.Date("2021-12-31") ~ "test",
  TRUE ~ "other"
)
panel_split <- as_tibble(panel_keys) %>% mutate(split = split_for_date(date)) %>%
  count(split, name = "panel_rows")
label_counts_by_split <- labels_model %>% mutate(split = split_for_date(date)) %>%
  count(variant_id, split, name = "positives") %>%
  complete(variant_id = variant_manifest$variant_id,
           split = c("train", "calibration_fit", "threshold_selection", "test"),
           fill = list(positives = 0L)) %>%
  left_join(panel_split, by = "split") %>% mutate(prevalence = positives / panel_rows)

event_counts <- bind_rows(lapply(seq_len(nrow(variant_manifest)), function(i) {
  variant <- variant_manifest[i, ]
  selected <- event_selected(variant$variant_id, variant$variant_family, variant$duration_minutes)
  tibble(variant_id = variant$variant_id, selected_event_rows = sum(selected),
         selected_mapped_event_rows = sum(selected & !is.na(raw$GID_2)))
}))
label_summary <- variant_manifest %>% left_join(event_counts, by = "variant_id") %>%
  left_join(labels_raw %>% count(variant_id, name = "raw_mapped_municipality_nights"), by = "variant_id") %>%
  left_join(labels_model %>% count(variant_id, name = "model_key_municipality_nights"), by = "variant_id") %>%
  mutate(
    dropped_on_missing_panel_dates = raw_mapped_municipality_nights - model_key_municipality_nights,
    share_of_current_headline_positives = model_key_municipality_nights / sum(panel_keys$current_label)
  )

# Reconciliation with the current headline and completed timing labels.
current_positive_keys <- as_tibble(panel_keys) %>%
  filter(current_label == 1L) %>%
  select(GID_2, date)
new_headline_keys <- labels_model %>% filter(variant_id == "night_gt180") %>% select(GID_2, date)
headline_intersection <- nrow(inner_join(
  current_positive_keys, new_headline_keys, by = c("GID_2", "date")
))
headline_current_only <- nrow(anti_join(
  current_positive_keys, new_headline_keys, by = c("GID_2", "date")
))
headline_rebuilt_only <- nrow(anti_join(
  new_headline_keys, current_positive_keys, by = c("GID_2", "date")
))
headline_reconciliation <- tibble(
  current_positive_keys = nrow(current_positive_keys),
  rebuilt_positive_keys = nrow(new_headline_keys),
  intersection = headline_intersection,
  current_only = headline_current_only,
  rebuilt_only = headline_rebuilt_only
) %>% mutate(jaccard = intersection / (intersection + current_only + rebuilt_only))

existing_timing <- readRDS(existing_timing_labels_path) %>%
  filter(variant_id == "op0130_b000") %>% select(GID_2, date)
diagnostic_active_gt180 <- raw %>%
  filter(active_at_0130, length_min > 180, !is.na(GID_2)) %>%
  distinct(GID_2, date = outage_date) %>% semi_join(as_tibble(panel_keys), by = c("GID_2", "date"))
timing_reconciliation <- tibble(
  existing_timing_keys = nrow(existing_timing),
  rebuilt_active_gt180_keys = nrow(diagnostic_active_gt180),
  intersection = nrow(inner_join(existing_timing, diagnostic_active_gt180, by = c("GID_2", "date"))),
  existing_only = nrow(anti_join(existing_timing, diagnostic_active_gt180, by = c("GID_2", "date"))),
  rebuilt_only = nrow(anti_join(diagnostic_active_gt180, existing_timing, by = c("GID_2", "date")))
) %>% mutate(jaccard = intersection / (intersection + existing_only + rebuilt_only))

current_mapping_check <- ground_truth %>% filter(!is.na(GID_2)) %>%
  select(raw_state_key, raw_municipality_key, expected_GID_2 = GID_2) %>% distinct() %>%
  left_join(crosswalk %>% select(raw_state_key, raw_municipality_key, rebuilt_GID_2 = GID_2),
            by = c("raw_state_key", "raw_municipality_key")) %>%
  mutate(matches = expected_GID_2 == rebuilt_GID_2)

event_audit <- raw %>% transmute(
  event_row_id, year, outage_date, time = as.character(time), length_min,
  clock_rollover_days,
  raw_state_key, raw_municipality_key, GID_2, mapping_method, tz_name,
  datetime_start_utc, datetime_end_utc, overpass_datetime_utc,
  overlaps_night, active_at_0130, classification_general, cause_mapping_method
)
saveRDS(event_audit, file.path(output_dir, "raw_event_mapping_timing_flags.rds"), compress = "gzip")
saveRDS(labels_raw, file.path(output_dir, "duration_labels_raw_mapped.rds"), compress = "gzip")
saveRDS(labels_model, file.path(output_dir, "duration_labels_model_keys.rds"), compress = "gzip")
write_csv(crosswalk, file.path(output_dir, "raw_municipality_crosswalk.csv"))
write_csv(manual_aliases, file.path(output_dir, "manual_aliases.csv"))
write_csv(mapping_coverage, file.path(output_dir, "mapping_coverage.csv"))
write_csv(variant_manifest, file.path(output_dir, "duration_variant_manifest.csv"))
write_csv(label_summary, file.path(output_dir, "duration_label_summary.csv"))
write_csv(label_counts_by_split, file.path(output_dir, "duration_label_counts_by_split.csv"))
write_csv(headline_reconciliation, file.path(output_dir, "headline_label_reconciliation.csv"))
write_csv(timing_reconciliation, file.path(output_dir, "active_overpass_label_reconciliation.csv"))
write_csv(current_mapping_check, file.path(output_dir, "existing_mapping_reconciliation.csv"))

config <- list(
  run_id = run_id, created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
  inputs = list(raw = raw_path, ground_truth = ground_truth_path, panel = panel_path,
                timezone_crosswalk = timezone_path),
  date_convention = "outage start date; nominal overpass is 01:30 local on the following date",
  nighttime_window = "22:00 on outage start date through 06:00 on following date",
  timezone = "municipality IANA/Olson zone with historical rules; event end is UTC start plus recorded duration",
  source_clock = "23 source clocks encoded as 24:xx are explicitly rolled to 00:xx on the following date",
  mapping = list(
    precedence = c("existing ground-truth mapping", "unique canonical municipality within state",
                   "documented manual alias"),
    unmapped_policy = "exclude; do not assign to a parent municipality",
    known_unmapped = "Puerto Morelos, Quintana Roo (absent from frozen panel geography)"
  ),
  cause_history = list(
    primary = "existing detailed-reason classification",
    fallback = "transparent CAUSA_SIMPLE mapping; otherwise Other"
  )
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
add_check("raw_support", nrow(raw) == 576717L && min(raw$length_min) == 6,
          paste(nrow(raw), min(raw$length_min), sep = "/"), "576717/6")
add_check("crosswalk_uniqueness", nrow(crosswalk) == 1417L &&
            !anyDuplicated(crosswalk[c("raw_state_key", "raw_municipality_key")]),
          paste(nrow(crosswalk), anyDuplicated(crosswalk[c("raw_state_key", "raw_municipality_key")]), sep = "/"),
          "1417/0")
add_check("mapping_coverage", mapping_coverage$mapped_share >= 0.999,
          sprintf("%.6f", mapping_coverage$mapped_share), ">=0.999")
add_check("existing_mapping_reproduction", all(current_mapping_check$matches),
          paste(sum(current_mapping_check$matches), nrow(current_mapping_check), sep = "/"), "all")
add_check("timezone_completeness", !any(is.na(raw$tz_name) & !is.na(raw$GID_2)),
          sum(is.na(raw$tz_name) & !is.na(raw$GID_2)), "0")
add_check("source_clock_rollover", sum(raw$clock_rollover_days) == 23L,
          sum(raw$clock_rollover_days), "23")
add_check("label_key_uniqueness", !anyDuplicated(labels_model[c("variant_id", "GID_2", "date")]),
          anyDuplicated(labels_model[c("variant_id", "GID_2", "date")]), "0")
night_counts <- label_summary %>% filter(variant_family == "night_duration_threshold") %>%
  arrange(duration_minutes) %>% pull(model_key_municipality_nights)
active_counts <- label_summary %>% filter(variant_family == "active_at_overpass_minimum_duration") %>%
  arrange(duration_minutes) %>% pull(model_key_municipality_nights)
add_check("duration_monotonicity", all(diff(night_counts) <= 0) && all(diff(active_counts) <= 0),
          paste(night_counts, active_counts, sep = "/", collapse = ";"), "nonincreasing")
add_check("headline_reconciliation", headline_reconciliation$jaccard >= 0.99,
          sprintf("%.6f", headline_reconciliation$jaccard), ">=0.99")
add_check("timing_reconciliation", timing_reconciliation$jaccard >= 0.99,
          sprintf("%.6f", timing_reconciliation$jaccard), ">=0.99")
add_check("split_support", all(label_counts_by_split$panel_rows > 0) &&
            nrow(label_counts_by_split) == 9L * 4L,
          nrow(label_counts_by_split), "36")
qa <- bind_rows(checks)
write_csv(qa, file.path(output_dir, "qa_checks.csv"))
status <- if (any(qa$status == "FAIL")) "FAIL" else "PASS"
report <- c(
  "# Phase 5 Duration Label Preparation", "",
  paste0("- Run ID: `", run_id, "`"), paste0("- Status: **", status, "**"),
  paste0("- Raw events: ", format(nrow(raw), big.mark = ",")),
  paste0("- Mapped events: ", format(mapping_coverage$mapped_event_rows, big.mark = ","),
         " (", sprintf("%.3f%%", 100 * mapping_coverage$mapped_share), ")"),
  "- Time conversion: municipality-specific IANA rules for 2017-2021",
  "- Variants: five night-overlap thresholds and four active-at-01:30 minimum durations", "",
  "## Label Counts", "",
  paste0("- ", label_summary$variant_id, ": ",
         format(label_summary$model_key_municipality_nights, big.mark = ","),
         " positive municipality-nights"), "",
  "## Reconciliation", "",
  paste0("- Current headline versus rebuilt >180-minute label Jaccard: ",
         sprintf("%.4f", headline_reconciliation$jaccard)),
  paste0("- Existing versus rebuilt active-at-overpass >180-minute label Jaccard: ",
         sprintf("%.4f", timing_reconciliation$jaccard)), "",
  "## QA", "", paste0("- PASS: ", sum(qa$status == "PASS")),
  paste0("- FAIL: ", sum(qa$status == "FAIL"))
)
writeLines(report, file.path(output_dir, "QA_REPORT.md"))
log_message("Phase 5 duration-label status: ", status)
if (status == "FAIL") stop("Phase 5 duration-label QA failed.")
