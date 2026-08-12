# Build overpass-timing outage labels without modifying the headline labels.
#
# The headline target remains outage_3h_or_more from the existing panel. This
# script derives separate robustness labels for outages visible around the
# nominal Black Marble / VIIRS overpass at 01:30 local wall-clock time.
#
# Outputs are written only under:
#   data/model_ready/timing_overpass/labels/

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lubridate)
  library(purrr)
  library(tibble)
})

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (basename(dirname(getwd())) == "scripts") return(normalizePath(file.path(getwd(), "..", "..")))
  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))
  normalizePath(getwd())
}

most_frequent <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[1]
}

flag_name <- function(buffer_minutes) sprintf("op0130_b%03d", as.integer(buffer_minutes))

normalise_ascii_upper <- function(x) {
  x <- toupper(trimws(iconv(as.character(x), to = "ASCII//TRANSLIT")))
  x <- gsub("\\.", "", x)
  x <- gsub("\\s+", " ", x)
  x
}

assign_mexico_iana_tz <- function(state_clean, municipality_clean) {
  state <- normalise_ascii_upper(state_clean)
  muni <- normalise_ascii_upper(municipality_clean)
  tz <- rep("America/Mexico_City", length(state))

  tz[state == "BAJA CALIFORNIA"] <- "America/Tijuana"
  tz[state == "BAJA CALIFORNIA SUR"] <- "America/Mazatlan"
  tz[state == "SINALOA"] <- "America/Mazatlan"
  tz[state == "SONORA"] <- "America/Hermosillo"
  tz[state == "QUINTANA ROO"] <- "America/Cancun"
  tz[state %in% c("CAMPECHE", "YUCATAN")] <- "America/Merida"
  tz[state %in% c("COAHUILA", "DURANGO", "NUEVO LEON", "TAMAULIPAS")] <- "America/Monterrey"

  nayarit <- state == "NAYARIT"
  tz[nayarit] <- "America/Mazatlan"
  tz[nayarit & muni == "BAHIA DE BANDERAS"] <- "America/Bahia_Banderas"

  chihuahua <- state == "CHIHUAHUA"
  tz[chihuahua] <- "America/Chihuahua"
  tz[chihuahua & muni %in% c("JANOS", "ASCENSION", "JUAREZ", "GUADALUPE",
                             "PRAXEDIS G GUERRERO")] <- "America/Ciudad_Juarez"
  tz[chihuahua & muni %in% c("COYAME DEL SOTOL", "OJINAGA",
                             "MANUEL BENAVIDES")] <- "America/Ojinaga"

  coah_border <- state == "COAHUILA" &
    muni %in% c("ACUNA", "ALLENDE", "GUERRERO", "HIDALGO", "JIMENEZ",
                "MORELOS", "NAVA", "OCAMPO", "PIEDRAS NEGRAS",
                "VILLA UNION", "ZARAGOZA")
  nl_border <- state == "NUEVO LEON" & muni == "ANAHUAC"
  tam_border <- state == "TAMAULIPAS" &
    muni %in% c("NUEVO LAREDO", "GUERRERO", "MIER", "MIGUEL ALEMAN",
                "CAMARGO", "GUSTAVO DIAZ ORDAZ", "REYNOSA", "RIO BRAVO",
                "VALLE HERMOSO", "MATAMOROS")
  tz[coah_border | nl_border | tam_border] <- "America/Matamoros"

  if (!all(tz %in% OlsonNames())) {
    stop("Internal error: some assigned IANA time zones are not available in OlsonNames().")
  }
  tz
}

local_clock_string <- function(x) {
  if (inherits(x, "POSIXt")) {
    return(format(x, "%Y-%m-%d %H:%M:%S", tz = "UTC"))
  }
  as.character(x)
}

local_clock_from_date_time <- function(date, time) {
  date_part <- as.Date(substr(as.character(date), 1, 10))
  time_part <- as.character(time)
  paste(date_part, time_part)
}

local_clock_to_utc <- function(clock_string, tz_name) {
  if (length(clock_string) != length(tz_name)) {
    stop("clock_string and tz_name must have the same length.")
  }
  out <- Map(function(x, z) {
    as.POSIXct(x, tz = z)
  }, clock_string, tz_name)
  as.POSIXct(as.numeric(unlist(out)), origin = "1970-01-01", tz = "UTC")
}

format_in_tz <- function(x, tz_name) {
  out <- Map(function(t, z) {
    format(t, "%Y-%m-%d %H:%M:%S", tz = z)
  }, x, tz_name)
  unlist(out, use.names = FALSE)
}

utc_offset_hours <- function(x, tz_name) {
  off <- Map(function(t, z) format(t, "%z", tz = z), x, tz_name)
  off <- unlist(off, use.names = FALSE)
  sign <- ifelse(substr(off, 1, 1) == "-", -1, 1)
  hours <- suppressWarnings(as.integer(substr(off, 2, 3)))
  mins <- suppressWarnings(as.integer(substr(off, 4, 5)))
  sign * (hours + mins / 60)
}

is_dst_in_tz <- function(x, tz_name) {
  out <- Map(function(t, z) {
    as.POSIXlt(t, tz = z)$isdst == 1L
  }, x, tz_name)
  unlist(out, use.names = FALSE)
}

add_timing_fields <- function(df, buffers) {
  df <- df %>%
    mutate(
      outage_date = as.Date(substr(as.character(date), 1, 10)),
      start_clock_local = if ("datetime_start" %in% names(.)) {
        local_clock_string(datetime_start)
      } else {
        local_clock_from_date_time(date, time)
      },
      end_clock_local = if ("datetime_end" %in% names(.)) {
        local_clock_string(datetime_end)
      } else {
        NA_character_
      },
      end_clock_local = if_else(
        is.na(end_clock_local),
        format(
          as.POSIXct(start_clock_local, tz = "UTC") + lubridate::minutes(length_min),
          "%Y-%m-%d %H:%M:%S",
          tz = "UTC"
        ),
        end_clock_local
      ),
      overpass_clock_local = paste(outage_date + 1L, "01:30:00"),
      datetime_start_utc = local_clock_to_utc(start_clock_local, tz_name),
      datetime_end_utc = local_clock_to_utc(end_clock_local, tz_name),
      overpass_datetime_utc = local_clock_to_utc(overpass_clock_local, tz_name),
      overpass_clock_local_check = format_in_tz(overpass_datetime_utc, tz_name),
      overpass_utc_offset_hours = utc_offset_hours(overpass_datetime_utc, tz_name),
      is_dst_at_overpass = is_dst_in_tz(overpass_datetime_utc, tz_name),
      minutes_start_to_overpass = as.numeric(
        difftime(overpass_datetime_utc, datetime_start_utc, units = "mins")
      ),
      minutes_end_to_overpass = as.numeric(
        difftime(overpass_datetime_utc, datetime_end_utc, units = "mins")
      )
    )

  for (b in buffers) {
    nm <- flag_name(b)
    df[[nm]] <- as.integer(
      !is.na(df$datetime_start_utc) &
        !is.na(df$datetime_end_utc) &
        df$datetime_start_utc <= df$overpass_datetime_utc &
        df$datetime_end_utc >= (df$overpass_datetime_utc - lubridate::minutes(b))
    )
  }
  df
}

variant_definition <- function(buffer_minutes) {
  if (buffer_minutes == 0L) {
    return("Outage active at nominal 01:30 local overpass.")
  }
  paste0(
    "Outage active at 01:30, or started before 01:30 and ended within ",
    buffer_minutes,
    " minutes before nominal 01:30 local overpass."
  )
}

build_muni_labels <- function(events, by_reason_events, variant_id) {
  selected <- events %>% filter(.data[[variant_id]] == 1L)

  if (nrow(selected) == 0L) {
    return(tibble())
  }

  out <- selected %>%
    group_by(GID_2, outage_date) %>%
    summarise(
      n_outages = n(),
      max_length_min = max(length_min, na.rm = TRUE),
      min_length_min = min(length_min, na.rm = TRUE),
      total_length_min = sum(length_min, na.rm = TRUE),
      mean_length_min = mean(length_min, na.rm = TRUE),
      median_length_min = median(length_min, na.rm = TRUE),
      earliest_start_hour = if (all(is.na(start_hour))) NA_real_ else min(start_hour, na.rm = TRUE),
      latest_start_hour = if (all(is.na(start_hour))) NA_real_ else max(start_hour, na.rm = TRUE),
      n_causes_general = n_distinct(classification_general[!is.na(classification_general)]),
      is_mixed_cause_general = as.integer(
        n_distinct(classification_general[!is.na(classification_general)]) > 1
      ),
      n_outages_environmental = sum(classification_general == "Environmental", na.rm = TRUE),
      n_outages_technical = sum(classification_general == "Technical", na.rm = TRUE),
      n_outages_planned = sum(classification_general == "Planned", na.rm = TRUE),
      n_outages_other = sum(classification_general == "Other", na.rm = TRUE),
      total_length_environmental = sum(length_min[classification_general == "Environmental"], na.rm = TRUE),
      total_length_technical = sum(length_min[classification_general == "Technical"], na.rm = TRUE),
      total_length_planned = sum(length_min[classification_general == "Planned"], na.rm = TRUE),
      total_length_other = sum(length_min[classification_general == "Other"], na.rm = TRUE),
      classification_general = most_frequent(classification_general),
      classification_detailed = most_frequent(classification_detailed),
      .groups = "drop"
    ) %>%
    mutate(
      variant_id = variant_id,
      date = outage_date,
      outage_3h_or_more = 1L
    )

  if (!is.null(by_reason_events) && nrow(by_reason_events) > 0L) {
    substation_summary <- by_reason_events %>%
      filter(.data[[variant_id]] == 1L) %>%
      group_by(GID_2, outage_date) %>%
      summarise(
        n_substations_affected = n_distinct(substation, na.rm = TRUE),
        n_distinct_nodes = n_distinct(node, na.rm = TRUE),
        primary_substation = most_frequent(substation),
        primary_node = most_frequent(node),
        dominant_reason = most_frequent(reason),
        .groups = "drop"
      )

    out <- out %>%
      left_join(substation_summary, by = c("GID_2", "outage_date")) %>%
      mutate(
        n_substations_affected = if_else(is.na(n_substations_affected), 0L, as.integer(n_substations_affected)),
        n_distinct_nodes = if_else(is.na(n_distinct_nodes), 0L, as.integer(n_distinct_nodes))
      )
  } else {
    out <- out %>%
      mutate(
        n_substations_affected = NA_integer_,
        n_distinct_nodes = NA_integer_,
        primary_substation = NA_character_,
        primary_node = NA_character_,
        dominant_reason = NA_character_
      )
  }

  out %>%
    select(
      variant_id, GID_2, date, outage_date,
      outage_3h_or_more, n_outages,
      max_length_min, min_length_min, total_length_min,
      mean_length_min, median_length_min,
      earliest_start_hour, latest_start_hour,
      n_causes_general, is_mixed_cause_general,
      n_outages_environmental, n_outages_technical,
      n_outages_planned, n_outages_other,
      total_length_environmental, total_length_technical,
      total_length_planned, total_length_other,
      classification_general, classification_detailed,
      n_substations_affected, n_distinct_nodes,
      primary_substation, primary_node, dominant_reason
    )
}

args <- commandArgs(trailingOnly = TRUE)
overwrite <- "--overwrite" %in% args

project_dir <- detect_project_dir()
data_dir <- file.path(project_dir, "data")
out_root <- file.path(data_dir, "model_ready", "timing_overpass")
label_dir <- file.path(out_root, "labels")
dir.create(label_dir, recursive = TRUE, showWarnings = FALSE)

main_path <- file.path(data_dir, "night_outages_3hrs_with_locations.csv")
reason_path <- file.path(data_dir, "night_outages_3hrs_with_locations_clean_by_reason.csv")
panel_path <- file.path(data_dir, "panel_mex_2017_2021_ntl_ghs.rds")
splits_path <- file.path(data_dir, "model_ready", "splits_fixed.rds")

required <- c(main_path, panel_path, splits_path)
missing <- required[!file.exists(required)]
if (length(missing) > 0L) {
  stop("Missing required files:\n", paste0("  - ", missing, collapse = "\n"))
}

output_files <- c(
  file.path(label_dir, "overpass_municipality_timezones.csv"),
  file.path(label_dir, "overpass_timezone_summary.csv"),
  file.path(label_dir, "overpass_variant_manifest.csv"),
  file.path(label_dir, "overpass_event_timing_flags.csv"),
  file.path(label_dir, "overpass_muni_night_labels_raw.rds"),
  file.path(label_dir, "overpass_muni_night_labels_model_keys.rds"),
  file.path(label_dir, "overpass_label_summary.csv"),
  file.path(label_dir, "overpass_label_counts_by_split.csv"),
  file.path(label_dir, "overpass_label_counts_by_year.csv"),
  file.path(label_dir, "overpass_label_qa.txt")
)
if (!overwrite && any(file.exists(output_files))) {
  stop(
    "Timing label outputs already exist. Re-run with --overwrite to replace only ",
    "files under: ", label_dir
  )
}

buffers <- c(0L, 30L, 60L, 120L, 180L, 240L)
variant_manifest <- tibble(
  variant_id = vapply(buffers, flag_name, character(1)),
  buffer_minutes = buffers,
  nominal_overpass_local_time = "01:30:00",
  time_rule = "civil_timezone_aware_nominal_0130",
  timezone_engine = "IANA/Olson time-zone database via base R",
  definition = vapply(buffers, variant_definition, character(1)),
  analysis_role = case_when(
    buffer_minutes %in% c(0L, 30L, 60L, 120L) ~ "recommended",
    buffer_minutes == 180L ~ "diagnostic_wide",
    TRUE ~ "diagnostic_near_current"
  )
)

message("Project dir: ", project_dir)
message("Output dir : ", label_dir)
message("Reading panel keys and assigning municipality time zones...")
panel <- readRDS(panel_path)
panel_keys <- panel %>%
  transmute(
    GID_2 = as.character(GID_2),
    date = as.Date(date),
    state_clean = as.character(state_clean),
    municipality_clean = as.character(municipality_clean),
    NAME_1 = as.character(NAME_1),
    NAME_2 = as.character(NAME_2),
    current_outage_3h_or_more = as.integer(outage_3h_or_more)
  )

municipality_tz <- panel_keys %>%
  distinct(GID_2, state_clean, municipality_clean, NAME_1, NAME_2) %>%
  mutate(tz_name = assign_mexico_iana_tz(state_clean, municipality_clean))

timezone_summary <- municipality_tz %>%
  count(tz_name, state_clean, name = "n_municipalities") %>%
  arrange(tz_name, state_clean)

message("Reading main outage ground truth...")

events <- read_csv(main_path, show_col_types = FALSE) %>%
  mutate(
    event_row_id = row_number(),
    GID_2 = as.character(GID_2),
    state_clean = as.character(state_clean),
    municipality_clean = as.character(municipality_clean)
  ) %>%
  left_join(municipality_tz %>% select(GID_2, tz_name), by = "GID_2") %>%
  mutate(tz_name = if_else(
    is.na(tz_name),
    assign_mexico_iana_tz(state_clean, municipality_clean),
    tz_name
  )) %>%
  add_timing_fields(buffers) %>%
  mutate(start_hour = lubridate::hour(as.POSIXct(start_clock_local, tz = "UTC")))

if (any(is.na(events$datetime_start_utc)) || any(is.na(events$datetime_end_utc))) {
  stop("Failed to parse some datetime_start/datetime_end values in main outage file.")
}

message("Reading cause/substation ground truth...")
if (file.exists(reason_path)) {
  by_reason_raw <- read_csv(reason_path, show_col_types = FALSE) %>%
    mutate(
      GID_2 = as.character(GID_2),
      state_clean = as.character(state_clean),
      municipality_clean = as.character(municipality_clean)
    ) %>%
    left_join(municipality_tz %>% select(GID_2, tz_name), by = "GID_2") %>%
    mutate(tz_name = if_else(
      is.na(tz_name),
      assign_mexico_iana_tz(state_clean, municipality_clean),
      tz_name
    )) %>%
    add_timing_fields(buffers)

  classification_lookup <- by_reason_raw %>%
    group_by(year, date, time, length_min, state, municipality) %>%
    summarise(
      classification_general = most_frequent(classification_general),
      classification_detailed = most_frequent(classification_detailed),
      .groups = "drop"
    )

  events <- events %>%
    left_join(
      classification_lookup,
      by = c("year", "date", "time", "length_min", "state", "municipality")
    )
} else {
  by_reason_raw <- NULL
  events <- events %>%
    mutate(
      classification_general = NA_character_,
      classification_detailed = NA_character_
    )
}

splits_fixed <- readRDS(splits_path)
split_for_date <- function(d) {
  case_when(
    d >= as.Date(splits_fixed$train_range[1]) & d <= as.Date(splits_fixed$train_range[2]) ~ "train",
    d >= as.Date(splits_fixed$val_range[1]) & d <= as.Date(splits_fixed$val_range[2]) ~ "val",
    d >= as.Date(splits_fixed$test_range[1]) & d <= as.Date(splits_fixed$test_range[2]) ~ "test",
    TRUE ~ "other"
  )
}

message("Aggregating timing labels by municipality-night...")
labels_raw <- map_dfr(variant_manifest$variant_id, function(v) {
  build_muni_labels(events, by_reason_raw, v)
})

if (nrow(labels_raw) == 0L) {
  stop("No timing labels were generated.")
}

dup_raw <- labels_raw %>%
  count(variant_id, GID_2, date) %>%
  filter(n > 1L)
if (nrow(dup_raw) > 0L) {
  stop("Duplicate raw timing labels detected for variant/GID_2/date.")
}

labels_model <- labels_raw %>%
  semi_join(panel_keys, by = c("GID_2", "date"))

dup_model <- labels_model %>%
  count(variant_id, GID_2, date) %>%
  filter(n > 1L)
if (nrow(dup_model) > 0L) {
  stop("Duplicate model-key timing labels detected for variant/GID_2/date.")
}

event_summary <- variant_manifest %>%
  rowwise() %>%
  mutate(
    event_rows_selected = sum(events[[variant_id]] == 1L, na.rm = TRUE),
    event_rows_total = nrow(events),
    event_rows_share = event_rows_selected / event_rows_total
  ) %>%
  ungroup()

raw_counts <- labels_raw %>%
  count(variant_id, name = "raw_muni_night_labels")

model_counts <- labels_model %>%
  count(variant_id, name = "model_key_muni_night_labels")

current_panel_positive <- sum(panel_keys$current_outage_3h_or_more == 1L, na.rm = TRUE)

label_summary <- variant_manifest %>%
  left_join(event_summary, by = c("variant_id", "buffer_minutes", "nominal_overpass_local_time",
                                  "time_rule", "timezone_engine", "definition", "analysis_role")) %>%
  left_join(raw_counts, by = "variant_id") %>%
  left_join(model_counts, by = "variant_id") %>%
  mutate(
    raw_muni_night_labels = coalesce(raw_muni_night_labels, 0L),
    model_key_muni_night_labels = coalesce(model_key_muni_night_labels, 0L),
    labels_dropped_not_in_panel = raw_muni_night_labels - model_key_muni_night_labels,
    current_panel_positive_labels = current_panel_positive,
    share_of_current_panel_positives = model_key_muni_night_labels / current_panel_positive
  )

panel_split_counts <- panel_keys %>%
  mutate(split = split_for_date(date)) %>%
  count(split, name = "panel_rows")

label_counts_by_split <- labels_model %>%
  mutate(split = split_for_date(date)) %>%
  count(variant_id, split, name = "positives") %>%
  tidyr::complete(variant_id = variant_manifest$variant_id, split = unique(panel_split_counts$split), fill = list(positives = 0L)) %>%
  left_join(panel_split_counts, by = "split") %>%
  mutate(prevalence = positives / panel_rows) %>%
  arrange(variant_id, split)

label_counts_by_year <- labels_model %>%
  mutate(year = lubridate::year(date)) %>%
  count(variant_id, year, name = "positives") %>%
  tidyr::complete(variant_id = variant_manifest$variant_id, year = 2017:2021, fill = list(positives = 0L)) %>%
  arrange(variant_id, year)

event_flags_out <- events %>%
  mutate(
    datetime_start_utc = format(datetime_start_utc, "%Y-%m-%d %H:%M:%S", tz = "UTC"),
    datetime_end_utc = format(datetime_end_utc, "%Y-%m-%d %H:%M:%S", tz = "UTC"),
    overpass_datetime_utc = format(overpass_datetime_utc, "%Y-%m-%d %H:%M:%S", tz = "UTC")
  )

write_csv(municipality_tz, file.path(label_dir, "overpass_municipality_timezones.csv"))
write_csv(timezone_summary, file.path(label_dir, "overpass_timezone_summary.csv"))
write_csv(variant_manifest, file.path(label_dir, "overpass_variant_manifest.csv"))
write_csv(event_flags_out, file.path(label_dir, "overpass_event_timing_flags.csv"))
saveRDS(labels_raw, file.path(label_dir, "overpass_muni_night_labels_raw.rds"))
saveRDS(labels_model, file.path(label_dir, "overpass_muni_night_labels_model_keys.rds"))
write_csv(label_summary, file.path(label_dir, "overpass_label_summary.csv"))
write_csv(label_counts_by_split, file.path(label_dir, "overpass_label_counts_by_split.csv"))
write_csv(label_counts_by_year, file.path(label_dir, "overpass_label_counts_by_year.csv"))

qa_lines <- c(
  "Overpass timing label QA",
  "",
  paste0("Generated on: ", format(Sys.time())),
  paste0("Project dir: ", project_dir),
  paste0("Main ground truth rows: ", nrow(events)),
  paste0("Panel rows: ", nrow(panel_keys)),
  paste0("Current panel positives: ", current_panel_positive),
  "",
  "Date convention:",
  "  outage_date = as.Date(substr(date, 1, 10)) from the existing ground truth.",
  "  nominal overpass = outage_date + 1 day at 01:30 local civil wall-clock time.",
  "  municipality time zones are assigned to IANA/Olson zones from state/municipality names.",
  "  outage and overpass wall-clock times are converted to UTC using historical IANA rules.",
  "  This handles 2017-2021 DST transitions and Mexico's multiple civil time zones.",
  "  It is still a nominal-overpass approximation; exact Black Marble acquisition-time layers would supersede it.",
  "",
  "Generated files:",
  paste0("  - ", basename(output_files))
)
writeLines(qa_lines, file.path(label_dir, "overpass_label_qa.txt"))

message("\nTiming label summary:")
print(label_summary %>%
        select(variant_id, buffer_minutes, analysis_role,
               event_rows_selected, model_key_muni_night_labels,
               share_of_current_panel_positives))

message("\nWrote timing label artifacts to: ", label_dir)
