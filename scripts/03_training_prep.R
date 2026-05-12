# 03_training_prep.R
# Goal:
#   draft for model training (panel + splits + QA) for outage prediction
#   we can run it even without all the data / while ML_Mexico_final.R is extracting
#   no feature engineering here => next script 04_feature_engineering.R
#
# what script does: 
#   ingest labels (outage ground truth)
#   ingest predictors (NTL features), using three modes:
#     A) full panel exists
#     B) partial daily files exist 
#     C) no NTL yet: skeleton panel 
#   validate schema  
#   add only basic calendar features 
#   save model mart + manifest + QA summary
#   save stable split definitions (date ranges, not row indices)
#   save reproducible rolling CV definition (dates used + params)
#
#   Note: all the lags/rolling/anomaly features part is in the next script (here is just preparation to model training)
#   because those require date-complete grids (or date-aware joins) 
# 
#
# SECTION 1: CONFIGURATION

# SAMPLE PANEL MODE
# Set to TRUE to run on stratified sample instead of full panel
# This uses data/panel_sample_20pct.rds and writes to data/model_ready_sample/
USE_SAMPLE_PANEL <- FALSE
SAMPLE_PCT <- 20  # must match the sample file created by 01_create_sample_panel.R

# pilot mode for faster iteration by subsetting to fewer municipalities
pilot_mode    <- FALSE # TRUE to enable pilot mode 
pilot_n_munis <- 100   # sample size if pilot selection still too large 
pilot_states  <- NULL  # Optional: vector of state_clean names to keep 
pilot_gid2    <- NULL  # Optional: explicit vector of GID_2 (municipalities) to keep
pilot_seed    <- 42    # Reproducibility seed for sampling

# date range (MUST match ML_Mexico_final.R extraction period)
# used for coverage QA and skeleton panel template
date_start <- as.Date("2017-01-01")
date_end   <- as.Date("2021-12-31")

# timezone for datetime => date conversion 
# Using UTC because ML_Mexico_final.R and ground truth timestamps are in UTC
label_tz <- "UTC"

# SECTION 2: LOAD PACKAGES

message("Loading packages...")

# suppressPackageStartupMessages() makes logs cleaner for repeated runs
suppressPackageStartupMessages({
  library(tidyverse) # all the usual dplyr... 
  library(lubridate) # date parsing and calendar features
  library(rsample)   # rolling-origin CV (cross vali) folds
  library(stringi)   # robust string normalization
})

# + helper for the join (weather data)
clean_key <- function(x) {
  x %>%
    as.character() %>%
    stringi::stri_trans_general("Latin-ASCII") %>%
    toupper() %>%
    stringr::str_squish()
}

# state name harmonisation: map panel short names to weather official names (I had a problem of names...)
harmonize_state_names <- function(state_vec) {
  # mapping from common short names to official full names used in weather data
  # confirmed by comparing unique(panel$state_clean) vs unique(weather_data$state_clean)
  state_map <- c(
    "COAHUILA"         = "COAHUILA DE ZARAGOZA",
    "MICHOACAN"        = "MICHOACAN DE OCAMPO",
    "VERACRUZ"         = "VERACRUZ DE IGNACIO DE LA LLAVE",
    "DISTRITO FEDERAL" = "CIUDAD DE MEXICO"
  )

  # applying mapping: if state is in map, replace it, otherwise keep original
  ifelse(state_vec %in% names(state_map), state_map[state_vec], state_vec)
}

# timezone-safe Date conversion: prevents date shifting from POSIXct with non-UTC timezones
to_utc_date <- function(x) {
  if (inherits(x, "POSIXt")) {
    as.Date(x, tz = "UTC")
  } else {
    as.Date(x)
  }
}

# SECTION 3: PATH HANDLING

# detect the project root robustly whether we run from:
#   - project root (contains data/ and scripts/)
#   - scripts/ subfolder
detect_project_dir <- function() {
  # if current working dir is ".../scripts", move up one level
  if (basename(getwd()) == "scripts") {
    return(normalizePath(file.path(getwd(), "..")))
  }
  # if we can see data/ or scripts/ in current directory, assume it's project root
  if (dir.exists("data") || dir.exists("scripts")) {
    return(normalizePath(getwd()))
  }
  # fallback: use current dir, but warn that assumption may be wrong
  warning("Could not determine project directory. Using current directory.")
  return(normalizePath(getwd()))
}

# resolve project and data directories once
project_dir <- detect_project_dir()
data_dir    <- file.path(project_dir, "data")

message("Project directory: ", project_dir)

# Input paths (read-only)
# muni boundaries (used only when no NTL yet => to create skeleton muni list)
gadm_path        <- file.path(data_dir, "gadm41_MEX.gpkg")
# ground truth outages with location and timestamps
outages_loc_path <- file.path(data_dir, "night_outages_3hrs_with_locations.csv")
# folder where ML_Mexico_final.R writes daily municipality-level features
features_dir     <- file.path(data_dir, "ntl_features_vnp46a2_ghs")
# final combined panel from ML_Mexico_final.R (if/when available).
# OR sample panel if USE_SAMPLE_PANEL is TRUE
if (USE_SAMPLE_PANEL) {
  panel_full_path <- file.path(data_dir, paste0("panel_sample_", SAMPLE_PCT, "pct.rds"))
  message("*** SAMPLE MODE: Using ", basename(panel_full_path), " ***")
} else {
  panel_full_path <- file.path(data_dir, "panel_mex_2017_2021_ntl_ghs.rds")
}
# weather data from ERA5
weather_path     <- file.path(data_dir, "weather", "full_weather_17_21.rds")

# output paths (new directories only)
# all outputs go into a folder that this script owns
if (USE_SAMPLE_PANEL) {
  model_ready_dir <- file.path(data_dir, "model_ready_sample")
  message("*** SAMPLE MODE: Outputs to ", model_ready_dir, " ***")
} else {
  model_ready_dir <- file.path(data_dir, "model_ready")
}

# create output directory if missing
dir.create(model_ready_dir, showWarnings = FALSE, recursive = TRUE)

# main outputs
mart_path          <- file.path(model_ready_dir, "model_mart.rds")
manifest_path      <- file.path(model_ready_dir, "model_mart_manifest.csv")
qa_path            <- file.path(model_ready_dir, "model_mart_qa.csv")
splits_fixed_path  <- file.path(model_ready_dir, "splits_fixed.rds")
splits_cv_path     <- file.path(model_ready_dir, "splits_cv_rolling.rds")
pilot_munis_path   <- file.path(model_ready_dir, "pilot_municipalities.csv")
cv_resamples_path <- file.path(model_ready_dir, "cv_resamples.rds")

# SECTION 4: EXPECTED SCHEMA (from ML_Mexico_final.R)

# identifier columns expected from GADM + cleaning step in extraction (TO CHECK AGAIN, but I think these are the right columns)
# NOTE: *_orig columns (from extraction) and *_pre_weather columns (from this script) are for QA/debugging only
# They must NOT be used as modeling features (high cardinality, leakage risk)
id_cols <- c("GID_2", "GID_1", "GID_0", "COUNTRY", "NAME_1", "NAME_2",
             "state_clean", "municipality_clean",
             "state_clean_orig", "municipality_clean_orig",
             "state_clean_pre_weather", "municipality_clean_pre_weather")

# NTL feature columns expected from the extraction step
# includes three weighting schemes and the built-up shares
ntl_cols <- c(
  # Scheme 1: all pixels in muni
  "ntl_mean_all", "ntl_sum_all", "ntl_sd_all", "wsum_all",
  # Scheme 2: weighted by built surface share
  "ntl_mean_built", "ntl_sum_built", "ntl_sd_built", "wsum_built",
  # Scheme 3: built-mask binary
  "ntl_mean_built_mask", "ntl_sum_built_mask", "ntl_sd_built_mask", "wsum_built_mask",
  # coverage / share variables for built weighting/mask
  "built_share_area", "built_share_mask"
)

# weather feature columns from ERA5 (preprocessed)
# these may affect NTL detected values (cloud cover, rain, etc.)
weather_cols <- c(
  # Temperature variables
  "temp", "max_temp", "min_temp", "skin_temp",
  # Humidity variables
  "dew", "max_dew", "min_dew", "rh",
  # Precipitation
  "rain",
  # Atmospheric pressure
  "atm",
  # Wind variables
  "wind_u", "wind_v", "wsp", "wdr",
  # Vegetation indices
  "lai_high", "lai_low"
)

# climate classification columns (categorical, may need encoding for modeling)
weather_climate_cols <- c("climate", "sub_climate", "rain_season", "koppen")

# label/outcome columns that the model mart should contain or create
outcome_cols <- c("outage_3h_or_more", "n_outages", "max_length_min", "min_length_min",
                   "total_length_min", "mean_length_min", "median_length_min",
                   "classification_general", "classification_detailed",
                   "n_causes_general", "is_mixed_cause_general",
                   "n_outages_environmental", "n_outages_technical",
                   "n_outages_planned", "n_outages_other",
                   "total_length_environmental", "total_length_technical",
                   "total_length_planned", "total_length_other",
                   "n_substations_affected")

# SECTION 5: LOAD OUTAGE LABELS

message("\n", strrep("=", 60))
message("STEP 1: Loading outage labels")
message(strrep("=", 60))

# ground truth required (even if not all NTL there yet)
if (!file.exists(outages_loc_path)) {
  stop("Required file not found: ", outages_loc_path)
}

# read outage CSV
outages_raw <- read_csv(outages_loc_path, show_col_types = FALSE)
message("Loaded ", nrow(outages_raw), " raw outage records")

# force-parse datetime_start (specific format expected)
# using label_tz for consistency
outages_raw <- outages_raw %>%
  mutate(datetime_start = ymd_hms(as.character(datetime_start), tz = label_tz, quiet = TRUE))

# guard against parse failures (NA datetimes)
n_bad_datetime <- sum(is.na(outages_raw$datetime_start))
if (n_bad_datetime > 0) {
  pct_bad_dt <- round(n_bad_datetime / nrow(outages_raw) * 100, 2)
  warning("Found ", n_bad_datetime, " rows (", pct_bad_dt,
          "%) with unparseable datetime_start. Dropping them.")
  outages_raw <- outages_raw %>% filter(!is.na(datetime_start))
  message("After dropping bad datetimes: ", nrow(outages_raw), " records remain")
}

# drop rows with missing GID_2 (failed municipality matching in preprocessing) => I saw there were around 8% 
# around 8.6% of rows have missing GID_2 => these can never join to NTL features
n_missing_gid2 <- sum(is.na(outages_raw$GID_2))
pct_missing_gid2 <- round(n_missing_gid2 / nrow(outages_raw) * 100, 1)
if (n_missing_gid2 > 0) {
  warning("Ground truth has ", n_missing_gid2, " rows (", pct_missing_gid2,
          "%) with missing GID_2 (won't join to NTL). Dropping them.")
  outages_raw <- outages_raw %>% filter(!is.na(GID_2))
  message("After dropping missing GID_2: ", nrow(outages_raw), " outage records remain")
}

# aggregate to municipality-date level to match the modeling unit:
# one row per (GID_2, date), label indicates any 3h+ outage that day
# designed to match ML_Mexico_final.R
# IMPORTANT: join by GID_2 + date ONLY (not names) to avoid silent label drops from name mismatches
outage_labels <- outages_raw %>%
  # convert timestamp to Date
  mutate(outage_date = as.Date(datetime_start)) %>%
  # group by GID_2 and date only (GID_2 is unique and trustworthy)
  group_by(GID_2, outage_date) %>%
  summarise(
    # number of outage events within the municipality-day
    n_outages      = n(),
    # max outage length that day (minutes)
    # guarding against -Inf when all length_min are NA
    max_length_min = if (all(is.na(length_min))) NA_real_ else max(length_min, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  # binary label: at least one outage >= 3h for that muni-day
  mutate(outage_3h_or_more = 1L)

message("Aggregated to ", nrow(outage_labels), " municipality-date outage events")

# keeping a unique municipality list derived from outage data (fallback if no GADM)
# NOTE: use outages_raw (not outage_labels) since outage_labels no longer has state_clean/municipality_clean
muni_from_outages <- outages_raw %>%
  filter(!is.na(GID_2)) %>%
  distinct(GID_2, state_clean, municipality_clean)

# SECTION 6: PILOT MODE (optional subsetting)

if (pilot_mode) {
  message("\n", strrep("=", 60))
  message("PILOT MODE: Selecting subset of municipalities")
  message(strrep("=", 60))
  # make sampling reproducible
  set.seed(pilot_seed)

  # def the universe of municipalities:
  # prefer GADM list (complete), otherwise fallback to municipalities seen in outages
  if (file.exists(gadm_path)) {
    # sf is only needed when reading GADM, so we load it conditionally
    suppressPackageStartupMessages(library(sf))
    muni_universe <- sf::st_read(gadm_path, layer = "ADM_ADM_2", quiet = TRUE) %>%
      sf::st_drop_geometry() %>% # remove geometry to keep memory small
      mutate(
        # normalise strings to match the extraction/label join conventions
        state_clean = stri_trans_general(toupper(NAME_1), "Latin-ASCII"),
        municipality_clean = stri_trans_general(toupper(NAME_2), "Latin-ASCII")
      ) %>%
      distinct(GID_2, GID_1, state_clean, municipality_clean, NAME_1, NAME_2)
  } else {
    muni_universe <- muni_from_outages
  }

  # starting from full universe, then applying filters and sampling
  pilot_munis <- muni_universe
  # if explicit set of GID_2 
  if (!is.null(pilot_gid2)) {
    pilot_munis <- pilot_munis %>% filter(GID_2 %in% pilot_gid2)
  } else if (!is.null(pilot_states)) {
    # otherwise, filter by state names (already normalised to uppercase before)
    pilot_munis <- pilot_munis %>% filter(state_clean %in% toupper(pilot_states))
  }
  # if still too large, sampling down to pilot_n_munis
  if (nrow(pilot_munis) > pilot_n_munis) {
    pilot_munis <- pilot_munis %>% slice_sample(n = pilot_n_munis)
  }
  # record selected municipality IDs for filtering the panel later
  pilot_gid2_selected <- pilot_munis$GID_2
  # save selection 
  write_csv(pilot_munis, pilot_munis_path)
  message("Selected ", length(pilot_gid2_selected), " pilot municipalities")

  # restrict label table to pilot municipalities
  outage_labels <- outage_labels %>% filter(GID_2 %in% pilot_gid2_selected)
} else {
  # when pilot mode is off, keep NULL 
  pilot_gid2_selected <- NULL
}

# SECTION 7: LOAD NTL DATA (three modes)

message("\n", strrep("=", 60))
message("STEP 2: Loading NTL data")
message(strrep("=", 60))

# initialise panel object and data provenance tag
panel <- NULL
data_source <- "none"

# availability checks:
#   - full combined panel (best case)
#   - directory containing daily files (what there is, even partial extraction)
has_full_panel <- file.exists(panel_full_path)
has_features_dir <- dir.exists(features_dir)

# when extraction running, a daily file could be partially written
# to reduce readRDS failures / corrupted reads, we ignore "too recent" files
min_file_age_seconds <- 120
# build list of safe daily files (if features_dir exists)
daily_files <- if (has_features_dir) {
  # getting all candidate daily files written by ML_Mexico_final.R
  all_files <- list.files(features_dir, pattern = "^ntl_features_\\d{4}_\\d{2}_\\d{2}\\.rds$", full.names = TRUE)
  if (length(all_files) > 0) {
    # get modification times to compute file age
    file_info <- file.info(all_files)
    # age = time since last modification (seconds)
    file_age_seconds <- as.numeric(difftime(Sys.time(), file_info$mtime, units = "secs"))
    # keep only files older than guard window
    safe_files <- all_files[file_age_seconds >= min_file_age_seconds]
    # log how many we skipped 
    n_skipped <- length(all_files) - length(safe_files)
    if (n_skipped > 0) {
      message("  Skipping ", n_skipped, " file(s) modified in last ", min_file_age_seconds, " seconds")
    }
    safe_files
  } else {
    character(0)
  }
} else {
  character(0)
}

# MODE A: Full panel exists
if (has_full_panel) {
  message("Found complete panel: ", panel_full_path)
  # read the combined panel
  panel <- readRDS(panel_full_path) %>%
    # CRITICAL: Convert date to Date type BEFORE joining labels
    # If panel$date is POSIXct/character and outage_labels$date is Date,
    # the join will silently fail and produce all-zero labels
    # Use timezone-safe conversion to prevent date shifting from non-UTC POSIXct
    mutate(date = to_utc_date(date))
  data_source <- "full_panel"

  # apply pilot filter
  if (pilot_mode && !is.null(pilot_gid2_selected)) {
    panel <- panel %>% filter(GID_2 %in% pilot_gid2_selected)
  }

  # full panel should already have labels,if not => join
  if (!"outage_3h_or_more" %in% names(panel)) {
    message("  Full panel missing labels - joining outage_labels...")
    # joining labels by GID_2 + date ONLY (safer than including names which can mismatch)
    panel <- panel %>%
      left_join(
        outage_labels %>% rename(date = outage_date),
        by = c("GID_2", "date")
      ) %>%
      # filling NAs for label outcomes for "no outage that day"
      # Guard against -Inf from max(length_min, na.rm=TRUE) on all-NA groups
      mutate(
        outage_3h_or_more = if_else(is.na(outage_3h_or_more), 0L, outage_3h_or_more),
        n_outages = if_else(is.na(n_outages), 0L, n_outages),
        max_length_min = if_else(is.na(max_length_min) | !is.finite(max_length_min), 0, max_length_min)
      )
  } else {
    # sanity check: labels exist, but verify prevalence isn't NA or exactly zero
    label_prevalence <- mean(panel$outage_3h_or_more == 1, na.rm = TRUE)
    if (is.na(label_prevalence) || label_prevalence == 0) {
      warning("Full panel has outage_3h_or_more column but prevalence is 0 or NA. Check data integrity.")
    }
  }

  message("Loaded ", nrow(panel), " rows from full panel")
} else if (length(daily_files) > 0) {
  # MODE B: Daily feature files exist
  message("No full panel. Found ", length(daily_files), " daily feature files")

  # read daily files and row-bind them
  # tryCatch ensures one bad file doesn't break the entire run
  panel <- map_dfr(daily_files, function(f) {
    tryCatch({
      df <- readRDS(f) %>%
        # CRITICAL: Convert date to Date type immediately after reading
        # If df$date is POSIXct/character, the join with labels will silently fail
        # Use timezone-safe conversion to prevent date shifting
        mutate(date = to_utc_date(date))
      # apply pilot filter inside the loop
      if (pilot_mode && !is.null(pilot_gid2_selected)) {
        df <- df %>% filter(GID_2 %in% pilot_gid2_selected)
      }
      df
    }, error = function(e) {
      message("Warning: Could not read ", basename(f))
      NULL
    })
  })

  # if we loaded anything, check for duplicates BEFORE joining labels
  if (nrow(panel) > 0) {
    # MODE B DUPLICATE CHECK: daily files can overlap if extraction is re-run
    # This can create multiple rows for the same (GID_2, date) with different values
    # We must detect and handle this BEFORE joining labels for data integrity
    n_rows_before_dedup <- nrow(panel)
    dup_keys <- panel %>%
      group_by(GID_2, date) %>%
      filter(n() > 1) %>%
      ungroup()

    n_dup_rows <- nrow(dup_keys)

    if (n_dup_rows > 0) {
      n_dup_keys <- dup_keys %>% distinct(GID_2, date) %>% nrow()
      warning("MODE B: Found ", n_dup_rows, " duplicate rows across ", n_dup_keys,
              " (GID_2, date) keys from overlapping daily files.")

      # STRATEGY: Keep only the first occurrence (arbitrary but deterministic)
      # BETTER ALTERNATIVES (if you want to implement later):
      #   1) Keep row from newest file (requires tracking file modification time)
      #   2) Average numeric features across duplicates
      #   3) Hard stop and force manual cleanup (safest for production)

      message("  Keeping first occurrence of each duplicate key...")

      # Save duplicate info for inspection
      dup_summary <- dup_keys %>%
        group_by(GID_2, date) %>%
        summarise(n_duplicates = n(), .groups = "drop") %>%
        arrange(desc(n_duplicates))

      write_csv(
        dup_summary,
        file.path(model_ready_dir, "mode_b_duplicate_keys.csv")
      )
      message("  Saved duplicate key summary to mode_b_duplicate_keys.csv")

      # Deduplicate: keep first occurrence
      panel <- panel %>%
        group_by(GID_2, date) %>%
        slice(1) %>%
        ungroup()

      n_rows_after_dedup <- nrow(panel)
      message("  Removed ", n_rows_before_dedup - n_rows_after_dedup, " duplicate rows")
    } else {
      message("  No duplicate (GID_2, date) keys found in daily files")
    }

    # NOW join labels and define non-outage days
    # join outage labels by GID_2 + date ONLY
    panel <- panel %>%
      left_join(
        outage_labels %>% rename(date = outage_date),
        by = c("GID_2", "date")
      ) %>%
      # Guard against -Inf from max(length_min, na.rm=TRUE) on all-NA groups
      mutate(
        outage_3h_or_more = if_else(is.na(outage_3h_or_more), 0L, outage_3h_or_more),
        n_outages = if_else(is.na(n_outages), 0L, n_outages),
        max_length_min = if_else(is.na(max_length_min) | !is.finite(max_length_min), 0, max_length_min)
      )
    data_source <- "partial_daily"
  }
}

# MODE C: No NTL data => building skeleton from outage labels only
if (is.null(panel) || nrow(panel) == 0) {
  message("No NTL data available. Building skeleton panel from outage labels...")

  # get municipality list: GADM is preferred option
  if (file.exists(gadm_path)) {
    suppressPackageStartupMessages(library(sf))
    muni_list <- sf::st_read(gadm_path, layer = "ADM_ADM_2", quiet = TRUE) %>%
      sf::st_drop_geometry() %>%
      mutate(
        state_clean = stri_trans_general(toupper(NAME_1), "Latin-ASCII"),
        municipality_clean = stri_trans_general(toupper(NAME_2), "Latin-ASCII")
      ) %>%
      # keeping only ID/name columns we need
      select(GID_2, GID_1, GID_0, COUNTRY, NAME_1, NAME_2, state_clean, municipality_clean) %>%
      distinct()
    # apply pilot filter if enabled
    if (pilot_mode && !is.null(pilot_gid2_selected)) {
      muni_list <- muni_list %>% filter(GID_2 %in% pilot_gid2_selected)
    }
  } else {
    # fallback: build minimal ID columns from outages
    # ROBUSTNESS: ensure unique by GID_2 (outages can have name variations)
    muni_list <- muni_from_outages %>%
      group_by(GID_2) %>%
      summarise(
        state_clean = first(state_clean),
        municipality_clean = first(municipality_clean),
        .groups = "drop"
      ) %>%
      mutate(GID_1 = NA_character_, GID_0 = "MEX", COUNTRY = "Mexico",
             NAME_1 = state_clean, NAME_2 = municipality_clean)
  }

  # Sparse skeleton approach:
  #   - include all observed outage rows (positive days)
  #   - plus one template row per municipality at date_start (a placeholder)
  # avoids building the full muni x date grid before NTL extraction is ready
  # Join by GID_2 only (safer than including names)
  outage_rows <- outage_labels %>%
    rename(date = outage_date) %>%
    left_join(muni_list, by = "GID_2")

  template_rows <- muni_list %>%
    mutate(date = date_start, outage_3h_or_more = 0L, n_outages = 0L, max_length_min = 0)
  
  # combine and keep one row per key
  panel <- bind_rows(outage_rows, template_rows) %>%
    distinct(GID_2, date, .keep_all = TRUE)

  # add placeholder NTL columns so downstream scripts see the expected schema
  for (col in ntl_cols) {
    panel[[col]] <- NA_real_
  }

  data_source <- "skeleton"
  message("Built skeleton with ", nrow(panel), " rows")
}

# NOTE: date conversion now happens immediately after reading panel in each mode
# This line is kept as a safety guard in case Mode C (skeleton) needs it
panel <- panel %>%
  mutate(date = to_utc_date(date))

# SECTION 7b: JOIN WEATHER DATA

message("\n", strrep("=", 60))
message("STEP 2b: Joining weather data")
message(strrep("=", 60))

# join keys for weather
weather_keys <- c("state_clean", "municipality_clean", "date")

# normalise join keys on PANEL side (before key checks / joining)
# IMPORTANT: preserve keys before transformation for weather join diagnostics
# NOTE: Do NOT overwrite *_orig columns if they exist (those come from extraction pipeline)
panel <- panel %>%
  mutate(
    state_clean_pre_weather = state_clean,
    municipality_clean_pre_weather = municipality_clean,
    state_clean = harmonize_state_names(clean_key(state_clean)),
    municipality_clean = clean_key(municipality_clean)
  )

if (file.exists(weather_path)) {
  message("Loading weather data: ", weather_path)
  
  # reading + min normalisation early to improve joins
  weather_data <- readRDS(weather_path) %>%
    mutate(
      date = to_utc_date(date),
      state_clean = clean_key(state_clean),
      municipality_clean = clean_key(municipality_clean)
    ) %>%
    # keeping only needed columns to save memory
    dplyr::select(any_of(c(weather_keys, weather_cols, weather_climate_cols)))
  
  # basic diagnostics
  message("Weather data: ", nrow(weather_data), " rows, ",
          n_distinct(weather_data$state_clean), " states, ",
          n_distinct(paste(weather_data$state_clean, weather_data$municipality_clean)), " municipalities")
  
  # JOIN: check for duplicate keys BEFORE joining (I had a problem of some missing muni...)
  dups <- weather_data %>%
    count(across(all_of(weather_keys)), name = "n") %>%
    filter(n > 1)
  
  if (nrow(dups) > 0) {
    warning("Weather has duplicate keys (", nrow(dups), " key(s)). Collapsing to unique keys by averaging numerics.")
    
    # safe mean that handles all-NA columns
    safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
    
    weather_data <- weather_data %>%
      group_by(across(all_of(weather_keys))) %>%
      summarise(
        across(any_of(weather_cols), safe_mean),
        across(any_of(weather_climate_cols), ~ dplyr::first(.x)),
        .groups = "drop"
      )
    message("  After collapsing: ", nrow(weather_data), " unique key rows")
  }
  
  # 2) PERSIST DIAGNOSTICS 
  #    - write muni-level counts to CSV
  #    - save full date-level unmatched keys as RDS (compressed), NOT CSV
  panel_keys <- panel %>%
    distinct(across(all_of(weather_keys)))
  
  weather_data_keys <- weather_data %>%
    distinct(across(all_of(weather_keys)))
  
  unmatched_panel_keys <- anti_join(panel_keys, weather_data_keys, by = weather_keys)
  
  if (nrow(unmatched_panel_keys) > 0) {
    # muni-level 
    unmatched_muni_counts <- unmatched_panel_keys %>%
      count(state_clean, municipality_clean, name = "n_unmatched_days") %>%
      arrange(desc(n_unmatched_days))
    
    write_csv(
      unmatched_muni_counts,
      file.path(model_ready_dir, "weather_unmatched_muni_counts.csv")
    )
    
    # save full date-level unmatched keys as RDS (compressed)
    saveRDS(
      unmatched_panel_keys,
      file.path(model_ready_dir, "weather_unmatched_panel_keys.rds")
    )
    
    # state-level rollup
    unmatched_state_summary <- unmatched_muni_counts %>%
      group_by(state_clean) %>%
      summarise(
        n_unmatched_munis = n(),
        total_unmatched_days = sum(n_unmatched_days),
        .groups = "drop"
      ) %>%
      arrange(desc(total_unmatched_days))
    
    write_csv(
      unmatched_state_summary,
      file.path(model_ready_dir, "weather_unmatched_state_summary.csv")
    )
    
    message("  Saved muni-level unmatched counts to weather_unmatched_muni_counts.csv")
    message("  Saved full date-level unmatched keys to weather_unmatched_panel_keys.rds")
    message("  Saved state-level summary to weather_unmatched_state_summary.csv")
  } else {
    message("  All panel keys matched to weather keys (no unmatched keys found).")
  }
  
  # Join weather to panel
  n_before <- nrow(panel)
  panel <- panel %>%
    left_join(weather_data, by = weather_keys)
  n_after <- nrow(panel)
  
  # Check join didn't create duplicates (should NEVER happen after key deduplication)
  # Hard stop for data integrity - training on duplicated data would be wrong
  if (n_after != n_before) {
    stop("FATAL: Weather join changed row count! Before: ", n_before, ", After: ", n_after,
         ". This indicates duplicate keys in weather_data that weren't caught by deduplication. ",
         "Check weather_data for duplicate (state_clean, municipality_clean, date) keys.")
  }
  
  # 3) IMPROVED MATCH RATE: match if ANY weather var (numeric or climate) joined
  weather_all_present <- intersect(c(weather_cols, weather_climate_cols), names(panel))
  if (length(weather_all_present) > 0) {
    # Count row as matched if ANY weather variable OR climate variable is non-NA
    weather_matched_flag <- rowSums(!is.na(panel[, weather_all_present, drop = FALSE])) > 0
    n_weather_matched <- sum(weather_matched_flag)

    # State-level match rate for diagnostics
    state_match_rates <- panel %>%
      mutate(has_weather = weather_matched_flag) %>%
      group_by(state_clean) %>%
      summarise(
        n_rows = n(),
        n_matched = sum(has_weather),
        pct_matched = round(n_matched / n_rows * 100, 1),
        .groups = "drop"
      ) %>%
      arrange(pct_matched, state_clean)

    write_csv(
      state_match_rates,
      file.path(model_ready_dir, "weather_state_match_rates.csv")
    )
    message("  Saved state-level match rates to weather_state_match_rates.csv")

    # Show states with low match rates
    low_match_states <- state_match_rates %>% filter(pct_matched < 50)
    if (nrow(low_match_states) > 0) {
      message("  WARNING: ", nrow(low_match_states), " state(s) with <50% weather match:")
      print(low_match_states)
    }
  } else {
    n_weather_matched <- 0
  }
  pct_weather_matched <- round(n_weather_matched / nrow(panel) * 100, 1)

  message("Weather matched: ", n_weather_matched, " / ", nrow(panel),
          " rows (", pct_weather_matched, "%)")
  
  weather_available <- TRUE
} else {
  message("Weather file not found: ", weather_path)
  message("Proceeding without weather data.")
  
  # Add NA placeholder columns for weather (typed)
  for (col in weather_cols) {
    panel[[col]] <- NA_real_
  }
  for (col in weather_climate_cols) {
    panel[[col]] <- NA_character_
  }
  
  weather_available <- FALSE
  n_weather_matched <- 0
  pct_weather_matched <- 0
}

# SECTION 8: SCHEMA VALIDATION

message("\n", strrep("=", 60))
message("STEP 3: Schema validation")
message(strrep("=", 60))

# min columns required for any downstream modeling
required_cols <- c("GID_2", "date", "outage_3h_or_more")

# if any required column missing, stop early
missing_required <- setdiff(required_cols, names(panel))
if (length(missing_required) > 0) {
  stop("Missing required columns: ", paste(missing_required, collapse = ", "))
}

# check which NTL columns are present 
present_ntl <- intersect(ntl_cols, names(panel))
missing_ntl <- setdiff(ntl_cols, names(panel))

message("Required columns: OK")
message("NTL columns present: ", length(present_ntl), " / ", length(ntl_cols))
if (length(missing_ntl) > 0) {
  message("  Missing: ", paste(missing_ntl, collapse = ", "))
}

# Check weather columns
present_weather <- intersect(weather_cols, names(panel))
missing_weather <- setdiff(weather_cols, names(panel))
message("Weather columns present: ", length(present_weather), " / ", length(weather_cols))
if (length(missing_weather) > 0) {
  message("  Missing: ", paste(missing_weather, collapse = ", "))
}

# key uniqueness check:
# we want at most one row per (GID_2, date)
# NOTE: MODE B duplicates should have been caught earlier (before label join)
# If we find duplicates here, they came from MODE A (full panel), MODE C (skeleton), or a join bug
n_duplicates <- panel %>%
  group_by(GID_2, date) %>%
  filter(n() > 1) %>% # keep only keys with duplicates
  nrow()              # count duplicate rows (not groups)

if (n_duplicates > 0) {
  warning("Found ", n_duplicates, " rows with duplicate (GID_2, date) keys after all joins!")
  message("  Data source: ", data_source)
  if (data_source == "partial_daily") {
    warning("  MODE B duplicates should have been caught earlier - this may indicate a join bug")
  }
  message("  Keeping first occurrence of each duplicate...")
  panel <- panel %>%
    group_by(GID_2, date) %>%
    slice(1) %>% # if we want deterministic pref need to add arrange() before slice()
    ungroup()
  message("  Panel now has ", nrow(panel), " rows after deduplication")
} else {
  message("Key uniqueness (GID_2, date): OK")
}

# SECTION 9: ADD BASIC CALENDAR FEATURES ONLY

message("\n", strrep("=", 60))
message("STEP 4: Adding basic calendar features")
message(strrep("=", 60))

# calendar covariates "safe" and do not depend on complete time series
# can be used immediately for baseline models and sanity checks 
# later can add holidays through API github : see link on our doc 
panel <- panel %>%
  mutate(
    # numeric day of week (1 = Monday, 7 = Sunday)
    dow_num = wday(date, week_start = 1),
    # weekend = Saturday (6) or Sunday (7): stored as integer (0/1) for modeling.
    is_weekend = as.integer(dow_num >= 6),
    # Month (1-12)
    month = month(date),
    # Year
    year = year(date),
    # Day of year (1-366)
    doy = yday(date)
  )

message("Added: dow_num, is_weekend, month, year, doy")

# SECTION 10: SAVE MODEL MART

message("\n", strrep("=", 60))
message("STEP 5: Saving model mart")
message(strrep("=", 60))

# saving mart as RDS for reload in later scripts
saveRDS(panel, mart_path)
message("Saved model mart to: ", mart_path)

# manifest is a readable inventory of columns:
# types, missingness, uniqueness, and category tag
manifest <- tibble(
  column = names(panel),
  type = sapply(panel, function(x) class(x)[1]),
  n_missing = sapply(panel, function(x) sum(is.na(x))),
  pct_missing = sapply(panel, function(x) mean(is.na(x)) * 100),
  n_unique = sapply(panel, function(x) n_distinct(x, na.rm = TRUE)),
  category = case_when(
    column %in% id_cols ~ "identifier",
    column %in% ntl_cols ~ "ntl_feature",
    column %in% weather_cols ~ "weather_feature",
    column %in% weather_climate_cols ~ "weather_climate",
    column %in% outcome_cols ~ "outcome",
    column %in% c("date", "dow_num", "is_weekend", "month", "year", "doy") ~ "calendar",
    TRUE ~ "other"
  )
)

# save manifest to CSV
write_csv(manifest, manifest_path)
message("Saved manifest to: ", manifest_path)

# SECTION 11: QA SUMMARY

message("\n", strrep("=", 60))
message("STEP 6: QA summary")
message(strrep("=", 60))

# dates present in the panel 
dates_in_panel <- sort(unique(panel$date))
# expected date sequence based on configured project range
expected_dates <- seq.Date(date_start, date_end, by = "day")
# Date-level coverage: "Do we have any data for each day?"
dates_present <- sum(expected_dates %in% dates_in_panel)
dates_missing <- sum(!expected_dates %in% dates_in_panel)

# NTL data availability (safe check for missing columns)
has_ntl_mean_built <- if ("ntl_mean_built" %in% names(panel)) {
  mean(!is.na(panel$ntl_mean_built))
} else {
  0
}
has_ntl_mean_all <- if ("ntl_mean_all" %in% names(panel)) {
  mean(!is.na(panel$ntl_mean_all))
} else {
  0
}

has_ntl <- has_ntl_mean_built

# muni-day coverage: compares current row count to a full muni x day grid
# helps quantify how complete the partial panel is
n_munis <- n_distinct(panel$GID_2)
n_expected_muni_days <- n_munis * length(expected_dates)
pct_muni_days_present <- nrow(panel) / n_expected_muni_days * 100

# QA summary
# Note: n_bad_datetime, n_missing_gid2, pct_missing_gid2 were computed during label loading
# Note: weather_available, n_weather_matched, pct_weather_matched were computed during weather join
qa_summary <- tibble(
  metric = c(
    "data_source",
    "n_rows",
    "n_municipalities",
    "n_dates_in_panel",
    "n_dates_expected",
    "pct_dates_covered",
    "n_dates_present",
    "n_dates_missing",
    "n_muni_days_expected",
    "pct_muni_days_present",
    "n_duplicate_keys_found",
    "date_min",
    "date_max",
    "n_outage_days",
    "n_total_outage_events",
    "outage_prevalence_pct",
    "pct_ntl_mean_built_available",
    "pct_ntl_mean_all_available",
    "weather_available",
    "pct_weather_matched",
    "n_bad_datetime_dropped",
    "n_missing_gid2_dropped",
    "pct_missing_gid2_dropped",
    "pilot_mode",
    "timestamp"
  ),
  value = c(
    data_source,
    as.character(nrow(panel)),
    as.character(n_munis),
    as.character(length(dates_in_panel)),
    as.character(length(expected_dates)),
    sprintf("%.1f", dates_present / length(expected_dates) * 100),
    as.character(dates_present),
    as.character(dates_missing),
    as.character(n_expected_muni_days),
    sprintf("%.2f", pct_muni_days_present),
    as.character(n_duplicates),
    as.character(min(panel$date, na.rm = TRUE)),
    as.character(max(panel$date, na.rm = TRUE)),
    as.character(sum(panel$outage_3h_or_more == 1, na.rm = TRUE)),
    as.character(sum(panel$n_outages, na.rm = TRUE)),
    sprintf("%.4f", mean(panel$outage_3h_or_more == 1, na.rm = TRUE) * 100),
    sprintf("%.1f", has_ntl_mean_built * 100),
    sprintf("%.1f", has_ntl_mean_all * 100),
    as.character(weather_available),
    sprintf("%.1f", pct_weather_matched),
    as.character(n_bad_datetime),
    as.character(n_missing_gid2),
    sprintf("%.1f", pct_missing_gid2),
    as.character(pilot_mode),
    as.character(Sys.time())
  )
)

# saving QA summary to CSV for review
write_csv(qa_summary, qa_path)
message("Saved QA summary to: ", qa_path)

print(qa_summary)

# SECTION 12: TRAIN/VAL/TEST SPLITS

message("\n", strrep("=", 60))
message("STEP 7: Creating train/val/test splits")
message(strrep("=", 60))

# fixed split boundaries: 
# - Train: 2017-2019
# - Val:   2020-01 to 2020-06
# - Test:  2020-07 to 2021-12
#
# IMPORTANT DESIGN CHOICE:
# we store ONLY DATE RANGES here, not row indices, because row indices change
# as extraction progresses and new dates/rows appear
train_start <- as.Date("2017-01-01")
train_end   <- as.Date("2019-12-31")
val_start   <- as.Date("2020-01-01")
val_end     <- as.Date("2020-06-30")
test_start  <- as.Date("2020-07-01")
test_end    <- as.Date("2021-12-31")

# optional COVID window definition 
# note: with the fixed split above, train ends in 2019 so excluding COVID does nothing
# if later we extend training into 2020, this becomes meaningful
covid_start <- as.Date("2020-03-01")
covid_end   <- as.Date("2020-06-30")

# save stable split definitions as date ranges
splits_fixed <- list(
  train_range = c(train_start, train_end),
  val_range   = c(val_start, val_end),
  test_range  = c(test_start, test_end),
  covid_range = c(covid_start, covid_end),

  # snapshot info is informational only 
  snapshot = list(
    n_train = sum(panel$date >= train_start & panel$date <= train_end),
    n_val   = sum(panel$date >= val_start & panel$date <= val_end),
    n_test  = sum(panel$date >= test_start & panel$date <= test_end),
    snapshot_time = Sys.time(),
    snapshot_n_rows = nrow(panel)
  )
)

# persist fixed split definitions
saveRDS(splits_fixed, splits_fixed_path)
message("Saved fixed splits to: ", splits_fixed_path)
message("  Train: ", splits_fixed$snapshot$n_train, " rows (", train_start, " to ", train_end, ")")
message("  Val:   ", splits_fixed$snapshot$n_val, " rows (", val_start, " to ", val_end, ")")
message("  Test:  ", splits_fixed$snapshot$n_test, " rows (", test_start, " to ", test_end, ")")
message("  NOTE: Row counts are snapshots. Training script should recompute indices from dates.")

# SECTION 13: ROLLING-ORIGIN CV FOLDS

message("\n", strrep("=", 60))
message("STEP 8: Creating rolling-origin CV folds")
message(strrep("=", 60))

# skip CV in skeleton mode (dates are sparse; folds are not meaningful yet)
if (data_source == "skeleton") {
  message("Skeleton mode: skipping rolling CV folds (need date-complete mart).")
  splits_cv <- NULL
} else {
  # rolling-origin CV requires a meaningful number of distinct dates
  n_unique_dates <- length(dates_in_panel)
  
  if (n_unique_dates < 100) {
    # if too few dates, folds would be unstable/uninformative
    message("Only ", n_unique_dates, " unique dates. Skipping rolling CV (need more data).")
    splits_cv <- NULL
  } else {
    # we create a date-level dataset to ensure folds split by date (not by row)
    # => to avoid having the same date present in both training and assessment
    dates_df <- tibble(
      date = dates_in_panel,
      date_idx = seq_along(dates_in_panel)
    )
    
    # choosing rolling-origin window sizes:
    # - initial around 2 years or half the available dates (whichever is smaller)
    # - assess around 3 months or 10%
    # - skip  around 3 months or 10% (gap between folds)
    initial_days <- min(730, floor(n_unique_dates * 0.5))  # 2 years or 50%
    assess_days  <- min(90,  floor(n_unique_dates * 0.1))  # 3 months
    skip_days    <- min(90,  floor(n_unique_dates * 0.1))  # gap between folds
    
    # building rolling-origin resamples
    cv_resamples <- rolling_origin(
      dates_df,
      initial = initial_days,
      assess = assess_days,
      skip = skip_days,
      cumulative = FALSE
    )
    
    cv_resamples_path <- file.path(model_ready_dir, "cv_resamples.rds")
    saveRDS(cv_resamples, cv_resamples_path)
    message("Saved rsample CV object to: ", cv_resamples_path)
    
    # store fold definitions reproducibly
    splits_cv <- list(
      # the exact ordered date vector used to build these folds
      dates_used = dates_in_panel,
      
      # rsample parameters for reproducibility
      params = list(
        initial = initial_days,
        assess = assess_days,
        skip = skip_days,
        cumulative = FALSE
      ),
      
      # per-fold date ranges 
      folds = map(seq_len(nrow(cv_resamples)), function(i) {
        split <- cv_resamples$splits[[i]]
        train_dates <- training(split)$date
        assess_dates <- testing(split)$date
        list(
          train_date_range  = c(min(train_dates),  max(train_dates)),
          assess_date_range = c(min(assess_dates), max(assess_dates)),
          n_train_dates  = length(train_dates),
          n_assess_dates = length(assess_dates)
        )
      }),
      
      metadata = list(
        n_folds = nrow(cv_resamples),
        n_dates_used = length(dates_in_panel),
        created_at = Sys.time()
      )
    )
    
    # persist rolling CV definition
    saveRDS(splits_cv, splits_cv_path)
    message("Saved rolling CV to: ", splits_cv_path)
    message("  Number of folds: ", splits_cv$metadata$n_folds)
  }
}

# SECTION 14: FINAL STATUS

message("\n", strrep("=", 60))
message("MODEL MART COMPLETE")
message(strrep("=", 60))

# summarise output artifacts => locate them 
message("\nOutputs created:")
message("  - ", mart_path)
message("  - ", manifest_path)
message("  - ", qa_path)
message("  - ", splits_fixed_path)
if (!is.null(splits_cv)) message("  - ", splits_cv_path)
if (!is.null(splits_cv)) message("  - ", cv_resamples_path)
if (pilot_mode) message("  - ", pilot_munis_path)

# summarise readiness metrics at end of run
message("\nData readiness:")
message("  - Date coverage: ", sprintf("%.1f", dates_present / length(expected_dates) * 100), "%")
message("  - Muni-day coverage: ", sprintf("%.2f", pct_muni_days_present), "%")
message("  - NTL data available: ", sprintf("%.1f", has_ntl_mean_built * 100), "%")
message("  - Weather data available: ", weather_available, " (", sprintf("%.1f", pct_weather_matched), "% matched)")

message("\nNext steps:")
if (has_ntl_mean_built < 0.5) {
  message("  - NTL data is ", sprintf("%.0f", has_ntl_mean_built * 100), "% available.")
  message("  - Wait for more extraction before running 04_feature_engineering.R") # other script I've began to write following this one
} else {
  message("  - NTL data is ", sprintf("%.0f", has_ntl_mean_built * 100), "% available.")
  message("  - Ready to run 04_feature_engineering.R")
}

message("\nSplit definitions:")
message("  - Fixed splits stored as DATE RANGES (not row indices)")
message("  - Training script should compute indices fresh from current mart")

message("\nScript completed at: ", Sys.time())
