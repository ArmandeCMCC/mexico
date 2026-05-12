# 04_feature_engineering_improved.R
# PURPOSE:
#   engineer temporal, anomaly, drop, spatial, and holiday features for outage prediction
#   what I added compared to weatherV1: 
#     1. Mexican holiday indicators (via holidayAPI) => just discovered that it was not free so I fixed this by fetching data
# and creating my own csv... => with only holidays potentially affecting power consumption patterns (don't think Mother's day does)
#     2. Spatial lag features using neighbor structure from 02_spatial_weights.R 
# script should run AFTER 02_spatial_weights.R and 03_training_prep.R
#
# KEY DESIGN CHOICES:
#   1. DATE-AWARE JOINS for lags (not lag() which fails with gaps)
#   => shift dates and join so "lag 1" truly means yesterday even with gaps
#   2. CLIMATOLOGY computed on TRAINING PERIOD ONLY (no future leakage)
#   3. ROLLING STATS computed on a complete date grid per municipality
#   => rolling windows calendar-based => temporarily fill missing dates per GID_2 so windows always represent the last N calendar days (not last N observed rows)
#   4. SPATIAL LAGS use LAGGED predictors (e.g., spatial lag of ntl_mean_built_lag1)
#   => prevents contemporaneous leakage from neighbors' current NTL values
#   5. Explicit data completeness checks before proceeding
#   => stop early if the date grid is too incomplete and warn if NTL is mostly missing.
#
# SECTION 1: CONFIGURATION

# SAMPLE PANEL MODE
# Set to TRUE to run on stratified sample instead of full panel
# Reads from data/model_ready_sample/, writes to data/model_ready_sample/
# STILL uses full spatial weights from data/model_ready/spatial_weights.rds
USE_SAMPLE_PANEL <- FALSE

# min data completeness to proceed (0-1)
min_date_coverage <- 0.50  # at least 50% of expected dates must be present

# feature engineering settings
lag_days <- c(1, 2, 7, 14)   # days to lag NTL features
rolling_windows <- c(7, 30)  # rolling windows (currently implemented explicitly as 7/30 below)

# NTL variables to engineer features
# primary variables: the ones we definitely lag (unless lag_all_ntl = TRUE)
ntl_vars_primary <- c("ntl_mean_built", "ntl_sum_built", "ntl_sd_built")
ntl_vars_secondary <- c("ntl_mean_built_mask", "ntl_mean_all")  # less critical (not used below yet) => TRUE if used

# weather feature engineering (V2 = expanded for operational mode)
# All weather variables that may have predictive power when lagged
weather_vars_primary <- c(
  "atm",           # atmospheric pressure
  "dew",           # dew point
  "max_dew",       # max dew point
  "min_dew",       # min dew point
  "lai_high",      # leaf area index (high veg)
  "lai_low",       # leaf area index (low veg)
  "rain",          # precipitation
  "rh",            # relative humidity
  "skin_temp",     # skin temperature
  "temp",          # temperature
  "wdr",           # wind direction
  "wind_u",        # wind u-component
  "wind_v",        # wind v-component
  "wsp",           # wind speed
  "max_temp",      # max temperature
  "min_temp"       # min temperature
)
weather_lag_days <- c(1, 7)                 # separate from NTL lag_days on purpose (more lag_days for NTL)

# optionally lag ALL NTL schemes (all/built/built_mask + wsum + built shares)
# default keeps current behavior (only primary)
# FALSE: only primary vars get lags
# TRUE = everything in ntl_cols_all gets lags => more features so heavier
lag_all_ntl <- FALSE

# mirrors the NTL schema in 03_training_prep.R
ntl_cols_all <- c(
  "ntl_mean_all", "ntl_sum_all", "ntl_sd_all", "wsum_all",
  "ntl_mean_built", "ntl_sum_built", "ntl_sd_built", "wsum_built",
  "ntl_mean_built_mask", "ntl_sum_built_mask", "ntl_sd_built_mask", "wsum_built_mask",
  "built_share_area", "built_share_mask"
)

# drop indicator thresholds
zscore_thresholds <- c(-1.0, -1.5, -2.0, -2.5, -3.0)
pct_drop_thresholds <- c(0.05, 0.10, 0.20, 0.30)

# SPATIAL FEATURES CONFIGURATION
# which lagged NTL features to compute spatial lags for
# IMPORTANT: spatial lag of LAGGED features (not current values) to prevent leakage
spatial_lag_features <- c("ntl_mean_built_lag1", "ntl_sum_built_lag1", "ntl_mean_built_lag7")

# SECTION 2: LOAD PACKAGES
message("Loading packages...")
suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(slider)
  library(Matrix)   # for spatial weights sparse matrix operations
})

# checking dplyr version for relationship= parameter support
USE_RELATIONSHIP_CHECK <- packageVersion("dplyr") >= "1.1.0"
if (!USE_RELATIONSHIP_CHECK) {
  warning("dplyr < 1.1.0 detected: 'relationship=' parameter not supported in joins. ",
          "Proceeding without it (still safe due to upstream duplicate key check).")
} else {
  message("dplyr ", as.character(packageVersion("dplyr")), " detected: using relationship= checks in joins.")
}

# SECTION 3: PATH HANDLING
# detect repo/project root FIRST (needed for holiday path below)
detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") {
    return(normalizePath(file.path(getwd(), "..")))
  }
  if (dir.exists("data") || dir.exists("scripts")) {
    return(normalizePath(getwd()))
  }
  warning("Could not determine project directory. Using current directory.")
  return(normalizePath(getwd()))
}

# timezone-safe Date conversion (same idea as 03)
to_utc_date <- function(x) {
  if (inherits(x, "POSIXt")) {
    as.Date(x, tz = "UTC")
  } else {
    as.Date(x)
  }
}

project_dir <- detect_project_dir()
data_dir    <- file.path(project_dir, "data")

# Use sample output directory if in sample mode
if (USE_SAMPLE_PANEL) {
  model_ready_dir <- file.path(data_dir, "model_ready_sample")
  message("*** SAMPLE MODE: Reading/writing from ", model_ready_dir, " ***")
} else {
  model_ready_dir <- file.path(data_dir, "model_ready")
}

# Spatial weights ALWAYS from full model_ready (uses full GADM, not affected by sample)
spatial_weights_dir <- file.path(data_dir, "model_ready")

message("Project directory: ", project_dir)

# HOLIDAY API CONFIGURATION
# enable holidays if API key is set OR if CSV file exists (priority to CSV)
holidays_csv_path <- file.path(data_dir, "holidays_mexico_2017_2021.csv")
ENABLE_HOLIDAYS <- Sys.getenv("HOLIDAYAPI_PAT", "") != "" || file.exists(holidays_csv_path)
HOLIDAY_COUNTRY <- "MX"  # Mexico ISO code

if (ENABLE_HOLIDAYS) {
  if (file.exists(holidays_csv_path)) {
    message("Holidays enabled: CSV file found (", basename(holidays_csv_path), ")")
  } else {
    message("Holidays enabled: HolidayAPI key detected")
  }
} else {
  message("Holidays disabled: No CSV file and no HOLIDAYAPI_PAT environment variable")
}

# SECTION 3.5: LOAD HOLIDAYAPI FUNCTIONS (if enabled)
if (ENABLE_HOLIDAYS) {
  message("Loading HolidayAPI functions...")
  # source the holidayAPI R scripts using project_dir
  holiday_api_dir <- file.path(project_dir, "holidayAPI")

  if (dir.exists(holiday_api_dir)) {
    tryCatch({
      source(file.path(holiday_api_dir, "utils-pipe.R"), local = TRUE)
      source(file.path(holiday_api_dir, "classes_methods_other.R"), local = TRUE)
      source(file.path(holiday_api_dir, "get_holidays.R"), local = TRUE)
      message("  HolidayAPI functions loaded successfully")
    }, error = function(e) {
      warning("Failed to load HolidayAPI functions: ", e$message, "\n",
              "Holidays will be disabled for this run.")
      ENABLE_HOLIDAYS <<- FALSE
    })
  } else {
    warning("HolidayAPI directory not found at: ", holiday_api_dir, "\n",
            "Holidays will be disabled for this run.")
    ENABLE_HOLIDAYS <- FALSE
  }
} else {
  message("Holidays disabled (HOLIDAYAPI_PAT environment variable not set)")
}

# input paths produced by 02 and 03
# NOTE: spatial_weights ALWAYS from full model_ready (not affected by sample)
spatial_weights_path <- file.path(spatial_weights_dir, "spatial_weights.rds")
mart_path         <- file.path(model_ready_dir, "model_mart.rds")
splits_fixed_path <- file.path(model_ready_dir, "splits_fixed.rds")

# output paths produced by this script
features_path      <- file.path(model_ready_dir, "features_engineered.rds")
feat_manifest_path <- file.path(model_ready_dir, "features_manifest.csv")
feat_qa_path       <- file.path(model_ready_dir, "features_qa.csv")

# SECTION 3.7: HELPER FUNCTIONS (defined once, used throughout)
# prevent NaN from mean(c(NA, NA), na.rm=TRUE) or sd() with < 2 observations
safe_mean <- function(x) {
  if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}

safe_sd <- function(x) {
  # need at least 2 non-NA values for meaningful SD
  if (sum(!is.na(x)) < 2) NA_real_ else sd(x, na.rm = TRUE)
}

# SECTION 4: LOAD MODEL MART AND CHECK COMPLETENESS
message("\n", strrep("=", 60))
message("STEP 1: Loading model mart and checking data completeness")
message(strrep("=", 60))

# fail fast if upstream outputs missing
if (!file.exists(mart_path)) {
  stop("Model mart not found: ", mart_path, "\nPlease run 03_training_prep.R first.")
}
panel <- readRDS(mart_path)
message("Loaded model mart: ", nrow(panel), " rows")

# ensure date is Date (03 should save Date, but keep this safe + UTC-safe)
# CRITICAL: coerce GID_2 to character (prevent factor-indexing bugs across all joins/filters)
panel <- panel %>%
  mutate(
    date = to_utc_date(date),
    GID_2 = as.character(GID_2)
  )
message("Type safety: GID_2 coerced to character (prevents factor-indexing bugs)")

# CRITICAL: Check for duplicate (GID_2, date) keys before any joins
# duplicates can silently multiply rows during lag joins
dup_n <- panel %>% count(GID_2, date) %>% filter(n > 1) %>% nrow()
if (dup_n > 0) {
  stop("Duplicate (GID_2, date) keys found in model_mart: ", dup_n, " duplicate keys.\n",
       "This will cause row multiplication during lag joins. Please fix upstream in 03_training_prep.R")
}
message("Key uniqueness check passed: no duplicate (GID_2, date) keys")

# loading splits early so we can derive train period + expected date range robustly
if (!file.exists(splits_fixed_path)) {
  stop("Splits not found: ", splits_fixed_path, "\nPlease run 03_training_prep.R first.")
}
splits_fixed <- readRDS(splits_fixed_path)

# training period for climatology (03 writes train_range => keep fallback for older format)
train_start <- if (!is.null(splits_fixed$train_range)) {
  as.Date(splits_fixed$train_range[1])
} else {
  as.Date(splits_fixed$metadata$train_period[1])
}
train_end <- if (!is.null(splits_fixed$train_range)) {
  as.Date(splits_fixed$train_range[2])
} else {
  as.Date(splits_fixed$metadata$train_period[2])
}
message("Training period for climatology: ", train_start, " to ", train_end)

# derive expected full date range (use train/val/test ranges from splits_fixed if we have them)
range_starts <- c(
  if (!is.null(splits_fixed$train_range)) as.Date(splits_fixed$train_range[1]) else NA,
  if (!is.null(splits_fixed$val_range))   as.Date(splits_fixed$val_range[1])   else NA,
  if (!is.null(splits_fixed$test_range))  as.Date(splits_fixed$test_range[1])  else NA
)
range_ends <- c(
  if (!is.null(splits_fixed$train_range)) as.Date(splits_fixed$train_range[2]) else NA,
  if (!is.null(splits_fixed$val_range))   as.Date(splits_fixed$val_range[2])   else NA,
  if (!is.null(splits_fixed$test_range))  as.Date(splits_fixed$test_range[2])  else NA
)

# if split ranges missing => fall back to min/max dates present
if (all(is.na(range_starts)) || all(is.na(range_ends))) {
  date_start <- as.Date(min(panel$date, na.rm = TRUE))
  date_end   <- as.Date(max(panel$date, na.rm = TRUE))
  message("Date range (fallback = mart min/max): ", date_start, " to ", date_end)
} else {
  date_start <- as.Date(min(range_starts, na.rm = TRUE))
  date_end   <- as.Date(max(range_ends, na.rm = TRUE))
  message("Date range (from splits_fixed): ", date_start, " to ", date_end)
}

# check date coverage
# how many unique dates do we have vs how many should exist in expected range
expected_dates <- seq.Date(date_start, date_end, by = "day")
dates_in_panel <- sort(unique(panel$date))
date_coverage <- length(dates_in_panel) / length(expected_dates)

message("Date coverage: ", sprintf("%.1f%%", date_coverage * 100),
        " (", length(dates_in_panel), " / ", length(expected_dates), " dates)")

# check NTL data availability
# sanity check using ntl_mean_built as proxy
ntl_available <- if ("ntl_mean_built" %in% names(panel)) {
  mean(!is.na(panel$ntl_mean_built))
} else {
  0
}
message("NTL data available (ntl_mean_built): ", sprintf("%.1f%%", ntl_available * 100))

# stop if tempo grid too incomplete to support temporal features
if (date_coverage < min_date_coverage) {
  stop("Insufficient date coverage (", sprintf("%.1f%%", date_coverage * 100), ").\n",
       "Need at least ", sprintf("%.0f%%", min_date_coverage * 100), " coverage.\n",
       "Wait for more extraction to complete before running feature engineering.")
}

# warn if NTL mostly missing
if (ntl_available < 0.1) {
  warning("Very low NTL data availability (", sprintf("%.1f%%", ntl_available * 100), ").\n",
          "Features will mostly be NA. Consider waiting for more extraction.")
}

# safety: ensure calendar fields exist (03 should provide them, but guard anyway)
# required for cyclical features and climatology day-of-year
need_calendar <- !all(c("dow_num", "month", "year", "doy") %in% names(panel))
if (need_calendar) {
  panel <- panel %>%
    mutate(
      dow_num = wday(date, week_start = 1),
      is_weekend = as.integer(dow_num >= 6),
      month = month(date),
      year = year(date),
      doy = yday(date)
    )
}

# weather availability flag
cols_present <- intersect(weather_vars_primary, names(panel))

panel <- panel %>%
  mutate(
    has_weather = if (length(cols_present) > 0) {
      as.integer(if_any(all_of(cols_present), ~ !is.na(.x)))
    } else {
      0L
    }
  )

if (length(cols_present) > 0) {
  message("Created has_weather flag (", length(cols_present), " weather variables checked)")
} else {
  message("No weather variables found; has_weather set to 0")
}

# SECTION 4.5: LOAD MEXICAN HOLIDAYS
if (ENABLE_HOLIDAYS) {
  message("\n", strrep("=", 60))
  message("STEP 1.5: Loading Mexican holidays")
  message(strrep("=", 60))

  # which years we need
  years_needed <- unique(year(c(date_start, date_end)))
  if (length(years_needed) == 1) {
    years_to_fetch <- years_needed
  } else {
    years_to_fetch <- seq(min(years_needed), max(years_needed))
  }

  # PRIORITY 1: Check for CSV file 
  if (file.exists(holidays_csv_path)) {
    message("Loading holidays from CSV: ", holidays_csv_path)

    holidays_by_date <- tryCatch({
      read_csv(holidays_csv_path, col_types = cols(
        date = col_date(format = ""),
        name = col_character(),
        public = col_integer(),
        type = col_character()
      )) %>%
        mutate(date = as.Date(date)) %>%
        filter(date >= date_start & date <= date_end) %>%
        # Group by date to ensure one row per date (some dates may have multiple holidays)
        group_by(date) %>%
        summarise(
          is_holiday = 1L,
          is_public_holiday = as.integer(any(public == 1, na.rm = TRUE)),
          holiday_names = paste(unique(name), collapse = "; "),
          .groups = "drop"
        )
    }, error = function(e) {
      warning("Failed to load holidays from CSV: ", e$message, ". Falling back to API/cache.")
      NULL
    })

    if (!is.null(holidays_by_date)) {
      message("Loaded ", nrow(holidays_by_date), " holiday dates from CSV (",
              sum(holidays_by_date$is_public_holiday), " public holidays)")
    }
  } else {
    holidays_by_date <- NULL
    message("CSV file not found: ", holidays_csv_path)
  }

  # PRIORITY 2: Fall back to cached API data or fetch from API
  if (is.null(holidays_by_date)) {
    message("Attempting to load from API cache or fetch from HolidayAPI...")
    holidays_cache_path <- file.path(model_ready_dir, "holidays_mx.rds")

  if (file.exists(holidays_cache_path)) {
    message("Loading holidays from cache: ", holidays_cache_path)

    # trying to load cache with robustness 
    holidays_by_date <- tryCatch({
      cache <- readRDS(holidays_cache_path)

      # validate cache schema (must have: date, is_holiday, is_public_holiday)
      required_cols <- c("date", "is_holiday", "is_public_holiday")
      if (!all(required_cols %in% names(cache))) {
        warning("Holiday cache has invalid schema (missing columns). Rebuilding from scratch.")
        NULL  # trigger rebuild
      } else {
        cache
      }
    }, error = function(e) {
      warning("Failed to load holiday cache: ", e$message, ". Rebuilding from scratch.")
      NULL  # trigger rebuild
    })

    if (is.null(holidays_by_date)) {
      # cache invalid or corrupt => rebuild from scratch
      message("  Rebuilding holiday cache from scratch...")
      years_missing <- years_to_fetch
    } else {
      # check if cache covers all needed years
      years_cached <- unique(year(holidays_by_date$date))
      years_missing <- setdiff(years_to_fetch, years_cached)
    }

    if (length(years_missing) > 0) {
      message("  Cache missing years: ", paste(years_missing, collapse = ", "))
      message("  Fetching missing years...")

      # fetching only missing years
      holidays_list <- list()
      for (yr in years_missing) {
        tryCatch({
          hol <- get_holidays(country = HOLIDAY_COUNTRY, year = yr)
          hol_df <- as.data.frame(hol)
          holidays_list[[as.character(yr)]] <- hol_df
          message("    ", yr, ": ", nrow(hol_df), " holidays fetched")
        }, error = function(e) {
          warning("Failed to fetch holidays for year ", yr, ": ", e$message)
        })
      }

      if (length(holidays_list) > 0) {
        # combine new fetches
        new_holidays <- bind_rows(holidays_list) %>%
          mutate(date = as.Date(date)) %>%
          group_by(date) %>%
          summarise(
            is_holiday = 1L,
            is_public_holiday = as.integer(any(public == TRUE, na.rm = TRUE)),
            holiday_names = paste(unique(name), collapse = "; "),
            .groups = "drop"
          )

        # merge with cache and re-save
        if (is.null(holidays_by_date)) {
          # cache was invalid/corrupt => use only new fetches
          holidays_by_date <- new_holidays
          message("  Created new cache with ", nrow(new_holidays), " holiday dates")
        } else {
          # merge new fetches with existing cache
          holidays_by_date <- bind_rows(holidays_by_date, new_holidays) %>%
            distinct(date, .keep_all = TRUE)  # in case of duplicates
          message("  Updated cache with ", nrow(new_holidays), " new holiday dates")
        }

        saveRDS(holidays_by_date, holidays_cache_path)
      }
    } else {
      message("  Cache covers all needed years")
    }

    # filter to only dates we need
    holidays_by_date <- holidays_by_date %>%
      filter(date >= date_start & date <= date_end)

    message("Loaded ", nrow(holidays_by_date), " holiday dates from cache")
  } else {
    message("Fetching holidays for years: ", paste(years_to_fetch, collapse = ", "))

    # fetch holidays for each year
    holidays_list <- list()
    for (yr in years_to_fetch) {
      tryCatch({
        hol <- get_holidays(country = HOLIDAY_COUNTRY, year = yr)
        hol_df <- as.data.frame(hol)
        holidays_list[[as.character(yr)]] <- hol_df
        message("  ", yr, ": ", nrow(hol_df), " holidays fetched")
      }, error = function(e) {
        warning("Failed to fetch holidays for year ", yr, ": ", e$message)
      })
    }

    if (length(holidays_list) > 0) {
      # combine all years and COLLAPSE to one row per date (avoid join multiplication)
      holidays_by_date <- bind_rows(holidays_list) %>%
        mutate(date = as.Date(date)) %>%
        group_by(date) %>%
        summarise(
          is_holiday = 1L,
          is_public_holiday = as.integer(any(public == TRUE, na.rm = TRUE)),
          holiday_names = paste(unique(name), collapse = "; "),
          .groups = "drop"
        )

      # saving to cache
      saveRDS(holidays_by_date, holidays_cache_path)
      message("Saved ", nrow(holidays_by_date), " holiday dates to cache: ", holidays_cache_path)
    } else {
      holidays_by_date <- NULL
    }
  }
  }  # end of API/cache fallback

  if (!is.null(holidays_by_date) && nrow(holidays_by_date) > 0) {
    message("Total unique holiday dates: ", nrow(holidays_by_date),
            " (", sum(holidays_by_date$is_public_holiday), " public)")

    # join holiday indicators: now safe because one row per date
    panel <- panel %>%
      left_join(
        holidays_by_date %>% select(date, is_holiday, is_public_holiday),
        by = "date"
      ) %>%
      mutate(
        is_holiday = coalesce(is_holiday, 0L),
        is_public_holiday = coalesce(is_public_holiday, 0L)
      )

    message("Created holiday indicators: is_holiday, is_public_holiday")
  } else {
    warning("No holidays fetched. Holiday indicators will be set to 0.")
    panel <- panel %>%
      mutate(
        is_holiday = 0L,
        is_public_holiday = 0L
      )
  }
} else {
  # if holidays disabled, setting indicators to 0
  panel <- panel %>%
    mutate(
      is_holiday = 0L,
      is_public_holiday = 0L
    )
  message("Holidays disabled: is_holiday and is_public_holiday set to 0")
}

# SECTION 4.6: LOAD SPATIAL WEIGHTS
message("\n", strrep("=", 60))
message("STEP 1.6: Loading spatial weights")
message(strrep("=", 60))

if (!file.exists(spatial_weights_path)) {
  warning("Spatial weights not found: ", spatial_weights_path, "\n",
          "Spatial features will be skipped. Run 02_spatial_weights.R first if want spatial features.")
  ENABLE_SPATIAL <- FALSE
} else {
  spatial_weights <- readRDS(spatial_weights_path)

  # extracting components
  gid2_order <- spatial_weights$gid2_order
  gid2_index <- spatial_weights$gid2_index
  W_sparse <- spatial_weights$W_sparse

  message("Loaded spatial weights:")
  message("  Municipalities: ", length(gid2_order))
  message("  Sparse matrix: ", nrow(W_sparse), "x", ncol(W_sparse))
  message("  Non-zero elements: ", Matrix::nnzero(W_sparse))

  # CRITICAL: Verify GID_2 alignment
  # every GID_2 in panel must exist in gid2_order
  panel_gids <- unique(panel$GID_2)
  missing_gids <- setdiff(panel_gids, gid2_order)

  if (length(missing_gids) > 0) {
    stop("GID_2 alignment failure: ", length(missing_gids), " municipalities in panel not found in spatial_weights.\n",
         "Missing GID_2 codes: ", paste(head(missing_gids, 10), collapse = ", "),
         if (length(missing_gids) > 10) " ..." else "", "\n",
         "This indicates spatial_weights.rds was built on different boundaries than current panel.\n",
         "Please re-run 02_spatial_weights.R with the same GADM file.")
  }

  extra_gids <- setdiff(gid2_order, panel_gids)
  if (length(extra_gids) > 0) {
    message("Note: ", length(extra_gids), " municipalities in spatial_weights not present in panel (OK if subset)")
  }

  message("GID_2 alignment verified: all panel municipalities present in spatial_weights")
  ENABLE_SPATIAL <- TRUE
}

# SECTION 5: DATE-AWARE LAG FEATURES
message("\n", strrep("=", 60))
message("STEP 2: Computing lag features (date-aware joins)")
message(strrep("=", 60))

# for each lag_d => lookup table where values from date moved to date + lag_d
# then join back by (GID_2, date)
for (lag_d in lag_days) {
  lag_suffix <- paste0("_lag", lag_d)
  # which col to lag, keep those present in panel
  cols_to_lag <- if (lag_all_ntl) ntl_cols_all else ntl_vars_primary
  cols_present <- intersect(cols_to_lag, names(panel))

  if (length(cols_present) == 0) {
    message("  No NTL columns found to lag. Skipping.")
    next
  }

  # guard against re-runs: only create missing lag columns
  target_names <- paste0(cols_present, lag_suffix)
  cols_missing <- cols_present[!target_names %in% names(panel)]

  if (length(cols_missing) == 0) {
    message("  NTL ", lag_d, "-day lags already exist. Skipping.")
    next
  }

  message("  Computing ", lag_d, "-day lags for: ", paste(cols_missing, collapse = ", "))

  # lookup = original values but with date shifted forward => align with "current" date on join
  lag_lookup <- panel %>%
    select(GID_2, date, all_of(cols_missing)) %>%
    mutate(date = date + lag_d)
  # rename lagged col
  names(lag_lookup)[names(lag_lookup) %in% cols_missing] <-
    paste0(cols_missing, lag_suffix)
  # join back to panel : for each (GID_2, date) we now have lagged features if past date existed
  # relationship="many-to-one" (requires dplyr >= 1.1.0) ensures lag_lookup has unique keys
  if (USE_RELATIONSHIP_CHECK) {
    panel <- panel %>%
      left_join(lag_lookup, by = c("GID_2", "date"), relationship = "many-to-one")
  } else {
    panel <- panel %>%
      left_join(lag_lookup, by = c("GID_2", "date"))
  }
}

message("Lag features created for days: ", paste(lag_days, collapse = ", "))

# DATE-AWARE WEATHER LAGS (V1 minimal)
cols_present <- intersect(weather_vars_primary, names(panel))

if (length(cols_present) == 0) {
  message("No weather columns found for lagging. Skipping weather lags.")
} else {
  for (lag_d in weather_lag_days) {
    lag_suffix <- paste0("_lag", lag_d)

    target_names <- paste0(cols_present, lag_suffix)
    cols_missing <- cols_present[!target_names %in% names(panel)]

    if (length(cols_missing) == 0) {
      message("  Weather ", lag_d, "-day lags already exist. Skipping.")
      next
    }

    message("  Computing weather ", lag_d, "-day lags for: ", paste(cols_missing, collapse = ", "))

    lag_lookup <- panel %>%
      select(GID_2, date, all_of(cols_missing)) %>%
      mutate(date = date + lag_d)

    names(lag_lookup)[names(lag_lookup) %in% cols_missing] <- paste0(cols_missing, lag_suffix)

    if (USE_RELATIONSHIP_CHECK) {
      panel <- panel %>%
        left_join(lag_lookup, by = c("GID_2", "date"), relationship = "many-to-one")
    } else {
      panel <- panel %>%
        left_join(lag_lookup, by = c("GID_2", "date"))
    }
  }

  message("Weather lag features created for days: ", paste(weather_lag_days, collapse = ", "))
}

# SECTION 6: DAY-OVER-DAY CHANGES
message("\n", strrep("=", 60))
message("STEP 3: Computing day-over-day changes")
message(strrep("=", 60))

# diff and percent changes where required lag col exist
# percent changes protected against division by zero and non-finite val
if (all(c("ntl_mean_built", "ntl_mean_built_lag1") %in% names(panel))) {
  panel <- panel %>%
    mutate(
      # absolute day over day change
      ntl_mean_built_diff1 = ntl_mean_built - ntl_mean_built_lag1,
      # rel day over day change (fraction)
      ntl_mean_built_pct_change1 = if_else(
        is.finite(ntl_mean_built_lag1) & ntl_mean_built_lag1 > 0,
        (ntl_mean_built - ntl_mean_built_lag1) / ntl_mean_built_lag1,
        NA_real_
      )
    )
}

if (all(c("ntl_sum_built", "ntl_sum_built_lag1") %in% names(panel))) {
  panel <- panel %>%
    mutate(
      ntl_sum_built_diff1 = ntl_sum_built - ntl_sum_built_lag1,
      ntl_sum_built_pct_change1 = if_else(
        is.finite(ntl_sum_built_lag1) & ntl_sum_built_lag1 > 0,
        (ntl_sum_built - ntl_sum_built_lag1) / ntl_sum_built_lag1,
        NA_real_
      )
    )
}

# 7 day changes capture weekly shifts
if (all(c("ntl_mean_built", "ntl_mean_built_lag7") %in% names(panel))) {
  panel <- panel %>%
    mutate(
      ntl_mean_built_diff7 = ntl_mean_built - ntl_mean_built_lag7,
      ntl_mean_built_pct_change7 = if_else(
        is.finite(ntl_mean_built_lag7) & ntl_mean_built_lag7 > 0,
        (ntl_mean_built - ntl_mean_built_lag7) / ntl_mean_built_lag7,
        NA_real_
      )
    )
}

if (all(c("ntl_sum_built", "ntl_sum_built_lag7") %in% names(panel))) {
  panel <- panel %>%
    mutate(
      ntl_sum_built_diff7 = ntl_sum_built - ntl_sum_built_lag7
    )
}

message("Day-over-day change features computed where inputs exist.")

# SECTION 7: ROLLING STATISTICS (DATE-AWARE)
message("\n", strrep("=", 60))
message("STEP 4: Computing rolling statistics")
message(strrep("=", 60))

# rolling stats => need complete date grid per muni so last 7 days is well def
# if required NTL col missing => skip
if (!all(c("ntl_mean_built", "ntl_sum_built") %in% names(panel))) {
  message("Skipping rolling stats: ntl_mean_built / ntl_sum_built not present in panel.")
} else {
  munis <- unique(panel$GID_2)
  all_dates <- seq.Date(date_start, date_end, by = "day")

  message("Completing date grid for rolling stats (", length(munis), " municipalities)...")

  # temporary complete grid: all muni x all dates
  # values NA where data missing
  panel_complete <- expand_grid(
    GID_2 = munis,
    date = all_dates
  ) %>%
    left_join(
      panel %>% select(GID_2, date, ntl_mean_built, ntl_sum_built),
      by = c("GID_2", "date")
    )
  # rolling mean/sd over previous N-1 days + current day
  message("  Computing 7-day rolling mean/sd...")
  panel_complete <- panel_complete %>%
    arrange(GID_2, date) %>%
    group_by(GID_2) %>%
    mutate(
      ntl_mean_built_roll7_mean = slider::slide_dbl(
        ntl_mean_built, safe_mean, .before = 6, .complete = FALSE
      ),
      ntl_mean_built_roll7_sd = slider::slide_dbl(
        ntl_mean_built, safe_sd, .before = 6, .complete = FALSE
      ),
      ntl_sum_built_roll7_mean = slider::slide_dbl(
        ntl_sum_built, safe_mean, .before = 6, .complete = FALSE
      )
    ) %>%
    ungroup()

  message("  Computing 30-day rolling mean/sd...")
  panel_complete <- panel_complete %>%
    arrange(GID_2, date) %>%
    group_by(GID_2) %>%
    mutate(
      ntl_mean_built_roll30_mean = slider::slide_dbl(
        ntl_mean_built, safe_mean, .before = 29, .complete = FALSE
      ),
      ntl_mean_built_roll30_sd = slider::slide_dbl(
        ntl_mean_built, safe_sd, .before = 29, .complete = FALSE
      )
    ) %>%
    ungroup()
  # keep only rolling feature col we created, then join back to original panel rows
  rolling_cols <- c(
    "ntl_mean_built_roll7_mean", "ntl_mean_built_roll7_sd",
    "ntl_sum_built_roll7_mean",
    "ntl_mean_built_roll30_mean", "ntl_mean_built_roll30_sd"
  )

  panel <- panel %>%
    left_join(
      panel_complete %>% select(GID_2, date, all_of(rolling_cols)),
      by = c("GID_2", "date")
    )

  rm(panel_complete)
  gc()

  message("Rolling statistics created for 7 and 30 day windows.")
}

# SECTION 8: CLIMATOLOGY (TRAINING PERIOD ONLY SO NO LEAKAGE)
message("\n", strrep("=", 60))
message("STEP 5: Computing climatology (training period only)")
message(strrep("=", 60))

# climatology requires ntl var + day of year
# if missing skip
if (!all(c("ntl_mean_built", "ntl_sum_built", "doy") %in% names(panel))) {
  message("Skipping climatology: required columns not present (ntl_mean_built/ntl_sum_built/doy).")
} else {
  # compute climatology using TRAINING PERIOD ONLY (avoid leakage)
  train_data <- panel %>%
    filter(date >= train_start & date <= train_end)

  message("Computing climatology from ", nrow(train_data), " training rows")

  # per-muni, per day-of-year mean and sd (TRAINING ONLY)
  # Using safe_mean/safe_sd defined at top to prevent NaN from all-NA groups
  climatology <- train_data %>%
    group_by(GID_2, doy) %>%
    summarise(
      ntl_mean_built_clim_mean = safe_mean(ntl_mean_built),
      ntl_mean_built_clim_sd = safe_sd(ntl_mean_built),
      ntl_sum_built_clim_mean = safe_mean(ntl_sum_built),
      ntl_sum_built_clim_sd = safe_sd(ntl_sum_built),
      # how many obs contributed
      clim_n_obs = sum(!is.na(ntl_mean_built)),
      .groups = "drop"
    )
  # join climatology back to all rows + compute anomalies and z-scores
  panel <- panel %>%
    left_join(climatology, by = c("GID_2", "doy")) %>%
    mutate(
      # anomaly = deviation from climatological mean
      ntl_mean_built_anomaly = ntl_mean_built - ntl_mean_built_clim_mean,
      ntl_sum_built_anomaly  = ntl_sum_built  - ntl_sum_built_clim_mean,
      # z-score: standardised anomaly
      ntl_mean_built_zscore = if_else(
        is.finite(ntl_mean_built_clim_sd) & ntl_mean_built_clim_sd > 0,
        (ntl_mean_built - ntl_mean_built_clim_mean) / ntl_mean_built_clim_sd,
        NA_real_
      ),
      ntl_sum_built_zscore = if_else(
        is.finite(ntl_sum_built_clim_sd) & ntl_sum_built_clim_sd > 0,
        (ntl_sum_built - ntl_sum_built_clim_mean) / ntl_sum_built_clim_sd,
        NA_real_
      )
    )

  message("Created anomaly and z-score features (leakage-free).")
}

# SECTION 8.5: STATIC BUILT-SHARE FEATURES
# built_share_area and built_share_mask vary daily due to cloud coverage
# (exact_extract sees different valid pixels each day).
# Compute a static per-municipality average using TRAINING data only:
# this captures the municipality's true built-up fraction, free of daily noise.
# Named without "ntl" so the forecast filter does not drop them.
message("\n", strrep("=", 60))
message("STEP 5.5: Computing static built-share features (training period only)")
message(strrep("=", 60))

if (all(c("built_share_area", "built_share_mask") %in% names(panel))) {
  train_built <- panel %>%
    filter(date >= train_start & date <= train_end) %>%
    group_by(GID_2) %>%
    summarise(
      built_share_area_static = safe_mean(built_share_area),
      built_share_mask_static = safe_mean(built_share_mask),
      .groups = "drop"
    )

  panel <- panel %>%
    left_join(train_built, by = "GID_2")

  message(sprintf("Created static built-share features from %d municipalities.",
                  nrow(train_built)))
  message(sprintf("  built_share_area_static: mean=%.4f, sd=%.4f",
                  mean(panel$built_share_area_static, na.rm = TRUE),
                  sd(panel$built_share_area_static, na.rm = TRUE)))
} else {
  message("Skipping static built-share: built_share_area/built_share_mask not in panel.")
}

# SECTION 9: DROP INDICATORS
message("\n", strrep("=", 60))
message("STEP 6: Computing drop indicators")
message(strrep("=", 60))

# drop indicators require z-scores, if not present, skip
if (!("ntl_mean_built_zscore" %in% names(panel))) {
  message("Skipping drop indicators: ntl_mean_built_zscore not present.")
} else {
  panel <- panel %>%
    mutate(
      # thresholded drop indicators based on climatological z-score
      ntl_drop_z1   = as.integer(!is.na(ntl_mean_built_zscore) & ntl_mean_built_zscore < -1.0),
      ntl_drop_z1_5 = as.integer(!is.na(ntl_mean_built_zscore) & ntl_mean_built_zscore < -1.5),
      ntl_drop_z2   = as.integer(!is.na(ntl_mean_built_zscore) & ntl_mean_built_zscore < -2.0),
      ntl_drop_z2_5 = as.integer(!is.na(ntl_mean_built_zscore) & ntl_mean_built_zscore < -2.5),
      ntl_drop_z3   = as.integer(!is.na(ntl_mean_built_zscore) & ntl_mean_built_zscore < -3.0),
      # continuous severity
      ntl_drop_severity_clim = pmin(0, ntl_mean_built_zscore),
      # percent-change drops vs yesterday
      ntl_drop_5pct_1d  = as.integer(!is.na(ntl_mean_built_pct_change1) & ntl_mean_built_pct_change1 < -0.05),
      ntl_drop_10pct_1d = as.integer(!is.na(ntl_mean_built_pct_change1) & ntl_mean_built_pct_change1 < -0.10),
      ntl_drop_20pct_1d = as.integer(!is.na(ntl_mean_built_pct_change1) & ntl_mean_built_pct_change1 < -0.20),
      ntl_drop_30pct_1d = as.integer(!is.na(ntl_mean_built_pct_change1) & ntl_mean_built_pct_change1 < -0.30),
      # continuous severity vs yesterday
      ntl_drop_severity_1d = pmin(0, ntl_mean_built_pct_change1),
      # drop vs recent rolling mean baseline
      ntl_drop_vs_roll7 = if_else(
        is.finite(ntl_mean_built_roll7_mean) & ntl_mean_built_roll7_mean > 0,
        (ntl_mean_built - ntl_mean_built_roll7_mean) / ntl_mean_built_roll7_mean,
        NA_real_
      ),
      # below rolling mean minus k*sd
      ntl_below_roll7_1sd = as.integer(
        is.finite(ntl_mean_built) & is.finite(ntl_mean_built_roll7_mean) &
          is.finite(ntl_mean_built_roll7_sd) & ntl_mean_built_roll7_sd > 0 &
          ntl_mean_built < (ntl_mean_built_roll7_mean - ntl_mean_built_roll7_sd)
      ),

      ntl_below_roll7_2sd = as.integer(
        is.finite(ntl_mean_built) & is.finite(ntl_mean_built_roll7_mean) &
          is.finite(ntl_mean_built_roll7_sd) & ntl_mean_built_roll7_sd > 0 &
          ntl_mean_built < (ntl_mean_built_roll7_mean - 2 * ntl_mean_built_roll7_sd)
      ),

      ntl_state = case_when(
        is.na(ntl_mean_built_zscore) ~ "unknown",
        ntl_mean_built_zscore < -3.0 ~ "severe_drop",
        ntl_mean_built_zscore < -2.0 ~ "significant_drop",
        ntl_mean_built_zscore < -1.0 ~ "mild_drop",
        ntl_mean_built_zscore > 2.0  ~ "unusually_bright",
        TRUE                         ~ "normal"
      ),

      ntl_drop_signal_count = coalesce(ntl_drop_z2, 0L) + coalesce(ntl_drop_10pct_1d, 0L) + coalesce(ntl_below_roll7_1sd, 0L),
      ntl_drop_confirmed = as.integer(coalesce(ntl_drop_z2, 0L) == 1 & coalesce(ntl_drop_10pct_1d, 0L) == 1),
      # NA-preserving drop indicator for spatial features (prevents bias in neighbor proportion)
      # ntl_drop_z2 treats NA as 0, which makes nb_prop downwardly biased when data missing
      ntl_drop_z2_obs = if_else(
        is.na(ntl_mean_built_zscore),
        NA_integer_,
        as.integer(ntl_mean_built_zscore < -2.0)
      )
    )

  message("Created drop indicators (including NA-preserving variant for spatial features).")
}

# SECTION 9.5: LAG DROP INDICATORS FOR SPATIAL FEATURES
message("\n", strrep("=", 60))
message("STEP 6.5: Creating lagged drop indicators for spatial features")
message(strrep("=", 60))

# creating ntl_drop_z2_lag1 for spatial neighbor proportion
# ensures spatial features use LAGGED predictors (no contemporaneous leakage)
if ("ntl_drop_z2" %in% names(panel)) {
  drop_lag_lookup <- panel %>%
    select(GID_2, date, ntl_drop_z2) %>%
    mutate(date = date + 1) %>%
    rename(ntl_drop_z2_lag1 = ntl_drop_z2)

  if (USE_RELATIONSHIP_CHECK) {
    panel <- panel %>%
      left_join(drop_lag_lookup, by = c("GID_2", "date"), relationship = "many-to-one")
  } else {
    panel <- panel %>%
      left_join(drop_lag_lookup, by = c("GID_2", "date"))
  }

  message("Created ntl_drop_z2_lag1 for spatial lag computation")
} else {
  message("Skipping ntl_drop_z2_lag1: ntl_drop_z2 not present")
}

# create ntl_drop_z2_obs_lag1 for NA-preserving spatial neighbor proportion
# ensures spatial features use LAGGED predictors + preserves NA (not treating unknown as "no drop")
if ("ntl_drop_z2_obs" %in% names(panel)) {
  drop_obs_lag_lookup <- panel %>%
    select(GID_2, date, ntl_drop_z2_obs) %>%
    mutate(date = date + 1) %>%
    rename(ntl_drop_z2_obs_lag1 = ntl_drop_z2_obs)

  if (USE_RELATIONSHIP_CHECK) {
    panel <- panel %>%
      left_join(drop_obs_lag_lookup, by = c("GID_2", "date"), relationship = "many-to-one")
  } else {
    panel <- panel %>%
      left_join(drop_obs_lag_lookup, by = c("GID_2", "date"))
  }

  message("Created ntl_drop_z2_obs_lag1 for NA-preserving spatial lag computation")
} else {
  message("Skipping ntl_drop_z2_obs_lag1: ntl_drop_z2_obs not present")
}

# SECTION 10: CYCLICAL CALENDAR FEATURES + HOLIDAYS
message("\n", strrep("=", 60))
message("STEP 7: Adding cyclical calendar features and holidays")
message(strrep("=", 60))

# covid range
covid_start <- if (!is.null(splits_fixed$covid_range)) as.Date(splits_fixed$covid_range[1]) else as.Date("2020-03-01")
covid_end   <- if (!is.null(splits_fixed$covid_range)) as.Date(splits_fixed$covid_range[2]) else as.Date("2020-06-30")

# cyclical encodings => seasonality
# leap day handling: use correct divisor (366 for leap years, 365 otherwise)
panel <- panel %>%
  mutate(
    # month
    month_sin = sin(2 * pi * month / 12),
    month_cos = cos(2 * pi * month / 12),
    # day of year (leap-year aware)
    days_in_year = if_else(leap_year(date), 366, 365),
    doy_sin = sin(2 * pi * doy / days_in_year),
    doy_cos = cos(2 * pi * doy / days_in_year),
    # day of week
    dow_sin = sin(2 * pi * dow_num / 7),
    dow_cos = cos(2 * pi * dow_num / 7),
    # binary indicator for COVID
    is_covid_period = as.integer(date >= covid_start & date <= covid_end)
  ) %>%
  select(-days_in_year)  # drop helper column

message("Created cyclical encodings, COVID indicator, and holiday indicators.")

# SECTION 10.5: SPATIAL LAG FEATURES
if (ENABLE_SPATIAL) {
  message("\n", strrep("=", 60))
  message("STEP 7.5: Computing spatial lag features")
  message(strrep("=", 60))

  # HELPER FUNCTION: fast spatial lag with correct missingness handling
  # computes renormalized weighted mean: numerator / denominator
  # returns NA where denominator = 0 (no valid neighbor data)
  spatial_lag_mean <- function(df_day, var, W, gid2_index) {
    n <- length(gid2_index)
    x <- rep(NA_real_, n)

    # filling in values for municipalities present in df_day
    # CRITICAL: coerce to character to avoid factor indexing bugs
    gids <- as.character(df_day$GID_2)
    idx <- unname(gid2_index[gids])

    # fail if any GID_2 not found in spatial weights => and tells it! very important to know 
    if (anyNA(idx)) {
      bad <- gids[is.na(idx)]
      stop("Spatial mapping failed for GID_2: ", paste(head(bad, 10), collapse = ", "),
           if (length(bad) > 10) " ..." else "", "\n",
           "These municipalities not found in gid2_index. ",
           "Re-run 02_spatial_weights.R with matching GADM boundaries.")
    }

    x[idx] <- df_day[[var]]

    # missingness mask
    m <- !is.na(x)
    x0 <- ifelse(m, x, 0)

    # renormalized spatial lag
    num <- as.numeric(W %*% x0)
    den <- as.numeric(W %*% as.numeric(m))

    out <- num / den
    out[den == 0] <- NA_real_
    out
  }

  # which lagged features to compute spatial lags for
  spatial_features_present <- intersect(spatial_lag_features, names(panel))

  if (length(spatial_features_present) == 0) {
    message("No spatial lag features to compute (required lagged features not present)")
  } else {
    message("Computing spatial lags for: ", paste(spatial_features_present, collapse = ", "))

    # adding ntl_drop_z2_obs_lag1 to features if it exists (for NA-preserving neighbor proportion)
    if ("ntl_drop_z2_obs_lag1" %in% names(panel)) {
      spatial_features_present <- c(spatial_features_present, "ntl_drop_z2_obs_lag1")
      message("  Including ntl_drop_z2_obs_lag1 for NA-preserving nb_prop_drop_z2_lag1")
    }

    # computing spatial lags using fast group_modify approach
    # avoids triple-nested loop and handles missingness correctly
    panel <- panel %>%
      group_by(date) %>%
      group_modify(~{
        df <- .x

        # type-safe indexing: coerce to character (even though already done upstream)
        gids_df <- as.character(df$GID_2)
        idx_df <- unname(gid2_index[gids_df])

        # fail if any GID_2 not found
        if (anyNA(idx_df)) {
          bad <- gids_df[is.na(idx_df)]
          stop("Spatial mapping failed for GID_2 in group_modify(): ",
               paste(head(bad, 10), collapse = ", "),
               if (length(bad) > 10) " ..." else "", "\n",
               "These municipalities not found in gid2_index.")
        }

        for (v in spatial_features_present) {
          # computing spatial lag for all municipalities (full gid2_order)
          full_spatial_lag <- spatial_lag_mean(df, v, W_sparse, gid2_index)

          # extracting only the municipalities present in this date's data
          df[[paste0("nb_mean_", v)]] <- full_spatial_lag[idx_df]
        }

        df
      }) %>%
      ungroup()

    message("Spatial lag features computed successfully")

    # renaming ntl_drop_z2_obs_lag1 spatial feature 
    if ("nb_mean_ntl_drop_z2_obs_lag1" %in% names(panel)) {
      panel <- panel %>%
        rename(nb_prop_drop_z2_lag1 = nb_mean_ntl_drop_z2_obs_lag1)
      message("  Created nb_prop_drop_z2_lag1 (NA-preserving proportion of neighbors with drops, lagged)")
    }
  }
} else {
  message("\n", strrep("=", 60))
  message("STEP 7.5: Spatial features SKIPPED (spatial weights not loaded)")
  message(strrep("=", 60))
}

# SECTION 11: SAVE ENGINEERED FEATURES
message("\n", strrep("=", 60))
message("STEP 8: Saving engineered features")
message(strrep("=", 60))

# CRITICAL INVARIANT CHECKS: ensure joins haven't multiplied or lost rows
# load original mart to compare row counts
mart_original <- readRDS(mart_path)
n_original <- nrow(mart_original)
n_final <- nrow(panel)

if (n_final != n_original) {
  stop("Row count mismatch! Original mart: ", n_original, " rows, final panel: ", n_final, " rows.\n",
       "This indicates a join has multiplied or lost rows. Check for duplicate keys or mismatched joins.")
}

# check for duplicate (GID_2, date) keys (double-check)
n_duplicates <- sum(duplicated(panel[, c("GID_2", "date")]))
if (n_duplicates > 0) {
  stop("Duplicate (GID_2, date) keys detected: ", n_duplicates, " duplicates.\n",
       "This will cause issues in modeling. Investigate which join created duplicates.")
}

# check neighbor proportion bounds (should be [0, 1] if weights are row-standardized)
if ("nb_prop_drop_z2_lag1" %in% names(panel)) {
  nb_prop_valid <- is.na(panel$nb_prop_drop_z2_lag1) |
                   (panel$nb_prop_drop_z2_lag1 >= 0 & panel$nb_prop_drop_z2_lag1 <= 1)
  if (!all(nb_prop_valid)) {
    n_invalid <- sum(!nb_prop_valid)
    stop("Neighbor proportion out of bounds: ", n_invalid, " rows have nb_prop_drop_z2_lag1 outside [0, 1].\n",
         "This suggests weights aren't row-standardized. Check spatial_weights.rds.")
  }
  message("Neighbor proportion bounds check passed: all values in [0, 1]")
}

message("Row-count invariant checks passed:")
message("  Original rows: ", n_original)
message("  Final rows: ", n_final)
message("  Duplicates: 0")

saveRDS(panel, features_path)
message("Saved features to: ", features_path)

manifest <- tibble(
  column = names(panel),
  type = sapply(panel, function(x) class(x)[1]),
  n_missing = sapply(panel, function(x) sum(is.na(x))),
  pct_missing = sapply(panel, function(x) mean(is.na(x)) * 100)
)
write_csv(manifest, feat_manifest_path)
message("Saved manifest to: ", feat_manifest_path)

# QA summary 
pct_drop_z2 <- if ("ntl_drop_z2" %in% names(panel)) mean(panel$ntl_drop_z2 == 1, na.rm = TRUE) * 100 else NA_real_

# dynamic weather lag pattern based on config
weather_lag_pattern <- paste0("^(", paste(weather_vars_primary, collapse = "|"), ")_lag\\d+$")

# count lag features separately for clarity
n_lag_total <- sum(grepl("_lag\\d+$", names(panel)))
n_weather_lags <- sum(grepl(weather_lag_pattern, names(panel)))
n_ntl_lags <- n_lag_total - n_weather_lags

# count spatial features
n_spatial_features <- sum(grepl("^nb_", names(panel)))

# count holiday features
n_holiday_features <- sum(grepl("^is_.*holiday", names(panel)))

qa_summary <- tibble(
  metric = c(
    "n_rows", "n_municipalities", "n_dates",
    "n_features_total", "n_lag_features_total", "n_lag_features_ntl", "n_weather_lag_features",
    "n_rolling_features", "n_drop_indicators", "n_spatial_features", "n_holiday_features",
    "pct_ntl_mean_built_na", "pct_ntl_mean_built_zscore_na", "pct_drop_z2_firing",
    "pct_temp_na", "pct_rain_na", "pct_has_weather",
    "pct_is_holiday", "pct_is_public_holiday",
    "outage_prevalence_pct", "climatology_source_period"
  ),
  value = c(
    as.character(nrow(panel)),
    as.character(n_distinct(panel$GID_2)),
    as.character(n_distinct(panel$date)),
    as.character(ncol(panel)),
    as.character(n_lag_total),
    as.character(n_ntl_lags),
    as.character(n_weather_lags),
    as.character(sum(grepl("_roll\\d+_", names(panel)))),
    as.character(sum(grepl("^ntl_drop_|^ntl_below_", names(panel)))),
    as.character(n_spatial_features),
    as.character(n_holiday_features),
    if ("ntl_mean_built" %in% names(panel)) sprintf("%.1f", mean(is.na(panel$ntl_mean_built)) * 100) else "NA",
    if ("ntl_mean_built_zscore" %in% names(panel)) sprintf("%.1f", mean(is.na(panel$ntl_mean_built_zscore)) * 100) else "NA",
    if (is.na(pct_drop_z2)) "NA" else sprintf("%.2f", pct_drop_z2),
    if ("temp" %in% names(panel)) sprintf("%.1f", mean(is.na(panel$temp)) * 100) else "NA",
    if ("rain" %in% names(panel)) sprintf("%.1f", mean(is.na(panel$rain)) * 100) else "NA",
    if ("has_weather" %in% names(panel)) sprintf("%.1f", mean(panel$has_weather == 1, na.rm = TRUE) * 100) else "NA",
    if ("is_holiday" %in% names(panel)) sprintf("%.1f", mean(panel$is_holiday == 1, na.rm = TRUE) * 100) else "NA",
    if ("is_public_holiday" %in% names(panel)) sprintf("%.1f", mean(panel$is_public_holiday == 1, na.rm = TRUE) * 100) else "NA",
    if ("outage_3h_or_more" %in% names(panel)) sprintf("%.4f", mean(panel$outage_3h_or_more == 1, na.rm = TRUE) * 100) else "NA",
    paste(train_start, "to", train_end)
  )
)

write_csv(qa_summary, feat_qa_path)
message("Saved QA summary to: ", feat_qa_path)
print(qa_summary)

# SECTION 12: FINAL STATUS
message("\n", strrep("=", 60))
message("FEATURE ENGINEERING COMPLETE (IMPROVED VERSION)")
message(strrep("=", 60))

message("\nKey design decisions applied:")
message("  - Lag features: date-aware joins (not row-based lag())")
message("  - Rolling stats: computed on complete date grid")
message("  - Climatology: computed from TRAINING period only (", train_start, " to ", train_end, ")")
message("  - No future information leakage in anomaly/z-score features")
message("  - Spatial lags: LAGGED predictors only + renormalized weights (correct NA handling)")
message("  - Holidays: Mexican holidays (cached, one row per date)")

message("\nImprovements over weatherV1:")
message("  - ", n_holiday_features, " holiday features (is_holiday, is_public_holiday)")
message("  - ", n_spatial_features, " spatial features (nb_mean_*_lag*, nb_prop_drop_z2_lag1)")
message("  - Fast spatial computation: group_modify + renormalized weighted mean")
message("  - Holiday caching: saves API calls on reruns")

message("\nOutputs created:")
message("  - ", features_path)
message("  - ", feat_manifest_path)
message("  - ", feat_qa_path)

message("\nScript completed at: ", Sys.time())
