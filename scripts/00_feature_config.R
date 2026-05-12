# 00_feature_config.R
# central feature set configuration for power outage prediction
#
# Two modes:
#   OPERATIONAL (forecast-safe): NTL + time + spatial + lagged weather only
#   DIAGNOSTIC (ex-post): Full model including same-day weather
#
# source file at the top of scripts 04, 05, 06, 07+ to ensure consistency

# FEATURE MODE SWITCH

# Set to "operational" for forecast-safe model (no same-day weather)
# Set to "diagnostic" for full ex-post analysis (includes same-day weather)
FEATURE_MODE <- "operational"  # Options: "operational", "diagnostic"

# DEFINE FEATURE GROUPS

define_feature_groups <- function(all_feature_names) {

  # NTL FEATURES
  # All features containing 'ntl' (includes spatial neighbor features)
  ntl_features <- all_feature_names[grepl("ntl", all_feature_names, ignore.case = TRUE)]

  # WEATHER FEATURES
  # Same-day weather (NOT forecast-safe - contemporaneous with outcome)
  weather_sameday <- c(
    "atm", "dew", "max_dew", "min_dew",
    "lai_high", "lai_low",
    "rain", "rh", "skin_temp", "temp",
    "wdr", "wind_u", "wind_v", "wsp",
    "max_temp", "min_temp"
  )
  weather_sameday <- intersect(weather_sameday, all_feature_names)

  # Lagged weather (forecast-safe - known at prediction time)
  # Auto-detect all weather lag features (pattern: *_lag1, *_lag7, etc.)
  # This includes both current (rain_lag1, temp_lag1) and any new ones added in 04
  weather_lagged <- all_feature_names[grepl("^(atm|dew|max_dew|min_dew|lai_high|lai_low|rain|rh|skin_temp|temp|wdr|wind_u|wind_v|wsp|max_temp|min_temp)_lag[0-9]+$", all_feature_names)]

  # Fallback: if no pattern matches, use explicit list

  if (length(weather_lagged) == 0) {
    weather_lagged <- c("rain_lag1", "rain_lag7", "temp_lag1", "temp_lag7")
    weather_lagged <- intersect(weather_lagged, all_feature_names)
  }

  # All weather
  weather_all <- c(weather_sameday, weather_lagged)

  # TIME/CALENDAR FEATURES
  time_features <- c(
    "dow_cos", "dow_num", "dow_sin",
    "doy", "doy_cos", "doy_sin",
    "is_holiday", "is_public_holiday", "is_weekend",
    "month", "month_cos", "month_sin",
    "year", "clim_n_obs"
  )
  time_features <- intersect(time_features, all_feature_names)

  list(
    ntl = ntl_features,
    weather_sameday = weather_sameday,
    weather_lagged = weather_lagged,
    weather_all = weather_all,
    time = time_features
  )
}

# BUILD FEATURE SETS BY MODE

get_feature_set <- function(all_feature_names, mode = FEATURE_MODE) {

  groups <- define_feature_groups(all_feature_names)

  if (mode == "operational") {
    # OPERATIONAL (forecast-safe): NTL + time + lagged weather only
    # NO same-day weather (contemporaneous correlation, not predictive)
    features <- c(
      groups$ntl,
      groups$time,
      groups$weather_lagged  # Only lagged weather
    )
    message("*** OPERATIONAL MODE: Using forecast-safe features (no same-day weather) ***")
    message(sprintf("    NTL: %d, Time: %d, Weather (lagged only): %d",
                    length(groups$ntl), length(groups$time), length(groups$weather_lagged)))

  } else if (mode == "diagnostic") {
    # DIAGNOSTIC (ex-post): Full model including same-day weather
    features <- c(
      groups$ntl,
      groups$time,
      groups$weather_all  # Includes same-day weather
    )
    message("*** DIAGNOSTIC MODE: Using full feature set (includes same-day weather) ***")
    message(sprintf("    NTL: %d, Time: %d, Weather (all): %d",
                    length(groups$ntl), length(groups$time), length(groups$weather_all)))
    message("    WARNING: Same-day weather is contemporaneous, not suitable for forecasting!")

  } else {
    stop("Invalid FEATURE_MODE. Use 'operational' or 'diagnostic'.")
  }

  # Remove any duplicates
  features <- unique(features)

  message(sprintf("    Total features: %d", length(features)))

  features
}

# CENTRALISED EXCLUSION LISTS
# Three categories of exclusion, controlled by (task_mode, weather_mode):
#   1. EXCLUDE_COLS_BASE: identifiers, label-side columns, dates, categoricals
#      - ALWAYS excluded regardless of task/weather mode
#   2. EXCLUDE_SAMEDAY_NTL: contemporaneous NTL values
#      - Excluded in FORECAST mode, allowed in NOWCAST
#   3. EXCLUDE_SAMEDAY_WEATHER: contemporaneous weather
#      - Excluded when weather_mode = "lagged" (true forecast-safe)
#      - Allowed when weather_mode = "sameday" (assumes weather forecast available)
#
# Build the final exclude list with: build_exclude_cols(task_mode, weather_mode)

# Same-day NTL: contemporaneous nighttime light measurements
EXCLUDE_SAMEDAY_NTL <- c(
  "ntl_mean_all", "ntl_sum_all", "ntl_sd_all", "wsum_all",
  "ntl_mean_built", "ntl_sum_built", "ntl_sd_built", "wsum_built",
  "ntl_mean_built_mask", "ntl_sum_built_mask", "ntl_sd_built_mask", "wsum_built_mask",
  "built_share_area", "built_share_mask"
)

# Same-day weather: contemporaneous with the outage night
# (lagged versions like rain_lag1, temp_lag7 remain available)
EXCLUDE_SAMEDAY_WEATHER <- c(
  "atm", "dew", "max_dew", "min_dew",
  "lai_high", "lai_low",
  "rain", "rh", "skin_temp", "temp",
  "wdr", "wind_u", "wind_v", "wsp",
  "max_temp", "min_temp"
)

# Base exclusions (always applied)
EXCLUDE_COLS_BASE <- c(
  # Identifiers / names
  "GID_2", "GID_1", "GID_0", "COUNTRY", "NAME_1", "NAME_2",
  "state_clean", "municipality_clean",
  "state_clean_orig", "municipality_clean_orig",
  "state_clean_pre_weather", "municipality_clean_pre_weather",

  # Outcomes / label-side columns (prevent leakage)
  "outage_3h_or_more", "n_outages", "max_length_min", "min_length_min",
  "total_length_min", "mean_length_min", "median_length_min",
  "classification_general", "classification_detailed",
  "n_causes_general", "is_mixed_cause_general",
  "n_outages_environmental", "n_outages_technical",
  "n_outages_planned", "n_outages_other",
  "total_length_environmental", "total_length_technical",
  "total_length_planned", "total_length_other",
  "n_substations_affected", "n_distinct_nodes",
  "primary_substation", "primary_node", "dominant_reason",
  "earliest_start_hour", "latest_start_hour",

  # Date + split helpers
  "date", "split",

  # Categorical fields that need encoding
  "climate", "sub_climate", "rain_season", "koppen",
  "ntl_state",

  # Text fields
  "holiday_names",

  # Categorical day-of-week (use encoded versions)
  "dow"
)

#' Build the exclude_cols list for a given configuration
#'
#' Single source of truth for feature exclusion across the pipeline.
#'
#' @param task_mode    "forecast" (drops same-day NTL) or "nowcast" (allows it)
#' @param weather_mode "lagged"   (drops same-day weather) or "sameday" (allows it)
#' @return Character vector of column names to exclude from feature selection
build_exclude_cols <- function(task_mode = "forecast", weather_mode = "lagged") {
  if (!task_mode %in% c("forecast", "nowcast")) {
    stop("task_mode must be 'forecast' or 'nowcast', got: ", task_mode)
  }
  if (!weather_mode %in% c("lagged", "sameday")) {
    stop("weather_mode must be 'lagged' or 'sameday', got: ", weather_mode)
  }

  ex <- EXCLUDE_COLS_BASE
  if (task_mode == "forecast") {
    ex <- c(ex, EXCLUDE_SAMEDAY_NTL)
  }
  if (weather_mode == "lagged") {
    ex <- c(ex, EXCLUDE_SAMEDAY_WEATHER)
  }
  unique(ex)
}

# Backward-compat alias for code that references EXCLUDE_COLS directly.
# DEFAULT = forecast + lagged (truly forecast-safe).
# This is intentionally STRICTER than the original alias, which used "sameday".
# All current consumers (10_operational_baseline.R, approach2_*) wrap this with
# get_feature_set("operational") which already drops same-day weather, so this
# tightening is a no-op for them. New code should call build_exclude_cols(...)
# explicitly with both task_mode and weather_mode.
EXCLUDE_COLS <- build_exclude_cols(task_mode = "forecast", weather_mode = "lagged")

# HELPER: GET ALL VALID FEATURES FROM DATA

get_all_valid_features <- function(df) {
  setdiff(names(df), EXCLUDE_COLS)
}

# PRINT CONFIG SUMMARY

print_feature_config <- function() {
  cat("================================================================================\n")
  cat("FEATURE CONFIGURATION\n")
  cat("================================================================================\n")
  cat(sprintf("Mode: %s\n", FEATURE_MODE))
  cat("\n")
  if (FEATURE_MODE == "operational") {
    cat("OPERATIONAL mode includes:\n")
    cat("  - NTL features (all, including spatial neighbors)\n")
    cat("  - Time/calendar features\n")
    cat("  - Lagged weather ONLY (rain_lag1/7, temp_lag1/7)\n")
    cat("  - NO same-day weather (prevents timing leakage)\n")
  } else {
    cat("DIAGNOSTIC mode includes:\n")
    cat("  - NTL features (all, including spatial neighbors)\n")
    cat("  - Time/calendar features\n")
    cat("  - ALL weather (same-day + lagged)\n")
    cat("  - WARNING: Same-day weather is contemporaneous!\n")
  }
  cat("================================================================================\n")
}

# LOG FEATURE SET FOR TRAINING (call from 05_model_training.R)


log_feature_set <- function(feature_cols, all_feature_names) {
  groups <- define_feature_groups(all_feature_names)

  n_total <- length(feature_cols)
  n_sameday_weather <- length(intersect(feature_cols, groups$weather_sameday))
  n_lagged_weather <- length(intersect(feature_cols, groups$weather_lagged))
  includes_sameday <- n_sameday_weather > 0

  cat("\n")
  cat("================================================================================\n")
  cat("FEATURE SET LOG (for reproducibility)\n")
  cat("================================================================================\n")
  cat(sprintf("Feature mode: %s\n", FEATURE_MODE))
  cat(sprintf("Total predictors: %d\n", n_total))
  cat(sprintf("Same-day weather included: %s (%d features)\n",
              ifelse(includes_sameday, "YES - WARNING!", "NO (forecast-safe)"),
              n_sameday_weather))
  cat(sprintf("Lagged weather included: %d features\n", n_lagged_weather))

  if (includes_sameday && FEATURE_MODE == "operational") {
    cat("\n*** ERROR: Same-day weather detected in OPERATIONAL mode! ***\n")
    cat("*** This should not happen - check feature selection logic. ***\n")
    stop("Same-day weather leakage in operational mode!")
  }

  cat("================================================================================\n\n")

  # Return summary for logging to file
  invisible(list(
    mode = FEATURE_MODE,
    n_predictors = n_total,
    sameday_weather = includes_sameday,
    n_sameday_weather = n_sameday_weather,
    n_lagged_weather = n_lagged_weather
  ))
}

# TASK MODE FILTERS (NOWCAST vs FORECAST)
#
# TWO TASKS:
#   NOWCAST (detect): Same-day NTL(t) signals allowed
#   FORECAST (prevent): Strict t-1 only. No NTL(t) at decision time.
#

#' Apply task mode filters to feature list
#'
#' For NOWCAST: returns features unchanged
#' For FORECAST: removes same-day NTL and anomaly features (keeps lagged versions)
#'
#' @param feature_cols Character vector of feature column names
#' @param task_mode Character, either "nowcast" or "forecast"
#' @param strict Logical, if TRUE (default) stops on any remaining same-day NTL
#' @param verbose Logical, if TRUE prints diagnostic messages
#' @return Filtered character vector of feature column names
apply_task_mode_filters <- function(feature_cols, task_mode = "nowcast",
                                     strict = TRUE, verbose = FALSE) {

  if (!task_mode %in% c("nowcast", "forecast")) {
    stop("Invalid task_mode: '", task_mode, "'. Use 'nowcast' or 'forecast'.")
  }

  if (task_mode == "nowcast") {
    if (verbose) message("TASK MODE: nowcast (same-day NTL allowed)")
    return(feature_cols)
  }

  # FORECAST: strict t-1 only
  if (verbose) message("TASK MODE: forecast (strict t-1)")

  # Patterns indicating "lagged" (safe for forecast)
  lag_pattern <- grepl("_lag[0-9]+$", feature_cols)
  strict_lag_pattern <- grepl("_lag1_strict$", feature_cols)
  is_lagged <- lag_pattern | strict_lag_pattern

  # RULE 1: Drop features containing "ntl" unless lagged
  ntl_pattern <- grepl("ntl", feature_cols, ignore.case = TRUE)
  sameday_ntl <- ntl_pattern & !is_lagged

  # RULE 2: Drop ^anomaly_ features unless lagged
  anomaly_pattern <- grepl("^anomaly_", feature_cols)
  sameday_anomaly <- anomaly_pattern & !is_lagged

  # RULE 3: Drop known same-day spatial features (without lag suffix)
  sameday_spatial <- grepl("^nb_prop_drop_z2$", feature_cols) |
                     grepl("^nb_mean_ntl", feature_cols) & !is_lagged

  # Combine all same-day flags
  sameday <- sameday_ntl | sameday_anomaly | sameday_spatial

  sameday_features <- feature_cols[sameday]
  safe_features <- feature_cols[!sameday]

  if (verbose && length(sameday_features) > 0) {
    message(sprintf("  Dropped %d same-day features", length(sameday_features)))
  }

  # DEFENSIVE: remaining features with "ntl" or "anomaly_" should all be lagged
  if (strict) {
    remaining_ntl <- safe_features[grepl("ntl", safe_features, ignore.case = TRUE)]
    remaining_anom <- safe_features[grepl("^anomaly_", safe_features)]
    remaining_suspect <- c(remaining_ntl, remaining_anom)
    remaining_no_lag <- remaining_suspect[!grepl("_lag[0-9]+$|_lag1_strict$", remaining_suspect)]

    if (length(remaining_no_lag) > 0) {
      stop("FORECAST LEAKAGE: features without lag suffix:\n  ",
           paste(remaining_no_lag, collapse = "\n  "))
    }
  }

  if (verbose) {
    message(sprintf("  Forecast-safe: %d features", length(safe_features)))
  }

  safe_features
}

#' List features that would be dropped in FORECAST mode
#'
#' @param feature_cols Character vector of feature column names
#' @return Character vector of same-day features
get_sameday_ntl_features <- function(feature_cols) {
  is_lagged <- grepl("_lag[0-9]+$|_lag1_strict$", feature_cols)
  ntl_pattern <- grepl("ntl", feature_cols, ignore.case = TRUE)
  anomaly_pattern <- grepl("^anomaly_", feature_cols)
  sameday <- (ntl_pattern | anomaly_pattern) & !is_lagged
  feature_cols[sameday]
}

# Print config when sourced
print_feature_config()
