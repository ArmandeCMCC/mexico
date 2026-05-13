# _ablation_helpers.R
# Shared utilities for ablation studies
#
# Provides:
#   - Feature family definitions (regex patterns for 05b filtering)
#   - Task-mode aware preliminary filtering (mirrors 05b's exclude_cols + apply_task_mode_filters)
#   - Ablation configuration specs
#   - Audit helpers for verifying regex-to-feature mapping
#
# Design: 07b uses these definitions to set env vars for 05b.
# 05b does actual filtering via FEATURE_INCLUDE_REGEX / FEATURE_EXCLUDE_REGEX,
# plus additional filtering (all-NA cols, character cols, factor cols) at runtime.
#
# Note: No ^anomaly_ prefixed features currently exist in the feature inventory.
# If such features are added, a dedicated anomaly family should be created.
#
# Important: 05b_ablation uses STRICT forecast mode which excludes:
#   - Same-day NTL (unless lagged)
#   - Same-day anomaly features (unless lagged)
#   - Same-day weather (contemporaneous weather has leakage risk)
# Lagged weather (rain_lag1, temp_lag7, etc.) is allowed in forecast mode.
# The `no_leakage_risk` field indicates conceptual forecast safety (no temporal leakage),
# and `usable_in_05b` indicates whether the family can appear in 05b input data.
#
# Dependencies: base R only

# 05b EXCLUDE_COLS — sourced from 00_feature_config.R (single source of truth)
# Provides EXCLUDE_COLS_BASE, EXCLUDE_SAMEDAY_NTL, EXCLUDE_SAMEDAY_WEATHER,
# build_exclude_cols(task_mode, weather_mode), apply_task_mode_filters().
# Do NOT redefine these locally - drift between this file and 05b is a footgun.
# Caller must have already source()'d 00_feature_config.R, OR have set
# working dir to the project root so the relative path resolves.
if (!exists("build_exclude_cols", mode = "function")) {
  config_path <- if (file.exists("scripts/00_feature_config.R")) {
    "scripts/00_feature_config.R"
  } else if (file.exists("00_feature_config.R")) {
    "00_feature_config.R"
  } else {
    stop("_ablation_helpers.R: cannot locate 00_feature_config.R. ",
         "source() it first or run from project root.")
  }
  source(config_path)
}

# TASK-MODE FILTERING (thin wrapper around centralised)

#' Apply task-mode filter (preserves singular-name API for back-compat)
#'
#' Wraps the centralized apply_task_mode_filters() from 00_feature_config.R.
#' In FORECAST mode: drops NTL/anomaly/nb_* features without lag suffix.
#' Same-day weather is dropped via the exclude list (build_exclude_cols), not here.
#' In NOWCAST mode: all features allowed.
apply_task_mode_filter <- function(features, task_mode = "forecast") {
  apply_task_mode_filters(features, task_mode = task_mode, strict = FALSE, verbose = FALSE)
}

#' Get preliminary 05b-eligible features
#'
#' Applies exclude_cols + task_mode filter via centralized config.
#' NOTE: "Preliminary" because 05b also removes all-NA/char/factor cols at runtime.
#' Always uses weather_mode="lagged" to mirror 05b_ablation's strict forecast behavior.
get_preliminary_eligible_features <- function(feature_names, task_mode = "forecast") {
  if (length(feature_names) == 1 && file.exists(feature_names)) {
    features_df <- readRDS(feature_names)
    feature_names <- names(features_df)
  }
  weather_mode <- if (task_mode == "forecast") "lagged" else "sameday"
  exclude_cols <- build_exclude_cols(task_mode = task_mode, weather_mode = weather_mode)
  features <- setdiff(feature_names, exclude_cols)
  apply_task_mode_filter(features, task_mode)
}


# FEATURE FAMILY DEFINITIONS

FEATURE_FAMILIES <- list(

  # NTL LAGS: lagged NTL values (forecast-safe) 
  ntl_lags = list(
    include = "^ntl_(mean|sum|sd)_built_lag[0-9]+$",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "Lagged NTL values (1d, 2d, 7d, 14d)"
  ),

  # NTL SAME-DAY: same-day NTL aggregates + wsum + built_share (nowcast only)
  ntl_sameday = list(
    include = "^(ntl_(mean|sum|sd)_(all|built|built_mask)|wsum_(all|built|built_mask)|built_share_(area|mask))$",
    exclude = NULL,
    no_leakage_risk = FALSE,
    usable_in_05b = TRUE,
    description = "Same-day NTL/wsum/built_share (nowcast only, excluded in forecast)"
  ),

  # NTL ROLLING: rolling window statistics (nowcast only)
  ntl_rolling = list(
    include = "^ntl_(mean|sum)_built_roll[0-9]+_(mean|sd)$",
    exclude = NULL,
    no_leakage_risk = FALSE,
    usable_in_05b = TRUE,
    description = "Rolling NTL stats (nowcast only)"
  ),

  # NTL DIFFS: short-term differences (nowcast only) 
  ntl_diffs = list(
    include = "^ntl_(mean|sum)_built_(diff[0-9]+|pct_change[0-9]+)$",
    exclude = NULL,
    no_leakage_risk = FALSE,
    usable_in_05b = TRUE,
    description = "NTL differences/pct_change (nowcast only)"
  ),

  # NTL ANOMALIES: z-scores, anomaly, climatology (nowcast only)
  ntl_anomalies = list(
    include = "^ntl_(mean|sum)_built_(anomaly|zscore|clim_mean|clim_sd)$",
    exclude = NULL,
    no_leakage_risk = FALSE,
    usable_in_05b = TRUE,
    description = "NTL anomaly indicators (nowcast only)"
  ),

  # NTL DROPS SAME-DAY: drop detection (nowcast only) 
  ntl_drops_sameday = list(
    include = "^ntl_(drop_|below_)",
    exclude = "_lag[0-9]+$",
    no_leakage_risk = FALSE,
    usable_in_05b = TRUE,
    description = "NTL drop detection, same-day (nowcast only)"
  ),

  # NTL DROPS LAGGED: lagged drop/below signals (forecast-safe) 
  ntl_drops_lagged = list(
    include = "^ntl_(drop_|below_).*_lag[0-9]+$",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "Lagged NTL drop/below signals (forecast-safe)"
  ),

  # NTL SPATIAL: neighborhood features (forecast-safe if lagged)
  ntl_spatial = list(
    include = "^nb_(mean|prop|sum)_.*_lag[0-9]+$",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "Spatial neighborhood NTL (lagged, forecast-safe)"
  ),

  # WEATHER SAME-DAY: contemporaneous weather (leakage risk, but 05b-eligible)
  weather_sameday = list(
    include = "^(atm|dew|max_dew|min_dew|lai_high|lai_low|rain|rh|skin_temp|temp|wdr|wind_u|wind_v|wsp|max_temp|min_temp)$",
    exclude = NULL,
    no_leakage_risk = FALSE,
    usable_in_05b = TRUE,
    description = "Same-day weather (05b-eligible but has leakage risk; excluded by strict forecast ablations)"
  ),

  # WEATHER LAGGED: lagged weather (forecast-safe) 
  weather_lagged = list(
    include = "^(atm|dew|max_dew|min_dew|lai_high|lai_low|rain|rh|skin_temp|temp|wdr|wind_u|wind_v|wsp|max_temp|min_temp)_lag[0-9]+$",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "Lagged weather (1d, 7d, forecast-safe)"
  ),
  
  # WEATHER AVAILABILITY: has_weather flag (forecast-safe, usable)
  weather_availability = list(
    include = "^has_weather$",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "Weather data availability flag (binary, usable)"
  ),

  # CLIMATE CATEGORICAL: climate zone indicators (AUDIT-ONLY, excluded by 05b) 
  climate_categorical = list(
    include = "^(climate|sub_climate|koppen|rain_season)$",
    exclude = NULL,
    no_leakage_risk = FALSE,
    usable_in_05b = FALSE,
    description = "Climate zones (AUDIT-ONLY: excluded by 05b as categorical)"
  ),

  # WEATHER CLIMATE: umbrella for all weather + climate features (orchestration/audit) 
  weather_climate = list(
    include = "^(atm|dew|max_dew|min_dew|lai_high|lai_low|rain|rh|skin_temp|temp|wdr|wind_u|wind_v|wsp|max_temp|min_temp|climate|sub_climate|koppen|rain_season|has_weather)(_lag[0-9]+)?$",
    exclude = NULL,
    no_leakage_risk = FALSE,
    usable_in_05b = FALSE,
    description = "Umbrella: all weather + climate features (AUDIT/ORCHESTRATION only)"
  ),

  # TIME/CALENDAR: temporal features (forecast-safe)
  time_calendar = list(
    include = "^(dow|doy|month|year|is_holiday|is_public_holiday|is_weekend|is_covid_period)(_|$)",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "Calendar and temporal features"
  ),

  # QUALITY FLAGS: observation counts (forecast-safe)
  quality_flags = list(
    include = "^clim_n_obs$",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "Quality/observation count indicators"
  ),

  # OUTAGE HISTORY: past failure indicators computed up to t-1 (forecast-safe)
  # Includes rolling 30d/90d counts, env/tech-stratified counts, rate, median duration,
  # cumulative minutes, and days_since_last_outage. All are strictly t-1 by construction.
  outage_history = list(
    include = "^(hist_|cumulative_outage_minutes|days_since_last_outage)",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "Past outage history features (counts/rates/duration up to t-1)"
  ),

  # BUILT INFRASTRUCTURE: static municipality-level built-up share 
  # Mean built_share_area/mask across training period (time-invariant per muni).
  # Captures grid density / urbanization as an infrastructure context variable.
  # Named without "ntl" so apply_task_mode_filters does not drop them.
  built_infrastructure = list(
    include = "^built_share_(area|mask)_static$",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "Static municipality built-up share (infrastructure context)"
  ),

  # NTL RATIO: log(NTL_lagged / NTL_climatological) 
  # Physics-motivated feature (Arora et al. 2025 Beer's Law analogy).
  # Captures proportional deviation from baseline, forecast-safe by construction.
  ntl_ratio = list(
    include = "^ntl_log_ratio_(mean|sum)_lag[0-9]+$",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "NTL log-ratio features: log(NTL_lagged / NTL_climatological)"
  ),

  # MOON FRACTION: lunar illumination confounder control 
  # Moonlight affects VIIRS NTL readings (Arora et al. 2025, Román et al. 2018).
  # Pure date-derived, forecast-safe.
  moon_phase = list(
    include = "^moon_fraction$",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "Moon illumination fraction (confounder control for NTL)"
  ),

  # SPATIAL OUTAGE: neighbor and state-level outage propagation (forecast-safe)
  # Captures crew competition, shared grid stress, and regional outage clustering.
  # All strictly t-1 by construction (uses lagged outage indicators).
  spatial_outage = list(
    include = "^(nb_outage_|state_outage_)",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "Spatial outage propagation: neighbor/state-level outage signals (t-1)"
  ),

  # CENACE PRICES: substation-level price/congestion features (forecast-safe)
  # Columns use lag1/lag7 values from CENACE marginal node prices plus
  # lagged rolling summaries (shifted to t-1), aggregated from substation
  # to municipality. Captures local grid stress
  # (congestion), demand pressure (peak/offpeak), and price anomalies.
  # Coverage: ~30% of panel (municipalities with CENACE substations).
  cenace_prices = list(
    include = "^cenace_",
    exclude = NULL,
    no_leakage_risk = TRUE,
    usable_in_05b = TRUE,
    description = "CENACE substation price/congestion features (lagged + lagged rolling, forecast-safe)"
  )
)

# ABLATION CONFIGURATIONS

ABLATION_SPECS <- list(

  full_model = list(
    name = "full_model",
    task_mode = "forecast",
    include_families = c("ntl_lags", "ntl_drops_lagged", "ntl_spatial",
                         "weather_lagged", "weather_availability", "time_calendar", "quality_flags"),
    description = "Full forecast-safe model"
  ),

  ntl_only = list(
    name = "ntl_only",
    task_mode = "forecast",
    include_families = c("ntl_lags", "ntl_drops_lagged", "ntl_spatial",
                         "time_calendar", "quality_flags"),
    description = "NTL features only (no weather)"
  ),

  ntl_core = list(
    name = "ntl_core",
    task_mode = "forecast",
    include_families = c("ntl_lags", "time_calendar"),
    description = "NTL lags only (minimal baseline)"
  ),

  ntl_plus_spatial = list(
    name = "ntl_plus_spatial",
    task_mode = "forecast",
    include_families = c("ntl_lags", "ntl_spatial", "time_calendar"),
    description = "NTL lags + spatial neighborhood"
  ),

  ntl_plus_drops = list(
    name = "ntl_plus_drops",
    task_mode = "forecast",
    include_families = c("ntl_lags", "ntl_drops_lagged", "time_calendar"),
    description = "NTL lags + lagged drop signals"
  ),

  weather_lagged_only = list(
    name = "weather_lagged_only",
    task_mode = "forecast",
    include_families = c("weather_lagged", "weather_availability", "time_calendar"),
    description = "Lagged weather only (no NTL)"
  ),

  ntl_plus_weather = list(
    name = "ntl_plus_weather",
    task_mode = "forecast",
    include_families = c("ntl_lags", "ntl_drops_lagged", "weather_lagged",
                         "weather_availability", "time_calendar", "quality_flags"),
    description = "NTL + lagged weather (no spatial)"
  ),

  full_nowcast = list(
    name = "full_nowcast",
    task_mode = "nowcast",
    include_families = c("ntl_sameday", "ntl_lags", "ntl_rolling", "ntl_diffs",
                         "ntl_anomalies", "ntl_drops_sameday", "ntl_spatial",
                         "weather_sameday", "weather_lagged", "weather_availability",
                         "time_calendar", "quality_flags"),
    description = "Full nowcast model (includes same-day features)"
  ),

  # All three are forecast-safe (strict t-1). They share NTL+weather+spatial+calendar
  # and differ only in whether outage-history features (and later, load) are included.
  #
  # bench_rs_only:        pure remote-sensing baseline
  # bench_rs_history:     deployable public-data model (adds outage history)
  # bench_rs_history_load: operational augmented model (adds CENACE price features)

  bench_rs_only = list(
    name = "bench_rs_only",
    task_mode = "forecast",
    include_families = c("ntl_lags", "ntl_drops_lagged", "ntl_spatial",
                         "weather_lagged", "weather_availability",
                         "time_calendar", "quality_flags",
                         "built_infrastructure"),
    description = "SUPERIMP benchmark 1: remote-sensing only (NTL+weather+spatial, NO history)"
  ),

  bench_rs_history = list(
    name = "bench_rs_history",
    task_mode = "forecast",
    include_families = c("ntl_lags", "ntl_drops_lagged", "ntl_spatial",
                         "weather_lagged", "weather_availability",
                         "time_calendar", "quality_flags",
                         "outage_history", "built_infrastructure",
                         "ntl_ratio", "moon_phase"),
    description = "SUPERIMP benchmark 2: remote-sensing + outage history (deployable public-data model)"
  ),

  bench_rs_history_spatial = list(
    name = "bench_rs_history_spatial",
    task_mode = "forecast",
    include_families = c("ntl_lags", "ntl_drops_lagged", "ntl_spatial",
                         "weather_lagged", "weather_availability",
                         "time_calendar", "quality_flags",
                         "outage_history", "built_infrastructure",
                         "ntl_ratio", "moon_phase",
                         "spatial_outage"),
    description = "Benchmark: RS + history + spatial outage propagation"
  ),

  bench_rs_history_load = list(
    name = "bench_rs_history_load",
    task_mode = "forecast",
    include_families = c("ntl_lags", "ntl_drops_lagged", "ntl_spatial",
                         "weather_lagged", "weather_availability",
                         "time_calendar", "quality_flags",
                         "outage_history", "built_infrastructure",
                         "ntl_ratio", "moon_phase",
                         "cenace_prices"),
    description = "benchmark 3: RS + history + CENACE prices (full augmented model)"
  )
)

# UTILITY FUNCTIONS

build_include_regex <- function(family_names) {
  patterns <- character(length(family_names))
  for (i in seq_along(family_names)) {
    name <- family_names[i]
    if (!name %in% names(FEATURE_FAMILIES)) {
      stop("Unknown feature family: ", name,
           ". Available: ", paste(names(FEATURE_FAMILIES), collapse = ", "))
    }
    patterns[i] <- FEATURE_FAMILIES[[name]]$include
  }
  paste0("(", paste(patterns, collapse = ")|("), ")")
}

build_exclude_regex <- function(family_names) {
  excludes <- character(0)
  for (name in family_names) {
    if (!name %in% names(FEATURE_FAMILIES)) stop("Unknown family: ", name)
    excl <- FEATURE_FAMILIES[[name]]$exclude
    if (!is.null(excl)) excludes <- c(excludes, excl)
  }
  if (length(excludes) == 0) return("")
  paste0("(", paste(excludes, collapse = ")|("), ")")
}

get_ablation_spec <- function(ablation_name) {
  if (!ablation_name %in% names(ABLATION_SPECS)) {
    stop("Unknown ablation: ", ablation_name,
         ". Available: ", paste(names(ABLATION_SPECS), collapse = ", "))
  }
  spec <- ABLATION_SPECS[[ablation_name]]

  # Guard: error if any included family is not usable in 05b

  for (fam_name in spec$include_families) {
    fam <- FEATURE_FAMILIES[[fam_name]]
    if (!isTRUE(fam$usable_in_05b)) {
      stop("Ablation '", ablation_name, "' includes family '", fam_name,
           "' which is marked usable_in_05b=FALSE (audit-only). ",
           "Remove this family from the ablation spec.")
    }
  }

  list(
    name = spec$name,
    task_mode = spec$task_mode,
    include_regex = build_include_regex(spec$include_families),
    exclude_regex = build_exclude_regex(spec$include_families),
    description = spec$description,
    families = spec$include_families
  )
}

list_ablations <- function() {
  data.frame(
    name = vapply(ABLATION_SPECS, function(s) s$name, character(1)),
    task_mode = vapply(ABLATION_SPECS, function(s) s$task_mode, character(1)),
    description = vapply(ABLATION_SPECS, function(s) s$description, character(1)),
    families = vapply(ABLATION_SPECS, function(s) paste(s$include_families, collapse = ", "), character(1)),
    stringsAsFactors = FALSE
  )
}

# AUDIT HELPERS

audit_feature_families <- function(feature_names, task_mode = NULL) {
  if (length(feature_names) == 1 && file.exists(feature_names)) {
    features_df <- readRDS(feature_names)
    feature_names <- names(features_df)
  }
  prelim_eligible <- if (!is.null(task_mode)) {
    get_preliminary_eligible_features(feature_names, task_mode)
  } else NULL

  results <- vector("list", length(FEATURE_FAMILIES))
  for (i in seq_along(FEATURE_FAMILIES)) {
    fam_name <- names(FEATURE_FAMILIES)[i]
    fam <- FEATURE_FAMILIES[[i]]

    raw_matches <- feature_names[grepl(fam$include, feature_names, perl = TRUE)]
    if (!is.null(fam$exclude) && length(raw_matches) > 0) {
      raw_matches <- raw_matches[!grepl(fam$exclude, raw_matches, perl = TRUE)]
    }

    n_prelim <- if (!is.null(prelim_eligible)) {
      pm <- prelim_eligible[grepl(fam$include, prelim_eligible, perl = TRUE)]
      if (!is.null(fam$exclude) && length(pm) > 0) pm <- pm[!grepl(fam$exclude, pm, perl = TRUE)]
      length(pm)
    } else NA_integer_

    examples <- if (length(raw_matches) > 3) {
      paste(c(head(raw_matches, 3), "..."), collapse = ", ")
    } else if (length(raw_matches) > 0) {
      paste(raw_matches, collapse = ", ")
    } else "(none)"

    results[[i]] <- data.frame(
      family = fam_name,
      pattern = fam$include,
      no_leakage_risk = fam$no_leakage_risk,
      usable_in_05b = fam$usable_in_05b,
      n_raw = length(raw_matches),
      n_prelim_eligible = n_prelim,
      examples = examples,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, results)
}

#' Audit a specific ablation against actual features
#'
#' @param ablation_name name from ABLATION_SPECS
#' @param feature_names character vector or path to features_engineered.rds
#' @return list with ablation info, regex patterns, raw/preliminary-eligible counts and feature vectors
audit_ablation <- function(ablation_name, feature_names) {
  if (length(feature_names) == 1 && file.exists(feature_names)) {
    features_df <- readRDS(feature_names)
    feature_names <- names(features_df)
  }
  spec <- get_ablation_spec(ablation_name)

  raw_matched <- feature_names[grepl(spec$include_regex, feature_names, perl = TRUE)]
  if (spec$exclude_regex != "") {
    raw_matched <- raw_matched[!grepl(spec$exclude_regex, raw_matched, perl = TRUE)]
  }

  prelim <- get_preliminary_eligible_features(feature_names, spec$task_mode)
  prelim_matched <- prelim[grepl(spec$include_regex, prelim, perl = TRUE)]
  if (spec$exclude_regex != "") {
    prelim_matched <- prelim_matched[!grepl(spec$exclude_regex, prelim_matched, perl = TRUE)]
  }

  list(
    ablation = ablation_name,
    task_mode = spec$task_mode,
    description = spec$description,
    families = spec$families,
    include_regex = spec$include_regex,
    exclude_regex = spec$exclude_regex,
    n_raw_total = length(feature_names),
    n_raw_matched = length(raw_matched),
    raw_matched_features = sort(raw_matched),
    n_prelim_eligible_total = length(prelim),
    n_prelim_eligible_matched = length(prelim_matched),
    prelim_eligible_matched_features = sort(prelim_matched)
  )
}

# DEFAULT POLICIES FOR ABLATION COMPARISONS

HEADLINE_POLICY <- list(type = "precision_floor", target = 0.10)
SECONDARY_POLICY <- list(type = "alerts_per_day_cap", target = 10)
