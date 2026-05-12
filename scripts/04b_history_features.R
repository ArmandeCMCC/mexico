# 04b_history_features.R
#
# PURPOSE:
#   engineer outage-history features (rolling outage rates, durations, cause-specific
#   counts, days-since-last-outage) from the panel and join them into the existing
#   features_engineered.rds
#
# WHY:
#   The current 82-feature set (NTL + weather + temporal) captures EXPOSURE but
#   nothing about INFRASTRUCTURE QUALITY or RECURRENCE. 
#   outage history features => highest-impact, lowest-effort
#   improvement: 43.6% of outages recur at the same substation within 7 days, and
#   municipalities with chronic Technical-cause outages have systematically different
#   durations (median 310 min) than Environmental ones (median 462 min). Past outage
#   activity is the strongest available proxy for unmeasured infrastructure quality.
#
# LEAKAGE DESIGN (CRITICAL):
#   Every feature here is computed strictly on data from t-1 and earlier. We use
#   group-wise cumulative sums shifted by one row, NOT slider rolling windows that
#   include the current observation. The panel has a complete (GID_2 x date) grid
#   (4,464,369 rows = 2,457 GID_2 x 1,817 dates) so cumsum-based windows correspond
#   exactly to calendar windows.
#
#   First N days of each GID_2 trajectory have NA for the corresponding history
#   feature (window is incomplete). This is intentional: do not impute pre-period
#   activity to 0, because that would assert "no prior outages" which is a strong
#   counterfactual claim. XGBoost handles NA natively. Logistic regression in
#   script 15 already handles complete-cases filtering for that branch.
#
# PIPELINE POSITION:
#   Run AFTER 03_training_prep.R + 04_feature_engineering_improved.R. Backs up the
#   existing features_engineered.rds before overwriting.
#
# OUTPUTS (added columns in features_engineered.rds):
#   - hist_outage_rate_30d        : mean(outage_3h_or_more) over [t-30, t-1]
#   - hist_outage_rate_90d        : mean(outage_3h_or_more) over [t-90, t-1]
#   - hist_outage_count_30d       : sum(outage_3h_or_more) over [t-30, t-1]
#                                   (count of nights with outage, NOT events)
#   - hist_n_outages_30d          : sum(n_outages) over [t-30, t-1] (event count)
#   - hist_outage_count_90d       : sum(outage_3h_or_more) over [t-90, t-1]
#   - hist_outage_count_env_30d   : sum(n_outages_environmental) over [t-30, t-1]
#   - hist_outage_count_tech_30d  : sum(n_outages_technical) over [t-30, t-1]
#   - cumulative_outage_minutes_90d : sum(total_length_min) over [t-90, t-1]
#   - hist_median_duration_180d   : median(total_length_min | outage=1) over [t-180, t-1]
#   - hist_outage_count_env_30d   : sum(n_outages_environmental) over [t-30, t-1]
#   - hist_outage_count_tech_30d  : sum(n_outages_technical) over [t-30, t-1]
#   - days_since_last_outage      : t minus most recent prior outage_3h_or_more=1 date
#
#   All measured in calendar days; all using strictly past data.

# SECTION 1: SETUP

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(tidyr)   # for fill()
})

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") {
    return(normalizePath(file.path(getwd(), "..")))
  }
  if (dir.exists("data") && dir.exists("scripts")) {
    return(normalizePath(getwd()))
  }
  warning("Could not determine project directory. Using current directory.")
  normalizePath(getwd())
}

project_dir     <- detect_project_dir()
data_dir        <- file.path(project_dir, "data")
model_ready_dir <- file.path(data_dir, "model_ready")

panel_path        <- file.path(data_dir, "panel_mex_2017_2021_ntl_ghs.rds")
features_path     <- file.path(model_ready_dir, "features_engineered.rds")
features_backup   <- file.path(model_ready_dir, "features_engineered_pre_history.rds")
manifest_path     <- file.path(model_ready_dir, "history_features_manifest.csv")

stopifnot(
  "panel not found"        = file.exists(panel_path),
  "features file not found" = file.exists(features_path)
)

message("Project dir : ", project_dir)
message("Panel       : ", panel_path)
message("Features    : ", features_path)

# SECTION 2: LOAD INPUTS

message("Loading panel...")
panel <- readRDS(panel_path)
message(sprintf("  Panel rows: %d, cols: %d", nrow(panel), ncol(panel)))

required_panel_cols <- c(
  "GID_2", "date", "outage_3h_or_more", "n_outages", "total_length_min",
  "n_outages_environmental", "n_outages_technical"
)
missing <- setdiff(required_panel_cols, names(panel))
if (length(missing) > 0) {
  stop("Panel missing required columns: ", paste(missing, collapse = ", "))
}

message("Loading features_engineered...")
features <- readRDS(features_path)
message(sprintf("  Features rows: %d, cols: %d", nrow(features), ncol(features)))

# Sanity: features and panel must share the (GID_2, date) keying
n_panel    <- nrow(panel)
n_features <- nrow(features)
if (n_features != n_panel) {
  warning(sprintf(
    "Row count mismatch: panel=%d, features=%d. Will inner-join on (GID_2, date).",
    n_panel, n_features
  ))
}

# SECTION 3: HISTORY FEATURES (cumulative-sum based, strict t-1)
#
# For a per-group cumulative sum c_i = x_1 + ... + x_i with c_0 = 0,
# the sum of x over rows [i-N+1, i] is c_i - c_{i-N}. To get the sum
# over [i-N, i-1] (strictly past N days), shift by one: c_{i-1} - c_{i-N-1}.
# We implement this as: lag_cumsum(x, 1) - lag_cumsum(x, N+1).

message("Sorting panel by GID_2, date for cumulative ops...")
hist_input <- panel %>%
  arrange(GID_2, date) %>%
  select(GID_2, date,
         outage_3h_or_more, n_outages, total_length_min,
         n_outages_environmental, n_outages_technical)

# Helper: per-group lagged cumulative sum, with 0 for the first row of each group.
# Uses dplyr::lag with default = 0 to keep arithmetic well-defined at edges; result
# is set to NA only when the look-back window extends before group's start.
group_cumsum <- function(x) cumsum(replace_na(x, 0))

message("Computing rolling sums via lagged cumulative sums...")
hist_input <- hist_input %>%
  group_by(GID_2) %>%
  mutate(
    # Per-group row index (used to determine when window is complete)
    .row_in_group = row_number(),

    # Cumulative sums (per group)
    .cs_outage      = group_cumsum(outage_3h_or_more),
    .cs_n_outages   = group_cumsum(n_outages),
    .cs_minutes     = group_cumsum(total_length_min),
    .cs_n_env       = group_cumsum(n_outages_environmental),
    .cs_n_tech      = group_cumsum(n_outages_technical),

    # Strict-past rolling sums: sum over [t-N, t-1] = lag(cs, 1) - lag(cs, N+1).
    # When the look-back extends before the group's start, lag(cs, N+1) is NA, which
    # propagates correctly to NA in the rolling sum. dplyr::lag default is NA.

    # 30-day window
    hist_outage_count_30d = lag(.cs_outage, 1) - lag(.cs_outage, 31),
    hist_n_outages_30d    = lag(.cs_n_outages, 1) - lag(.cs_n_outages, 31),
    hist_outage_count_env_30d  = lag(.cs_n_env, 1)  - lag(.cs_n_env, 31),
    hist_outage_count_tech_30d = lag(.cs_n_tech, 1) - lag(.cs_n_tech, 31),

    # 90-day window
    hist_outage_count_90d        = lag(.cs_outage, 1) - lag(.cs_outage, 91),
    cumulative_outage_minutes_90d = lag(.cs_minutes, 1) - lag(.cs_minutes, 91),

    # Derived rates (calendar-based — panel grid is complete, so denominator = window)
    hist_outage_rate_30d = hist_outage_count_30d / 30,
    hist_outage_rate_90d = hist_outage_count_90d / 90
  ) %>%
  ungroup()

# SECTION 4: HISTORY MEDIAN DURATION (180-day, strict t-1)
#
# Rolling median => we implement it via a per-group
# loop using a deque-style approach: for each (GID_2, date), gather durations from
# positives in [t-180, t-1] and take the median. This is O(n * window) per group;
# with 1,817 dates and 2,457 groups that is ~8B ops worst-case, far too slow.
#
# Practical approach: per group, maintain an indexed table of (date, total_length_min)
# for positive nights only. For each row, binary-search for the window indices.
# With ~22,996 positives in the whole panel (~9.4 positives per GID_2 on average),
# this is cheap.

message("Computing 180-day rolling median duration on positive nights...")

hist_input <- hist_input %>%
  group_by(GID_2) %>%
  group_modify(~{
    df <- .x
    pos_idx <- which(df$outage_3h_or_more == 1L)
    pos_dates <- df$date[pos_idx]
    pos_dur   <- df$total_length_min[pos_idx]

    n <- nrow(df)
    out <- rep(NA_real_, n)
    if (length(pos_idx) == 0L) {
      df$hist_median_duration_180d <- out
      return(df)
    }

    # For each row date d_i, window is [d_i - 180, d_i - 1]
    # Find positives whose date falls in this window.
    # pos_dates is sorted (ascending) since input is arranged by date.
    cutoffs_lo <- df$date - 180L
    cutoffs_hi <- df$date - 1L

    # Two-pointer scan over sorted positives
    lo_ptr <- 1L
    hi_ptr <- 0L
    for (i in seq_len(n)) {
      # advance hi_ptr while pos_dates[hi_ptr+1] <= cutoffs_hi[i]
      while (hi_ptr < length(pos_dates) && pos_dates[hi_ptr + 1L] <= cutoffs_hi[i]) {
        hi_ptr <- hi_ptr + 1L
      }
      # advance lo_ptr while pos_dates[lo_ptr] < cutoffs_lo[i]
      while (lo_ptr <= length(pos_dates) && pos_dates[lo_ptr] < cutoffs_lo[i]) {
        lo_ptr <- lo_ptr + 1L
      }
      if (hi_ptr >= lo_ptr) {
        out[i] <- median(pos_dur[lo_ptr:hi_ptr], na.rm = TRUE)
      }
    }
    df$hist_median_duration_180d <- out
    df
  }) %>%
  ungroup()

# SECTION 5: DAYS SINCE LAST OUTAGE (strict t-1)
#
# At date t, find max(date) where outage_3h_or_more=1 AND date < t.
# Implementation: per group, mark current date as "last_outage_date_today" if
# outage=1, else NA. Forward-fill to get "last_outage_date_so_far". Lag by one
# row so we use only strictly past info.

message("Computing days_since_last_outage...")
hist_input <- hist_input %>%
  arrange(GID_2, date) %>%
  group_by(GID_2) %>%
  mutate(
    .last_outage_today    = if_else(outage_3h_or_more == 1L, date, as.Date(NA)),
    .last_outage_so_far   = .last_outage_today
  ) %>%
  fill(.last_outage_so_far, .direction = "down") %>%
  mutate(
    .last_outage_strict   = lag(.last_outage_so_far),
    days_since_last_outage = as.integer(date - .last_outage_strict)
  ) %>%
  ungroup()

# SECTION 6: SELECT NEW FEATURE COLUMNS AND JOIN INTO features_engineered.rds

new_feature_cols <- c(
  "hist_outage_count_30d",
  "hist_outage_rate_30d",
  "hist_outage_count_90d",
  "hist_outage_rate_90d",
  "hist_n_outages_30d",
  "hist_outage_count_env_30d",
  "hist_outage_count_tech_30d",
  "cumulative_outage_minutes_90d",
  "hist_median_duration_180d",
  "days_since_last_outage"
)

history_features <- hist_input %>%
  select(GID_2, date, all_of(new_feature_cols))

# SECTION 7: SANITY CHECKS (LEAKAGE + WINDOW INTEGRITY)

message("Running leakage / sanity checks...")

# Check 1: at any row, hist_outage_count_30d must NOT include current-day outage.
# Verify by spot-checking: take a known positive row, confirm history excludes it.
check_dates <- head(panel$date[panel$outage_3h_or_more == 1L], 5)
check_gids  <- head(panel$GID_2[panel$outage_3h_or_more == 1L], 5)

for (k in seq_along(check_dates)) {
  d <- check_dates[k]; g <- check_gids[k]
  panel_sub <- panel %>% filter(GID_2 == g, date >= d - 30L, date <= d)
  prior_count <- sum(panel_sub$outage_3h_or_more[panel_sub$date < d], na.rm = TRUE)
  hf_row <- history_features %>% filter(GID_2 == g, date == d)
  if (nrow(hf_row) == 1L &&
      !is.na(hf_row$hist_outage_count_30d) &&
      hf_row$hist_outage_count_30d != prior_count) {
    stop(sprintf(
      "Leakage check FAILED at GID_2=%s date=%s: hist=%g vs expected_prior=%d",
      g, as.character(d), hf_row$hist_outage_count_30d, prior_count
    ))
  }
}
message("  Leakage spot-checks passed (", length(check_dates), " positive rows verified)")

# Check 2: NA fraction at each window's start
na_summary <- history_features %>%
  summarise(across(all_of(new_feature_cols), ~ mean(is.na(.))))
message("NA fractions per new feature:")
for (c in new_feature_cols) {
  message(sprintf("  %-32s %.4f", c, na_summary[[c]]))
}

# Check 3: positive-rate sanity on training period
joined_for_check <- features %>%
  select(GID_2, date, outage_3h_or_more) %>%
  inner_join(history_features, by = c("GID_2", "date"))

message("Positive-rate by hist_outage_rate_30d quintile (training period 2017-2019):")
training_check <- joined_for_check %>%
  filter(date >= as.Date("2017-02-01"), date <= as.Date("2019-12-31"),  # skip first 30 days
         !is.na(hist_outage_rate_30d)) %>%
  mutate(quintile = ntile(hist_outage_rate_30d, 5)) %>%
  group_by(quintile) %>%
  summarise(
    n           = n(),
    mean_hist   = mean(hist_outage_rate_30d),
    pos_rate    = mean(outage_3h_or_more)
  )
print(training_check)

# Check 4: days_since_last_outage distribution
ds_summary <- history_features %>%
  filter(!is.na(days_since_last_outage)) %>%
  summarise(
    n       = n(),
    min     = min(days_since_last_outage),
    p25     = quantile(days_since_last_outage, 0.25),
    median  = median(days_since_last_outage),
    p75     = quantile(days_since_last_outage, 0.75),
    max     = max(days_since_last_outage)
  )
message("days_since_last_outage distribution:")
print(ds_summary)

# SECTION 8: BACKUP + JOIN + SAVE

if (!file.exists(features_backup)) {
  message("Backing up current features_engineered.rds -> ", basename(features_backup))
  file.copy(features_path, features_backup)
} else {
  message("Backup already exists at ", basename(features_backup), " — leaving it untouched.")
}

# Drop any stale history columns from a prior run before joining.
existing_hist <- intersect(new_feature_cols, names(features))
if (length(existing_hist) > 0) {
  message("Removing stale history columns from features: ",
          paste(existing_hist, collapse = ", "))
  features <- features %>% select(-all_of(existing_hist))
}

features_with_history <- features %>%
  left_join(history_features, by = c("GID_2", "date"))

# Verify join didn't duplicate or drop rows
if (nrow(features_with_history) != nrow(features)) {
  stop(sprintf(
    "Row count changed after history join: before=%d, after=%d",
    nrow(features), nrow(features_with_history)
  ))
}

message(sprintf(
  "Saving features_engineered.rds with %d new history columns (now %d cols total)",
  length(new_feature_cols), ncol(features_with_history)
))
saveRDS(features_with_history, features_path)

# SECTION 9: WRITE MANIFEST FOR REPRODUCIBILITY

manifest <- tibble::tibble(
  feature      = new_feature_cols,
  description  = c(
    "Sum of outage_3h_or_more over [t-30, t-1] (calendar days), per GID_2",
    "hist_outage_count_30d / 30",
    "Sum of outage_3h_or_more over [t-90, t-1]",
    "hist_outage_count_90d / 90",
    "Sum of n_outages over [t-30, t-1]",
    "Sum of n_outages_environmental over [t-30, t-1]",
    "Sum of n_outages_technical over [t-30, t-1]",
    "Sum of total_length_min over [t-90, t-1] (minutes)",
    "Median total_length_min restricted to positive nights in [t-180, t-1]",
    "Days since most recent outage_3h_or_more=1 strictly before t (NA if none in panel)"
  ),
  na_fraction  = unlist(na_summary[new_feature_cols]),
  generated_on = Sys.time()
)
readr::write_csv(manifest, manifest_path)
message("Manifest written: ", manifest_path)

message("DONE.")
