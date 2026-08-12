# Build isolated feature files for overpass-timing robustness labels.
#
# This script does not modify data/model_ready/features_engineered.rds or any
# existing output. It starts from features_engineered_pre_history.rds, replaces
# the target/label-side columns with one timing label variant, recomputes
# strictly-past outage-history features from that same variant target, and saves
# the result under:
#   data/model_ready/timing_overpass/features/<variant_id>/features_engineered.rds

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lubridate)
  library(tidyr)
  library(tibble)
})

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (basename(dirname(getwd())) == "scripts") return(normalizePath(file.path(getwd(), "..", "..")))
  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))
  normalizePath(getwd())
}

label_side_cols <- c(
  "outage_3h_or_more", "n_outages", "max_length_min", "min_length_min",
  "total_length_min", "mean_length_min", "median_length_min",
  "earliest_start_hour", "latest_start_hour",
  "n_causes_general", "is_mixed_cause_general",
  "n_outages_environmental", "n_outages_technical",
  "n_outages_planned", "n_outages_other",
  "total_length_environmental", "total_length_technical",
  "total_length_planned", "total_length_other",
  "classification_general", "classification_detailed",
  "n_substations_affected", "n_distinct_nodes",
  "primary_substation", "primary_node", "dominant_reason"
)

history_feature_cols <- c(
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

zero_integer_cols <- c(
  "outage_3h_or_more", "n_outages",
  "n_causes_general", "is_mixed_cause_general",
  "n_outages_environmental", "n_outages_technical",
  "n_outages_planned", "n_outages_other",
  "n_substations_affected", "n_distinct_nodes"
)

zero_numeric_cols <- c(
  "max_length_min", "min_length_min", "total_length_min",
  "mean_length_min", "median_length_min",
  "total_length_environmental", "total_length_technical",
  "total_length_planned", "total_length_other"
)

fill_missing_label_side <- function(df) {
  for (nm in label_side_cols) {
    if (!nm %in% names(df)) {
      if (nm %in% c(zero_integer_cols, zero_numeric_cols)) {
        df[[nm]] <- 0
      } else {
        df[[nm]] <- NA
      }
    }
  }
  for (nm in intersect(zero_integer_cols, names(df))) {
    df[[nm]] <- as.integer(replace_na(df[[nm]], 0L))
  }
  for (nm in intersect(zero_numeric_cols, names(df))) {
    df[[nm]] <- as.numeric(replace_na(df[[nm]], 0))
  }
  df
}

group_cumsum <- function(x) cumsum(replace_na(x, 0))

compute_history_features <- function(df) {
  required <- c(
    "GID_2", "date", "outage_3h_or_more", "n_outages", "total_length_min",
    "n_outages_environmental", "n_outages_technical"
  )
  missing <- setdiff(required, names(df))
  if (length(missing) > 0L) {
    stop("Cannot compute history features; missing columns: ", paste(missing, collapse = ", "))
  }

  hist_input <- df %>%
    arrange(GID_2, date) %>%
    select(
      GID_2, date, outage_3h_or_more, n_outages, total_length_min,
      n_outages_environmental, n_outages_technical
    ) %>%
    group_by(GID_2) %>%
    mutate(
      .cs_outage = group_cumsum(outage_3h_or_more),
      .cs_n_outages = group_cumsum(n_outages),
      .cs_minutes = group_cumsum(total_length_min),
      .cs_n_env = group_cumsum(n_outages_environmental),
      .cs_n_tech = group_cumsum(n_outages_technical),
      hist_outage_count_30d = lag(.cs_outage, 1) - lag(.cs_outage, 31),
      hist_n_outages_30d = lag(.cs_n_outages, 1) - lag(.cs_n_outages, 31),
      hist_outage_count_env_30d = lag(.cs_n_env, 1) - lag(.cs_n_env, 31),
      hist_outage_count_tech_30d = lag(.cs_n_tech, 1) - lag(.cs_n_tech, 31),
      hist_outage_count_90d = lag(.cs_outage, 1) - lag(.cs_outage, 91),
      cumulative_outage_minutes_90d = lag(.cs_minutes, 1) - lag(.cs_minutes, 91),
      hist_outage_rate_30d = hist_outage_count_30d / 30,
      hist_outage_rate_90d = hist_outage_count_90d / 90
    ) %>%
    ungroup()

  message("  Computing 180-day rolling median duration...")
  hist_input <- hist_input %>%
    group_by(GID_2) %>%
    group_modify(~{
      dat <- .x
      pos_idx <- which(dat$outage_3h_or_more == 1L)
      pos_dates <- dat$date[pos_idx]
      pos_dur <- dat$total_length_min[pos_idx]
      n <- nrow(dat)
      out <- rep(NA_real_, n)

      if (length(pos_idx) > 0L) {
        lo_ptr <- 1L
        hi_ptr <- 0L
        cutoffs_lo <- dat$date - 180L
        cutoffs_hi <- dat$date - 1L

        for (i in seq_len(n)) {
          while (hi_ptr < length(pos_dates) && pos_dates[hi_ptr + 1L] <= cutoffs_hi[i]) {
            hi_ptr <- hi_ptr + 1L
          }
          while (lo_ptr <= length(pos_dates) && pos_dates[lo_ptr] < cutoffs_lo[i]) {
            lo_ptr <- lo_ptr + 1L
          }
          if (hi_ptr >= lo_ptr) {
            out[i] <- median(pos_dur[lo_ptr:hi_ptr], na.rm = TRUE)
          }
        }
      }

      dat$hist_median_duration_180d <- out
      dat
    }) %>%
    ungroup()

  message("  Computing days_since_last_outage...")
  hist_input <- hist_input %>%
    arrange(GID_2, date) %>%
    group_by(GID_2) %>%
    mutate(
      .last_outage_today = if_else(outage_3h_or_more == 1L, date, as.Date(NA)),
      .last_outage_so_far = .last_outage_today
    ) %>%
    fill(.last_outage_so_far, .direction = "down") %>%
    mutate(
      .last_outage_strict = lag(.last_outage_so_far),
      days_since_last_outage = as.integer(date - .last_outage_strict)
    ) %>%
    ungroup()

  hist_input %>%
    select(GID_2, date, all_of(history_feature_cols))
}

run_leakage_spot_check <- function(df, history_features) {
  positives <- df %>%
    filter(outage_3h_or_more == 1L) %>%
    arrange(date) %>%
    slice_head(n = 5)

  if (nrow(positives) == 0L) return(invisible(TRUE))

  for (i in seq_len(nrow(positives))) {
    g <- positives$GID_2[[i]]
    d <- positives$date[[i]]
    expected <- df %>%
      filter(GID_2 == g, date >= d - 30L, date < d) %>%
      summarise(x = sum(outage_3h_or_more, na.rm = TRUE)) %>%
      pull(x)
    observed <- history_features %>%
      filter(GID_2 == g, date == d) %>%
      pull(hist_outage_count_30d)
    if (length(observed) == 1L && !is.na(observed) && observed != expected) {
      stop(
        "Leakage check failed for GID_2=", g, " date=", d,
        ": observed hist_outage_count_30d=", observed,
        ", expected strictly-prior count=", expected
      )
    }
  }
  invisible(TRUE)
}

args <- commandArgs(trailingOnly = TRUE)
overwrite <- "--overwrite" %in% args
dry_run <- "--dry-run" %in% args
build_all <- "--all" %in% args
args <- setdiff(args, c("--overwrite", "--dry-run", "--all"))

project_dir <- detect_project_dir()
data_dir <- file.path(project_dir, "data")
timing_root <- file.path(data_dir, "model_ready", "timing_overpass")
label_dir <- file.path(timing_root, "labels")
features_root <- file.path(timing_root, "features")
dir.create(features_root, recursive = TRUE, showWarnings = FALSE)

base_features_path <- file.path(data_dir, "model_ready", "features_engineered_pre_history.rds")
labels_path <- file.path(label_dir, "overpass_muni_night_labels_model_keys.rds")
variant_manifest_path <- file.path(label_dir, "overpass_variant_manifest.csv")

required <- c(base_features_path, labels_path, variant_manifest_path)
missing <- required[!file.exists(required)]
if (length(missing) > 0L) {
  stop("Missing required files:\n", paste0("  - ", missing, collapse = "\n"),
       "\nRun scripts/timing_overpass/01_build_overpass_labels.R first.")
}

variant_manifest <- read_csv(variant_manifest_path, show_col_types = FALSE)
recommended_variants <- variant_manifest %>%
  filter(analysis_role == "recommended") %>%
  pull(variant_id)

variants <- if (build_all) {
  recommended_variants
} else if (length(args) > 0L) {
  args
} else {
  variant_manifest %>%
    filter(buffer_minutes == 0L) %>%
    pull(variant_id)
}

invalid <- setdiff(variants, variant_manifest$variant_id)
if (length(invalid) > 0L) {
  stop("Unknown variant(s): ", paste(invalid, collapse = ", "),
       "\nAvailable: ", paste(variant_manifest$variant_id, collapse = ", "))
}

message("Project dir: ", project_dir)
message("Base features: ", base_features_path)
message("Timing labels: ", labels_path)
message("Variants: ", paste(variants, collapse = ", "))
message("Dry run: ", dry_run)

for (variant_id in variants) {
  out_dir <- file.path(features_root, variant_id)
  out_features_path <- file.path(out_dir, "features_engineered.rds")
  out_manifest_path <- file.path(out_dir, "variant_feature_manifest.csv")
  out_history_manifest_path <- file.path(out_dir, "history_features_manifest.csv")
  out_qa_path <- file.path(out_dir, "variant_feature_qa.csv")

  message("\n", strrep("=", 72))
  message("Variant: ", variant_id)
  message(strrep("=", 72))
  message("Output: ", out_features_path)

  if (!overwrite && file.exists(out_features_path)) {
    stop("Variant feature file already exists: ", out_features_path,
         "\nRe-run with --overwrite to replace only this timing-variant output.")
  }

  if (dry_run) {
    next
  }

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  message("Loading base pre-history features...")
  base <- readRDS(base_features_path) %>%
    mutate(
      .row_id = row_number(),
      GID_2 = as.character(GID_2),
      date = as.Date(date)
    )
  base_n <- nrow(base)

  drop_cols <- intersect(c(label_side_cols, history_feature_cols), names(base))
  if (length(drop_cols) > 0L) {
    message("Dropping old label/history columns from isolated copy: ", paste(drop_cols, collapse = ", "))
    base <- base %>% select(-all_of(drop_cols))
  }

  message("Loading timing labels...")
  labels <- readRDS(labels_path) %>%
    filter(variant_id == .env$variant_id) %>%
    mutate(
      GID_2 = as.character(GID_2),
      date = as.Date(date)
    ) %>%
    select(-variant_id, -outage_date)

  dup_labels <- labels %>%
    count(GID_2, date) %>%
    filter(n > 1L)
  if (nrow(dup_labels) > 0L) {
    stop("Duplicate labels for ", variant_id, ".")
  }

  message("Joining timing label-side columns into isolated feature copy...")
  variant_features <- base %>%
    left_join(labels, by = c("GID_2", "date")) %>%
    arrange(.row_id) %>%
    select(-.row_id) %>%
    fill_missing_label_side()

  if (nrow(variant_features) != base_n) {
    stop("Row count changed after label join: before=", base_n,
         ", after=", nrow(variant_features))
  }

  positives <- sum(variant_features$outage_3h_or_more == 1L, na.rm = TRUE)
  if (positives != nrow(labels)) {
    stop("Positive count mismatch after join: features=", positives,
         ", labels=", nrow(labels))
  }
  message("Positive rows: ", positives)

  message("Computing variant-specific strictly-past history features...")
  history_features <- compute_history_features(variant_features)
  run_leakage_spot_check(variant_features, history_features)

  existing_hist <- intersect(history_feature_cols, names(variant_features))
  if (length(existing_hist) > 0L) {
    variant_features <- variant_features %>% select(-all_of(existing_hist))
  }

  variant_features <- variant_features %>%
    left_join(history_features, by = c("GID_2", "date"))

  if (nrow(variant_features) != base_n) {
    stop("Row count changed after history join.")
  }

  dup_keys <- variant_features %>%
    count(GID_2, date) %>%
    filter(n > 1L) %>%
    nrow()
  if (dup_keys > 0L) stop("Duplicate GID_2/date keys in variant features.")

  qa <- tibble(
    variant_id = variant_id,
    rows = nrow(variant_features),
    cols = ncol(variant_features),
    positives = positives,
    prevalence = positives / nrow(variant_features),
    n_history_features = length(intersect(history_feature_cols, names(variant_features))),
    min_date = as.character(min(variant_features$date, na.rm = TRUE)),
    max_date = as.character(max(variant_features$date, na.rm = TRUE)),
    duplicate_keys = dup_keys,
    generated_on = as.character(Sys.time())
  )

  hist_manifest <- tibble(
    feature = history_feature_cols,
    description = c(
      "Sum of variant outage_3h_or_more over [t-30, t-1], per GID_2",
      "hist_outage_count_30d / 30",
      "Sum of variant outage_3h_or_more over [t-90, t-1], per GID_2",
      "hist_outage_count_90d / 90",
      "Sum of variant n_outages over [t-30, t-1], per GID_2",
      "Sum of variant n_outages_environmental over [t-30, t-1], per GID_2",
      "Sum of variant n_outages_technical over [t-30, t-1], per GID_2",
      "Sum of variant total_length_min over [t-90, t-1], per GID_2",
      "Median variant total_length_min on positive nights in [t-180, t-1], per GID_2",
      "Days since most recent variant outage_3h_or_more=1 strictly before t"
    ),
    generated_from_variant = variant_id,
    generated_on = as.character(Sys.time())
  )

  feature_manifest <- tibble(
    variant_id = variant_id,
    base_features_path = base_features_path,
    labels_path = labels_path,
    output_features_path = out_features_path,
    n_rows = nrow(variant_features),
    n_cols = ncol(variant_features),
    positives = positives,
    generated_on = as.character(Sys.time())
  )

  message("Saving variant feature file...")
  saveRDS(variant_features, out_features_path)
  write_csv(feature_manifest, out_manifest_path)
  write_csv(hist_manifest, out_history_manifest_path)
  write_csv(qa, out_qa_path)

  message("Wrote: ", out_features_path)
  rm(base, labels, variant_features, history_features)
  gc()
}

message("\nDone.")
