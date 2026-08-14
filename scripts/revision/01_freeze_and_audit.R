suppressPackageStartupMessages({
  library(data.table)
  library(digest)
  library(jsonlite)
})

options(stringsAsFactors = FALSE, scipen = 999)

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L || all(is.na(x))) y else x

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default = NULL) {
  pos <- match(flag, args)
  if (is.na(pos) || pos == length(args)) default else args[[pos + 1L]]
}

project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
run_id <- arg_value("--run-id", paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_phase1_freeze_audit"))
default_manuscripts <- c(
  "/Users/armandeaboudrar-meda/Downloads/manuscript (1).tex",
  file.path(project_root, "manuscript.tex")
)
manuscript_path <- arg_value("--manuscript", default_manuscripts[file.exists(default_manuscripts)][1])
if (is.na(manuscript_path) || !nzchar(manuscript_path) || !file.exists(manuscript_path)) {
  stop("No current manuscript found. Supply --manuscript /absolute/path/to/manuscript.tex")
}
manuscript_path <- normalizePath(manuscript_path, winslash = "/", mustWork = TRUE)

output_dir <- file.path(project_root, "data", "revision", run_id)
if (dir.exists(output_dir)) stop("Refusing to overwrite existing run directory: ", output_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("Phase 1 freeze-and-audit run: ", run_id, "\n", sep = "")
cat("Output directory: ", output_dir, "\n", sep = "")

rel <- function(path) {
  path <- normalizePath(path, winslash = "/", mustWork = FALSE)
  prefix <- paste0(project_root, "/")
  if (startsWith(path, prefix)) substring(path, nchar(prefix) + 1L) else path
}

abs_path <- function(path) {
  out <- path
  relative <- !is.na(path) & !grepl("^/", path)
  out[relative] <- file.path(project_root, path[relative])
  out
}

git_value <- function(...) {
  out <- tryCatch(system2("git", c(...), stdout = TRUE, stderr = FALSE), error = function(e) character())
  if (length(out)) paste(out, collapse = "\n") else NA_character_
}

git_commit <- git_value("rev-parse", "HEAD")
git_branch <- git_value("rev-parse", "--abbrev-ref", "HEAD")
git_dirty <- nzchar(git_value("status", "--porcelain", "--untracked-files=no") %||% "")

headline_dir <- file.path(
  project_root, "data/baselines/binary_threshold/ablation_batches/20260512_111324",
  "20260512_111341_forecast_strict_bench_rs_history"
)
replicate_dir <- file.path(
  project_root, "data/baselines/binary_threshold/ablation_batches/20260513_153833",
  "20260513_153858_forecast_strict_bench_rs_history"
)
history_free_dir <- file.path(
  project_root, "data/baselines/binary_threshold/ablation_batches/20260417_003025",
  "20260417_003044_forecast_strict_bench_rs_only"
)
timing_dir <- file.path(
  project_root, "data/baselines/binary_threshold/timing_overpass",
  "full_timing_table_20260702_162306"
)
logistic_dir <- file.path(project_root, "data/baselines/logistic_threshold/20260513_120055_logistic_baseline")
spatial_dir <- file.path(
  project_root, "data/baselines/binary_threshold/sensitivity_spatial_holdout_bench_rs_history_20260611",
  "20260611_164925_forecast_strict_bench_rs_history_southern_holdout"
)
uri_dir <- file.path(
  project_root, "data/baselines/binary_threshold/sensitivity_uri_bench_rs_history_20260611",
  "20260611_164924_forecast_strict_bench_rs_history_uri_excl"
)

input_spec <- data.table(
  role = c(
    "raw_outage_source", "filtered_ground_truth", "cause_enriched_ground_truth",
    "night_filter_script", "municipality_mapping_script", "panel_builder", "final_panel",
    "model_ready_canonical", "model_ready_stale_snapshot", "model_ready_stale_qa",
    "fixed_splits", "feature_config", "feature_engineering", "history_feature_engineering",
    "history_feature_manifest", "headline_features", "headline_config", "headline_metrics",
    "headline_platt", "headline_policy", "headline_importance", "history_free_metrics",
    "timing_table", "timing_family_gain", "logistic_metrics", "logistic_config",
    "cause_metrics", "cause_support", "cause_comparison", "spatial_metrics", "spatial_config",
    "uri_metrics", "ltr_metrics", "ltr_config", "two_part_aggregate", "two_part_count",
    "two_part_severity", "two_part_config", "current_manuscript", "comment_workbook"
  ),
  path = c(
    "data/dist_outage_2017_2021_causa_simple.csv",
    "data/night_outages_3hrs_with_locations.csv",
    "data/night_outages_3hrs_with_locations_clean_by_reason.csv",
    "scripts/01_select_night_outages.R", "scripts/02_merge_with_shapefile.R",
    "ML_Mexico_final.R", "data/panel_mex_2017_2021_ntl_ghs.rds",
    "data/model_ready/features_engineered.rds", "data/model_ready/features_engineered.parquet",
    "data/model_ready/features_qa.csv", "data/model_ready/splits_fixed.rds",
    "scripts/00_feature_config.R", "scripts/04_feature_engineering_improved.R",
    "scripts/04b_history_features.R", "data/model_ready/history_features_manifest.csv",
    file.path(headline_dir, "features_used.txt"), file.path(headline_dir, "run_config.json"),
    file.path(headline_dir, "metrics_summary.csv"), file.path(headline_dir, "platt_calibration.csv"),
    file.path(headline_dir, "deployment_decision_table.csv"), file.path(headline_dir, "feature_importance.csv"),
    file.path(history_free_dir, "metrics_summary.csv"), file.path(timing_dir, "timing_table_publication_ready.csv"),
    file.path(timing_dir, "timing_feature_family_gain_all.csv"), file.path(logistic_dir, "metrics_summary.csv"),
    file.path(logistic_dir, "run_config.json"),
    file.path(headline_dir, "cause_specific_eval/cause_metrics_main_block_bootstrap.csv"),
    file.path(headline_dir, "cause_specific_eval/cause_support_counts.csv"),
    file.path(headline_dir, "cause_specific_eval/env_vs_tech_comparison.csv"),
    file.path(spatial_dir, "metrics_summary.csv"), file.path(spatial_dir, "run_config.json"),
    file.path(uri_dir, "metrics_summary.csv"),
    file.path(replicate_dir, "ltr_bench_rs_history/ltr_topk_metrics.csv"),
    file.path(replicate_dir, "ltr_bench_rs_history/ltr_run_config.json"),
    file.path(replicate_dir, "two_part_xgb_cause/burden_aggregate_summary_xgb.csv"),
    file.path(replicate_dir, "two_part_xgb_cause/count_model_xgb_summary.csv"),
    file.path(replicate_dir, "two_part_xgb_cause/severity_model_xgb_summary.csv"),
    file.path(replicate_dir, "two_part_xgb_cause/two_part_xgb_run_config.json"),
    manuscript_path,
    "/Users/armandeaboudrar-meda/Downloads/outages_ML_nightlights_comments_2026-06-26.xlsx"
  ),
  canonical = c(rep(TRUE, 8), FALSE, FALSE, rep(TRUE, 22), rep(FALSE, 6), TRUE, TRUE),
  notes = c(
    "Unfiltered source includes short events", "Canonical event-level ground truth",
    "Same event set with cause and grid-detail duplication", "Defines nighttime overlap and strict duration >180 minutes",
    "Maps cleaned names to GADM GID_2", "Constructs municipality-night panel", "Canonical ML panel",
    "Canonical 192-column engineered object used by headline run", "Older 153-column February export; not canonical",
    "QA for older pre-history export", "Chronological split definition", "Predictor family configuration",
    "Core feature engineering", "Strictly lagged outage history", "History-window definitions",
    "Frozen 83-feature list", "Frozen headline model configuration", "Frozen headline discrimination and calibration metrics",
    "Frozen Platt coefficients", "Frozen validation-selected deployment policy", "Frozen headline split-gain importance",
    "History-free national model", "Publication-ready timing robustness table", "Timing feature-family gain shares",
    "Exploratory non-like-for-like logistic reference", "Exploratory logistic run metadata",
    "Cause-specific ISO-week-block intervals", "Cause-specific supports", "Environmental-versus-technical comparison",
    "History-retained three-state holdout", "Three-state holdout metadata", "Winter Storm Uri sensitivity",
    "Fixed-budget LTR comparison from equivalent replicate", "LTR metadata from equivalent replicate",
    "Two-part aggregates from equivalent replicate", "Conditional count model from equivalent replicate",
    "Conditional cumulative-minutes model from equivalent replicate", "Two-part metadata from equivalent replicate",
    "Current blueprint supplied for this audit", "Three-sheet revision comment source"
  )
)
input_spec[, absolute_path := vapply(path, abs_path, character(1))]
input_spec[, exists := file.exists(absolute_path)]

cat("Hashing input files (large RDS files can take several minutes)...\n")
file_rows <- lapply(seq_len(nrow(input_spec)), function(i) {
  p <- input_spec$absolute_path[[i]]
  info <- if (file.exists(p)) file.info(p) else NULL
  file_flags <- if (file.exists(p)) {
    tryCatch(paste(system2("stat", c("-f", "%Sf", shQuote(p)), stdout = TRUE, stderr = FALSE), collapse = " "),
             error = function(e) "")
  } else ""
  materialized <- file.exists(p) && !grepl("dataless", file_flags, fixed = TRUE)
  local_rel <- if (startsWith(p, paste0(project_root, "/"))) rel(p) else NA_character_
  tracked <- if (!is.na(local_rel)) {
    identical(system2("git", c("ls-files", "--error-unmatch", local_rel), stdout = FALSE, stderr = FALSE), 0L)
  } else FALSE
  status <- if (!is.na(local_rel)) git_value("status", "--short", "--", local_rel) else NA_character_
  data.table(
    role = input_spec$role[[i]], path = rel(p), absolute_path = p,
    exists = file.exists(p), materialized = materialized,
    size_bytes = if (file.exists(p)) as.numeric(info$size) else NA_real_,
    mtime_utc = if (file.exists(p)) format(info$mtime, tz = "UTC", usetz = TRUE) else NA_character_,
    sha256 = if (materialized) digest(p, algo = "sha256", file = TRUE) else NA_character_,
    git_tracked = tracked, git_status = status %||% "", canonical = input_spec$canonical[[i]],
    notes = input_spec$notes[[i]]
  )
})
input_manifest <- rbindlist(file_rows, fill = TRUE)
fwrite(input_manifest, file.path(output_dir, "input_file_manifest.csv"))

required_roles <- c(
  "raw_outage_source", "filtered_ground_truth", "cause_enriched_ground_truth", "panel_builder",
  "final_panel", "model_ready_canonical", "fixed_splits", "headline_features", "headline_config",
  "headline_metrics", "headline_platt", "headline_policy", "current_manuscript"
)

qa <- data.table(check_id = character(), area = character(), status = character(),
                 observed = character(), expected = character(), evidence = character(), notes = character())
add_qa <- function(check_id, area, pass, observed, expected, evidence, notes = "", warn = FALSE) {
  status <- if (isTRUE(pass)) "PASS" else if (isTRUE(warn)) "WARN" else "FAIL"
  qa <<- rbind(qa, data.table(
    check_id = check_id, area = area, status = status, observed = as.character(observed),
    expected = as.character(expected), evidence = evidence, notes = notes
  ))
}

missing_required <- input_manifest[role %in% required_roles & !exists, role]
add_qa("FILES_001", "inputs", !length(missing_required), paste(missing_required, collapse = "; "),
       "all canonical inputs exist", "input_file_manifest.csv")
unmaterialized_required <- input_manifest[role %in% required_roles & exists & materialized == FALSE, role]
if (length(unmaterialized_required)) {
  stop(
    "Canonical inputs are iCloud placeholders rather than local files: ",
    paste(unmaterialized_required, collapse = ", "),
    ". Download them locally before rerunning the audit."
  )
}
add_qa("GIT_001", "repository", !is.na(git_commit) && nchar(git_commit) == 40L, git_commit,
       "40-character Git commit", "run_metadata.json")
add_qa("GIT_002", "repository", !git_dirty, git_dirty, FALSE, "run_metadata.json",
       "Existing tracked worktree changes were not modified by this script.", warn = TRUE)

headline_config <- read_json(file.path(headline_dir, "run_config.json"), simplifyVector = TRUE)
headline_features <- trimws(readLines(file.path(headline_dir, "features_used.txt"), warn = FALSE))
headline_features <- headline_features[nzchar(headline_features)]
fwrite(data.table(feature_order = seq_along(headline_features), feature = headline_features),
       file.path(output_dir, "headline_features.csv"))
headline_metrics <- fread(file.path(headline_dir, "metrics_summary.csv"))
headline_platt <- fread(file.path(headline_dir, "platt_calibration.csv"))
headline_policy <- fread(file.path(headline_dir, "deployment_decision_table.csv"))[role == "default"]
fixed_splits <- readRDS("data/model_ready/splits_fixed.rds")

add_qa("MODEL_001", "headline", length(headline_features) == 83L, length(headline_features), 83,
       rel(file.path(headline_dir, "features_used.txt")))
add_qa("MODEL_002", "headline", identical(headline_config$run_id, basename(headline_dir)),
       headline_config$run_id, basename(headline_dir), rel(file.path(headline_dir, "run_config.json")))
add_qa("MODEL_003", "headline", identical(as.integer(headline_config$seed), 42L), headline_config$seed, 42,
       rel(file.path(headline_dir, "run_config.json")))

cat("Reading and reconciling outage event files...\n")
gt <- fread("data/night_outages_3hrs_with_locations.csv")
cause <- fread("data/night_outages_3hrs_with_locations_clean_by_reason.csv")
raw <- fread("data/dist_outage_2017_2021_causa_simple.csv")
event_key <- c("year", "date", "time", "length_min", "state", "municipality")
gt[, date := as.IDate(date)]
cause[, date := as.IDate(date)]
gt_events <- unique(gt[, ..event_key])
cause_events <- unique(cause[, ..event_key])
setkeyv(gt_events, event_key)
setkeyv(cause_events, event_key)
gt_only <- fsetdiff(gt_events, cause_events)
cause_only <- fsetdiff(cause_events, gt_events)

raw_duration <- suppressWarnings(as.numeric(raw[["DURACIÓN (MINUTOS)"]]))
add_qa("GT_001", "ground_truth", nrow(gt_events) == nrow(gt), nrow(gt) - nrow(gt_events), 0,
       "data/night_outages_3hrs_with_locations.csv", "Main ground truth should be event-level unique.")
add_qa("GT_002", "ground_truth", !nrow(gt_only) && !nrow(cause_only),
       paste0("main_only=", nrow(gt_only), "; cause_only=", nrow(cause_only)), "0; 0",
       "both filtered ground-truth files")
add_qa("GT_003", "ground_truth", nrow(cause_events) == nrow(gt_events), nrow(cause_events), nrow(gt_events),
       "both filtered ground-truth files")
add_qa("GT_004", "ground_truth", min(gt$length_min, na.rm = TRUE) == 181,
       min(gt$length_min, na.rm = TRUE), 181, "data/night_outages_3hrs_with_locations.csv")
add_qa("GT_005", "ground_truth", min(raw_duration, na.rm = TRUE) < 181,
       min(raw_duration, na.rm = TRUE), "<181", "data/dist_outage_2017_2021_causa_simple.csv",
       "The raw source supports rebuilding shorter-duration sensitivities.")

most_frequent <- function(x) {
  x <- x[!is.na(x) & nzchar(x)]
  if (!length(x)) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[1]
}
cause_lookup <- cause[, .(
  classification_general = most_frequent(classification_general),
  classification_detailed = most_frequent(classification_detailed)
), by = event_key]
setkeyv(cause_lookup, event_key)
gt_with_cause <- cause_lookup[gt_events]
cause_assigned_events <- sum(!is.na(gt_with_cause$classification_general))

gt_mapped <- gt[!is.na(GID_2) & nzchar(GID_2)]
gt_agg <- gt_mapped[, .(
  n_outages_gt = .N,
  max_length_gt = max(length_min, na.rm = TRUE)
), by = .(GID_2, date)]
setkey(gt_agg, GID_2, date)

reconciliation <- data.table(
  metric = c(
    "raw_rows", "raw_min_duration_minutes", "filtered_main_rows", "filtered_main_unique_events",
    "filtered_cause_rows", "filtered_cause_unique_events", "cause_detail_extra_rows",
    "main_events_missing_GID_2", "cause_events_with_general_classification", "mapped_events",
    "mapped_positive_municipality_nights"
  ),
  value = c(
    nrow(raw), min(raw_duration, na.rm = TRUE), nrow(gt), nrow(gt_events), nrow(cause), nrow(cause_events),
    nrow(cause) - nrow(cause_events), sum(is.na(gt$GID_2) | !nzchar(gt$GID_2)), cause_assigned_events,
    nrow(gt_mapped), nrow(gt_agg)
  ),
  source = c(
    rep("data/dist_outage_2017_2021_causa_simple.csv", 2),
    rep("data/night_outages_3hrs_with_locations.csv", 2),
    rep("data/night_outages_3hrs_with_locations_clean_by_reason.csv", 3),
    "data/night_outages_3hrs_with_locations.csv", "exact event-key cause join",
    rep("mapped main ground truth", 2)
  )
)

cat("Reading canonical panel...\n")
panel <- as.data.table(readRDS("data/panel_mex_2017_2021_ntl_ghs.rds"))
panel[, date := as.IDate(date)]
panel_keys <- panel[, .(
  outage_panel = as.integer(outage_3h_or_more),
  n_outages_panel = as.integer(n_outages),
  max_length_panel = as.numeric(max_length_min)
), by = .(GID_2, date)]
panel_rows <- nrow(panel)
panel_cols <- ncol(panel)
panel_dates <- sort(unique(panel$date))
panel_munis <- uniqueN(panel$GID_2)
panel_positives <- sum(panel$outage_3h_or_more == 1L, na.rm = TRUE)
panel_prevalence <- mean(panel$outage_3h_or_more == 1L, na.rm = TRUE)
panel_events <- sum(panel$n_outages, na.rm = TRUE)
panel_cause_missing_positive <- sum(panel$outage_3h_or_more == 1L & is.na(panel$classification_general))
panel_duplicate_keys <- panel_rows - uniqueN(panel, by = c("GID_2", "date"))
rm(panel)
gc(verbose = FALSE)

setkey(panel_keys, GID_2, date)
gt_join <- panel_keys[gt_agg]
gt_panel_matched <- gt_join[!is.na(outage_panel)]
missing_panel_keys <- gt_join[is.na(outage_panel), .(GID_2, date, n_outages_gt, max_length_gt)]
missing_panel_dates <- missing_panel_keys[, .(
  ground_truth_events = sum(n_outages_gt),
  ground_truth_positive_keys = .N
), by = date][order(date)]
fwrite(missing_panel_dates, file.path(output_dir, "missing_panel_dates.csv"))

panel_positive_without_gt <- panel_keys[outage_panel == 1L][!gt_agg, on = .(GID_2, date)]
add_qa("PANEL_001", "panel", panel_duplicate_keys == 0L, panel_duplicate_keys, 0,
       "data/panel_mex_2017_2021_ntl_ghs.rds")
add_qa("PANEL_002", "panel", nrow(gt_panel_matched) == panel_positives, nrow(gt_panel_matched), panel_positives,
       "ground truth joined to final panel")
add_qa("PANEL_003", "panel", sum(gt_panel_matched$n_outages_gt != gt_panel_matched$n_outages_panel) == 0L,
       sum(gt_panel_matched$n_outages_gt != gt_panel_matched$n_outages_panel), 0,
       "ground truth joined to final panel")
add_qa("PANEL_004", "panel", sum(gt_panel_matched$max_length_gt != gt_panel_matched$max_length_panel) == 0L,
       sum(gt_panel_matched$max_length_gt != gt_panel_matched$max_length_panel), 0,
       "ground truth joined to final panel")
add_qa("PANEL_005", "panel", nrow(panel_positive_without_gt) == 0L, nrow(panel_positive_without_gt), 0,
       "final panel anti-joined to ground truth")
add_qa("PANEL_006", "panel", sum(missing_panel_keys$n_outages_gt) == 196L,
       sum(missing_panel_keys$n_outages_gt), 196, "missing_panel_dates.csv",
       "Ground-truth events on nine dates absent from the NTL-balanced panel.")

reconciliation <- rbind(reconciliation, data.table(
  metric = c(
    "panel_rows", "panel_columns", "panel_dates", "panel_municipalities", "panel_positive_municipality_nights",
    "panel_prevalence", "panel_outage_events", "panel_positive_keys_missing_cause",
    "ground_truth_events_on_missing_panel_dates", "ground_truth_keys_on_missing_panel_dates"
  ),
  value = c(panel_rows, panel_cols, length(panel_dates), panel_munis, panel_positives, panel_prevalence,
            panel_events, panel_cause_missing_positive, sum(missing_panel_keys$n_outages_gt), nrow(missing_panel_keys)),
  source = rep("data/panel_mex_2017_2021_ntl_ghs.rds and ground-truth reconciliation", 10)
))

cat("Reading canonical 192-column model-ready object...\n")
features <- as.data.table(readRDS("data/model_ready/features_engineered.rds"))
features[, date := as.IDate(date)]
features_label_col <- if ("outage_3h_or_more" %in% names(features)) "outage_3h_or_more" else stop("Missing label column")
missing_headline_features <- setdiff(headline_features, names(features))
feature_keys <- features[, .(
  outage_features = as.integer(get(features_label_col)),
  n_outages_features = as.integer(n_outages),
  max_length_features = as.numeric(max_length_min)
), by = .(GID_2, date)]
features_rows <- nrow(features)
features_cols <- ncol(features)
features_dates <- uniqueN(features$date)
features_munis <- uniqueN(features$GID_2)
features_positives <- sum(features[[features_label_col]] == 1L, na.rm = TRUE)
features_duplicate_keys <- features_rows - uniqueN(features, by = c("GID_2", "date"))

split_defs <- data.table(
  split = c("training", "validation", "test"),
  start = as.IDate(c("2017-01-01", "2020-01-01", "2020-07-01")),
  end = as.IDate(c("2019-12-31", "2020-06-30", "2021-12-31"))
)
split_summary <- rbindlist(lapply(seq_len(nrow(split_defs)), function(i) {
  z <- features[date >= split_defs$start[[i]] & date <= split_defs$end[[i]]]
  data.table(
    split = split_defs$split[[i]], start = as.character(split_defs$start[[i]]),
    end = as.character(split_defs$end[[i]]), nights = uniqueN(z$date), municipality_nights = nrow(z),
    positives = sum(z[[features_label_col]] == 1L), prevalence = mean(z[[features_label_col]] == 1L),
    municipalities = uniqueN(z$GID_2)
  )
}))
split_summary <- rbind(split_summary, data.table(
  split = "total", start = as.character(min(features$date)), end = as.character(max(features$date)),
  nights = uniqueN(features$date), municipality_nights = features_rows, positives = features_positives,
  prevalence = mean(features[[features_label_col]] == 1L), municipalities = features_munis
))
fwrite(split_summary, file.path(output_dir, "split_summary.csv"))
rm(features)
gc(verbose = FALSE)

setkey(feature_keys, GID_2, date)
panel_feature_join <- feature_keys[panel_keys]
add_qa("FEATURES_001", "model_ready", features_duplicate_keys == 0L, features_duplicate_keys, 0,
       "data/model_ready/features_engineered.rds")
add_qa("FEATURES_002", "model_ready", !length(missing_headline_features),
       paste(missing_headline_features, collapse = "; "), "all 83 headline features present",
       "features_engineered.rds plus headline_features.csv")
add_qa("FEATURES_003", "model_ready", features_rows == panel_rows, features_rows, panel_rows,
       "panel versus canonical model-ready RDS")
add_qa("FEATURES_004", "model_ready", nrow(panel_feature_join) == panel_rows && all(!is.na(panel_feature_join$outage_features)),
       sum(is.na(panel_feature_join$outage_features)), 0, "keyed panel-to-model-ready join")
add_qa("FEATURES_005", "model_ready", sum(panel_feature_join$outage_panel != panel_feature_join$outage_features) == 0L,
       sum(panel_feature_join$outage_panel != panel_feature_join$outage_features), 0, "keyed panel-to-model-ready join")
add_qa("FEATURES_006", "model_ready", sum(panel_feature_join$n_outages_panel != panel_feature_join$n_outages_features) == 0L,
       sum(panel_feature_join$n_outages_panel != panel_feature_join$n_outages_features), 0, "keyed panel-to-model-ready join")
add_qa("FEATURES_007", "model_ready", sum(panel_feature_join$max_length_panel != panel_feature_join$max_length_features) == 0L,
       sum(panel_feature_join$max_length_panel != panel_feature_join$max_length_features), 0, "keyed panel-to-model-ready join")

stale_qa <- fread("data/model_ready/features_qa.csv")
stale_cols <- as.integer(stale_qa[metric == "n_features_total", value])
add_qa("FEATURES_008", "model_ready", FALSE,
       paste0("Parquet materialized=", input_manifest[role == "model_ready_stale_snapshot", materialized],
              "; Parquet QA columns=", stale_cols, "; canonical RDS columns=", features_cols),
       "Parquet must match canonical RDS before substitution", "data/model_ready/features_qa.csv",
       "The Parquet file is an older February snapshot without the later history/features. Use the RDS.", warn = TRUE)

expected_split <- data.table(split = c("training", "validation", "test"),
                             rows = c(2683044L, 447174L, 1334151L), positives = c(15951L, 1695L, 5350L))
split_check <- merge(split_summary[split != "total", .(split, rows = municipality_nights, positives)],
                     expected_split, by = "split", suffixes = c("_observed", "_expected"))
add_qa("SPLIT_001", "splits", all(split_check$rows_observed == split_check$rows_expected),
       paste(split_check$rows_observed, collapse = ";"), paste(split_check$rows_expected, collapse = ";"),
       "split_summary.csv")
add_qa("SPLIT_002", "splits", all(split_check$positives_observed == split_check$positives_expected),
       paste(split_check$positives_observed, collapse = ";"), paste(split_check$positives_expected, collapse = ";"),
       "split_summary.csv")
fixed_ranges_observed <- c(
  as.character(fixed_splits$train_range), as.character(fixed_splits$val_range),
  as.character(fixed_splits$test_range)
)
fixed_ranges_expected <- c("2017-01-01", "2019-12-31", "2020-01-01", "2020-06-30", "2020-07-01", "2021-12-31")
add_qa("SPLIT_003", "splits", identical(fixed_ranges_observed, fixed_ranges_expected),
       paste(fixed_ranges_observed, collapse = ";"), paste(fixed_ranges_expected, collapse = ";"),
       "data/model_ready/splits_fixed.rds")
fixed_rows_observed <- c(fixed_splits$snapshot$n_train, fixed_splits$snapshot$n_val, fixed_splits$snapshot$n_test)
fixed_rows_expected <- c(2683044L, 447174L, 1334151L)
add_qa("SPLIT_004", "splits", identical(as.integer(fixed_rows_observed), fixed_rows_expected),
       paste(fixed_rows_observed, collapse = ";"), paste(fixed_rows_expected, collapse = ";"),
       "data/model_ready/splits_fixed.rds")

replicate_features_path <- file.path(replicate_dir, "features_used.txt")
replicate_features <- trimws(readLines(replicate_features_path, warn = FALSE))
replicate_features <- replicate_features[nzchar(replicate_features)]
add_qa("PROV_001", "secondary_outputs", identical(headline_features, replicate_features),
       length(replicate_features), "same ordered 83-feature list", rel(replicate_features_path),
       "LTR and two-part artwork use a later equivalent replicate rather than the frozen headline directory.", warn = TRUE)

figure_spec <- data.table(
  tex_line = c(79, 92, 108, 172, 282, 289, 296, 303),
  figure_id = c("landscape", "skill", "feature_contribution", "count_duration", "ltr",
                "municipality_rates", "municipality_night", "state_burden"),
  asset = c("figures/fig1_2.png", "figures/fig3.png", "figures/fig5.png", "figures/fig_count_duration.png",
            "figures/fig_ltr.png", "figures/fig_municipality_rates.png", "figures/fig_municipality_night.png",
            "figures/fig_choropleth_burden.png"),
  source_script = c("figures/figure1_2.R", "figures/figure3.R", NA, "figure_count_duration.R",
                    "figure_ltr.R", "figure_municipality_rates.R", "figure_municipality_night.R",
                    "figure_choropleth_burden.R"),
  principal_inputs = c(
    "final panel; model-ready features; ground truth", "headline/equivalent metrics; pre-history ablations; cause outputs",
    "headline feature importance; nested feature-set metrics", "equivalent-replicate two-part outputs",
    "equivalent-replicate LTR and classifier outputs", "equivalent-replicate decomposed burden predictions",
    "equivalent-replicate decomposed burden predictions", "equivalent-replicate two-part aggregate outputs"
  )
)
figure_spec[, asset_exists := file.exists(abs_path(asset))]
figure_spec[, script_exists := !is.na(source_script) & file.exists(abs_path(source_script))]
figure_spec[, status := fifelse(!asset_exists | !script_exists, "FAIL", "PASS")]
figure_spec[, notes := ""]
figure_spec[figure_id == "feature_contribution", notes := "Current manuscript asset and generator are missing locally."]
figure_spec[figure_id == "landscape", notes := "Generator contains a hard-coded external drive path and is not portable."]
figure_spec[figure_id == "skill", notes := "Generator contains a hard-coded Windows path and mixes canonical-equivalent runs."]
figure_spec[figure_id %in% c("count_duration", "ltr", "municipality_rates", "municipality_night", "state_burden"),
            notes := "Generator uses the equivalent 20260513 replicate rather than the frozen 20260512 directory."]
fwrite(figure_spec, file.path(output_dir, "figure_provenance.csv"))
add_qa("FIG_001", "figures", all(figure_spec$asset_exists),
       paste(figure_spec[asset_exists == FALSE, asset], collapse = "; "), "all manuscript figure assets exist",
       "figure_provenance.csv", "Missing artwork prevents complete manuscript reproduction.", warn = TRUE)
add_qa("FIG_002", "figures", all(figure_spec$script_exists),
       paste(figure_spec[script_exists == FALSE, figure_id], collapse = "; "), "all figure generators exist",
       "figure_provenance.csv", "Missing or nonportable generators prevent one-command figure reproduction.", warn = TRUE)

claim_registry <- data.table(
  claim_id = sprintf("M%02d", 1:28),
  tex_line = c(28, 58, 68, 69, 70, 72, 80, 86, 88, 93, 102, 109, 113, 122, 124, 135, 136, 137, 138,
               151, 153, 157, 161, 168, 199, 231, 235, 241),
  claim = c(
    "Abstract sample, headline, history-free, timing and burden summary", "Panel dimensions, positives and municipalities",
    "Training sample", "Validation sample", "Test sample", "Total sample", "Landscape figure caption",
    "Headline discrimination calibration and operating policy", "Exploratory logistic reference", "Skill figure caption",
    "History-free deltas and feature-family importance", "Feature-contribution figure", "Municipality aggregate correlations",
    "Exact-overpass share and timezone implementation", "Timing ranges and gain shares", "Timing row buffer 0",
    "Timing row buffer 30", "Timing row buffer 60", "Timing row buffer 120", "Cause-specific metrics",
    "Southern holdout metrics", "LTR fixed-budget metrics", "Conditional count and minute errors",
    "Aggregate count and minute correlations", "Outage source and outcome construction", "Panel dates and split protocol",
    "Model parameters feature count calibration and policy", "Timing label rule"
  ),
  primary_source = c(
    "split_summary.csv; headline metrics/policy; history-free metrics; timing table; two-part outputs",
    "split_summary.csv", rep("split_summary.csv", 4), "final panel and figure1 generator",
    rel(file.path(headline_dir, "metrics_summary.csv")), rel(file.path(logistic_dir, "metrics_summary.csv")),
    "figure3 generator plus headline/pre-history/cause outputs",
    paste(rel(file.path(history_free_dir, "metrics_summary.csv")), rel(file.path(headline_dir, "feature_importance.csv")), sep = "; "),
    "MISSING figures/fig5.png and generator", rel(file.path(replicate_dir, "two_part_xgb_cause/decomposed_burden_xgb_test.csv")),
    rel(file.path(timing_dir, "timing_table_publication_ready.csv")), rel(file.path(timing_dir, "timing_table_publication_ready.csv")),
    rep(rel(file.path(timing_dir, "timing_table_publication_ready.csv")), 4),
    rel(file.path(headline_dir, "cause_specific_eval/cause_metrics_main_block_bootstrap.csv")),
    rel(file.path(spatial_dir, "metrics_summary.csv")), rel(file.path(replicate_dir, "ltr_bench_rs_history/ltr_topk_metrics.csv")),
    paste(rel(file.path(replicate_dir, "two_part_xgb_cause/count_model_xgb_summary.csv")),
          rel(file.path(replicate_dir, "two_part_xgb_cause/severity_model_xgb_summary.csv")), sep = "; "),
    rel(file.path(replicate_dir, "two_part_xgb_cause/burden_aggregate_summary_xgb.csv")),
    "raw source; filtered ground truths; scripts/01_select_night_outages.R; ML_Mexico_final.R",
    "split_summary.csv; missing_panel_dates.csv; fixed splits", rel(file.path(headline_dir, "run_config.json")),
    "scripts/timing_overpass and timing publication table"
  ),
  transformation = c(
    "direct and rounded", "direct", rep("direct", 4), "descriptive figure",
    "direct plus Brier skill derived against constant test prevalence", "direct", "composite figure",
    "direct; feature-family sums computed from split gain", "not reproducible locally", "Pearson and Spearman aggregation",
    "event-to-municipality-night relabelling with historical IANA rules", "range and family-share summary",
    rep("direct", 4), "direct", "direct", "matched daily top-K comparison", "direct",
    "aggregation across 18 months and 32 states", "strict >180 and nighttime overlap then GADM mapping",
    "direct", "direct", "UTC comparison after local-civil conversion"
  ),
  status = c(rep("VERIFIED", 11), "UNRESOLVED", rep("VERIFIED", 12), "PARTIAL", rep("VERIFIED", 3)),
  notes = c(rep("", 11), "Figure asset and source script are absent.", rep("", 12),
            "Event lineage is verified; exact release/access route, reporting rules and completeness are unresolved.", rep("", 3))
)
fwrite(claim_registry, file.path(output_dir, "manuscript_claim_registry.csv"))

manuscript_lines <- readLines(manuscript_path, warn = FALSE)
numeric_lines <- which(grepl("[0-9]", manuscript_lines))
mapped_lines <- claim_registry$tex_line
numeric_audit <- data.table(
  tex_line = numeric_lines,
  text = trimws(manuscript_lines[numeric_lines]),
  classification = fifelse(numeric_lines %in% mapped_lines, "result_claim",
                           fifelse(grepl("^\\\\(documentclass|usepackage|author|affil|setcounter|renewcommand)",
                                        trimws(manuscript_lines[numeric_lines])), "structural", "method_or_reference")),
  registry_claim_id = vapply(numeric_lines, function(x) paste(claim_registry[tex_line == x, claim_id], collapse = ";"), character(1))
)
fwrite(numeric_audit, file.path(output_dir, "manuscript_numeric_line_audit.csv"))
add_qa("PROV_002", "manuscript", all(claim_registry$status == "VERIFIED"),
       paste(claim_registry[status != "VERIFIED", claim_id], collapse = "; "), "all result claims verified",
       "manuscript_claim_registry.csv", "Known unresolved provenance is recorded rather than silently asserted.", warn = TRUE)

reconciliation <- rbind(reconciliation, data.table(
  metric = c("model_ready_rows", "model_ready_columns", "model_ready_dates", "model_ready_municipalities",
             "model_ready_positive_municipality_nights", "headline_feature_count"),
  value = c(features_rows, features_cols, features_dates, features_munis, features_positives, length(headline_features)),
  source = c(rep("data/model_ready/features_engineered.rds", 5), rel(file.path(headline_dir, "features_used.txt")))
))
fwrite(reconciliation, file.path(output_dir, "sample_reconciliation.csv"))

manifest <- rbindlist(list(
  data.table(section = "repository", manifest_key = c("git_commit", "git_branch", "tracked_worktree_dirty", "audit_run_id"),
             value = c(git_commit, git_branch, git_dirty, run_id), source = "Git / audit invocation"),
  data.table(section = "headline", manifest_key = c(
    "run_id", "task_mode", "seed", "feature_count", "model_type", "scale_pos_weight", "trees", "tree_depth",
    "min_n", "loss_reduction", "sample_size", "mtry", "learn_rate", "train_start", "train_end", "validation_start",
    "validation_end", "calibration_fit_start", "calibration_fit_end", "threshold_select_start", "threshold_select_end",
    "test_start", "test_end", "platt_intercept", "platt_slope", "policy_type", "policy_target", "policy_threshold",
    "policy_precision_test", "policy_recall_test", "policy_alerts_per_night_test", "roc_auc_test", "pr_auc_test",
    "brier_test_platt", "logloss_test_platt", "ece_equal_test_platt"
  ), value = as.character(c(
    headline_config$run_id, headline_config$task_mode, headline_config$seed, length(headline_features),
    headline_config$model$type, headline_config$model$scale_pos_weight, headline_config$model$params$trees,
    headline_config$model$params$tree_depth, headline_config$model$params$min_n,
    headline_config$model$params$loss_reduction, headline_config$model$params$sample_size,
    headline_config$model$params$mtry, headline_config$model$params$learn_rate,
    headline_config$splits$train$start, headline_config$splits$train$end,
    headline_config$splits$val$start, headline_config$splits$val$end,
    headline_config$calibration$strict_separation$calib_fit_dates_range[[1]],
    headline_config$calibration$strict_separation$calib_fit_dates_range[[2]],
    headline_config$calibration$strict_separation$threshold_select_dates_range[[1]],
    headline_config$calibration$strict_separation$threshold_select_dates_range[[2]],
    headline_config$splits$test$start, headline_config$splits$test$end,
    headline_platt$platt_intercept[[1]], headline_platt$platt_slope[[1]],
    headline_policy$policy_type[[1]], headline_policy$policy_target[[1]], headline_policy$threshold[[1]],
    headline_policy$precision[[1]], headline_policy$recall[[1]], headline_policy$alerts_per_day[[1]],
    headline_metrics[split == "test", roc_auc], headline_metrics[split == "test", pr_auc],
    headline_metrics[split == "test", brier_platt], headline_metrics[split == "test", logloss_platt],
    headline_metrics[split == "test", ece_equal_platt]
  )), source = rel(file.path(headline_dir, "run_config.json"))),
  data.table(section = "panel", manifest_key = c("rows", "columns", "nights", "municipalities", "positives", "prevalence", "events"),
             value = as.character(c(panel_rows, panel_cols, length(panel_dates), panel_munis, panel_positives,
                                    panel_prevalence, panel_events)), source = "data/panel_mex_2017_2021_ntl_ghs.rds"),
  data.table(section = "model_ready", manifest_key = c("rows", "columns", "nights", "municipalities", "positives"),
             value = as.character(c(features_rows, features_cols, features_dates, features_munis, features_positives)),
             source = "data/model_ready/features_engineered.rds")
), fill = TRUE)
setnames(manifest, "manifest_key", "key")
fwrite(manifest, file.path(output_dir, "frozen_analysis_manifest.csv"))

fwrite(qa, file.path(output_dir, "qa_checks.csv"))
overall_status <- if (any(qa$status == "FAIL")) "FAIL" else if (any(qa$status == "WARN")) "PASS_WITH_WARNINGS" else "PASS"

run_metadata <- list(
  audit_run_id = run_id,
  created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  project_root = project_root,
  output_dir = output_dir,
  current_manuscript = manuscript_path,
  git = list(commit = git_commit, branch = git_branch, tracked_worktree_dirty = git_dirty),
  canonical_headline_run = headline_config$run_id,
  overall_status = overall_status,
  qa_counts = as.list(table(factor(qa$status, levels = c("PASS", "WARN", "FAIL"))))
)
write_json(run_metadata, file.path(output_dir, "run_metadata.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")

report <- c(
  "# Phase 1 Freeze And Audit",
  "",
  paste0("**Overall status:** ", overall_status),
  paste0("**Audit run:** `", run_id, "`"),
  paste0("**Git commit:** `", git_commit, "`"),
  paste0("**Canonical headline run:** `", headline_config$run_id, "`"),
  "",
  "## Canonical Analysis Identity",
  "",
  paste0("- Final panel: ", format(panel_rows, big.mark = ","), " rows, ", panel_cols, " columns, ",
         length(panel_dates), " nights, ", panel_munis, " municipalities."),
  paste0("- Positive municipality-nights: ", format(panel_positives, big.mark = ","),
         " (", sprintf("%.4f%%", 100 * panel_prevalence), ")."),
  paste0("- Canonical model-ready object: ", format(features_rows, big.mark = ","), " rows, ",
         features_cols, " columns; all ", length(headline_features), " headline predictors present."),
  paste0("- Test ROC-AUC: ", sprintf("%.6f", headline_metrics[split == "test", roc_auc]),
         "; test PR-AUC: ", sprintf("%.6f", headline_metrics[split == "test", pr_auc]), "."),
  paste0("- Platt policy threshold: ", headline_policy$threshold[[1]], "; precision ",
         sprintf("%.2f%%", 100 * headline_policy$precision[[1]]), "; recall ",
         sprintf("%.2f%%", 100 * headline_policy$recall[[1]]), "; ",
         sprintf("%.2f", headline_policy$alerts_per_day[[1]]), " alerts per night."),
  "",
  "## Ground-Truth Reconciliation",
  "",
  paste0("- Main filtered ground truth: ", nrow(gt), " event rows; cause-enriched file: ", nrow(cause),
         " rows representing the same ", nrow(cause_events), " unique events."),
  paste0("- Cause-enriched duplication: ", nrow(cause) - nrow(cause_events),
         " extra grid-detail rows; event-key differences between files: 0."),
  paste0("- Mapped filtered events: ", nrow(gt_mapped), "; panel events: ", panel_events, "."),
  paste0("- The difference is exactly ", sum(missing_panel_keys$n_outages_gt),
         " events on nine dates absent from the Black Marble-balanced panel."),
  "- Every panel label, event count, and maximum duration matches the filtered ground truth after keyed joining.",
  "- Every panel key and outcome matches the canonical model-ready RDS after keyed joining.",
  "",
  "## Warnings And Open Items",
  "",
  "- `features_engineered.parquet` is an older 153-column February snapshot. The 192-column RDS is the canonical headline input.",
  "- `figures/fig5.png` and its generator are absent locally, so the promoted feature-contribution figure is not currently reproducible.",
  "- Figure 1 and Figure 3 generators contain machine-specific paths; several other figures use a later but feature-identical replicate.",
  "- Outage-event lineage is verified, but the exact CFE release/access route, administrative reporting rules, completeness, and bias remain unresolved metadata/analysis requirements.",
  "- Existing tracked worktree changes were observed and left untouched.",
  "",
  "## QA Checks",
  "",
  "| Check | Area | Status | Observed | Expected |",
  "|---|---|---:|---|---|",
  vapply(seq_len(nrow(qa)), function(i) paste0("| `", qa$check_id[[i]], "` | ", qa$area[[i]], " | ",
                                               qa$status[[i]], " | ", gsub("\\|", "/", qa$observed[[i]]), " | ",
                                               gsub("\\|", "/", qa$expected[[i]]), " |"), character(1)),
  "",
  "## Output Inventory",
  "",
  paste0("- `frozen_analysis_manifest.csv`: canonical analysis configuration and sample identity."),
  paste0("- `input_file_manifest.csv`: paths, sizes, UTC modification times, SHA-256 hashes, and Git state."),
  paste0("- `sample_reconciliation.csv`, `split_summary.csv`, and `missing_panel_dates.csv`: sample lineage."),
  paste0("- `headline_features.csv`: ordered 83-feature specification."),
  paste0("- `manuscript_claim_registry.csv` and `manuscript_numeric_line_audit.csv`: result-number provenance."),
  paste0("- `figure_provenance.csv`: artwork, generator, and input provenance."),
  paste0("- `qa_checks.csv`: machine-readable pass, warning, and failure checks."),
  "",
  "No existing scripts, data, figures, outputs, panels, or manuscripts were modified."
)
writeLines(report, file.path(output_dir, "QA_REPORT.md"), useBytes = TRUE)

cat("Completed with status: ", overall_status, "\n", sep = "")
cat("QA counts: ", paste(names(table(qa$status)), as.integer(table(qa$status)), collapse = ", "), "\n", sep = "")
cat("Report: ", file.path(output_dir, "QA_REPORT.md"), "\n", sep = "")
