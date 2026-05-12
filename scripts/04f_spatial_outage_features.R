# 04f_spatial_outage_features.R
#
# Spatial outage propagation features: neighbor and state-level outage signals.
#
# Input:
#   - data/model_ready/features_engineered.rds (has outage_3h_or_more, hist_outage_rate_30d, GID_1)
#   - data/model_ready/spatial_weights.rds (W_sparse, gid2_order, gid2_index)
#
# Output:
#   - data/model_ready/features_engineered.rds (updated with spatial outage features)
#   - data/model_ready/features_engineered_pre_spatial_outage.rds (backup)
#
# Features added (all strictly t-1, forecast-safe):
#   - nb_outage_lag1:         neighbor mean of outage_3h_or_more at t-1
#   - nb_outage_rate_30d:     neighbor mean of hist_outage_rate_30d (already t-1 by construction)
#   - state_outage_count_lag1: count of other municipalities in same state with outage at t-1
#   - state_outage_rate_lag1:  fraction of other municipalities in same state with outage at t-1
#
# Leakage design:
#   - nb_outage_lag1: uses dplyr::lag(outage_3h_or_more, 1) per GID_2 before spatial aggregation
#   - nb_outage_rate_30d: hist_outage_rate_30d is already strictly t-1 (built by 04b)
#   - state_outage_*_lag1: uses lag(outage_3h_or_more, 1) per GID_2, then state-level sum
#
# Pipeline position: run AFTER 04b_history_features.R (needs hist_outage_rate_30d)

suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
})

cat("=== 04f_spatial_outage_features.R ===\n")
cat("Started:", format(Sys.time()), "\n\n")

# Setup
detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))
  normalizePath(getwd())
}

project_dir <- detect_project_dir()
features_path <- file.path(project_dir, "data", "model_ready", "features_engineered.rds")
sw_path <- file.path(project_dir, "data", "model_ready", "spatial_weights.rds")
backup_path <- file.path(project_dir, "data", "model_ready", "features_engineered_pre_spatial_outage.rds")

stopifnot(file.exists(features_path), file.exists(sw_path))

# Load data
cat("Loading features_engineered.rds...\n")
fe <- as.data.table(readRDS(features_path))
cat(sprintf("  Dimensions: %d x %d\n", nrow(fe), ncol(fe)))

# Check if already patched
if ("nb_outage_lag1" %in% names(fe)) {
  cat("Spatial outage features already present. Skipping.\n")
  quit(save = "no", status = 0)
}

# Verify required columns
required <- c("GID_2", "GID_1", "date", "outage_3h_or_more", "hist_outage_rate_30d")
missing <- setdiff(required, names(fe))
if (length(missing) > 0) stop("Missing required columns: ", paste(missing, collapse = ", "))

cat("Loading spatial_weights.rds...\n")
sw <- readRDS(sw_path)
W_sparse <- sw$W_sparse
gid2_order <- sw$gid2_order
gid2_index <- sw$gid2_index
cat(sprintf("  Spatial weights: %d municipalities, style: %s\n",
            length(gid2_order), sw$metadata$weights_style))

# Backup 
cat("Saving backup: features_engineered_pre_spatial_outage.rds\n")
saveRDS(readRDS(features_path), backup_path)

# PART 1: Lagged outage column (t-1)
cat("\n--- Part 1: Computing lagged outage (t-1) ---\n")

setorder(fe, GID_2, date)
fe[, outage_lag1 := shift(outage_3h_or_more, n = 1L, type = "lag"), by = GID_2]
cat(sprintf("  outage_lag1: non-NA = %d of %d\n", sum(!is.na(fe$outage_lag1)), nrow(fe)))

# PART 2: Spatial neighbor features via W_sparse
cat("\n--- Part 2: Computing spatial neighbor outage features ---\n")

# Reusable spatial lag function (same logic as 04_feature_engineering_improved.R)
spatial_lag_mean <- function(values_by_gid2, W, gid2_idx, gid2_ord) {
  n <- length(gid2_ord)
  x <- rep(NA_real_, n)

  gids <- names(values_by_gid2)
  idx <- unname(gid2_idx[gids])
  x[idx] <- values_by_gid2

  m <- !is.na(x)
  x0 <- ifelse(m, x, 0)

  num <- as.numeric(W %*% x0)
  den <- as.numeric(W %*% as.numeric(m))

  out <- num / den
  out[den == 0] <- NA_real_
  out
}

# Process date-by-date
dates <- sort(unique(fe$date))
cat(sprintf("  Processing %d dates...\n", length(dates)))

# Pre-allocate result columns
fe[, nb_outage_lag1 := NA_real_]
fe[, nb_outage_rate_30d := NA_real_]

t0 <- Sys.time()
for (i in seq_along(dates)) {
  d <- dates[i]
  rows_idx <- which(fe$date == d)
  sub <- fe[rows_idx]

  gids <- as.character(sub$GID_2)

  # nb_outage_lag1: neighbor mean of yesterday's outage
  vals1 <- setNames(as.numeric(sub$outage_lag1), gids)
  full_lag <- spatial_lag_mean(vals1, W_sparse, gid2_index, gid2_order)
  fe_idx <- unname(gid2_index[gids])
  set(fe, i = rows_idx, j = "nb_outage_lag1", value = full_lag[fe_idx])

  # nb_outage_rate_30d: neighbor mean of hist_outage_rate_30d
  vals2 <- setNames(as.numeric(sub$hist_outage_rate_30d), gids)
  full_rate <- spatial_lag_mean(vals2, W_sparse, gid2_index, gid2_order)
  set(fe, i = rows_idx, j = "nb_outage_rate_30d", value = full_rate[fe_idx])

  if (i %% 500 == 0) cat(sprintf("    %d / %d dates done\n", i, length(dates)))
}
t1 <- Sys.time()
cat(sprintf("  Spatial features computed in %.1f min\n", difftime(t1, t0, units = "mins")))

# PART 3: State-level concurrent outages (t-1)
cat("\n--- Part 3: State-level concurrent outages ---\n")

# Count outages at t-1 per state, excluding the focal municipality
fe[, state_outage_count_lag1 := NA_real_]
fe[, state_outage_rate_lag1 := NA_real_]

# State-level aggregation per date using outage_lag1
state_agg <- fe[!is.na(outage_lag1),
                .(state_total = sum(outage_lag1, na.rm = TRUE),
                  state_n = .N),
                by = .(GID_1, date)]

fe <- merge(fe, state_agg, by = c("GID_1", "date"), all.x = TRUE)

# Exclude focal municipality: state count - own contribution
fe[, state_outage_count_lag1 := fifelse(
  is.na(state_total) | is.na(outage_lag1), NA_real_,
  state_total - outage_lag1
)]
fe[, state_outage_rate_lag1 := fifelse(
  is.na(state_n) | state_n <= 1L, NA_real_,
  state_outage_count_lag1 / (state_n - 1L)
)]

# Clean up temp columns
fe[, c("state_total", "state_n", "outage_lag1") := NULL]

# PART 4: Save
cat("\n--- Part 4: Saving ---\n")

new_cols <- c("nb_outage_lag1", "nb_outage_rate_30d",
              "state_outage_count_lag1", "state_outage_rate_lag1")
for (col in new_cols) {
  pct <- round(100 * mean(!is.na(fe[[col]])), 1)
  cat(sprintf("  %-30s %5.1f%% non-NA\n", col, pct))
}

# Restore original row order (by GID_2, date)
setorder(fe, GID_2, date)

# Convert back to data.frame (05b expects data.frame)
fe <- as.data.frame(fe)

cat(sprintf("\nSaving updated features_engineered.rds: %d x %d\n", nrow(fe), ncol(fe)))
saveRDS(fe, features_path)

cat("\n=== 04f_spatial_outage_features.R complete ===\n")
cat("Finished:", format(Sys.time()), "\n")
