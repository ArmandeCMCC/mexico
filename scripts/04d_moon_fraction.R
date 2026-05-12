# 04d_moon_fraction.R
# PURPOSE:
#   Add moon illumination fraction as a confounder control for NTL measurements.
#   Motivated by Arora et al. (2025): moonlight affects VIIRS sensor readings (sent by Giacomo)
#   potentially masking real NTL drops or inflating apparent brightness.
#
#   Moon fraction is computed from date using a standard astronomical algorithm
#   (simplified synodic period calculation). Values range from 0 (new moon)
#   to 1 (full moon).
#
#   This is a pure date-derived feature: forecast-safe and leakage-free.

library(data.table)

# Moon fraction calculation using synodic period
# Reference new moon: Jan 6, 2000 18:14 UTC (Julian date 2451550.1)
moon_fraction <- function(dates) {
  # Days since reference new moon
  jd <- as.numeric(as.Date(dates)) + 2440587.5  # convert to Julian date
  ref_new_moon_jd <- 2451550.1
  days_since <- jd - ref_new_moon_jd
  synodic_period <- 29.53058770576  # mean synodic month in days
  phase <- (days_since %% synodic_period) / synodic_period
  # Illumination fraction: 0 at new moon, 1 at full moon
  # phase=0 → new moon, phase=0.5 → full moon
  illumination <- 0.5 * (1 - cos(2 * pi * phase))
  illumination
}

# Detect and patch data
fe_path <- "data/model_ready/features_engineered.rds"
if (!file.exists(fe_path)) stop("features_engineered.rds not found")

message("Loading ", fe_path, " ...")
fe <- readRDS(fe_path)
orig_class <- class(fe)
setDT(fe)
cat("Rows:", nrow(fe), " Cols:", ncol(fe), "\n")

if ("moon_fraction" %in% names(fe)) {
  stop("Already patched: moon_fraction exists.")
}

# Compute moon fraction
message("Computing moon_fraction from date ...")
fe[, moon_fraction := moon_fraction(date)]

cat(sprintf("  moon_fraction: mean=%.4f, sd=%.4f, min=%.4f, max=%.4f, NAs=%d\n",
            mean(fe$moon_fraction, na.rm = TRUE),
            sd(fe$moon_fraction, na.rm = TRUE),
            min(fe$moon_fraction, na.rm = TRUE),
            max(fe$moon_fraction, na.rm = TRUE),
            sum(is.na(fe$moon_fraction))))

# Restore original class
if ("data.frame" %in% orig_class && !"data.table" %in% orig_class) {
  setDF(fe)
}

cat("After patch: Cols:", ncol(fe), "\n")
message("Saving ...")
saveRDS(fe, fe_path)
message("Done. Added moon_fraction.")

# Also patch sample version
sample_path <- "data/model_ready_sample/features_engineered.rds"
if (file.exists(sample_path)) {
  message("\nPatching sample version ...")
  fe_s <- readRDS(sample_path)
  orig_class_s <- class(fe_s)
  setDT(fe_s)
  if (!"moon_fraction" %in% names(fe_s)) {
    fe_s[, moon_fraction := moon_fraction(date)]
    if ("data.frame" %in% orig_class_s && !"data.table" %in% orig_class_s) setDF(fe_s)
    saveRDS(fe_s, sample_path)
    message("Sample patched.")
  } else {
    message("Sample already patched.")
  }
}
