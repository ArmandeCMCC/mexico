# 04c_ntl_ratio_features.R
# PURPOSE:
#   Compute physics-motivated NTL ratio features: log(NTL_lagged / NTL_climatological).
#   Motivated by Arora et al. (2025): Beer's Law analogy where log(I/Ī) correlates (article that Giacomo sent)
#   with outage fraction (Spearman ρ=0.65 in their disaster-mapping setting).
#
#   In our forecast-safe setting, we use:
#     - Numerator:   NTL_lag1 or NTL_lag7 (known at prediction time)
#     - Denominator:  NTL_clim_mean (municipality × DOY climatological average, training-only)
#
#   These are conceptually different from the existing anomaly/z-score features:
#     - anomaly = NTL_sameday - NTL_clim  (additive, same-day, NOT forecast-safe)
#     - z-score = (NTL_sameday - NTL_clim) / NTL_clim_sd  (same-day)
#     - log ratio = log(NTL_lagged / NTL_clim)  (multiplicative, lagged, forecast-safe)
#
#   The log-ratio captures PROPORTIONAL deviations from baseline, which is more
#   physically meaningful for severity: a drop from 10→5 (50%) and 100→50 (50%)
#   produce the same ratio, whereas the anomaly treats them very differently.
#
# REQUIRES:
#   features_engineered.rds must already contain:
#     ntl_mean_built_lag1, ntl_sum_built_lag1   (from 04)
#     ntl_mean_built_lag7, ntl_sum_built_lag7   (from 04)
#     ntl_mean_built_clim_mean, ntl_sum_built_clim_mean  (from 04, Section 8)
#
# OUTPUT:
#   Adds columns to features_engineered.rds (in-place patch):
#     ntl_log_ratio_mean_lag1, ntl_log_ratio_sum_lag1
#     ntl_log_ratio_mean_lag7, ntl_log_ratio_sum_lag7
#
#   Named with _lag suffix so apply_task_mode_filters() keeps them in forecast mode.
#
# NOTES ON eps:
#   We add eps = 1e-6 to both numerator and denominator to handle zero/near-zero NTL.
#   This is conservative: log(eps/eps) = 0 (neutral), and for typical NTL values
#   (order 0.1-10 nW/cm²/sr), eps has negligible effect.

library(data.table)

eps <- 1e-6  # small constant to prevent log(0)

# Detect data path
if (file.exists("data/model_ready/features_engineered.rds")) {
  fe_path <- "data/model_ready/features_engineered.rds"
} else {
  stop("features_engineered.rds not found in data/model_ready/")
}

message("Loading ", fe_path, " ...")
fe <- readRDS(fe_path)
orig_class <- class(fe)
setDT(fe)
cat("Rows:", nrow(fe), " Cols:", ncol(fe), "\n")

# Check if already patched
if ("ntl_log_ratio_mean_lag1" %in% names(fe)) {
  stop("Already patched: ntl_log_ratio_mean_lag1 exists.")
}

# Verify required columns exist
required <- c("ntl_mean_built_lag1", "ntl_sum_built_lag1",
              "ntl_mean_built_lag7", "ntl_sum_built_lag7",
              "ntl_mean_built_clim_mean", "ntl_sum_built_clim_mean")
missing <- setdiff(required, names(fe))
if (length(missing) > 0) {
  stop("Missing required columns: ", paste(missing, collapse = ", "))
}

# Compute ratio features
message("Computing NTL log-ratio features (eps = ", eps, ") ...")

fe[, ntl_log_ratio_mean_lag1 := log((ntl_mean_built_lag1 + eps) / (ntl_mean_built_clim_mean + eps))]
fe[, ntl_log_ratio_sum_lag1  := log((ntl_sum_built_lag1  + eps) / (ntl_sum_built_clim_mean  + eps))]
fe[, ntl_log_ratio_mean_lag7 := log((ntl_mean_built_lag7 + eps) / (ntl_mean_built_clim_mean + eps))]
fe[, ntl_log_ratio_sum_lag7  := log((ntl_sum_built_lag7  + eps) / (ntl_sum_built_clim_mean  + eps))]

# Summary statistics
for (col in c("ntl_log_ratio_mean_lag1", "ntl_log_ratio_sum_lag1",
              "ntl_log_ratio_mean_lag7", "ntl_log_ratio_sum_lag7")) {
  vals <- fe[[col]]
  cat(sprintf("  %s: mean=%.4f, sd=%.4f, median=%.4f, NAs=%d, Inf=%d\n",
              col,
              mean(vals, na.rm = TRUE),
              sd(vals, na.rm = TRUE),
              median(vals, na.rm = TRUE),
              sum(is.na(vals)),
              sum(is.infinite(vals), na.rm = TRUE)))
}

# Replace any Inf/-Inf with NA (can happen if clim_mean is 0 or lag is 0)
for (col in c("ntl_log_ratio_mean_lag1", "ntl_log_ratio_sum_lag1",
              "ntl_log_ratio_mean_lag7", "ntl_log_ratio_sum_lag7")) {
  n_inf <- sum(is.infinite(fe[[col]]), na.rm = TRUE)
  if (n_inf > 0) {
    message("  Replacing ", n_inf, " Inf values in ", col, " with NA")
    set(fe, which(is.infinite(fe[[col]])), col, NA_real_)
  }
}

# Restore original class
if ("data.frame" %in% orig_class && !"data.table" %in% orig_class) {
  setDF(fe)
}

cat("After patch: Cols:", ncol(fe), "\n")

message("Saving updated features_engineered.rds ...")
saveRDS(fe, fe_path)
message("Done. Added 4 NTL log-ratio features.")

# Also patch sample version if it exists
sample_path <- "data/model_ready_sample/features_engineered.rds"
if (file.exists(sample_path)) {
  message("\nAlso patching sample version ...")
  fe_s <- readRDS(sample_path)
  orig_class_s <- class(fe_s)
  setDT(fe_s)

  if ("ntl_log_ratio_mean_lag1" %in% names(fe_s)) {
    message("Sample already patched, skipping.")
  } else {
    fe_s[, ntl_log_ratio_mean_lag1 := log((ntl_mean_built_lag1 + eps) / (ntl_mean_built_clim_mean + eps))]
    fe_s[, ntl_log_ratio_sum_lag1  := log((ntl_sum_built_lag1  + eps) / (ntl_sum_built_clim_mean  + eps))]
    fe_s[, ntl_log_ratio_mean_lag7 := log((ntl_mean_built_lag7 + eps) / (ntl_mean_built_clim_mean + eps))]
    fe_s[, ntl_log_ratio_sum_lag7  := log((ntl_sum_built_lag7  + eps) / (ntl_sum_built_clim_mean  + eps))]

    for (col in c("ntl_log_ratio_mean_lag1", "ntl_log_ratio_sum_lag1",
                  "ntl_log_ratio_mean_lag7", "ntl_log_ratio_sum_lag7")) {
      n_inf <- sum(is.infinite(fe_s[[col]]), na.rm = TRUE)
      if (n_inf > 0) set(fe_s, which(is.infinite(fe_s[[col]])), col, NA_real_)
    }

    if ("data.frame" %in% orig_class_s && !"data.table" %in% orig_class_s) {
      setDF(fe_s)
    }
    saveRDS(fe_s, sample_path)
    message("Sample patched.")
  }
}
