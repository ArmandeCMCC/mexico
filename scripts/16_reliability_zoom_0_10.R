# 16_reliability_zoom_0_10.R

# Fine-grained reliability / calibration analysis in the
# 0-10% predicted-probability range, where 99%+ of test
# predictions fall (test prevalence ~0.40%).
#
# Produces:
#   reliability_zoom_0_10_bins.csv  -- binned calibration table
#   reliability_zoom_0_10.png      -- reliability diagram plot
#
# Both saved in the strict-run output directory.
#
# Usage:
#   Rscript scripts/16_reliability_zoom_0_10.R [run_id]
#   Rscript scripts/16_reliability_zoom_0_10.R ablation_batches/20260512_111324/20260512_111341_forecast_strict_bench_rs_history

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(jsonlite)
  library(tidymodels)
})

set.seed(42)

# HELPERS

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (dir.exists("data") || dir.exists("scripts")) return(normalizePath(getwd()))
  normalizePath(getwd())
}

apply_platt_with_params <- function(prob, intercept, slope, eps = 1e-15) {
  p <- pmin(pmax(prob, eps), 1 - eps)
  plogis(intercept + slope * qlogis(p))
}

get_nthread <- function() {
  override <- suppressWarnings(as.integer(Sys.getenv("XGB_NTHREAD", "")))
  if (!is.na(override) && override >= 1) return(as.integer(override))
  n <- suppressWarnings(parallel::detectCores())
  if (is.na(n) || n < 2) return(1L)
  max(1L, as.integer(n - 1))
}

# PATHS

args <- commandArgs(trailingOnly = TRUE)
run_id <- if (length(args) >= 1 && nzchar(args[1])) args[1] else "20260223_125430_forecast_strict"

project_dir <- detect_project_dir()
run_dir     <- file.path(project_dir, "data", "baselines", "binary_threshold", run_id)

features_path      <- file.path(project_dir, "data", "model_ready", "features_engineered.rds")
splits_path        <- file.path(project_dir, "data", "model_ready", "splits_fixed.rds")
features_used_path <- file.path(run_dir, "features_used.txt")
run_config_path    <- file.path(run_dir, "run_config.json")
platt_path         <- file.path(run_dir, "platt_calibration.csv")

required_files <- c(features_path, splits_path, features_used_path, run_config_path, platt_path)
missing_files  <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop("Missing required files:\n", paste0(" - ", missing_files, collapse = "\n"))
}

message(strrep("=", 70))
message("16_reliability_zoom_0_10.R")
message(strrep("=", 70))
message("Run ID:       ", run_id)
message("Project dir:  ", project_dir)

# LOAD ARTIFACTS

feature_cols <- read_lines(features_used_path)
feature_cols <- unique(feature_cols[nzchar(feature_cols)])

run_config <- jsonlite::read_json(run_config_path)
xgb_params <- run_config$model$params

platt_params    <- read_csv(platt_path, show_col_types = FALSE) %>% slice(1)
platt_intercept <- as.numeric(platt_params$platt_intercept)
platt_slope     <- as.numeric(platt_params$platt_slope)
if (!is.finite(platt_intercept) || !is.finite(platt_slope)) {
  stop("Invalid Platt parameters in platt_calibration.csv")
}
message("Platt params:  intercept=", sprintf("%.4f", platt_intercept),
        ", slope=", sprintf("%.4f", platt_slope))

features     <- readRDS(features_path)
splits_fixed <- readRDS(splits_path)

train_start <- as.Date(splits_fixed$train_range[1])
train_end   <- as.Date(splits_fixed$train_range[2])
test_start  <- as.Date(splits_fixed$test_range[1])
test_end    <- as.Date(splits_fixed$test_range[2])

features <- features %>%
  mutate(
    date = as.Date(date),
    split = case_when(
      date >= train_start & date <= train_end ~ "train",
      date >= test_start  & date <= test_end  ~ "test",
      TRUE ~ "other"
    )
  )

missing_feats <- setdiff(feature_cols, names(features))
if (length(missing_feats) > 0) {
  stop("Missing features:\n", paste0(" - ", head(missing_feats, 20), collapse = "\n"))
}

train_data <- features %>% filter(split == "train")
test_data  <- features %>% filter(split == "test")

message("Train rows: ", format(nrow(train_data), big.mark = ","))
message("Test rows:  ", format(nrow(test_data), big.mark = ","))

# REBUILD XGBOOST PREDICTIONS ON TEST SET

prep_model_df <- function(df, model_features) {
  df %>%
    select(all_of(c("outage_3h_or_more", model_features))) %>%
    mutate(
      outage_3h_or_more = factor(as.character(outage_3h_or_more), levels = c("1", "0"))
    ) %>%
    mutate(across(where(is.logical), as.integer))
}

train_model <- prep_model_df(train_data, feature_cols)
test_model  <- prep_model_df(test_data, feature_cols)

n_pos <- sum(train_model$outage_3h_or_more == "1", na.rm = TRUE)
n_neg <- sum(train_model$outage_3h_or_more == "0", na.rm = TRUE)
scale_pos_weight <- n_neg / max(1, n_pos)

n_features <- length(feature_cols)
mtry_count <- max(1, floor(xgb_params$mtry * n_features))

model_recipe <- recipe(outage_3h_or_more ~ ., data = train_model) %>%
  step_zv(all_predictors())

xgb_spec <- boost_tree(
  trees          = xgb_params$trees,
  tree_depth     = xgb_params$tree_depth,
  min_n          = xgb_params$min_n,
  loss_reduction = xgb_params$loss_reduction,
  sample_size    = xgb_params$sample_size,
  mtry           = mtry_count,
  learn_rate     = xgb_params$learn_rate
) %>%
  set_engine(
    "xgboost",
    scale_pos_weight = scale_pos_weight,
    nthread          = get_nthread(),
    verbosity        = 0
  ) %>%
  set_mode("classification")

message("Training XGBoost...")
xgb_fit <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(xgb_spec) %>%
  fit(data = train_model)

test_prob_raw <- predict(xgb_fit, new_data = test_model, type = "prob") %>%
  pull(.pred_1)

# APPLY PLATT CALIBRATION

test_prob_platt <- apply_platt_with_params(test_prob_raw, platt_intercept, platt_slope)
test_label      <- as.integer(as.character(test_model$outage_3h_or_more))

message("Platt-calibrated predictions -- range: [",
        sprintf("%.6f", min(test_prob_platt)), ", ",
        sprintf("%.6f", max(test_prob_platt)), "]")
message("Fraction <= 0.10: ",
        sprintf("%.4f", mean(test_prob_platt <= 0.10)))

# BUILD FINE-GRAINED BINS IN [0, 0.10]

# Bin breakpoints:
#   [0, 0.01] at 0.001 spacing  -> 10 bins
#   (0.01, 0.05] at 0.005       ->  8 bins
#   (0.05, 0.10] at 0.01        ->  5 bins
#   >0.10                       ->  1 overflow bin

breaks_fine <- c(
  seq(0, 0.010, by = 0.001),               # 0.000, 0.001, ..., 0.010
  seq(0.015, 0.050, by = 0.005),            # 0.015, 0.020, ..., 0.050
  seq(0.060, 0.100, by = 0.010),            # 0.060, 0.070, ..., 0.100
  Inf                                       # overflow
)
breaks_fine <- unique(breaks_fine)

n_total <- length(test_prob_platt)

bin_idx <- findInterval(test_prob_platt, breaks_fine, rightmost.closed = FALSE, left.open = FALSE)
# findInterval: value in [breaks[i], breaks[i+1]) gets index i.
# Adjust: bin_idx == 0 should not happen since min break is 0.
# The last bin covers [0.10, Inf).

n_bins <- length(breaks_fine) - 1

bins_list <- lapply(seq_len(n_bins), function(i) {
  in_bin <- (bin_idx == i)
  probs  <- test_prob_platt[in_bin]
  labels <- test_label[in_bin]

  n_obs      <- length(probs)
  n_positive <- sum(labels == 1L, na.rm = TRUE)

  mean_predicted <- if (n_obs > 0) mean(probs) else NA_real_
  observed_rate  <- if (n_obs > 0) mean(labels, na.rm = TRUE) else NA_real_

  # Wilson confidence interval for observed rate
  if (n_obs > 0 && !is.na(observed_rate)) {
    wilson <- prop.test(n_positive, n_obs, conf.level = 0.95, correct = FALSE)
    ci_lower <- wilson$conf.int[1]
    ci_upper <- wilson$conf.int[2]
  } else {
    ci_lower <- NA_real_
    ci_upper <- NA_real_
  }

  bin_lower <- breaks_fine[i]
  bin_upper <- breaks_fine[i + 1]

  tibble(
    bin_lower        = bin_lower,
    bin_upper        = if (is.infinite(bin_upper)) NA_real_ else bin_upper,
    bin_label        = if (is.infinite(bin_upper)) sprintf(">%.3f", bin_lower) else sprintf("[%.3f, %.3f)", bin_lower, bin_upper),
    bin_midpoint     = if (is.infinite(bin_upper)) NA_real_ else (bin_lower + bin_upper) / 2,
    n_obs            = n_obs,
    n_positive       = n_positive,
    mean_predicted   = mean_predicted,
    observed_rate    = observed_rate,
    calibration_error = if (!is.na(mean_predicted) && !is.na(observed_rate)) abs(mean_predicted - observed_rate) else NA_real_,
    ci_lower         = ci_lower,
    ci_upper         = ci_upper,
    share_of_total_obs = n_obs / n_total
  )
})

bins_df <- bind_rows(bins_list)

message("\nBinned calibration table (", nrow(bins_df), " bins):")
print(bins_df %>% select(bin_label, n_obs, n_positive, mean_predicted, observed_rate, calibration_error), n = 30)

# SAVE CALIBRATION TABLE

csv_path <- file.path(run_dir, "reliability_zoom_0_10_bins.csv")
write_csv(bins_df, csv_path)
message("\nSaved: ", csv_path)

# RELIABILITY DIAGRAM PLOT

# Plot only bins within [0, 0.10] (exclude overflow and empty bins)
plot_df <- bins_df %>%
  filter(!is.na(bin_midpoint), n_obs > 0)

# Diagonal line range
diag_max <- max(plot_df$bin_upper, na.rm = TRUE)

p <- ggplot(plot_df, aes(x = mean_predicted, y = observed_rate)) +
  # Perfect calibration line
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.5) +
  # Error bars (Wilson CIs)
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.0005, colour = "steelblue", alpha = 0.6) +
  # Points sized by number of observations

  geom_point(aes(size = n_obs), colour = "steelblue", alpha = 0.8) +
  scale_size_continuous(
    name   = "N observations",
    range  = c(1.5, 8),
    labels = scales::comma
  ) +
  scale_x_continuous(
    name   = "Mean predicted probability",
    labels = scales::percent_format(accuracy = 0.1),
    limits = c(0, diag_max),
    expand = expansion(mult = c(0.01, 0.05))
  ) +
  scale_y_continuous(
    name   = "Observed rate",
    labels = scales::percent_format(accuracy = 0.1),
    limits = c(0, NA),
    expand = expansion(mult = c(0.01, 0.05))
  ) +
  labs(
    title    = "Reliability Diagram (Zoom: 0-10%)",
    subtitle = paste0("Platt-calibrated XGBoost | Test set (",
                      format(n_total, big.mark = ","), " obs, prevalence ",
                      sprintf("%.2f%%", mean(test_label) * 100), ")"),
    caption  = paste0("Run: ", run_id)
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold"),
    legend.position = "right"
  )

png_path <- file.path(run_dir, "reliability_zoom_0_10.png")
ggsave(png_path, plot = p, width = 8, height = 6, dpi = 300, bg = "white")
message("Saved: ", png_path)

# SUMMARY

# Quick overall calibration stats in the 0-10% range
zoom_df <- bins_df %>% filter(!is.na(bin_midpoint), n_obs > 0)
weighted_ace <- sum(zoom_df$calibration_error * zoom_df$n_obs, na.rm = TRUE) / sum(zoom_df$n_obs, na.rm = TRUE)
max_ce <- max(zoom_df$calibration_error, na.rm = TRUE)

message("\n", strrep("-", 50))
message("Summary (0-10% range):")
message("  Bins with data:               ", nrow(zoom_df))
message("  Observations in 0-10% range:  ", format(sum(zoom_df$n_obs), big.mark = ","),
        " (", sprintf("%.2f%%", sum(zoom_df$share_of_total_obs) * 100), " of total)")
message("  Weighted avg calibration err: ", sprintf("%.6f", weighted_ace))
message("  Max calibration error:        ", sprintf("%.6f", max_ce))

# Overflow bin summary
overflow <- bins_df %>% filter(is.na(bin_midpoint))
if (nrow(overflow) > 0 && overflow$n_obs[1] > 0) {
  message("  Overflow (>10%) observations: ", format(overflow$n_obs[1], big.mark = ","),
          " (", sprintf("%.2f%%", overflow$share_of_total_obs[1] * 100), " of total)")
}

message("\nDone.")
