# =============================================================================
# figure3.R
# Figure 3: Probabilistic model skill
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})

# ── Paths ─────────────────────────────────────────────────────────────────────

proj_dir <- normalizePath(
  if (basename(getwd()) == "figures") file.path(getwd(), "..") else getwd()
)

# Auto-detect data_dir => first existing path wins
# Can add or edit candidates if differenty local layout
candidate_data_dirs <- "C:/Users/giaco/Documents/Github/mexico/data"    # Win Drive Stream

data_dir <- candidate_data_dirs[dir.exists(candidate_data_dirs)][1]
if (is.na(data_dir)) {
  stop("Could not find mexico/data directory. Edit candidate_data_dirs above.")
}
message("Using data_dir = ", data_dir)

fig_dir <- file.path(proj_dir, "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# Headline run is two-level nested inside ablation_batches
# These names are the folder names on disk
HEADLINE_BATCH    <- "20260513_153833"
HEADLINE_RUN      <- "20260513_153858_forecast_strict_bench_rs_history"
# Pre-history ablation batch (7 feature subsets) for Panel C:
PREHISTORY_BATCH  <- "20260302_122455"

run_dir <- file.path(data_dir, "baselines", "binary_threshold",
                     "ablation_batches", HEADLINE_BATCH, HEADLINE_RUN)
abl_dir <- file.path(data_dir, "baselines", "binary_threshold",
                     "ablation_batches", PREHISTORY_BATCH)

# Logistic baseline: pick most recent run directory
log_root <- file.path(data_dir, "baselines", "logistic_threshold")
log_dirs <- sort(list.dirs(log_root, recursive = FALSE, full.names = TRUE))
log_dir  <- if (length(log_dirs) > 0) rev(log_dirs)[1] else NULL

# Test-set prevalence (verified: 5,350 / 1,334,151 = 0.401%)
PREVALENCE <- 0.401 / 100

# Overall test ROC-AUC of the headline classifier (cf run_config.json).
# Used as the dashed reference line in Panel D and the Panel A ROC inset.
OVERALL_ROC <- 0.919

# Logistic baseline test ROC-AUC (cf logistic_threshold run metrics_summary.csv).
# Panel A inset uses it to show the discrimination gap (0.92 vs 0.78): this is
# where XGBoost dominates, even though PR-AUC is a near-tie at 0.4% prevalence.
LOG_ROC <- 0.776

# ── Sanity-check that the files we need exist before plotting ────────────────

required <- c(
  pr_xgb    = file.path(run_dir, "pr_curve.csv"),
  cal_bins  = file.path(run_dir, "calibration_bins_quantile_all_methods.csv"),
  metrics   = file.path(run_dir, "metrics_summary.csv"),
  abl_lb    = file.path(abl_dir, "ablation_leaderboard_by_pr_auc.csv"),
  cause_bb  = file.path(run_dir, "cause_specific_eval",
                        "cause_metrics_main_block_bootstrap.csv")
)
missing <- required[!file.exists(required)]
if (length(missing) > 0) {
  stop("Missing required files:\n",
       paste0("  - ", names(missing), ": ", missing, collapse = "\n"))
}

# ── Theme ─────────────────────────────────────────────────────────────────────

theme_nat <- function(base = 8) {
  theme_classic(base_size = base, base_family = "sans") +
    theme(
      axis.line         = element_line(linewidth = 0.3, colour = "grey30"),
      axis.ticks        = element_line(linewidth = 0.3, colour = "grey30"),
      axis.text         = element_text(size = base - 1, colour = "grey20"),
      axis.title        = element_text(size = base,     colour = "grey10"),
      legend.key.size   = unit(3, "mm"),
      legend.text       = element_text(size = base - 1),
      legend.title      = element_text(size = base - 1, face = "bold"),
      legend.background = element_rect(fill = "white", colour = NA),
      plot.title        = element_text(size = base, face = "bold",
                                       hjust = 0, vjust = 1),
      plot.background   = element_rect(fill = "white", colour = NA),
      panel.background  = element_rect(fill = "white", colour = NA),
      plot.margin       = margin(2, 4, 2, 4, "mm")
    )
}

# Cause colours (shared with fig 1)
cause_pal <- c(
  "Environmental" = "#2166ac",
  "Technical"     = "#d6604d",
  "Planned"       = "#4dac26",
  "Other"         = "#999999"
)

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Load data
# ═════════════════════════════════════════════════════════════════════════════

message("Loading PR curve data ...")
pr_xgb <- read_csv(required[["pr_xgb"]], show_col_types = FALSE) |>
  filter(split == "test") |>
  # yardstick names columns .threshold / recall / precision
  rename_with(~ str_remove(., "^\\."), starts_with("."))

pr_log <- if (!is.null(log_dir) &&
              file.exists(file.path(log_dir, "pr_curve.csv"))) {
  read_csv(file.path(log_dir, "pr_curve.csv"), show_col_types = FALSE) |>
    filter(split == "test") |>
    rename_with(~ str_remove(., "^\\."), starts_with("."))
} else {
  message("  Logistic pr_curve.csv not found - baseline omitted from panel A.")
  NULL
}

message("Loading ROC curves (for panel A discrimination inset) ...")
roc_xgb <- read_csv(file.path(run_dir, "roc_curve.csv"), show_col_types = FALSE) |>
  filter(split == "test") |>
  rename_with(~ str_remove(., "^\\."), starts_with("."))

roc_log <- if (!is.null(log_dir) &&
               file.exists(file.path(log_dir, "roc_curve.csv"))) {
  read_csv(file.path(log_dir, "roc_curve.csv"), show_col_types = FALSE) |>
    filter(split == "test") |>
    rename_with(~ str_remove(., "^\\."), starts_with("."))
} else NULL

message("Loading calibration bins ...")
cal_bins <- read_csv(required[["cal_bins"]], show_col_types = FALSE) |>
  filter(split == "test", method %in% c("raw", "platt"))

message("Loading ablation leaderboard ...")
abl_df <- read_csv(required[["abl_lb"]], show_col_types = FALSE)

message("Loading cause-specific metrics (block bootstrap) ...")
cause_df <- read_csv(required[["cause_bb"]], show_col_types = FALSE)

message("Loading headline metrics summary ...")
# Test-row metrics drive the annotations below, so the figure tracks the data.
met_xgb <- read_csv(required[["metrics"]], show_col_types = FALSE) |>
  filter(split == "test")

# Expected calibration error (quantile bins), raw scores vs after Platt scaling.
ECE_RAW   <- met_xgb$ece_quantile[1]          # ~0.139  (wildly overconfident)
ECE_PLATT <- met_xgb$ece_quantile_platt[1]    # ~0.0002 (essentially perfect)
# Deployment operating point (threshold selected on the validation set).
OP_PREC   <- met_xgb$precision_at_best_t[1]   # ~0.105
OP_RECALL <- met_xgb$recall_at_best_t[1]      # ~0.254
OP_LIFT   <- OP_PREC / PREVALENCE             # ~26x the no-skill rate

# ═════════════════════════════════════════════════════════════════════════════
# 2.  Pre-process
# ═════════════════════════════════════════════════════════════════════════════

# --- Ablation labels & ordering ---
# These 7 ablation_name values correspond to the sub-runs inside
# data/baselines/binary_threshold/ablation_batches/20260302_122455/
abl_labels <- c(
  full_model          = "Full model",
  ntl_plus_weather    = "NTL + weather",
  ntl_only            = "NTL only",
  ntl_plus_spatial    = "NTL + spatial",
  ntl_core            = "NTL core lags",
  ntl_plus_drops      = "NTL + drop flags",
  weather_lagged_only = "Weather only"
)

# Colour encodes the panel's message: night-lights presence. The full model is
# the headline; every other NTL-bearing subset shares the mid tone; the
# weather-only subset (no NTL) is greyed to read as the no-NTL floor.
abl_plot <- abl_df |>
  mutate(
    label = recode(ablation_name, !!!abl_labels),
    label = if_else(is.na(label), ablation_name, label),
    tier  = case_when(
      ablation_name == "full_model"          ~ "Full model",
      ablation_name == "weather_lagged_only" ~ "No NTL (weather only)",
      TRUE                                   ~ "NTL subset"
    )
  ) |>
  arrange(pr_auc_test) |>
  mutate(label = factor(label, levels = label))

# --- Cause metrics ---
# IMPORTANT: analysis_priority values in the CSV are lowercase
# ('primary', 'exploratory'); compare on the lowercase form.
cause_plot <- cause_df |>
  filter(cause %in% c("Environmental", "Technical", "Planned", "Other")) |>
  mutate(
    cause    = factor(cause, levels = c("Environmental", "Technical",
                                        "Planned", "Other")),
    priority = if_else(tolower(analysis_priority) == "primary",
                       "Primary", "Exploratory")
  )

# ═════════════════════════════════════════════════════════════════════════════
# 3.  Panel A - Precision-Recall curve
# ═════════════════════════════════════════════════════════════════════════════
# Note: PR-AUC is invariant to post-hoc calibration (monotone transform),
# so a single XGBoost curve represents all calibration variants.

# Log y-axis: at 0.4% prevalence the curve otherwise collapses onto the x-axis
# and the two models are indistinguishable. On a log scale the vertical gap to
# the no-skill line reads directly as the precision lift over chance.
pA <- ggplot() +
  geom_hline(yintercept = PREVALENCE * 100,
             linetype = "dashed", linewidth = 0.4, colour = "grey55") +
  annotate("text",
           x = 0.98, y = PREVALENCE * 100,
           label = sprintf("No skill (%.1f%%)", PREVALENCE * 100),
           hjust = 1, vjust = -0.55, size = 2.05, colour = "grey45") +
  {if (!is.null(pr_log))
    geom_path(data = filter(pr_log, precision > 0),
              aes(x = recall, y = precision * 100),
              colour = "#4393c3", linewidth = 0.5, linetype = "21")
  } +
  geom_path(data = filter(pr_xgb, precision > 0),
            aes(x = recall, y = precision * 100),
            colour = "#b2182b", linewidth = 0.8) +
  # Deployment operating point + leader label
  annotate("point", x = OP_RECALL, y = OP_PREC * 100,
           shape = 21, fill = "white", colour = "#b2182b",
           stroke = 0.8, size = 2) +
  annotate("text", x = OP_RECALL + 0.04, y = OP_PREC * 100 * 1.9,
           label = sprintf("Operating point\n%.0f%% precision @ %.0f%% recall\n(%.0f× no skill)",
                           OP_PREC * 100, OP_RECALL * 100, OP_LIFT),
           hjust = 0, vjust = 0.5, size = 1.95, colour = "grey25",
           lineheight = 0.95) +
  # Mini-legend in the empty upper-right (the curves overlap, so floating
  # colour-coded labels alone read as ambiguous -- show the line keys instead)
  annotate("segment", x = 0.58, xend = 0.66, y = 55, yend = 55,
           colour = "#b2182b", linewidth = 0.8) +
  annotate("text", x = 0.68, y = 55, label = "XGBoost", colour = "#b2182b",
           hjust = 0, size = 2.4, fontface = "bold") +
  {if (!is.null(pr_log)) list(
    annotate("segment", x = 0.58, xend = 0.66, y = 36, yend = 36,
             colour = "#4393c3", linewidth = 0.5, linetype = "21"),
    annotate("text", x = 0.68, y = 36, label = "Logistic", colour = "#4393c3",
             hjust = 0, size = 2.4)
  )} +
  scale_x_continuous(limits = c(0, 1), expand = c(0.01, 0),
                     labels = label_percent()) +
  scale_y_log10(limits = c(PREVALENCE * 100 * 0.85, 100),
                breaks = c(0.5, 1, 2, 5, 10, 25, 50, 100),
                labels = function(x) paste0(x, "%"),
                expand = expansion(mult = c(0.02, 0.04))) +
  annotation_logticks(sides = "l", linewidth = 0.2, colour = "grey55",
                      short = unit(0.4, "mm"), mid = unit(0.7, "mm"),
                      long = unit(1.1, "mm")) +
  coord_cartesian(clip = "off") +
  theme_nat() +
  labs(title = "A  Precision-recall curve (test set, log scale)",
       x     = "Recall",
       y     = "Precision")

# ROC inset: PR-AUC is a near-tie, but discrimination is not -- XGBoost's ROC
# curve bows hard to the top-left (AUC 0.92) while logistic stays near the
# chance diagonal (0.78). This is where the model's superiority lives.
roc_inset <- ggplot() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              linewidth = 0.25, colour = "grey60") +
  {if (!is.null(roc_log))
    geom_path(data = roc_log, aes(x = 1 - specificity, y = sensitivity),
              colour = "#4393c3", linewidth = 0.45, linetype = "21")} +
  geom_path(data = roc_xgb, aes(x = 1 - specificity, y = sensitivity),
            colour = "#b2182b", linewidth = 0.6) +
  annotate("text", x = 0.96, y = 0.38, hjust = 1, size = 1.75,
           colour = "#b2182b", fontface = "bold",
           label = sprintf("XGBoost %.2f", OVERALL_ROC)) +
  {if (!is.null(roc_log))
    annotate("text", x = 0.96, y = 0.18, hjust = 1, size = 1.75,
             colour = "#4393c3", label = sprintf("Logistic %.2f", LOG_ROC))} +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE, clip = "off") +
  labs(title = "ROC (AUC)", x = NULL, y = NULL) +
  theme_classic(base_size = 6, base_family = "sans") +
  theme(
    plot.title       = element_text(size = 5.5, face = "bold", colour = "grey25",
                                     hjust = 0.5, margin = margin(0, 0, 0.4, 0, "mm")),
    axis.text        = element_blank(),
    axis.ticks       = element_blank(),
    axis.line        = element_line(linewidth = 0.2, colour = "grey45"),
    plot.background  = element_rect(fill = alpha("white", 0.85),
                                    colour = "grey70", linewidth = 0.2),
    plot.margin      = margin(0.6, 0.6, 0.6, 0.6, "mm")
  )

pA <- pA + inset_element(roc_inset, left = 0.105, bottom = 0.07,
                         right = 0.46, top = 0.46, align_to = "panel")

# ═════════════════════════════════════════════════════════════════════════════
# 4.  Panel B - Reliability (calibration) diagram
# ═════════════════════════════════════════════════════════════════════════════

# Raw scores are de-emphasised (grey); Platt is the deployed model (hero red).
cal_pal  <- c("raw" = "#9e9e9e", "platt" = "#b2182b")
cal_labs <- c("raw" = "Raw XGBoost", "platt" = "XGBoost + Platt")

# Zoom window: 0-3.5% on both axes.
# Every Platt bin lives below 3.1% predicted, so a tight window is what makes
# the on-diagonal calibration visible (the old 0-15% window crushed all ten
# Platt points into the corner). Raw bins climb to ~71% predicted while the
# observed rate stays near zero -- those run off-frame and are flagged with an
# arrow rather than shown, since plotting them would re-flatten the panel.
pB_zoom <- 3.5

pB <- ggplot(cal_bins,
             aes(x = mean_predicted * 100,
                 y = mean_observed  * 100,
                 colour = method)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", linewidth = 0.35, colour = "grey50") +
  annotate("text", x = pB_zoom * 0.96, y = pB_zoom * 0.96,
           label = "perfect calibration", angle = 45,
           hjust = 1, vjust = -0.4, size = 1.95, colour = "grey55") +
  geom_line(aes(group = method), linewidth = 0.4, show.legend = FALSE) +
  geom_point(size = 1.9, alpha = 0.9, show.legend = TRUE) +
  # Raw bins run off the top-right of the zoom window
  annotate("segment", x = 2.05, xend = 2.95, y = 0.12, yend = 0.12,
           arrow = arrow(length = unit(1.1, "mm"), type = "closed"),
           colour = "#9e9e9e", linewidth = 0.4) +
  annotate("text", x = 2.5, y = 0.34,
           label = "raw scores reach 71%\n(off scale, far below line)",
           hjust = 0.5, size = 1.85, colour = "#777777", lineheight = 0.95) +
  # Calibration error before/after Platt
  annotate("text", x = 0.1, y = pB_zoom * 0.99,
           label = sprintf("Calibration error (ECE)\nraw %.3f → Platt %.4f",
                           ECE_RAW, ECE_PLATT),
           hjust = 0, vjust = 1, size = 1.95, colour = "grey20",
           lineheight = 1.05) +
  annotate("text", x = 0.1, y = pB_zoom * 0.73,
           label = "points = deciles of predicted risk",
           hjust = 0, vjust = 1, size = 1.8, colour = "grey50",
           fontface = "italic") +
  scale_colour_manual(values = cal_pal, labels = cal_labs, name = NULL) +
  scale_x_continuous(expand = c(0.01, 0), labels = label_number(suffix = "%")) +
  scale_y_continuous(expand = c(0.01, 0), labels = label_number(suffix = "%")) +
  coord_equal(xlim = c(0, pB_zoom), ylim = c(0, pB_zoom), clip = "on") +
  theme_nat() +
  theme(legend.position = c(0.32, 0.48),
        legend.key.size = unit(3, "mm")) +
  labs(title = "B  Reliability diagram (test set)",
       x     = "Mean predicted probability",
       y     = "Observed outage rate")

# ═════════════════════════════════════════════════════════════════════════════
# 5.  Panel C - Ablation: PR-AUC by feature subset
# ═════════════════════════════════════════════════════════════════════════════

tier_pal <- c("Full model"            = "#b2182b",
              "NTL subset"            = "#d6604d",
              "No NTL (weather only)" = "#9e9e9e")

# No legend: the y-axis labels already name the feature families, and the grey
# "Weather only" bar is self-evidently the night-lights-free subset. Colour just
# reinforces it (darkest = deployed full model, grey = no NTL).
pC <- ggplot(abl_plot,
             aes(x = pr_auc_test, y = label, fill = tier)) +
  geom_col(width = 0.62, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.3f", pr_auc_test)),
            hjust = -0.18, size = 2.15, colour = "grey25") +
  scale_fill_manual(values = tier_pal) +
  scale_x_continuous(
    expand = expansion(mult = c(0, 0.16)),
    labels = label_number(accuracy = 0.01)
  ) +
  theme_nat() +
  theme(
    axis.line.y  = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title.y = element_blank()
  ) +
  labs(title = "C  Feature-family ablation",
       x     = "PR-AUC (test set)")

# ═════════════════════════════════════════════════════════════════════════════
# 6.  Panel D - Cause-stratified ROC-AUC with 95% CIs
# ═════════════════════════════════════════════════════════════════════════════
# Current verified cause ROC-AUCs (block-bootstrap CIs):
#   Environmental 0.925 [0.920, 0.929]
#   Technical     0.895 [0.886, 0.903]
#   Planned       0.894 [0.871, 0.917]
#   Other         0.886 [0.838, 0.925]
# Overall reference: 0.919 (full-test ROC-AUC of the headline classifier).

# y-floor at 0.81 so the wide "Other" CI (lower bound ~0.827) is shown in full
# rather than clipped; faded points/bars flag the small-n exploratory causes.
pD_floor <- 0.81

pD <- ggplot(cause_plot,
             aes(x = cause, y = roc_auc,
                 colour = cause, alpha = priority)) +
  geom_hline(yintercept = OVERALL_ROC,
             linetype = "dashed", linewidth = 0.4, colour = "grey50") +
  annotate("text",
           x = Inf, y = OVERALL_ROC,
           label = sprintf("Overall %.3f", OVERALL_ROC),
           hjust = 1.05, vjust = -0.5, size = 2.05, colour = "grey45") +
  annotate("text", x = -Inf, y = Inf,
           label = "faded = exploratory (small n)",
           hjust = -0.05, vjust = 1.5, size = 1.85, colour = "grey55") +
  geom_errorbar(aes(ymin = roc_auc_ci_lower, ymax = roc_auc_ci_upper),
                width = 0.14, linewidth = 0.5, show.legend = FALSE) +
  geom_point(size = 2.4, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.3f", roc_auc)),
            vjust = -1.1, size = 2.15, colour = "grey20", alpha = 1,
            show.legend = FALSE) +
  geom_text(aes(y = pD_floor + 0.008, label = paste0("n=", comma(n_positive))),
            size = 1.85, colour = "grey45", alpha = 1, show.legend = FALSE) +
  scale_colour_manual(values = cause_pal) +
  scale_alpha_manual(values = c("Primary" = 1.0, "Exploratory" = 0.42),
                     guide = "none") +
  scale_y_continuous(
    limits = c(pD_floor, 0.95),
    breaks = seq(0.82, 0.94, by = 0.02),
    labels = label_number(accuracy = 0.01),
    expand = expansion(mult = c(0, 0.04))
  ) +
  coord_cartesian(clip = "off") +
  theme_nat() +
  theme(axis.title.x = element_blank()) +
  labs(title = "D  Cause-stratified ROC-AUC (test set, 95% CI)",
       y     = "ROC-AUC")

# ═════════════════════════════════════════════════════════════════════════════
# 7.  Assemble & save
# ═════════════════════════════════════════════════════════════════════════════

fig3 <- (pA | pB) / (pC | pD) &
  theme(plot.background = element_rect(fill = "white", colour = NA))

W <- 180; H <- 140

ggsave(file.path(fig_dir, "fig3.pdf"),
       fig3, width = W, height = H, units = "mm",
       device = cairo_pdf)  # cairo: embeds fonts + renders Unicode (arrows, ×, –)

ggsave(file.path(fig_dir, "fig3.png"),
       fig3, width = W, height = H, units = "mm",
       dpi = 300)  # ragg (ggplot2 default) renders Unicode; falls back gracefully

message(sprintf("Saved  figures/fig3.pdf  and  figures/fig3.png  (%d x %d mm)", W, H))
