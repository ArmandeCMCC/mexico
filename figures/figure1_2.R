# =============================================================================
# figure1_2.R
# Combined Figure 1+2: Outage characterisation & feature landscape
# =============================================================================
#
# Layout (two rows, each row = large map | three stacked panels):
#
#   Row 1 — Outage characterisation (figure 1 content)
#     A (left, large) — Municipality outage prevalence map
#     B (top-right)   — Share of events by cause
#     C (mid-right)   — Mean outage duration distribution
#     D (bot-right)   — Monthly seasonality
#
#   Row 2 — Feature landscape (figure 2 content)
#     E (left, large) — Municipality mean NTL map
#     F (top-right)   — NTL z-score density by outage status
#     G (mid-right)   — Outage frequency by daily max. temperature bin
#     H (bot-right)   — Outage rate by municipality NTL quintile
#
# Outputs (figures/):
#   fig1_2.pdf  — vector, 180 × 220 mm
#   fig1_2.png  — 300 dpi raster
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(sf)
  library(patchwork)
  library(scales)
})

# ── Paths ─────────────────────────────────────────────────────────────────────

proj_dir <- normalizePath(
  if (basename(getwd()) == "figures") file.path(getwd(), "..") else getwd()
)
data_dir  <- "F:/.shortcut-targets-by-id/16GZLjPN8g4v5Bye7uQb14tkxc0xxUQf3/mexico/data"
feat_path <- file.path(data_dir, "model_ready", "features_engineered.rds")
fig_dir   <- file.path(proj_dir, "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ── Shared palette & theme ────────────────────────────────────────────────────

cause_pal <- c(
  "Environmental" = "#2166ac",
  "Technical"     = "#d6604d",
  "Planned"       = "#4dac26",
  "Other"         = "#999999"
)

outage_pal <- c(
  "No outage" = "#4393c3",
  "Outage"    = "#b2182b"
)

theme_nat <- function(base = 8) {
  theme_classic(base_size = base, base_family = "sans") +
    theme(
      axis.line        = element_line(linewidth = 0.3, colour = "grey30"),
      axis.ticks       = element_line(linewidth = 0.3, colour = "grey30"),
      axis.text        = element_text(size = base - 1, colour = "grey20"),
      axis.title       = element_text(size = base,     colour = "grey10"),
      legend.key.size  = unit(3,  "mm"),
      legend.text      = element_text(size = base - 1),
      legend.title     = element_text(size = base - 1, face = "bold"),
      legend.background = element_rect(fill = "white", colour = NA),
      plot.title       = element_text(size = base, face = "bold",
                                      hjust = 0, vjust = 1),
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.margin      = margin(2, 3, 2, 3, "mm")
    )
}

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Load data
# ═════════════════════════════════════════════════════════════════════════════

message("Loading municipality boundaries ...")
mex_mun <- st_read(file.path(data_dir, "gadm41_MEX.gpkg"),
                   layer = "ADM_ADM_2", quiet = TRUE) |>
  st_make_valid() |>
  st_zm(drop = TRUE) |>
  select(GID_2, GID_1, NAME_1, NAME_2)

mex_state <- mex_mun |>
  group_by(GID_1) |>
  summarise(geometry = st_union(geom), .groups = "drop") |>
  st_make_valid()

message("Loading original panel ...")
panel <- readRDS(file.path(data_dir, "panel_mex_2017_2021_ntl_ghs.rds")) |>
  select(GID_2, NAME_1, date, month, year,
         outage_3h_or_more, mean_length_min, total_length_min,
         n_outages, classification_general) |>
  mutate(
    GID_2 = as.character(GID_2),
    date  = as.Date(date),
    cause = case_when(
      classification_general == "Environmental" ~ "Environmental",
      classification_general == "Technical"     ~ "Technical",
      classification_general == "Planned"       ~ "Planned",
      TRUE                                       ~ "Other"
    ),
    cause = factor(cause, levels = names(cause_pal))
  )

message("Loading engineered features ...")
feat <- readRDS(feat_path) |>
  mutate(GID_2 = as.character(GID_2),
         date  = as.Date(date))

# Join classification_general into features (excluded as label-side in pipeline)
feat <- feat |>
  left_join(panel |> select(GID_2, date, classification_general),
            by = c("GID_2", "date")) |>
  mutate(
    cause = case_when(
      classification_general == "Environmental" ~ "Environmental",
      classification_general == "Technical"     ~ "Technical",
      classification_general == "Planned"       ~ "Planned",
      !is.na(classification_general)            ~ "Other",
      TRUE                                       ~ NA_character_
    ),
    cause  = factor(cause, levels = names(cause_pal)),
    status = if_else(outage_3h_or_more == 1L, "Outage", "No outage"),
    status = factor(status, levels = c("No outage", "Outage"))
  )

message(sprintf("Panel: %s rows | Features: %s rows",
                format(nrow(panel), big.mark = ","),
                format(nrow(feat),  big.mark = ",")))

# ═════════════════════════════════════════════════════════════════════════════
# 2.  Summaries — Row 1 (outage characterisation)
# ═════════════════════════════════════════════════════════════════════════════

outage_nights <- panel |>
  filter(outage_3h_or_more == 1L) |>
  mutate(
    duration_h  = mean_length_min / 60,
    total_dur_h = total_length_min / 60
  )

# A: prevalence map
prev_mun <- panel |>
  group_by(GID_2) |>
  summarise(
    n_nights   = n(),
    n_outage   = sum(outage_3h_or_more, na.rm = TRUE),
    prevalence = 100 * n_outage / n_nights,
    .groups    = "drop"
  )

prev_sf <- mex_mun |>
  left_join(prev_mun, by = "GID_2") |>
  mutate(
    prev_bin = cut(prevalence,
                   breaks         = c(0, 0.2, 0.5, 1.0, 2.0, 4.0, Inf),
                   labels         = c("0–0.2", "0.2–0.5", "0.5–1.0",
                                      "1.0–2.0", "2.0–4.0", ">4.0"),
                   right          = FALSE,
                   include.lowest = TRUE)
  )

# B: cause breakdown
cause_df <- outage_nights |> count(cause) |> mutate(pct = 100 * n / sum(n))

# C: duration distribution
dur_cap    <- quantile(outage_nights$duration_h, 0.99, na.rm = TRUE)
dur_median <- median(outage_nights$duration_h, na.rm = TRUE)
dur_df     <- outage_nights |>
  filter(!is.na(duration_h), duration_h > 0) |>
  mutate(duration_h = pmin(duration_h, dur_cap))

# D: monthly seasonality
month_labs <- c("Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec")
month_df <- outage_nights |>
  mutate(mon = factor(month, 1:12, month_labs)) |>
  count(mon, .drop = FALSE) |>
  mutate(pct = 100 * n / sum(n))

# ═════════════════════════════════════════════════════════════════════════════
# 3.  Summaries — Row 2 (feature landscape)
# ═════════════════════════════════════════════════════════════════════════════

# E: mean NTL map
ntl_mun <- feat |>
  group_by(GID_2) |>
  summarise(ntl_mean = mean(ntl_mean_built, na.rm = TRUE), .groups = "drop")

ntl_sf <- mex_mun |>
  left_join(ntl_mun, by = "GID_2") |>
  mutate(
    ntl_bin = cut(ntl_mean,
                  breaks         = c(0, 0.5, 1.5, 3, 6, 12, Inf),
                  labels         = c("<0.5", "0.5–1.5", "1.5–3",
                                     "3–6", "6–12", ">12"),
                  right          = FALSE,
                  include.lowest = TRUE)
  )

# F: NTL z-score density by outage status
set.seed(42)
zscore_samp <- feat |>
  filter(!is.na(ntl_mean_built_zscore), is.finite(ntl_mean_built_zscore)) |>
  mutate(zscore_cap = pmax(pmin(ntl_mean_built_zscore, 5), -5)) |>
  group_by(status) |>
  slice_sample(prop = 0.05) |>
  ungroup()

# G: outage frequency by tmax bin
tmax_df <- feat |>
  filter(!is.na(max_temp), is.finite(max_temp)) |>
  mutate(tmax_bin = cut(max_temp, breaks = seq(5, 45, by = 5),
                        include.lowest = TRUE, right = FALSE)) |>
  filter(!is.na(tmax_bin)) |>
  group_by(tmax_bin) |>
  summarise(outage_rate = mean(outage_3h_or_more, na.rm = TRUE) * 100,
            n = n(), .groups = "drop")

# H: municipality-level NTL quintile vs outage rate
mun_summary <- feat |>
  group_by(GID_2) |>
  summarise(
    mean_ntl    = mean(ntl_mean_built, na.rm = TRUE),
    outage_rate = mean(outage_3h_or_more, na.rm = TRUE) * 100,
    n_nights    = n(),
    .groups     = "drop"
  ) |>
  filter(!is.na(mean_ntl)) |>
  mutate(ntl_quintile = ntile(mean_ntl, 5))

ntl_quintile_df <- mun_summary |>
  group_by(ntl_quintile) |>
  summarise(outage_rate = weighted.mean(outage_rate, n_nights),
            n_munis = n(), .groups = "drop")

# ═════════════════════════════════════════════════════════════════════════════
# 4.  Row 1 panels
# ═════════════════════════════════════════════════════════════════════════════

prev_pal <- c("#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f")

pA <- ggplot() +
  geom_sf(data = prev_sf, aes(fill = prev_bin),
          colour = NA, linewidth = 0) +
  geom_sf(data = mex_state, fill = NA, colour = "white",  linewidth = 0.3) +
  geom_sf(data = mex_state, fill = NA, colour = "grey40", linewidth = 0.1) +
  scale_fill_manual(
    values = prev_pal, na.value = "grey88",
    name   = "Outage nights\n(% of total)", drop = FALSE,
    guide  = guide_legend(title.position = "top",
                          keyheight = unit(3, "mm"), keywidth = unit(4, "mm"),
                          ncol = 1)
  ) +
  coord_sf(expand = FALSE) +
  theme_void(base_size = 8) +
  theme(
    legend.position = c(0.14, 0.28),
    legend.text     = element_text(size = 7),
    legend.title    = element_text(size = 7, face = "bold"),
    plot.background = element_rect(fill = "white", colour = NA),
    plot.title      = element_text(size = 8, face = "bold", hjust = 0),
    plot.margin     = margin(1, 1, 1, 1, "mm")
  ) +
  labs(title = "A  Outage prevalence by municipality")

pB <- cause_df |>
  mutate(cause = fct_reorder(cause, pct)) |>
  ggplot(aes(x = pct, y = cause, fill = cause)) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.1f%%", pct)),
            hjust = -0.1, size = 2.3, colour = "grey20") +
  scale_fill_manual(values = cause_pal) +
  scale_x_continuous(limits = c(0, 92), expand = c(0, 0),
                     labels = label_percent(scale = 1, suffix = "%")) +
  theme_nat() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        axis.title.y = element_blank()) +
  labs(title = "B  Events by cause", x = "Share of outage events (%)")

pC <- ggplot(dur_df, aes(x = duration_h)) +
  geom_histogram(aes(y = after_stat(count / sum(count) * 100)),
                 binwidth = 0.5, fill = "#4393c3",
                 colour = "white", linewidth = 0.15) +
  geom_vline(xintercept = dur_median, linetype = "dashed",
             colour = "#b2182b", linewidth = 0.55) +
  annotate("text", x = dur_median + 0.25, y = Inf,
           label = sprintf("Median\n%.1f h", dur_median),
           hjust = 0, vjust = 1.25, size = 2.2, colour = "#b2182b") +
  scale_x_continuous(breaks = seq(0, floor(dur_cap) + 1, by = 2),
                     labels = label_number(suffix = " h"), expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0), labels = label_number(suffix = "%")) +
  coord_cartesian(clip = "off") +
  theme_nat() +
  labs(title = "C  Outage duration distribution",
       x = "Mean event duration", y = "Share of nights (%)")

pD_ymax <- max(month_df$pct, na.rm = TRUE) * 1.35

pD <- ggplot(month_df, aes(x = mon, y = pct, group = 1)) +
  annotate("rect", xmin = 5.5, xmax = 11.5, ymin = 0, ymax = pD_ymax,
           fill = "#fc8d59", alpha = 0.10) +
  annotate("text", x = 8.5, y = pD_ymax * 0.93,
           label = "Hurricane\nseason", size = 2.1,
           colour = "#e34a33", lineheight = 0.9) +
  geom_area(fill = "#d1e5f0", alpha = 0.7) +
  geom_line(colour = "#2166ac", linewidth = 0.65) +
  geom_point(colour = "#2166ac", size = 0.9) +
  scale_y_continuous(expand = expansion(mult = c(0, 0)),
                     limits = c(0, pD_ymax),
                     labels = label_number(suffix = "%")) +
  scale_x_discrete(expand = c(0.02, 0.02)) +
  coord_cartesian(clip = "off") +
  theme_nat() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6.5)) +
  labs(title = "D  Monthly outage frequency", x = NULL,
       y = "Share of annual events (%)")

# ═════════════════════════════════════════════════════════════════════════════
# 5.  Row 2 panels
# ═════════════════════════════════════════════════════════════════════════════

ntl_map_pal <- c("#f7f7f7", "#d9f0d3", "#a6dba0", "#5aae61", "#1b7837", "#00441b")

pE <- ggplot() +
  geom_sf(data = ntl_sf, aes(fill = ntl_bin),
          colour = NA, linewidth = 0) +
  geom_sf(data = mex_state, fill = NA, colour = "white",  linewidth = 0.3) +
  geom_sf(data = mex_state, fill = NA, colour = "grey40", linewidth = 0.1) +
  scale_fill_manual(
    values = ntl_map_pal, na.value = "grey88",
    name   = "Mean NTL\n(nW/cm²/sr)", drop = FALSE,
    guide  = guide_legend(title.position = "top",
                          keyheight = unit(3, "mm"), keywidth = unit(4, "mm"),
                          ncol = 1)
  ) +
  coord_sf(expand = FALSE) +
  theme_void(base_size = 8) +
  theme(
    legend.position = c(0.14, 0.28),
    legend.text     = element_text(size = 7),
    legend.title    = element_text(size = 7, face = "bold"),
    plot.background = element_rect(fill = "white", colour = NA),
    plot.title      = element_text(size = 8, face = "bold", hjust = 0),
    plot.margin     = margin(1, 1, 1, 1, "mm")
  ) +
  labs(title = "E  Mean NTL radiance by municipality")

pF <- ggplot(zscore_samp, aes(x = zscore_cap, colour = status, fill = status)) +
  geom_density(alpha = 0.25, linewidth = 0.55, adjust = 1.2) +
  geom_vline(xintercept = 0, linetype = "dotted",
             colour = "grey50", linewidth = 0.35) +
  scale_colour_manual(values = outage_pal, name = NULL) +
  scale_fill_manual(  values = outage_pal, name = NULL) +
  scale_x_continuous(breaks = seq(-4, 4, by = 2),
                     labels = label_number(suffix = " σ"),
                     expand = c(0.02, 0)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  coord_cartesian(clip = "off") +
  theme_nat() +
  theme(
    legend.position = c(0.78, 0.82),
    legend.key.size = unit(3, "mm"),
    axis.title.y    = element_blank(),
    axis.text.y     = element_blank(),
    axis.ticks.y    = element_blank(),
    axis.line.y     = element_blank()
  ) +
  labs(title = "F  NTL z-score by outage status",
       x     = "NTL z-score (σ from climatology)")

pG_ymax <- max(tmax_df$outage_rate, na.rm = TRUE) * 1.25

pG <- ggplot(tmax_df, aes(x = tmax_bin, y = outage_rate, group = 1)) +
  geom_col(fill = "#d6604d", alpha = 0.75, width = 0.75) +
  geom_line(colour = "#b2182b", linewidth = 0.6) +
  geom_point(colour = "#b2182b", size = 1.2) +
  scale_y_continuous(limits = c(0, pG_ymax), expand = c(0, 0),
                     labels = label_number(suffix = "%")) +
  coord_cartesian(clip = "off") +
  theme_nat() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6.5)) +
  labs(title = "G  Outage rate by max. temperature bin",
       x     = "Max. temperature bin (°C)",
       y     = "Outage rate (%)")

quintile_labs <- c("Q1\n(darkest)", "Q2", "Q3", "Q4", "Q5\n(brightest)")

pH <- ggplot(ntl_quintile_df, aes(x = ntl_quintile, y = outage_rate)) +
  geom_col(fill = "#4393c3", width = 0.65, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.2f%%", outage_rate)),
            vjust = -0.4, size = 2.3, colour = "grey20") +
  scale_x_continuous(breaks = 1:5, labels = quintile_labs, expand = c(0.07, 0)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18)),
                     labels = label_number(suffix = "%")) +
  coord_cartesian(clip = "off") +
  theme_nat() +
  labs(title = "H  Outage rate by municipality NTL level",
       x     = "Mean NTL quintile (municipality)",
       y     = "Outage rate (%)")

# ═════════════════════════════════════════════════════════════════════════════
# 6.  Assemble & save
# ═════════════════════════════════════════════════════════════════════════════

row1 <- pA + (pB / pC / pD) + plot_layout(widths = c(2.1, 1))
row2 <- pE + (pF / pG / pH) + plot_layout(widths = c(2.1, 1))

fig <- (row1 / row2) &
  theme(plot.background = element_rect(fill = "white", colour = NA))

W <- 180; H <- 220  # mm

ggsave(file.path(fig_dir, "fig1_2.pdf"),
       fig, width = W, height = H, units = "mm",
       device = cairo_pdf)

ggsave(file.path(fig_dir, "fig1_2.png"),
       fig, width = W, height = H, units = "mm",
       dpi = 300)

message(sprintf("Saved  figures/fig1_2.pdf  and  figures/fig1_2.png  (%d x %d mm)", W, H))
