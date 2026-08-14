suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
})

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))
  stop("Run from the project root or scripts directory.")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: Rscript scripts/revision/07_render_dynamic_skill_figure.R <run-id>")
}
project_dir <- detect_project_dir()
run_id <- args[[1]]
run_dir <- file.path(project_dir, "data", "revision", run_id)
figure_dir <- file.path(project_dir, "figures", "revision", run_id)
table_path <- file.path(run_dir, "dynamic_skill_table.csv")
if (!file.exists(table_path)) stop("Missing dynamic skill table: ", table_path)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

dynamic_table <- read_csv(table_path, show_col_types = FALSE)
model_ids <- c(
  "municipality_climatology", "history_only_xgb", "history_free_xgb", "full_83_xgb",
  "full_no_days_since_xgb", "full_explicit_missing_xgb"
)
model_labels <- c(
  municipality_climatology = "Municipality climatology",
  history_only_xgb = "History-only XGBoost",
  history_free_xgb = "History-free remote-sensing",
  full_83_xgb = "Full 83-feature XGBoost",
  full_no_days_since_xgb = "Full model without days-since",
  full_explicit_missing_xgb = "Full model with missing indicators"
)
model_order <- rev(unname(model_labels[model_ids]))

all_test_plot <- dynamic_table %>%
  filter(evaluation_scope == "all_test") %>%
  select(model_label, pooled_roc_auc, centered_pooled_roc_auc,
         within_macro_roc_auc, within_pair_weighted_roc_auc) %>%
  tidyr::pivot_longer(-model_label, names_to = "measure", values_to = "auc") %>%
  mutate(
    model_label = factor(model_label, levels = model_order),
    measure = recode(
      measure,
      pooled_roc_auc = "Pooled ROC-AUC",
      centered_pooled_roc_auc = "Municipality-centered pooled",
      within_macro_roc_auc = "Within-municipality macro",
      within_pair_weighted_roc_auc = "Within-municipality pair-weighted"
    )
  )

scope_order <- c(
  "All test municipality-nights", "Municipalities with both test classes",
  "Municipalities with a training outage", "No outage observed during training",
  "Ever-outage municipalities only", "Up to and including first observed outage",
  "Strictly after first observed outage"
)
history_scope_plot <- dynamic_table %>%
  filter(model_id == "full_83_xgb") %>%
  select(scope_label, pooled_roc_auc, centered_pooled_roc_auc) %>%
  tidyr::pivot_longer(-scope_label, names_to = "measure", values_to = "auc") %>%
  mutate(
    scope_label = factor(scope_label, levels = rev(scope_order)),
    measure = recode(
      measure,
      pooled_roc_auc = "Pooled ROC-AUC",
      centered_pooled_roc_auc = "Municipality-centered pooled"
    )
  )

palette_a <- c(
  "Pooled ROC-AUC" = "#0072B2", "Municipality-centered pooled" = "#D55E00",
  "Within-municipality macro" = "#009E73", "Within-municipality pair-weighted" = "#CC79A7"
)
plot_a <- ggplot(all_test_plot, aes(x = auc, y = model_label, colour = measure, shape = measure)) +
  geom_vline(xintercept = 0.5, colour = "grey65", linewidth = 0.45, linetype = "dashed") +
  geom_point(size = 2.7, position = position_dodge(width = 0.55)) +
  scale_colour_manual(values = palette_a) +
  scale_x_continuous(limits = c(0.30, 0.95), breaks = seq(0.3, 0.9, 0.1)) +
  labs(title = "A. Persistent versus within-municipality discrimination", x = "ROC-AUC", y = NULL,
       colour = NULL, shape = NULL) +
  theme_minimal(base_size = 10.5) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"), axis.text.y = element_text(colour = "black"))

palette_b <- c("Pooled ROC-AUC" = "#0072B2", "Municipality-centered pooled" = "#D55E00")
plot_b <- ggplot(history_scope_plot, aes(x = auc, y = scope_label, colour = measure, shape = measure)) +
  geom_vline(xintercept = 0.5, colour = "grey65", linewidth = 0.45, linetype = "dashed") +
  geom_point(size = 2.9, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = palette_b) +
  scale_x_continuous(limits = c(0.30, 0.95), breaks = seq(0.3, 0.9, 0.1)) +
  labs(title = "B. Headline model by outage-history status", x = "ROC-AUC", y = NULL,
       colour = NULL, shape = NULL) +
  theme_minimal(base_size = 10.5) +
  theme(legend.position = "none", panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"), axis.text.y = element_text(colour = "black"))

combined_plot <- plot_a / plot_b + plot_layout(heights = c(1, 1.05), guides = "keep")
png_path <- file.path(figure_dir, "dynamic_skill_figure.png")
pdf_path <- file.path(figure_dir, "dynamic_skill_figure.pdf")
ggsave(png_path, combined_plot, width = 10.5, height = 8.2, dpi = 320, bg = "white")
ggsave(pdf_path, combined_plot, width = 10.5, height = 8.2, device = grDevices::pdf, bg = "white")

writeLines(c(
  paste0("Run ID: ", run_id),
  paste0("Rendered at: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")),
  "Source: dynamic_skill_table.csv",
  paste0("PNG bytes: ", file.info(png_path)$size),
  paste0("PDF bytes: ", file.info(pdf_path)$size)
), file.path(run_dir, "figure_render_log.txt"))
cat("Rendered Phase 3 figure for ", run_id, "\n", sep = "")
