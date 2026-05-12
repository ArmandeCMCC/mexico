# 04e_cenace_price_features.R
# Add CENACE substation-level price features to features_engineered.rds
#
# Input:
#   - data/model_ready/features_engineered.rds (current ML panel, GID_2 x date)
#   - data/Node-Prices/ml_panel_node_daily.rds  (Luis's panel, ss_node x date)
#   - data/gadm41_MEX.gpkg (for GID_2 <-> INEGI crosswalk)
#
# Output:
#   - data/model_ready/features_engineered.rds (updated with price features)
#   - data/model_ready/features_engineered_pre_prices.rds (backup)
#   - data/model_ready/gid2_agem_crosswalk.rds (reusable crosswalk)
#
# Design:
#   - All price features use LAG1 columns from Luis's panel (forecast-safe)
#   - Aggregated from ss_node to agem (municipality) via mean/max/sd
#   - Joined to our panel via GID_2 <-> agem crosswalk
#   - Municipalities without substations -> NA (XGBoost handles natively)

suppressPackageStartupMessages({
  library(data.table)
  library(sf)
})

cat("=== 04e_cenace_price_features.R ===\n")
cat("Started:", format(Sys.time()), "\n\n")

# PART 1: Build GID_2 <-> agem (INEGI) crosswalk

cat("--- Part 1: Building GID_2 <-> agem crosswalk ---\n")

# Read GADM shapefile
gadm <- st_read("data/gadm41_MEX.gpkg", layer = "ADM_ADM_2", quiet = TRUE)
gadm_df <- as.data.frame(gadm)[, c("GID_2", "GID_1", "NAME_1", "NAME_2")]

# Extract GADM state and municipality numbers
parts <- strsplit(gsub("_2$", "", gadm_df$GID_2), "[.]")
gadm_df$gadm_state <- as.integer(sapply(parts, function(x) x[2]))
gadm_df$gadm_muni  <- as.integer(sapply(parts, function(x) x[3]))

# Name normalisation: remove accents, lowercase, strip punctuation
normalize_name <- function(x) {
  x <- tolower(x)
  x <- chartr(
    "\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1",
    "aeiouunAEIOUUN", x
  )
  x <- gsub("[[:punct:]]", "", x)
  x <- gsub("[[:space:]]+", " ", trimws(x))
  x
}

# INEGI official state codes (alphabetical with historical exceptions)
inegi_states <- data.frame(
  inegi_state = sprintf("%02d", 1:32),
  inegi_name = c(
    "Aguascalientes", "Baja California", "Baja California Sur", "Campeche",
    "Coahuila de Zaragoza", "Colima", "Chiapas", "Chihuahua",
    "Ciudad de Mexico", "Durango", "Guanajuato", "Guerrero",
    "Hidalgo", "Jalisco", "Mexico", "Michoacan de Ocampo",
    "Morelos", "Nayarit", "Nuevo Leon", "Oaxaca",
    "Puebla", "Queretaro", "Quintana Roo", "San Luis Potosi",
    "Sinaloa", "Sonora", "Tabasco", "Tamaulipas",
    "Tlaxcala", "Veracruz de Ignacio de la Llave", "Yucatan", "Zacatecas"
  ),
  stringsAsFactors = FALSE
)
inegi_states$norm <- normalize_name(inegi_states$inegi_name)

# GADM state names
gadm_states <- unique(gadm_df[, c("gadm_state", "NAME_1")])
gadm_states$norm <- normalize_name(gadm_states$NAME_1)
# Fix known GADM naming: "Distrito Federal" -> "Ciudad de Mexico"
gadm_states$norm[gadm_states$norm == "distrito federal"] <- "ciudad de mexico"

# Match states: try exact first, then substring
state_map <- data.frame(gadm_state = gadm_states$gadm_state,
                        gadm_name = gadm_states$NAME_1,
                        gadm_norm = gadm_states$norm,
                        inegi_state = NA_character_,
                        stringsAsFactors = FALSE)

for (i in seq_len(nrow(state_map))) {
  gn <- state_map$gadm_norm[i]
  # Exact match
  idx <- which(inegi_states$norm == gn)
  if (length(idx) == 1) {
    state_map$inegi_state[i] <- inegi_states$inegi_state[idx]
    next
  }
  # Substring match: GADM name contained in INEGI name or vice versa
  idx <- which(grepl(gn, inegi_states$norm, fixed = TRUE) |
               sapply(inegi_states$norm, function(n) grepl(n, gn, fixed = TRUE)))
  if (length(idx) == 1) {
    state_map$inegi_state[i] <- inegi_states$inegi_state[idx]
  }
}

n_unmatched_states <- sum(is.na(state_map$inegi_state))
if (n_unmatched_states > 0) {
  cat("WARNING: Unmatched states:\n")
  print(state_map[is.na(state_map$inegi_state), c("gadm_state", "gadm_name")])
  stop("Cannot proceed with unmatched states. Fix the state mapping.")
}
cat("All 32 states matched.\n")

# Now build municipality-level crosswalk
# Strategy: within each state, match GADM municipalities to INEGI codes
# INEGI muni codes are sequential within state (001, 002, ...)
# GADM muni codes are also sequential (1, 2, ...)
# If the counts match, assume 1:1 correspondence (both alphabetical)
# If not, fall back to name matching

# Load Luis's data to get the agem codes we need to match
node_panel <- readRDS("data/Node-Prices/ml_panel_node_daily.rds")
target_agems <- sort(unique(node_panel$agem))
cat("Target agem codes to match:", length(target_agems), "\n")

# Build agem -> GID_2 mapping
# For each GADM municipality, construct the hypothetical agem code
gadm_df <- merge(gadm_df, state_map[, c("gadm_state", "inegi_state")],
                 by = "gadm_state", all.x = TRUE)
gadm_df$agem <- sprintf("%s%03d", gadm_df$inegi_state, gadm_df$gadm_muni)

# Check overlap
overlap <- intersect(gadm_df$agem, target_agems)
missing <- setdiff(target_agems, gadm_df$agem)
cat("Direct positional match:", length(overlap), "of", length(target_agems),
    "(", round(100 * length(overlap) / length(target_agems), 1), "%)\n")
cat("Missing:", length(missing), "\n")

if (length(missing) > 0) {
  cat("Attempting to resolve", length(missing), "missing codes...\n")

  # For each state with missing codes, rebuild the full mapping using
  # name-based matching between GADM and INEGI municipality names.
  # The issue: GADM uses sequential 1..N numbering while INEGI may skip codes.
  missing_by_state <- split(missing, substr(missing, 1, 2))

  for (st_code in names(missing_by_state)) {
    gadm_st <- state_map$gadm_state[state_map$inegi_state == st_code]
    gadm_munis <- gadm_df[gadm_df$gadm_state == gadm_st, ]

    # All INEGI codes for this state in Luis's data
    all_inegi_in_state <- sort(target_agems[startsWith(target_agems, st_code)])
    inegi_muni_nums <- sort(as.integer(substr(all_inegi_in_state, 3, 5)))
    gadm_muni_nums  <- sort(gadm_munis$gadm_muni)

    cat(sprintf("  State %s (%s): GADM has %d munis, INEGI target has %d codes (max: %d)\n",
                st_code,
                state_map$gadm_name[state_map$inegi_state == st_code],
                nrow(gadm_munis), length(all_inegi_in_state),
                max(inegi_muni_nums)))

    # Rebuild: if GADM and INEGI have the same count of municipalities,
    # align sorted GADM names with sorted INEGI codes.
    # If counts differ, only the positional matches (already done) hold;
    # unmatched codes are logged and skipped.
    if (nrow(gadm_munis) == length(all_inegi_in_state)) {
      gadm_sorted <- gadm_munis[order(gadm_munis$NAME_2), ]
      for (j in seq_len(nrow(gadm_sorted))) {
        idx <- which(gadm_df$GID_2 == gadm_sorted$GID_2[j])
        gadm_df$agem[idx] <- all_inegi_in_state[j]
      }
      cat("    -> Rebuilt via sorted-name matching\n")
    }
  }

  # Recheck
  overlap2 <- intersect(gadm_df$agem, target_agems)
  missing2 <- setdiff(target_agems, gadm_df$agem)
  cat("After fixes:", length(overlap2), "of", length(target_agems),
      "(", round(100 * length(overlap2) / length(target_agems), 1), "%)\n")
  if (length(missing2) > 0) {
    cat("Unresolvable codes (", length(missing2), "):",
        paste(missing2, collapse = ", "), "\n")
    cat("These municipalities will have NA price features (XGBoost handles natively).\n")
  }
}

# Build final crosswalk
crosswalk <- gadm_df[, c("GID_2", "agem")]
cat("\nCrosswalk built:", nrow(crosswalk), "rows\n")

# Save crosswalk for reuse
saveRDS(crosswalk, "data/model_ready/gid2_agem_crosswalk.rds")
cat("Saved: data/model_ready/gid2_agem_crosswalk.rds\n")

# PART 2: Aggregate price features from ss_node to municipality

cat("\n--- Part 2: Aggregating price features ---\n")

dt <- as.data.table(node_panel)

# Use only FORECAST-SAFE features (lag1/lag7 + rolling)
# Same-day prices would be leakage (prices observed same day as outage)
price_lag_cols <- c(
  "price_mean_lag1", "price_mean_lag7",
  "price_sd_lag1", "price_sd_lag7",
  "price_max_lag1", "price_max_lag7",
  "price_range_lag1", "price_range_lag7",
  "congestion_mean_lag1", "congestion_mean_lag7",
  "congestion_share_lag1", "congestion_share_lag7"
)

# Verify these columns exist
stopifnot(all(price_lag_cols %in% names(dt)))

# Also include rolling features (these use past data, forecast-safe)
rolling_cols <- c("price_roll7", "price_dev_roll7")
stopifnot(all(rolling_cols %in% names(dt)))

all_price_cols <- c(price_lag_cols, rolling_cols)

# Aggregate to agem (municipality) x date
# Multiple substations per municipality -> take mean, max, sd
cat("Aggregating", length(all_price_cols), "price columns across",
    uniqueN(dt$ss_node), "substations to", uniqueN(dt$agem), "municipalities...\n")

agg_list <- list()

# Mean across substations
agg_mean <- dt[, lapply(.SD, mean, na.rm = TRUE), by = .(agem, date), .SDcols = all_price_cols]
# Rename: prefix with cenace_
old_names <- all_price_cols
new_names_mean <- paste0("cenace_", old_names)
setnames(agg_mean, old_names, new_names_mean)

# Max congestion (captures worst bottleneck in municipality)
agg_cong_max <- dt[, .(
  cenace_congestion_max_lag1 = max(congestion_mean_lag1, na.rm = TRUE)
), by = .(agem, date)]
# Fix -Inf from all-NA groups
agg_cong_max[is.infinite(cenace_congestion_max_lag1),
             cenace_congestion_max_lag1 := NA_real_]

# SD across substations (within-municipality heterogeneity)
# Only meaningful for municipalities with >1 substation
agg_sd <- dt[, .(
  cenace_price_mean_lag1_sd = sd(price_mean_lag1, na.rm = TRUE),
  cenace_congestion_mean_lag1_sd = sd(congestion_mean_lag1, na.rm = TRUE)
), by = .(agem, date)]

# Number of substations with data (coverage indicator)
agg_n <- dt[, .(cenace_n_substations = sum(!is.na(price_mean_lag1))),
            by = .(agem, date)]

# Merge all aggregations
price_features <- merge(agg_mean, agg_cong_max, by = c("agem", "date"), all = TRUE)
price_features <- merge(price_features, agg_sd, by = c("agem", "date"), all = TRUE)
price_features <- merge(price_features, agg_n, by = c("agem", "date"), all = TRUE)

cat("Price features aggregated:", nrow(price_features), "rows,",
    ncol(price_features) - 2, "feature columns\n")

# PART 3: Join to features_engineered.rds

cat("\n--- Part 3: Joining to features_engineered.rds ---\n")

# Read current features
fe <- readRDS("data/model_ready/features_engineered.rds")
cat("Current features_engineered:", nrow(fe), "x", ncol(fe), "\n")

# Backup
cat("Saving backup: features_engineered_pre_prices.rds\n")
saveRDS(fe, "data/model_ready/features_engineered_pre_prices.rds")

fe <- as.data.table(fe)

# Add agem via crosswalk
fe <- merge(fe, as.data.table(crosswalk), by = "GID_2", all.x = TRUE)
cat("GID_2 matched to agem:", sum(!is.na(fe$agem)), "of", nrow(fe),
    "(", round(100 * mean(!is.na(fe$agem)), 1), "%)\n")

# Ensure date types match
fe[, date := as.Date(date)]
price_features[, date := as.Date(date)]

# Left join price features
n_before <- ncol(fe)
fe <- merge(fe, price_features, by = c("agem", "date"), all.x = TRUE)
n_after <- ncol(fe)
cat("Added", n_after - n_before, "price feature columns\n")

# Drop the agem column (not needed in final panel)
fe[, agem := NULL]

# Report coverage
cenace_cols <- grep("^cenace_", names(fe), value = TRUE)
cat("\nCENACE feature coverage (% non-NA):\n")
for (col in cenace_cols) {
  pct <- round(100 * mean(!is.na(fe[[col]])), 1)
  cat(sprintf("  %-40s %5.1f%%\n", col, pct))
}

# Convert back to data.frame (05b expects data.frame, not data.table)
fe <- as.data.frame(fe)

# Save
cat("\nSaving updated features_engineered.rds...\n")
saveRDS(fe, "data/model_ready/features_engineered.rds")
cat("Done:", nrow(fe), "x", ncol(fe), "\n")

cat("\n=== 04e_cenace_price_features.R complete ===\n")
cat("Finished:", format(Sys.time()), "\n")
