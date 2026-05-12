# ML_Mexico_final.R
# PURPOSE:
#   download NASA Black Marble VNP46A2 daily nighttime lights for Mexico,
#   extract municipality-level features using GHS-BUILT-S weighting,
#   and build a panel dataset for outage prediction modeling (will probably be binary, XGBoost or Random Forest).
#
# OUTPUTS:
#   - data/ntl_features_vnp46a2_ghs/ntl_features_YYYY_MM_DD.rds (daily features)
#   - data/panel_mex_2017_2021_ntl_ghs.rds (final panel with outage labels)
#
# DATA SOURCES
#
# 1. NASA Black Marble VNP46A2 (Daily gap-filled BRDF-corrected NTL)
#    - Source: NASA LAADS DAAC 
#    - Product: VNP46A2 (VIIRS/NPP Gap-Filled Lunar BRDF-Adjusted NTL)
#    - Resolution: 15 arc-seconds (500m)
#    - Variable: Gap_Filled_DNB_BRDF-Corrected_NTL (radiance)
#    - Quality filtering: Remove pixels with Mandatory_Quality_Flag = 1 or 2
#    - Why VNP46A2? Daily temporal resolution
#
# 2. GHS-BUILT-S E2020 (Built-up surface fraction)
#    - Source: EU JRC Global Human Settlement Layer
#    - Product: GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0
#    - Resolution: 100m (Mollweide projection, reprojected to match NTL)
#    - Values: Fraction of pixel that is built-up (0-1)
#    - Purpose: Weight NTL extraction to focus on populated/electrified areas
#
# 3. GADM Mexico Administrative Boundaries
#    - Source: GADM (Francesco sent it)
#    - Level: ADM_2 (municipalities, around 2,469 units)
#
# 4. Outage Labels
#    - File: night_outages_3hrs_with_locations.csv (Francesco, Luis)
#    - Definition: Outages >= 3 hours during nighttime
#
# - The script is RESTARTABLE: it skips days that already have .rds files
#
# - Tile efficiency: Downloads around 10 tiles ONCE per day for all of Mexico,
#   then extracts features for all 2,469 municipalities in a single pass.
#   Complexity: O(tiles x days), NOT O(tiles x municipalities x days)
#
# FEATURE EXTRACTION: THREE WEIGHTING SCHEMES
#
# For each municipality, NTL statistics computed with 3 weighting schemes:
#
# 1. _all: standard zonal statistics (weight = coverage_fraction)
#    - all pixels weighted equally by overlap with municipality
#    - baseline, but diluted by dark rural pixels
#
# 2. _built_mask: binary built-up mask (weight = coverage * 1{built >= threshold})
#    - Only pixels with >= 1% built-up contribute (threshold = 0.01) => question of this threshold... We can change it later when doing ML model
#    - what is the average brightness of built-up areas?
#    - good for outage detection
#
# 3. _built_area: continuous built-up weighting (weight = coverage * built_fraction)
#    - highly built pixels contribute more than sparsely built pixels
#    - What is the density-weighted average brightness?
#
# Statistics computed for each scheme:
#   - mean: weighted mean radiance (brightness level)
#   - sum:  weighted sum (total light output, scales with area)
#   - sd:   weighted standard deviation (variability)
#   - wsum: sum of weights (effective area)
#
# NOTE: built_thr = 0.01 (1%) threshold => starting point. At 500m 
# need sensitivity analysis with higher thresholds (0.05, 0.10) for robustness

# SECTION 1: LOAD PACKAGES

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)

  library(sf)
  library(terra)
  library(exactextractr)

  library(blackmarbler)

  library(tidymodels)
  library(xgboost)


  library(slider)
})

# increase timeout for large file downloads 
options(timeout = 600)

# SECTION 2: CONFIG AND PATHS

# project directory 
project_dir <- normalizePath(if (basename(getwd()) == "scripts") file.path(getwd(), "..") else getwd())
data_dir    <- file.path(project_dir, "data")
scripts_dir <- file.path(project_dir, "scripts")

# input data paths
gadm_path <- file.path(data_dir, "gadm41_MEX.gpkg") 

outages_loc_path <- file.path(data_dir, "night_outages_3hrs_with_locations.csv")
outages_3h_path  <- file.path(data_dir, "night_outages_3hrs.csv")

ghs_built_dir <- file.path(data_dir, "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0")

# output paths
features_dir <- file.path(data_dir, "ntl_features_vnp46a2_ghs")
dir.create(features_dir, showWarnings = FALSE, recursive = TRUE)

panel_out_path <- file.path(data_dir, "panel_mex_2017_2021_ntl_ghs.rds")
model_out_path <- file.path(data_dir, "xgb_outage_model.rds")

# date range for extraction => we want 2017-2021 to match Francesco ground truth data 
date_start <- as.Date("2017-01-01")
date_end   <- as.Date("2021-12-31")
dates_all  <- seq.Date(date_start, date_end, by = "day")

# SECTION 3: NASA EARTHDATA AUTHENTICATION
# generate token => put it in .Renviron 
# you need to generate your token on NASA website, copy it and paste it like this in a .Renviron file : 
# EARTHDATA_BEARER=token (without any "")

for (p in c("~/.Renviron", file.path(project_dir, ".Renviron"))) {
  p2 <- path.expand(p)
  if (file.exists(p2)) try(readRenviron(p2), silent = TRUE)
}
bearer <- Sys.getenv("EARTHDATA_BEARER")
stopifnot(nchar(bearer) > 0)
message("Earthdata bearer token loaded (", nchar(bearer), " chars).")

# GHS-BUILT threshold for defining "built-up" pixels
# Pixels with built fraction >= this value are considered "built"
# NOTE: 0.01 (1%) is a starting point 
built_thr <- 0.01

# wget (I tried other, was not working well)
download_method <- if (nzchar(Sys.which("wget"))) "wget" else "httr"
message("Download method: ", download_method)

# SECTION 4:
# cache for Black Marble tiles and processed GHS
bm_cache_dir <- file.path(data_dir, "blackmarble_cache")
bm_h5_dir    <- file.path(bm_cache_dir, "h5")  # downloaded HDF5 tiles
dir.create(bm_h5_dir, showWarnings = FALSE, recursive = TRUE)

# terra temporary directory 
terra_tmp <- file.path(data_dir, "terra_tmp")
dir.create(terra_tmp, showWarnings = FALSE, recursive = TRUE)
terra::terraOptions(tempdir = terra_tmp, memfrac = 0.7)

# to fix a problem I had with PROJ_LIB... 
ensure_proj_db <- function() {
  if (nzchar(Sys.getenv("PROJ_LIB"))) return(invisible(TRUE))

  candidates <- c(
    "/opt/homebrew/share/proj",
    "/usr/local/share/proj",
    "/Library/Frameworks/R.framework/Resources/library/sf/proj",
    "/Library/Frameworks/R.framework/Resources/library/terra/proj"
  )

  ok <- FALSE
  for (p in candidates) {
    if (file.exists(file.path(p, "proj.db"))) {
      Sys.setenv(PROJ_LIB = p)
      message("Set PROJ_LIB = ", p)
      ok <- TRUE
      break
    }
  }
  invisible(ok)
}
ensure_proj_db()

# SECTION 5: I had some pbs with blackmarbler because of a change of structure of way of handling NASA's data 

tiles_cache_path <- file.path(bm_cache_dir, "blackmarbletiles.geojson")
dir.create(dirname(tiles_cache_path), recursive = TRUE, showWarnings = FALSE)

# tile index GeoJSON with fallback URLs
download_first_ok <- function(urls, dest) {
  for (u in urls) {
    ok <- tryCatch({
      suppressWarnings(download.file(u, destfile = dest, mode = "wb", quiet = TRUE))
      file.exists(dest) && file.info(dest)$size > 1000
    }, error = function(e) FALSE, warning = function(w) FALSE)
    if (isTRUE(ok)) return(u)
  }
  NULL
}

ensure_tiles_geojson <- function(dest = tiles_cache_path) {
  if (file.exists(dest) && file.info(dest)$size > 1000) {
    return(dest)
  }

  urls <- c(
    "https://raw.githubusercontent.com/worldbank/blackmarbler/main/data/blackmarbletiles.geojson",
    "https://raw.githubusercontent.com/worldbank/blackmarbler/master/data/blackmarbletiles.geojson",
    "https://raw.githubusercontent.com/ramarty/download_blackmarble/main/data/blackmarbletiles.geojson",
    "https://raw.githubusercontent.com/ramarty/download_blackmarble/master/data/blackmarbletiles.geojson"
  )

  u <- download_first_ok(urls, dest)
  if (is.null(u)) {
    stop(
      "Could not download blackmarbler tiles geojson.\n",
      "Tried:\n  - ", paste(urls, collapse = "\n  - "), "\n\n",
      "Fix: ensure internet access to GitHub raw OR manually download one of these URLs\n",
      "and save it to:\n  ", dest
    )
  }

  message("Cached tiles geojson from: ", u)
  dest
}

# cached geojson
patch_blackmarbler_tiles <- function(tiles_sf) {
  ns <- asNamespace("blackmarbler")
  nms <- ls(ns, all.names = TRUE)

  find_target <- function() {
    excluded <- c("file_to_raster")
    for (nm in nms) {
      if (nm %in% excluded) next
      obj <- tryCatch(get(nm, envir = ns), error = function(e) NULL)
      if (!is.function(obj)) next
      txt <- tryCatch(paste(deparse(body(obj)), collapse = "\n"), error = function(e) "")
      if (nzchar(txt) && grepl("blackmarbletiles\\.geojson", txt)) return(nm)
    }
    NULL
  }

  target <- find_target()
  if (is.null(target)) {
    message("NOTE: Could not locate blackmarbler internal tile loader to patch. Proceeding without patch.")
    return(invisible(FALSE))
  }

  ok <- tryCatch({
    unlockBinding(target, ns)
    assign(target, function(...) tiles_sf, envir = ns)
    lockBinding(target, ns)
    TRUE
  }, error = function(e) {
    message("NOTE: Failed to patch blackmarbler tile loader (", target, "): ", conditionMessage(e))
    FALSE
  })

  if (ok) message("Patched blackmarbler tile loader: ", target, " (now using local cached tiles)")
  invisible(ok)
}

# patch file_to_raster to handle new HDF5 paths
# still about HDF5 structure and NASA VNP46A2 => handle old and new way 

file_to_raster_fixed <- function(h5_file, variable, quality_flag_rm) {
  tile_i <- h5_file %>% stringr::str_extract("h\\d{2}v\\d{2}")

  # loading tile grid to get bounding box (local cache)
  bm_tiles_sf <- tryCatch({
    if (file.exists(tiles_cache_path)) {
      sf::read_sf(tiles_cache_path, quiet = TRUE)
    } else {
      sf::read_sf("https://raw.githubusercontent.com/worldbank/blackmarbler/main/data/blackmarbletiles.geojson")
    }
  }, error = function(e) {
    stop("Could not load tile grid: ", conditionMessage(e))
  })

  # getting tile bounding box
  grid_i_sf <- bm_tiles_sf[bm_tiles_sf$TileID %in% tile_i, ]
  grid_i_sf_box <- grid_i_sf %>% sf::st_bbox()
  xMin <- min(grid_i_sf_box$xmin) %>% round()
  yMin <- min(grid_i_sf_box$ymin) %>% round()
  xMax <- max(grid_i_sf_box$xmax) %>% round()
  yMax <- max(grid_i_sf_box$ymax) %>% round()

  # trying both HDF5 grid structures
  grid_structures <- c(
    "VIIRS_Grid_DNB_2d",  # new 
    "VNP_Grid_DNB"        # old 
  )

  out <- NULL
  qf <- NULL

  for (grid_struct in grid_structures) {
    var_path <- paste0('HDF5:"', h5_file, '"://HDFEOS/GRIDS/', grid_struct, '/Data_Fields/', variable)

    out <- tryCatch({
      terra::rast(var_path)
    }, error = function(e) NULL)

    if (!is.null(out) && terra::hasValues(out)) {
      # quality flag for filtering
      if (length(quality_flag_rm) > 0) {
        qf_var <- if (stringr::str_detect(h5_file, "VNP46A2")) "Mandatory_Quality_Flag" else paste0(stringr::str_replace_all(stringr::str_replace_all(variable, "_Num", ""), "_Std", ""), "_Quality")
        qf_path <- paste0('HDF5:"', h5_file, '"://HDFEOS/GRIDS/', grid_struct, '/Data_Fields/', qf_var)
        qf <- tryCatch(terra::rast(qf_path), error = function(e) NULL)
      }
      break
    }
  }

  if (is.null(out) || !terra::hasValues(out)) {
    warning("Could not read variable '", variable, "' from ", basename(h5_file))
    return(NULL)
  }

  # quality filtering (remove poor quality pixels) => 1,2
  if (!is.null(qf) && length(quality_flag_rm) > 0) {
    for (val in quality_flag_rm) {
      out[qf == val] <- NA
    }
  }

  # set coordinate reference system and extent
  terra::crs(out) <- "EPSG:4326"
  terra::ext(out) <- c(xMin, xMax, yMin, yMax)

  # remove fill values (65535 = no data in VNP46A2)
  out[out == 65535] <- NA

  return(out)
}

patch_file_to_raster <- function() {
  ns <- asNamespace("blackmarbler")

  ok <- tryCatch({
    unlockBinding("file_to_raster", ns)
    assign("file_to_raster", file_to_raster_fixed, envir = ns)
    lockBinding("file_to_raster", ns)
    TRUE
  }, error = function(e) {
    message("Failed to patch file_to_raster: ", conditionMessage(e))
    FALSE
  })

  if (ok) message("Patched blackmarbler::file_to_raster (fixed HDF5 path handling)")
  invisible(ok)
}

# patches
tiles_path <- ensure_tiles_geojson(tiles_cache_path)
tiles_sf   <- sf::st_read(tiles_path, quiet = TRUE)

patch_file_to_raster()

# SECTION 6: HDF5 FILE VALIDATION HELPERS
# helpers detect and clean up invalid files (because otherwise it seemed like HDF5 not downloaded ie NAs were downloaded...)

is_hdf5_file <- function(f) {
  if (!file.exists(f)) return(FALSE)
  if (is.na(file.info(f)$size) || file.info(f)$size < 1e6) return(FALSE)  
  # Check HDF5 magic number (first 8 bytes)
  sig <- try(readBin(f, what = "raw", n = 8), silent = TRUE)
  if (inherits(sig, "try-error") || length(sig) < 8) return(FALSE)
  identical(as.integer(sig), c(137L, 72L, 68L, 70L, 13L, 10L, 26L, 10L))
}

# converting date to day-of-year (3-digit string)
doy3 <- function(d) sprintf("%03d", as.integer(strftime(as.Date(d), "%j")))
yr4  <- function(d) format(as.Date(d), "%Y")

# find H5 files for a specific date
h5_files_for_date <- function(d, h5_dir = bm_h5_dir) {
  pat <- paste0("^VNP46A2\\.A", yr4(d), doy3(d), "\\..*\\.h5$")
  list.files(h5_dir, pattern = pat, full.names = TRUE)
}

# delete invalid H5 files for a date (re-download on next run)
delete_invalid_h5_for_date <- function(d, h5_dir = bm_h5_dir) {
  h5s <- h5_files_for_date(d, h5_dir)
  if (length(h5s) == 0) return(0L)
  invalid <- h5s[!vapply(h5s, is_hdf5_file, logical(1))]
  if (length(invalid) > 0) {
    message("  Deleting ", length(invalid), " invalid h5 files for ", as.character(d))
    unlink(invalid)
  }
  length(invalid)
}

# SECTION 7: LOAD MEXICO MUNICIPALITY BOUNDARIES

mex_mun <- st_read(gadm_path, layer = "ADM_ADM_2", quiet = TRUE) %>%
  st_make_valid() %>%
  st_zm(drop = TRUE, what = "ZM")  

# handle geometry collections (convert to polygons)
gt <- unique(as.character(st_geometry_type(mex_mun)))
if (any(gt == "GEOMETRYCOLLECTION")) {
  mex_mun <- st_collection_extract(mex_mun, "POLYGON", warn = FALSE)
}

# remove empty geometries
mex_mun <- mex_mun %>%
  filter(!st_is_empty(.))

# dissolve duplicate GID_2 (shouldn't happen, but just in case)
if (any(duplicated(mex_mun$GID_2))) {
  mex_mun <- mex_mun %>%
    group_by(GID_2) %>%
    summarise(
      across(where(~ !inherits(.x, "sfc")), dplyr::first),
      geometry = st_union(st_geometry(.)),
      .groups = "drop"
    ) %>%
    st_as_sf()
}

# ensuring consistent geometry type
mex_mun <- mex_mun %>%
  st_cast("MULTIPOLYGON", warn = FALSE) %>%
  filter(!st_is_empty(.))

# transform to WGS84 (required by blackmarbler)
mex_mun <- st_transform(mex_mun, 4326)

# clean name columns for joining 
mex_mun <- mex_mun %>%
  mutate(
    state_clean        = stringi::stri_trans_general(toupper(NAME_1), "Latin-ASCII"),
    municipality_clean = stringi::stri_trans_general(toupper(NAME_2), "Latin-ASCII")
  )

# extracting key columns (without geometry) for output
mex_key <- mex_mun %>%
  st_drop_geometry() %>%
  select(GID_2, GID_1, GID_0, COUNTRY, NAME_1, NAME_2, state_clean, municipality_clean)

# single ROI polygon for tile downloads (TILE EFFICIENCY)
# download tiles for the ENTIRE country at once, then extract per municipality.
# tile-efficient: O(tiles x days), NOT O(tiles x municipalities x days)

mex_roi <- mex_mun %>%
  summarise(geometry = st_union(st_geometry(.))) %>%
  st_make_valid()

# SECTION 8: LOAD OUTAGE LABELS

# if processed outage file doesn't exist, run preprocessing scripts
if (!file.exists(outages_loc_path)) {
  message("Missing night_outages_3hrs_with_locations.csv -> rebuilding via scripts 01/02...")
  source(file.path(scripts_dir, "01_select_night_outages.R"))
  source(file.path(scripts_dir, "02_merge_with_shapefile.R"))
}
stopifnot(file.exists(outages_loc_path))

# load and aggregate outages by municipality-date
# start_hour is the wall-clock hour (0-23) at which the outage began; used downstream
# to build earliest/latest-start-hour features. datetime_start is POSIXct in the CSV.
night_long <- readr::read_csv(outages_loc_path, show_col_types = FALSE) %>%
  mutate(
    outage_date = as.Date(substr(as.character(date), 1, 10)),
    start_hour  = lubridate::hour(datetime_start)
  )

# loading classification from the detailed file (with causes)
# file has multiple rows per outage (one per substation), so we aggregate
# to get one classification per unique outage event
outages_by_reason_path <- file.path(data_dir, "night_outages_3hrs_with_locations_clean_by_reason.csv")

# helper function to get most frequent value, returning NA if all values are NA
most_frequent <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[1]
}

if (file.exists(outages_by_reason_path)) {
  message("Loading outage classifications from night_outages_3hrs_with_locations_clean_by_reason.csv...")
  # Read the by-reason file ONCE and derive both the classification lookup and the
  # municipality-night substation/node/reason summary from the same frame.
  by_reason_raw <- readr::read_csv(outages_by_reason_path, show_col_types = FALSE) %>%
    mutate(outage_date = as.Date(substr(as.character(date), 1, 10)))

  classification_lookup <- by_reason_raw %>%
    group_by(year, date, time, length_min, state, municipality) %>%
    summarise(
      classification_general  = most_frequent(classification_general),
      classification_detailed = most_frequent(classification_detailed),
      .groups = "drop"
    )

  # joining classification to ground truth
  night_long <- night_long %>%
    left_join(classification_lookup, by = c("year", "date", "time", "length_min", "state", "municipality"))

  message("  -> Joined classifications: ", sum(!is.na(night_long$classification_general)),
          " / ", nrow(night_long), " outages have classification_general")

  # Municipality-night summary of grid-side identity.
  # clean_by_reason has one row per (outage x substation), so n_distinct gives spatial
  # severity (how many substations/nodes failed that night) and most_frequent picks the
  # dominant grid element / reason per (GID_2, outage_date).
  substation_summary <- by_reason_raw %>%
    group_by(GID_2, outage_date) %>%
    summarise(
      n_substations_affected = n_distinct(substation, na.rm = TRUE),
      n_distinct_nodes       = n_distinct(node, na.rm = TRUE),
      primary_substation     = most_frequent(substation),
      primary_node           = most_frequent(node),
      dominant_reason        = most_frequent(reason),
      .groups = "drop"
    )
  message("  -> Substation-level summary rows: ", nrow(substation_summary))
} else {
  message("NOTE: night_outages_3hrs_with_locations_clean_by_reason.csv not found.")
  message("      Panel will be built without classification columns.")
  night_long$classification_general <- NA_character_
  night_long$classification_detailed <- NA_character_
  substation_summary <- NULL
}

outage_labels <- night_long %>%
  group_by(GID_2, outage_date) %>%
  summarise(
    n_outages               = n(),  # number of events that night
    max_length_min          = max(length_min, na.rm = TRUE),
    min_length_min          = min(length_min, na.rm = TRUE),
    total_length_min        = sum(length_min, na.rm = TRUE),
    mean_length_min         = mean(length_min, na.rm = TRUE),
    median_length_min       = median(length_min, na.rm = TRUE),

    # start-hour summaries: outages beginning close to midnight tend to be longer
    # than those beginning in the early evening (exploratory finding).
    # Safe min/max: returns NA_real_ when all start_hour values are NA
    # (avoids Inf/-Inf from min(NA, na.rm=TRUE) when no observed start times).
    earliest_start_hour     = if (all(is.na(start_hour))) NA_real_ else min(start_hour, na.rm = TRUE),
    latest_start_hour       = if (all(is.na(start_hour))) NA_real_ else max(start_hour, na.rm = TRUE),

    # cause ambiguity: how many distinct causes that night, and is it mixed?
    # NOTE: these MUST come BEFORE classification_general = most_frequent(...)
    # because dplyr evaluates summarise() expressions sequentially — later
    # expressions see earlier results, not the original column.
    n_causes_general        = n_distinct(classification_general[!is.na(classification_general)]),
    is_mixed_cause_general  = as.integer(n_distinct(classification_general[!is.na(classification_general)]) > 1),

    # cause-specific counts (must also precede classification_general aggregation)
    n_outages_environmental = sum(classification_general == "Environmental", na.rm = TRUE),
    n_outages_technical     = sum(classification_general == "Technical", na.rm = TRUE),
    n_outages_planned       = sum(classification_general == "Planned", na.rm = TRUE),
    n_outages_other         = sum(classification_general == "Other", na.rm = TRUE),

    # cause-specific total durations (for cause-specific severity models)
    total_length_environmental = sum(length_min[classification_general == "Environmental"], na.rm = TRUE),
    total_length_technical     = sum(length_min[classification_general == "Technical"], na.rm = TRUE),
    total_length_planned       = sum(length_min[classification_general == "Planned"], na.rm = TRUE),
    total_length_other         = sum(length_min[classification_general == "Other"], na.rm = TRUE),

    # dominant classification for that municipality-date (most frequent if multiple outages)
    # MUST come LAST: overwrites column name, would mask original for later expressions
    classification_general  = most_frequent(classification_general),
    classification_detailed = most_frequent(classification_detailed),

    .groups = "drop"
  ) %>%
  mutate(outage_3h_or_more = 1L)  # at least one ≥3h nighttime outage

# join substation/node/reason summary (spatial severity + grid identity) if available
if (!is.null(substation_summary)) {
  outage_labels <- outage_labels %>%
    left_join(substation_summary, by = c("GID_2", "outage_date")) %>%
    mutate(
      n_substations_affected = if_else(is.na(n_substations_affected), 0L,
                                       as.integer(n_substations_affected)),
      n_distinct_nodes       = if_else(is.na(n_distinct_nodes), 0L,
                                       as.integer(n_distinct_nodes))
      # primary_substation, primary_node, dominant_reason stay NA when missing
      # (will be filled appropriately at the panel-join stage for non-outage days).
    )
} else {
  outage_labels$n_substations_affected <- NA_integer_
  outage_labels$n_distinct_nodes       <- NA_integer_
  outage_labels$primary_substation     <- NA_character_
  outage_labels$primary_node           <- NA_character_
  outage_labels$dominant_reason        <- NA_character_
}

# SECTION 9: PREPARE GHS-BUILT-S RASTER
# GHS-BUILT-S gives fraction of each pixel that is built-up (0-1).
# We use it to weight NTL extraction so that:
#   - Built-up areas (where people/electricity are) get higher weight
#   - Rural/forest pixels (always dark) are downweighted or excluded
#
# Processing steps:
#   1. Load global GHS raster (100m, Mollweide projection)
#   2. Crop to Mexico bounding box
#   3. Reproject to WGS84 (EPSG:4326) to match NTL
#   4. Resample to NTL resolution (500m)
#   5. Cache result for reuse

ghs_cache_path <- file.path(bm_cache_dir, "ghs_on_ntl_cached.tif")
mex_mun_ntl_cache <- file.path(bm_cache_dir, "mex_mun_ntl_cached.rds")

if (file.exists(ghs_cache_path) && file.exists(mex_mun_ntl_cache)) {
  message("Loading cached GHS and municipality geometries...")
  ghs_on_ntl <- terra::rast(ghs_cache_path)
  mex_mun_ntl <- readRDS(mex_mun_ntl_cache)
  message("Loaded from cache.")
} else {
  message("Building GHS cache (first run - takes a few minutes)...")

  stopifnot(dir.exists(ghs_built_dir))
  ghs_tif <- list.files(ghs_built_dir, pattern = "\\.(tif|tiff)$", full.names = TRUE, ignore.case = TRUE)
  stopifnot(length(ghs_tif) >= 1)
  if (length(ghs_tif) > 1) message("Multiple GeoTIFFs found; using first: ", basename(ghs_tif[1]))
  ghs_path <- ghs_tif[1]
  ghs_raw <- terra::rast(ghs_path)

  # create NTL template grid covering all of Mexico
  message("Creating NTL template grid for all of Mexico...")

  # getting Mexico bounding box with small buffer
  mex_bb <- st_bbox(mex_mun)
  xmin_t <- floor(mex_bb["xmin"]) - 1
  xmax_t <- ceiling(mex_bb["xmax"]) + 1
  ymin_t <- floor(mex_bb["ymin"]) - 1
  ymax_t <- ceiling(mex_bb["ymax"]) + 1

  # VNP46A2 resolution: 15 arc-seconds = 0.004166667 degrees (500m)
  ntl_res <- 0.004166667

  ntl_template <- terra::rast(
    xmin = xmin_t, xmax = xmax_t,
    ymin = ymin_t, ymax = ymax_t,
    resolution = ntl_res,
    crs = "EPSG:4326"
  )
  terra::values(ntl_template) <- 1  # dummy values

  message("Template extent: ", paste(as.vector(terra::ext(ntl_template)), collapse=", "))
  message("Template dimensions: ", paste(dim(ntl_template), collapse=" x "))

  mex_mun_ntl <- st_transform(mex_mun, terra::crs(ntl_template))

  roi_in_ghs <- st_transform(mex_mun, st_crs(terra::crs(ghs_raw)))
  bb <- st_bbox(roi_in_ghs)
  ghs_crop <- terra::crop(ghs_raw, terra::ext(bb["xmin"], bb["xmax"], bb["ymin"], bb["ymax"]))

  ghs_on_ntl <- terra::project(ghs_crop, ntl_template, method = "bilinear")
  ghs_on_ntl <- terra::resample(ghs_on_ntl, ntl_template, method = "bilinear")

  # normalisation to 0-1 range (some versions have values 0-100)
  mx <- terra::global(ghs_on_ntl, "max", na.rm = TRUE)[1, 1]
  if (is.finite(mx) && mx > 1.5) ghs_on_ntl <- ghs_on_ntl / 100
  ghs_on_ntl <- terra::clamp(ghs_on_ntl, 0, 1)

  # saing to cache
  message("Saving GHS cache...")
  terra::writeRaster(ghs_on_ntl, ghs_cache_path, overwrite = TRUE)
  saveRDS(mex_mun_ntl, mex_mun_ntl_cache)
  message("GHS cache saved.")
}

# SECTION 10: exactextractr
# summary function for exact_extract that computes statistics
# using the three weighting schemes

ntl_summary_fun <- function(values, coverage_fraction, weights) {
  if (is.data.frame(values)) values <- values[[1]]

  v  <- values            # NTL radiance values
  cf <- coverage_fraction # how much pixel overlaps municipality (0-1)
  g  <- weights           # GHS built-up fraction (0-1)

  # Scheme 1: All pixels (standard zonal stats)
  w_all <- cf

  # Scheme 2: Built-up area weighted (continuous)
  # pixels below threshold get zero weight
  w_built_area <- cf * if_else(g >= built_thr, g, 0)

  # Scheme 3: Built-up binary mask
  # pixels are either fully included (if built >= threshold) or excluded
  w_built_mask <- cf * if_else(g >= built_thr, 1.0, 0.0)

  # helper to compute weighted statistics
  wstats <- function(v, w) {
    ok <- is.finite(v) & is.finite(w) & (w > 0)
    v <- v[ok]; w <- w[ok]
    if (length(v) == 0 || sum(w) == 0) {
      return(tibble(mean = NA_real_, sum = NA_real_, sd = NA_real_, wsum = 0))
    }
    m <- sum(w * v) / sum(w)           # weighted mean
    s <- sum(w * v)                     # weighted sum
    sdv <- sqrt(sum(w * (v - m)^2) / sum(w))  # weighted SD
    tibble(mean = m, sum = s, sd = sdv, wsum = sum(w))
  }

  a <- wstats(v, w_all)
  b <- wstats(v, w_built_area)
  c <- wstats(v, w_built_mask)

  tibble(
    ntl_mean_all = a$mean,
    ntl_sum_all  = a$sum,
    ntl_sd_all   = a$sd,
    wsum_all     = a$wsum,

    ntl_mean_built = b$mean,
    ntl_sum_built  = b$sum,
    ntl_sd_built   = b$sd,
    wsum_built     = b$wsum,

    ntl_mean_built_mask = c$mean,
    ntl_sum_built_mask  = c$sum,
    ntl_sd_built_mask   = c$sd,
    wsum_built_mask     = c$wsum,

    built_share_area = if_else(a$wsum > 0, b$wsum / a$wsum, NA_real_),
    built_share_mask = if_else(a$wsum > 0, c$wsum / a$wsum, NA_real_)
  )
}

# SECTION 11: DOWNLOAD AND EXTRACTION HELPERS

# retry with exponential backoff
retry <- function(expr, times = 5, base_sleep = 2, envir = parent.frame()) {
  last <- NULL
  for (i in seq_len(times)) {
    out <- try(eval(expr, envir = envir), silent = TRUE)
    if (!inherits(out, "try-error")) return(out)
    last <- out
    Sys.sleep(base_sleep * 2^(i-1))  # exponential backoff
  }
  stop(as.character(last))
}

# template for NA features (used when extraction fails)
na_feature_row <- tibble(
  ntl_mean_all = NA_real_, ntl_sum_all = NA_real_, ntl_sd_all = NA_real_, wsum_all = 0,
  ntl_mean_built = NA_real_, ntl_sum_built = NA_real_, ntl_sd_built = NA_real_, wsum_built = 0,
  ntl_mean_built_mask = NA_real_, ntl_sum_built_mask = NA_real_, ntl_sd_built_mask = NA_real_, wsum_built_mask = 0,
  built_share_area = NA_real_, built_share_mask = NA_real_
)

# main extraction function for one day
extract_features_one_day <- function(d) {
  d <- as.Date(d, origin = "1970-01-01")

  out_file <- file.path(features_dir, paste0("ntl_features_", format(d, "%Y_%m_%d"), ".rds"))
  if (file.exists(out_file)) return(out_file)

  message("Day: ", as.character(d), "  (download NTL -> exact_extract -> save)")

  # clean up any invalid H5 files from previous failed attempts
  delete_invalid_h5_for_date(d)

  # download NTL raster for entire Mexico ROI (all tiles at once)
  ntl_obj <- tryCatch(
    retry(
      quote(
        blackmarbler::bm_raster(
          roi_sf = mex_roi,
          product_id = "VNP46A2",
          date = d,
          bearer = bearer,
          variable = NULL,  
          quality_flag_rm = c(1, 2),  # Remove poor quality pixels
          check_all_tiles_exist = FALSE,
          h5_dir = bm_h5_dir,
          download_method = download_method,
          quiet = TRUE
        )
      ),
      times = 3,
      base_sleep = 5,
      envir = environment()
    ),
    error = function(e) {
      msg <- conditionMessage(e)

      # if the tiles geojson caused this, refresh cache + patch once and retry
      if (grepl("blackmarbletiles\\.geojson", msg, ignore.case = TRUE)) {
        message("Tile-index fetch failed. Refreshing local tiles cache + patching blackmarbler...")
        try({
          tiles_path2 <- ensure_tiles_geojson(tiles_cache_path)
          tiles_sf2   <- sf::st_read(tiles_path2, quiet = TRUE)
          patch_blackmarbler_tiles(tiles_sf2)
        }, silent = TRUE)
      }

      # delete possibly corrupted h5 files and retry once more
      delete_invalid_h5_for_date(d)

      out2 <- try(
        retry(
          quote(
            blackmarbler::bm_raster(
              roi_sf = mex_roi,
              product_id = "VNP46A2",
              date = d,
              bearer = bearer,
              variable = NULL,
              quality_flag_rm = c(1, 2),
              check_all_tiles_exist = FALSE,
              h5_dir = bm_h5_dir,
              download_method = download_method,
              quiet = TRUE
            )
          ),
          times = 2,
          base_sleep = 10,
          envir = environment()
        ),
        silent = TRUE
      )
      if (!inherits(out2, "try-error")) return(out2)

      message("bm_raster failed on ", as.character(d), ": ", msg)
      NULL
    }
  )

  if (is.null(ntl_obj)) {
    # DO NOT save placeholder - skip this day and let it be retried later (at some point, was just saving placehodler)
    message("  -> SKIPPING ", as.character(d), " (download failed, will retry later)")
    return(NULL)
  }

  # CAREFUL: WE SHOULDN'T run terra::rast() on an already-valid SpatRaster => destroys values!!!
  # bm_raster returns a SpatRaster directly => converting only if necessary
  ntl <- if (inherits(ntl_obj, "SpatRaster")) {
    ntl_obj
  } else {
    terra::rast(ntl_obj)
  }
  if (terra::nlyr(ntl) > 1) ntl <- ntl[[1]]

  # check if raster has values
  if (!terra::hasValues(ntl) || all(is.na(terra::values(ntl)))) {
    # skip this day
    message("  -> SKIPPING ", as.character(d), " (no values after quality filtering)")
    return(NULL)
  }

  # aligning GHS raster to NTL grid if geometries don't match
  built_w <- if (!terra::compareGeom(ntl, ghs_on_ntl, stopOnError = FALSE)) {
    terra::resample(ghs_on_ntl, ntl, method = "bilinear")
  } else {
    ghs_on_ntl
  }

  # Extract features for ALL municipalities in ONE call (efficient!)
  stats <- tryCatch(
    exactextractr::exact_extract(
      x = ntl,
      y = mex_mun_ntl,
      weights = built_w,
      fun = ntl_summary_fun,
      progress = FALSE,
      grid_compat_tol = 0.01
    ),
    error = function(e) {
      message("exact_extract failed on ", as.character(d), ": ", conditionMessage(e))
      na_feature_row[rep(1, nrow(mex_key)), ]
    }
  )

  out <- bind_cols(mex_key, stats) %>%
    mutate(date = as.Date(d))

  saveRDS(out, out_file)
  out_file
}

# SECTION 12: EXTRACTION LOOP

run_extraction <- TRUE

if (run_extraction) {
  expected_files <- file.path(features_dir, paste0("ntl_features_", format(dates_all, "%Y_%m_%d"), ".rds"))
  missing_dates  <- dates_all[!file.exists(expected_files)]
  message("Missing days to extract: ", length(missing_dates), " / ", length(dates_all))

  if (length(missing_dates) > 0) {
    message("First missing: ", min(missing_dates), ", Last missing: ", max(missing_dates))
  }

  failed <- character(0)

  for (i in seq_along(missing_dates)) {
    d <- missing_dates[i]  # keeps Date class
    ok <- tryCatch({ extract_features_one_day(d); TRUE },
                   error = function(e) {
                     message("FAILED on ", as.character(d), " : ", conditionMessage(e))
                     FALSE
                   }
    )
    if (!ok) failed <- c(failed, as.character(d))

    # progress update every 10 days
    if (i %% 10 == 0) {
      message("Progress: ", i, "/", length(missing_dates), " days processed")
    }
  }

  if (length(failed) > 0) {
    warning("Finished with failures on ", length(failed), " days. First few: ",
            paste(head(failed, 10), collapse = ", "))
  } else {
    message("Extraction completed successfully!")
  }
}

# SECTION 13: FINAL PANEL
# combining all daily feature files and join with outage labels

feature_files <- list.files(features_dir, pattern = "^ntl_features_\\d{4}_\\d{2}_\\d{2}\\.rds$", full.names = TRUE)
if (length(feature_files) == 0) stop("No feature files were created.")

ntl_features <- purrr::map_dfr(feature_files, readRDS)

panel <- ntl_features %>%
  left_join(
    outage_labels,
    by = c("GID_2", "date" = "outage_date")
  ) %>%
  mutate(
    # NA with 0 for municipality-dates without outages
    outage_3h_or_more       = if_else(is.na(outage_3h_or_more), 0L, outage_3h_or_more),
    n_outages               = if_else(is.na(n_outages), 0L, n_outages),
    max_length_min          = if_else(is.na(max_length_min), 0, max_length_min),
    min_length_min          = if_else(is.na(min_length_min), 0, min_length_min),
    total_length_min        = if_else(is.na(total_length_min), 0, total_length_min),
    mean_length_min         = if_else(is.na(mean_length_min), 0, mean_length_min),
    median_length_min       = if_else(is.na(median_length_min), 0, median_length_min),
    # start-hour features: NA for non-outage days (correct — no event means no start).
    # Keep as NA rather than 0 to avoid leaking a sentinel value (0 = midnight) into
    # downstream feature engineering; model code should gate on outage presence.
    # cause columns: NA for non-outage days (correct — no outage = no cause)
    # classification_general, classification_detailed stay NA for non-outage days
    n_causes_general        = if_else(is.na(n_causes_general), 0L, n_causes_general),
    is_mixed_cause_general  = if_else(is.na(is_mixed_cause_general), 0L, is_mixed_cause_general),
    n_outages_environmental = if_else(is.na(n_outages_environmental), 0L, n_outages_environmental),
    n_outages_technical     = if_else(is.na(n_outages_technical), 0L, n_outages_technical),
    n_outages_planned       = if_else(is.na(n_outages_planned), 0L, n_outages_planned),
    n_outages_other         = if_else(is.na(n_outages_other), 0L, n_outages_other),
    total_length_environmental = if_else(is.na(total_length_environmental), 0, total_length_environmental),
    total_length_technical     = if_else(is.na(total_length_technical), 0, total_length_technical),
    total_length_planned       = if_else(is.na(total_length_planned), 0, total_length_planned),
    total_length_other         = if_else(is.na(total_length_other), 0, total_length_other),
    n_substations_affected  = if_else(is.na(n_substations_affected), 0L, as.integer(n_substations_affected)),
    n_distinct_nodes        = if_else(is.na(n_distinct_nodes), 0L, as.integer(n_distinct_nodes)),
    # primary_substation, primary_node, dominant_reason: stay NA for non-outage days.

    # Time features
    dow   = wday(date, label = TRUE, week_start = 1),
    month = lubridate::month(date),
    year  = lubridate::year(date)
  )

saveRDS(panel, panel_out_path)
message("Saved panel: ", panel_out_path)
