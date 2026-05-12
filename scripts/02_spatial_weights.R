# 02_spatial_weights.R
# PURPOSE:
#   compute spatial neighbor structure (weights matrix) for Mexican municipalities (tackle spillover btw municipalities => spatial feature)
#   STATIC preprocessing step => run once when boundaries are set
#   output used downstream in 04_feature_engineering.R for spatial lag features
#
# KEY DESIGN CHOICES:
#   1. QUEEN CONTIGUITY: municipalities sharing edge OR vertex are neighbors
#   2. KNN FALLBACK for islands: municipalities with no queen neighbors get k nearest neighbors
#   3. STABLE GID_2 ORDER: ensures consistent row/column ordering across reruns
#   4. ROW-STANDARDISED WEIGHTS: each row sums to 1 
#
# INPUT:
#   data/gadm41_MEX.gpkg (ADM2 administrative boundaries)
#
# OUTPUT:
#   data/model_ready/spatial_weights.rds containing:
#     - gid2_order: character vector of GID_2 codes in stable sort order
#     - nb: spdep neighbor list object (queen + kNN fallback for islands)
#     - listw: spdep row-standardized weights list object
#     - W_sparse: sparse matrix (dgCMatrix) for fast spatial lag computation
#     - metadata: creation info (timestamp, method, n_islands, etc.)
# cf documentation: https://r-spatial.github.io/spdep/articles/nb_sf.html

# SECTION 1: LOAD PACKAGES
message("Loading packages...")
suppressPackageStartupMessages({
  library(sf)        # spatial data structures
  library(spdep)     # spatial weights matrices
  library(tidyverse) # data manipulation
  library(Matrix)    # sparse matrix support
})

# SECTION 2: PATH HANDLING
detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") {
    return(normalizePath(file.path(getwd(), "..")))
  }
  if (dir.exists("data") || dir.exists("scripts")) {
    return(normalizePath(getwd()))
  }
  warning("Could not determine project directory. Using current directory.")
  return(normalizePath(getwd()))
}

project_dir <- detect_project_dir()
data_dir    <- file.path(project_dir, "data")
model_ready_dir <- file.path(data_dir, "model_ready")

message("Project directory: ", project_dir)

# input paths
gadm_path <- file.path(data_dir, "gadm41_MEX.gpkg")

# output paths
spatial_weights_path <- file.path(model_ready_dir, "spatial_weights.rds")

# creating output directory if needed 
if (!dir.exists(model_ready_dir)) {
  dir.create(model_ready_dir, recursive = TRUE)
  message("Created directory: ", model_ready_dir)
}

# SECTION 3: CONFIGURATION
k_nearest <- 5  # number of nearest neighbors for islands (municipalities with no contiguous neighbors)

# SECTION 4: LOAD GADM BOUNDARIES
message("\n", strrep("=", 60))
message("STEP 1: Loading GADM administrative boundaries")
message(strrep("=", 60))

if (!file.exists(gadm_path)) {
  stop("GADM file not found: ", gadm_path)
}

# read ADM2 (municipality level) polygons
muni_sf <- st_read(gadm_path, layer = "ADM_ADM_2", quiet = TRUE)

n_muni_initial <- nrow(muni_sf)
message("Loaded ", n_muni_initial, " municipalities from GADM")

# check for required columns
if (!"GID_2" %in% names(muni_sf)) {
  stop("GID_2 column not found in GADM data. Check layer name or file structure.")
}

# keeping only essential columns + geometry
muni_sf <- muni_sf %>%
  select(GID_2, NAME_2) %>%
  arrange(GID_2)  # stable sort order

# ensuring valid geometries before computing neighbors
# invalid geometries = pbs 
message("Validating geometries...")
n_invalid <- sum(!st_is_valid(muni_sf))
if (n_invalid > 0) {
  message("  Found ", n_invalid, " invalid geometries - fixing with st_make_valid()")
  muni_sf <- st_make_valid(muni_sf)
}
# ensuring consistent geometry type
muni_sf <- st_cast(muni_sf, "MULTIPOLYGON")
message("  All geometries valid and cast to MULTIPOLYGON")

# guard against st_make_valid() + st_cast() changing row count
if (nrow(muni_sf) != n_muni_initial) {
  stop("Geometry fixing changed row count (", nrow(muni_sf), " vs expected ", n_muni_initial, "). ",
       "This can break gid2_order alignment. Consider re-aggregating by GID_2.")
}
if (anyDuplicated(muni_sf$GID_2) > 0) {
  stop("Duplicate GID_2 detected after geometry fixing. Re-aggregate by GID_2.")
}

# create stable GID_2 ordering vector
gid2_order <- muni_sf$GID_2
n_muni <- length(gid2_order)

# create index mapping: GID_2 -> row number (we will use it in script 04)
gid2_index <- setNames(seq_along(gid2_order), gid2_order)

message("Stable ordering created for ", n_muni, " municipalities")
message("  (Note: GID_2 codes are character strings)")

# SECTION 5: COMPUTE QUEEN CONTIGUITY NEIGHBORS
message("\n", strrep("=", 60))
message("STEP 2: Computing queen contiguity neighbors")
message(strrep("=", 60))

# queen contiguity: shares edge OR vertex
# row.names attaches GID_2 codes for debug 
nb_queen <- poly2nb(muni_sf, queen = TRUE, snap = 1e-6, row.names = muni_sf$GID_2)

# diagnostics
n_no_neighbors <- sum(card(nb_queen) == 0)
n_with_neighbors <- n_muni - n_no_neighbors

message("Queen contiguity neighbors computed:")
message("  Municipalities with ≥1 neighbor: ", n_with_neighbors)
message("  Islands (no contiguous neighbors): ", n_no_neighbors)

if (n_no_neighbors > 0) {
  island_ids <- which(card(nb_queen) == 0)
  island_names <- muni_sf$NAME_2[island_ids]
  message("\n  Island municipalities:")
  for (i in seq_along(island_ids)) {
    message("    ", gid2_order[island_ids[i]], " (", island_names[i], ")")
  }
}

# summary statistics
neighbor_counts <- card(nb_queen)
message("\nNeighbor count summary:")
message("  Min: ", min(neighbor_counts))
message("  Median: ", median(neighbor_counts))
message("  Mean: ", sprintf("%.1f", mean(neighbor_counts)))
message("  Max: ", max(neighbor_counts))

# SECTION 6: HANDLE ISLANDS WITH KNN FALLBACK
message("\n", strrep("=", 60))
message("STEP 3: Handling islands with kNN fallback")
message(strrep("=", 60))

nb_final <- nb_queen  # start with queen neighbors

if (n_no_neighbors > 0) {
  message("Applying kNN fallback (k=", k_nearest, ") to ", n_no_neighbors, " island(s)...")

  # for kNN, we need points in a METRIC CRS and as coordinate matrix for stability
  # checking if current CRS is defined
  current_crs <- st_crs(muni_sf)
  if (is.na(current_crs)) {
    stop("CRS is missing in muni_sf. Cannot compute kNN distances without defined CRS.\n",
         "ensure input GPKG has valid CRS metadata.")
  }

  # check if current CRS is projected (metric) or geographic (lon/lat)
  is_longlat <- st_is_longlat(muni_sf)

  if (is_longlat) {
    # transform to metric CRS for distance calculation
    # EPSG:3857 (Web Mercator) is acceptable for kNN fallback (edge case only)
    # for production with many islands => would be better to use Mexico-specific UTM zones
    message("  Transforming to EPSG:3857 (Web Mercator) for metric kNN distances...")
    muni_sf_proj <- st_transform(muni_sf, 3857)
  } else {
    # already in projected/metric CRS
    muni_sf_proj <- muni_sf
  }

  # use st_point_on_surface() => fall on polygon
  # extract coordinates as numeric matrix for stable knearneigh() behavior
  pts <- st_point_on_surface(st_geometry(muni_sf_proj))
  pts_coords <- st_coordinates(pts)

  # compute k-nearest neighbors for ALL municipalities
  # will only use it for islands, but computing once is cleaner
  nb_knn <- knn2nb(knearneigh(pts_coords, k = k_nearest))

  # for each island, replacing empty neighbor list with kNN neighbors
  island_ids <- which(card(nb_queen) == 0)

  for (idx in island_ids) {
    nb_final[[idx]] <- nb_knn[[idx]]
    message("  ", gid2_order[idx], ": added ", length(nb_knn[[idx]]), " nearest neighbors")
  }

  message("kNN fallback applied successfully")
} else {
  message("No islands detected - kNN fallback not needed")
}

# check no municipality is left without neighbors
final_neighbor_counts <- card(nb_final)
if (any(final_neighbor_counts == 0)) {
  stop("FATAL: Some municipalities still have no neighbors after kNN fallback. ",
       "Increase k_nearest or check geometry.")
}

message("\nFinal neighbor count summary:")
message("  Min: ", min(final_neighbor_counts))
message("  Median: ", median(final_neighbor_counts))
message("  Mean: ", sprintf("%.1f", mean(final_neighbor_counts)))
message("  Max: ", max(final_neighbor_counts))

# SECTION 7: CREATE ROW-STANDARDIZED WEIGHTS
message("\n", strrep("=", 60))
message("STEP 4: Creating row-standardized spatial weights")
message(strrep("=", 60))

# row-standardized weights: each row sums to 1
# makes spatial lag interpretable as "average of neighbors"
# style="W" = row-standardized, zero.policy=TRUE allows asymmetric weights if needed
listw <- nb2listw(nb_final, style = "W", zero.policy = TRUE)

message("Row-standardized weights matrix (listw) created")
message("  Style: W (row-standardized)")
message("  Each municipality's neighbors sum to weight = 1")

# SECTION 8: CREATE SPARSE MATRIX REPRESENTATION
message("\n", strrep("=", 60))
message("STEP 5: Creating sparse matrix representation")
message(strrep("=", 60))

# converting to sparse matrix => fast computation
# good for large-scale spatial lag computation in 04
# try direct sparse conversion first, fall back to dense->sparse if not available
W_sparse <- tryCatch(
  {
    # attempt direct sparse matrix creation
    result <- spdep::listw2dgCMatrix(listw)
    message("Created sparse matrix directly using listw2dgCMatrix()")
    result
  },
  error = function(e) {
    # fallback: create dense then convert to sparse
    message("Direct sparse conversion not available, using dense->sparse fallback")
    W_dense <- nb2mat(nb_final, style = "W", zero.policy = TRUE)
    W_sparse_fallback <- Matrix(W_dense, sparse = TRUE)
    rm(W_dense) 
    W_sparse_fallback
  }
)

message("Sparse weights matrix created:")
message("  Dimensions: ", nrow(W_sparse), " x ", ncol(W_sparse))
message("  Non-zero elements: ", nnzero(W_sparse))
message("  Sparsity: ", sprintf("%.2f%%", 100 * (1 - nnzero(W_sparse) / (nrow(W_sparse)^2))))

# SECTION 9: SAVE OUTPUTS
message("\n", strrep("=", 60))
message("STEP 6: Saving spatial weights")
message(strrep("=", 60))

# metadata for reproducibility
metadata <- list(
  created_timestamp = Sys.time(),
  n_municipalities = n_muni,
  method = "queen_contiguity",
  knn_fallback = n_no_neighbors > 0,
  k_nearest = if (n_no_neighbors > 0) k_nearest else NA,
  n_islands = n_no_neighbors,
  island_gid2 = if (n_no_neighbors > 0) gid2_order[which(card(nb_queen) == 0)] else character(0),
  neighbor_count_min = min(final_neighbor_counts),
  neighbor_count_max = max(final_neighbor_counts),
  neighbor_count_mean = mean(final_neighbor_counts),
  weights_style = "W (row-standardized)",
  gadm_source = basename(gadm_path),
  package_versions = list(
    sf = as.character(packageVersion("sf")),
    spdep = as.character(packageVersion("spdep")),
    Matrix = as.character(packageVersion("Matrix"))
  ),
  notes = "Queen contiguity with kNN fallback for islands. Use gid2_order for consistent indexing."
)

# package everything
spatial_weights <- list(
  gid2_order = gid2_order,      # stable municipality ordering (character vector)
  gid2_index = gid2_index,      # GID_2 -> row index mapping (named integer vector)
  nb = nb_final,                 # neighbor list (class: nb)
  listw = listw,                 # row-standardized weights (class: listw)
  W_sparse = W_sparse,           # sparse matrix (class: dgCMatrix)
  metadata = metadata
)

# save
saveRDS(spatial_weights, spatial_weights_path)
message("Saved spatial weights to: ", spatial_weights_path)

# SECTION 10: FINAL SUMMARY
message("\n", strrep("=", 60))
message("SPATIAL WEIGHTS COMPUTATION COMPLETE")
message(strrep("=", 60))

message("\nOutputs saved:")
message("  ", spatial_weights_path)

message("\nContents:")
message("  gid2_order: ", length(gid2_order), " municipality codes (stable order)")
message("  gid2_index: named vector mapping GID_2 -> row index (for fast lookup)")
message("  nb: neighbor list (", class(nb_final)[1], ")")
message("  listw: spatial weights (", class(listw)[1], ", style=W)")
message("  W_sparse: ", nrow(W_sparse), "x", ncol(W_sparse), " sparse matrix (",
        sprintf("%.1f%%", 100 * nnzero(W_sparse) / (nrow(W_sparse)^2)), " filled)")
message("  metadata: creation info and diagnostics")

message("\nKey statistics:")
message("  Total municipalities: ", n_muni)
message("  Islands (kNN fallback): ", n_no_neighbors)
message("  Avg neighbors per municipality: ", sprintf("%.1f", mean(final_neighbor_counts)))
message("  Max neighbors: ", max(final_neighbor_counts))

message("\nNext steps:")
message("  1. This script creates STATIC neighbor structure")
message("  2. Use outputs in 04_feature_engineering.R to compute spatial lag features")
message("  3. Examples: nb_mean_ntl_lag1, nb_prop_drop_z2, nb_mean_temp_lag1")

message("\nScript completed at: ", Sys.time())
