# Run probability-calibrated XGBoost ablations on timing-variant feature files.
#
# This is an additive wrapper around the existing 05b script. It does not modify
# existing modeling scripts or outputs. It passes FEATURES_PATH_OVERRIDE and
# OUTPUT_ROOT_OVERRIDE so timing outputs are isolated under:
#   data/baselines/binary_threshold/timing_overpass/<batch>/<variant_id>/

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(jsonlite)
})

detect_project_dir <- function() {
  if (basename(getwd()) == "scripts") return(normalizePath(file.path(getwd(), "..")))
  if (basename(dirname(getwd())) == "scripts") return(normalizePath(file.path(getwd(), "..", "..")))
  if (dir.exists("data") && dir.exists("scripts")) return(normalizePath(getwd()))
  normalizePath(getwd())
}

args <- commandArgs(trailingOnly = TRUE)
dry_run <- "--dry-run" %in% args
build_all <- "--all-variants" %in% args

bootstrap_n <- 100L
if ("--bootstrap-n" %in% args) {
  idx <- which(args == "--bootstrap-n")
  if (idx == length(args)) stop("--bootstrap-n requires an integer value.")
  bootstrap_n <- suppressWarnings(as.integer(args[idx + 1L]))
  if (is.na(bootstrap_n) || bootstrap_n <= 0L) stop("Invalid --bootstrap-n value.")
  args <- args[-c(idx, idx + 1L)]
}

policy_type <- "precision_floor"
policy_target <- 0.10

if ("--policy-type" %in% args) {
  idx <- which(args == "--policy-type")
  if (idx == length(args)) stop("--policy-type requires a value.")
  policy_type <- args[idx + 1L]
  args <- args[-c(idx, idx + 1L)]
}

if ("--policy-target" %in% args) {
  idx <- which(args == "--policy-target")
  if (idx == length(args)) stop("--policy-target requires a numeric value.")
  policy_target <- suppressWarnings(as.numeric(args[idx + 1L]))
  if (is.na(policy_target)) stop("Invalid --policy-target value.")
  args <- args[-c(idx, idx + 1L)]
}

if ("--ablations" %in% args) {
  idx <- which(args == "--ablations")
  if (idx == length(args)) stop("--ablations requires a comma-separated value.")
  requested_ablations <- trimws(strsplit(args[idx + 1L], ",")[[1]])
  args <- args[-c(idx, idx + 1L)]
} else {
  requested_ablations <- c("bench_rs_only", "bench_rs_history")
}

args <- setdiff(args, c("--dry-run", "--all-variants"))

project_dir <- detect_project_dir()
script_dir <- file.path(project_dir, "scripts")
data_dir <- file.path(project_dir, "data")

source(file.path(script_dir, "00_feature_config.R"))
source(file.path(script_dir, "_ablation_helpers.R"))

model_script <- file.path(script_dir, "05b_binary_threshold_eval_ablation.R")
variant_manifest_path <- file.path(data_dir, "model_ready", "timing_overpass", "labels", "overpass_variant_manifest.csv")
features_root <- file.path(data_dir, "model_ready", "timing_overpass", "features")
batch_root <- file.path(data_dir, "baselines", "binary_threshold", "timing_overpass")

required <- c(model_script, variant_manifest_path)
missing <- required[!file.exists(required)]
if (length(missing) > 0L) {
  stop("Missing required files:\n", paste0("  - ", missing, collapse = "\n"))
}

variant_manifest <- read_csv(variant_manifest_path, show_col_types = FALSE)

variants <- if (build_all) {
  variant_manifest %>%
    filter(analysis_role == "recommended") %>%
    pull(variant_id)
} else if (length(args) > 0L) {
  args
} else {
  variant_manifest %>%
    filter(buffer_minutes == 0L) %>%
    pull(variant_id)
}

invalid_variants <- setdiff(variants, variant_manifest$variant_id)
if (length(invalid_variants) > 0L) {
  stop("Unknown variant(s): ", paste(invalid_variants, collapse = ", "),
       "\nAvailable: ", paste(variant_manifest$variant_id, collapse = ", "))
}

invalid_ablations <- setdiff(requested_ablations, names(ABLATION_SPECS))
if (length(invalid_ablations) > 0L) {
  stop("Unknown ablation(s): ", paste(invalid_ablations, collapse = ", "),
       "\nAvailable: ", paste(names(ABLATION_SPECS), collapse = ", "))
}

batch_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
batch_dir <- file.path(batch_root, batch_id)
dir.create(batch_dir, recursive = TRUE, showWarnings = FALSE)
manifest_path <- file.path(batch_dir, "timing_ablation_manifest.csv")

infer_run_dir <- function(root, ablation_run_name) {
  if (!dir.exists(root)) return(NA_character_)
  dirs <- list.dirs(root, full.names = TRUE, recursive = FALSE)
  if (length(dirs) == 0L) return(NA_character_)
  matched <- dirs[grepl(paste0("_", ablation_run_name, "$"), basename(dirs))]
  if (length(matched) == 0L) return(NA_character_)
  info <- file.info(matched)
  matched[order(info$mtime, decreasing = TRUE)][[1]]
}

message(strrep("=", 72))
message("Timing overpass ablation runner")
message(strrep("=", 72))
message("Project dir: ", project_dir)
message("Batch dir:   ", batch_dir)
message("Variants:    ", paste(variants, collapse = ", "))
message("Ablations:   ", paste(requested_ablations, collapse = ", "))
message("Bootstrap N: ", bootstrap_n)
message("Policy:      ", policy_type, " = ", policy_target)
message("Dry run:     ", dry_run)

manifest <- tibble()

for (variant_id in variants) {
  features_path <- file.path(features_root, variant_id, "features_engineered.rds")
  if (!file.exists(features_path)) {
    stop("Missing timing feature file for ", variant_id, ":\n  ", features_path,
         "\nRun scripts/timing_overpass/02_build_timing_variant_features.R ", variant_id)
  }

  variant_out_root <- file.path(batch_dir, variant_id)
  dir.create(variant_out_root, recursive = TRUE, showWarnings = FALSE)

  for (abl_name in requested_ablations) {
    spec <- get_ablation_spec(abl_name)
    strict_arg <- if (spec$task_mode == "forecast") "strict" else "standard"
    ablation_run_name <- paste(variant_id, abl_name, sep = "_")

    env_vars <- list(
      ABLATION_NAME = ablation_run_name,
      REFERENCE_POLICY_TYPE = policy_type,
      REFERENCE_POLICY_TARGET = as.character(policy_target),
      BOOTSTRAP_N = as.character(bootstrap_n),
      FEATURE_INCLUDE_REGEX = spec$include_regex,
      FEATURE_EXCLUDE_REGEX = if (spec$exclude_regex == "") "" else spec$exclude_regex,
      OUTPUT_ROOT_OVERRIDE = variant_out_root,
      FEATURES_PATH_OVERRIDE = features_path
    )

    message("\n", strrep("-", 72))
    message("Variant: ", variant_id, " | Ablation: ", abl_name)
    message("Features: ", features_path)
    message("Output root: ", variant_out_root)
    message("Command: Rscript ", model_script, " ", spec$task_mode, " ", strict_arg, " ", bootstrap_n)

    start_time <- Sys.time()
    status <- "skipped_dry_run"
    run_id <- NA_character_
    output_dir <- NA_character_
    exit_code <- NA_integer_
    run_log <- file.path(variant_out_root, paste0(ablation_run_name, "_system_output.log"))

    if (!dry_run) {
      env_names <- names(env_vars)
      old_env <- Sys.getenv(env_names, unset = NA)

      result <- tryCatch({
        do.call(Sys.setenv, env_vars)
        out <- system2(
          "Rscript",
          args = c(model_script, spec$task_mode, strict_arg, as.character(bootstrap_n)),
          stdout = TRUE,
          stderr = TRUE
        )
        ec <- attr(out, "status")
        if (is.null(ec)) ec <- 0L
        list(output = out, exit_code = ec)
      }, error = function(e) {
        list(output = as.character(e), exit_code = -1L)
      })

      for (j in seq_along(env_names)) {
        if (is.na(old_env[j])) {
          Sys.unsetenv(env_names[j])
        } else {
          do.call(Sys.setenv, setNames(list(old_env[j]), env_names[j]))
        }
      }

      exit_code <- result$exit_code
      writeLines(as.character(result$output), run_log)
      if (identical(exit_code, 0L)) {
        status <- "success"
        run_line <- grep("^Run ID:", result$output, value = TRUE)
        run_id <- if (length(run_line) > 0L) {
          trimws(sub("^Run ID:", "", run_line[[1]]))
        } else {
          paste0(format(start_time, "%Y%m%d_%H%M%S"), "_", spec$task_mode, "_", strict_arg, "_", ablation_run_name)
        }
        output_dir <- file.path(variant_out_root, run_id)
      } else {
        status <- "failed"
        inferred <- infer_run_dir(variant_out_root, ablation_run_name)
        if (!is.na(inferred)) {
          output_dir <- inferred
          run_id <- basename(inferred)
        }
        message("Run failed. Last output lines:")
        message(paste(tail(result$output, 20), collapse = "\n"))
      }
    }

    end_time <- Sys.time()
    manifest <- bind_rows(
      manifest,
      tibble(
        batch_id = batch_id,
        variant_id = variant_id,
        ablation_name = abl_name,
        task_mode = spec$task_mode,
        strict_mode = strict_arg,
        features_path = features_path,
        output_root = variant_out_root,
        run_id = run_id,
        output_dir = output_dir,
        status = status,
        exit_code = exit_code,
        bootstrap_n = bootstrap_n,
        reference_policy_type = policy_type,
        reference_policy_target = policy_target,
        run_log = run_log,
        start_time = as.character(start_time),
        end_time = as.character(end_time),
        duration_min = as.numeric(difftime(end_time, start_time, units = "mins"))
      )
    )
    write_csv(manifest, manifest_path)
  }
}

message("\nManifest written: ", manifest_path)
message("Done.")
