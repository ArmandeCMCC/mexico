# ltr_utils.R
# Pure utility functions for Learning-to-Rank pipeline
#
# IMPORTANT: This file contains ONLY function definitions.
# It MUST NOT execute any training or main logic when source()d.
#
# Functions exported:
#   - load_model_ready_data()
#   - detect_state_col()
#   - get_ltr_features()
#   - prepare_rank_data()
#   - train_ltr()
#   - predict_ltr()
#   - compute_topk_single()
#   - evaluate_topk()

suppressPackageStartupMessages({
  library(tidyverse)
  library(xgboost)
})

# DATA LOADING

#' Load model-ready data from a run tag
#'
#' @param run_tag Character, the run tag (timestamp). If NULL, uses latest.
#' @param data_dir Character, base directory for model-ready data
#' @return List with train, val, test data frames plus metadata
load_model_ready_data <- function(run_tag = NULL,
                                   data_dir = "data/processed/model_ready_anomaly") {
  if (is.null(run_tag)) {
    tag_file <- file.path(data_dir, "latest_run_tag.txt")
    if (!file.exists(tag_file)) {
      stop("No latest_run_tag.txt found in ", data_dir)
    }
    run_tag <- readLines(tag_file, warn = FALSE)[1]
  }

  data_path <- file.path(data_dir, run_tag)

  if (!dir.exists(data_path)) {
    stop("Data directory not found: ", data_path)
  }

  list(
    train = readRDS(file.path(data_path, "train.rds")),
    val = readRDS(file.path(data_path, "val.rds")),
    test = readRDS(file.path(data_path, "test.rds")),
    run_tag = run_tag,
    data_path = data_path
  )
}

# STATE DETECTION

#' Detect the state column in a dataframe
#'
#' Searches for common state column names and validates by checking
#' for ~32 unique values (Mexican states) with low missingness.
#'
#' @param df Data frame to search
#' @return Character, the detected state column name
detect_state_col <- function(df) {
  # Candidates in order of preference
  candidates <- c("state", "state_name", "state_code", "GID_1", "NAME_1")

  for (col in candidates) {
    if (col %in% names(df)) {
      n_unique <- n_distinct(df[[col]], na.rm = TRUE)
      pct_missing <- mean(is.na(df[[col]]))

      # Mexico has 32 states; allow some flexibility

      if (n_unique >= 28 && n_unique <= 35 && pct_missing < 0.05) {
        return(col)
      }
    }
  }

  # Fallback: derive from GID_2 (format: MEX.1.1_1 -> MEX.1)
  if ("GID_2" %in% names(df)) {
    message("Deriving state from GID_2 column")
    return("GID_2_derived_state")
  }

  stop("Could not detect state column. Available columns: ",
       paste(head(names(df), 20), collapse = ", "))
}

#' Add derived state column if needed
#'
#' @param df Data frame
#' @param state_col Result from detect_state_col()
#' @return Data frame with state column added if derived
add_state_col_if_needed <- function(df, state_col) {
  if (state_col == "GID_2_derived_state") {
    df$state <- sub("^(MEX\\.[0-9]+)\\.[0-9]+_[0-9]+$", "\\1", df$GID_2)
    return(list(df = df, state_col = "state"))
  }
  list(df = df, state_col = state_col)
}

# FEATURE SELECTION

#' Get feature columns for LTR (baseline + lag1 anomaly features)
#'
#' Uses scripts/00_feature_config.R for centralised feature definitions.
#' Enforces warning-safe rules: only lag1 anomaly features allowed.
#'
#' @param df Data frame to extract feature names from
#' @param mode Character, feature mode ("operational" or "diagnostic")
#' @return Character vector of feature column names
get_ltr_features <- function(df, mode = "operational") {

  # Source feature config (suppress output)
  config_env <- new.env()
  suppressMessages({
    source("scripts/00_feature_config.R", local = config_env)
  })

  all_cols <- names(df)

  # Detect anomaly features
  all_anomaly <- grep("^anomaly_", all_cols, value = TRUE)
  anomaly_lag1 <- all_anomaly[grepl("_lag1$", all_anomaly)]

  # Baseline features (from config)
  potential_features <- setdiff(all_cols, config_env$EXCLUDE_COLS)
  potential_features <- setdiff(potential_features, all_anomaly)
  potential_features <- potential_features[sapply(df[potential_features], is.numeric)]

  baseline_features <- config_env$get_feature_set(potential_features, mode = mode)
  baseline_features <- setdiff(baseline_features,
                                c("has_weather", "is_covid_period", "nb_prop_drop_z2_lag1"))

  # Combined: baseline + lag1 anomaly (warning-safe)
  combined_features <- union(baseline_features, anomaly_lag1)
  combined_features <- combined_features[sapply(df[combined_features], is.numeric)]

  combined_features
}

# DATA PREPARATION FOR XGBOOST RANKING

#' Prepare data for XGBoost ranking
#'
#' @param data Data frame with features, labels, and group column
#' @param feature_cols Character vector of feature column names
#' @param group_col Character, column name for grouping (default "date")
#' @param label_col Character, column name for labels (default "outage_3h_or_more")
#' @return List with X matrix, y vector, group sizes, and sorted data
prepare_rank_data <- function(data, feature_cols,
                               group_col = "date",
                               label_col = "outage_3h_or_more") {
  # Sort by group
  data_sorted <- data %>%
    arrange(across(all_of(group_col))) %>%
    mutate(.row_id = row_number())

  # Get group sizes
  group_info <- data_sorted %>%
    group_by(across(all_of(group_col))) %>%
    summarise(.n = n(), .groups = "drop")

  group_sizes <- group_info$.n

  # Prepare feature matrix
  X <- as.matrix(data_sorted[, feature_cols, drop = FALSE])
  X[is.na(X)] <- 0  # Replace NA with 0

  # Labels
  y <- as.numeric(data_sorted[[label_col]])

  list(
    X = X,
    y = y,
    group = group_sizes,
    data = data_sorted,
    feature_cols = feature_cols
  )
}

# MODEL TRAINING
#' Train a Learning-to-Rank model
#'
#' @param train_data Data frame for training
#' @param val_data Data frame for validation (optional, for early stopping)
#' @param feature_cols Character vector of feature column names
#' @param group_col Character, column name for grouping (default "date")
#' @param label_col Character, column name for labels (default "outage_3h_or_more")
#' @param xgb_params List of XGBoost parameters (default uses rank:pairwise)
#' @param nrounds Integer, max number of boosting rounds (default 500)
#' @param early_stopping Integer, early stopping rounds (default 50, NULL to disable)
#' @param seed Integer, random seed (default 42)
#' @param verbose Integer, verbosity level (default 1)
#' @return List with model, training metadata
train_ltr <- function(train_data, val_data = NULL, feature_cols,
                      group_col = "date",
                      label_col = "outage_3h_or_more",
                      xgb_params = NULL,
                      nrounds = 500,
                      early_stopping = 50,
                      seed = 42,
                      verbose = 1) {

  # Default XGBoost parameters for ranking
  if (is.null(xgb_params)) {
    xgb_params <- list(
      objective = "rank:pairwise",
      eval_metric = "ndcg@10",
      max_depth = 6,
      eta = 0.05,
      subsample = 0.8,
      colsample_bytree = 0.8
    )
  }

  # Prepare training data
  train_rank <- prepare_rank_data(train_data, feature_cols, group_col, label_col)

  dtrain <- xgb.DMatrix(
    data = train_rank$X,
    label = train_rank$y,
    group = train_rank$group
  )

  # Prepare validation data if provided
  watchlist <- list(train = dtrain)
  val_rank <- NULL

  if (!is.null(val_data) && nrow(val_data) > 0) {
    val_rank <- prepare_rank_data(val_data, feature_cols, group_col, label_col)
    dval <- xgb.DMatrix(
      data = val_rank$X,
      label = val_rank$y,
      group = val_rank$group
    )
    watchlist <- list(train = dtrain, val = dval)
  }

  # Train model
  set.seed(seed)
  start_time <- Sys.time()

  # Handle early stopping
  es_rounds <- if (is.null(early_stopping) || is.null(val_data)) NULL else early_stopping

  model <- xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = nrounds,
    watchlist = watchlist,
    early_stopping_rounds = es_rounds,
    verbose = verbose,
    print_every_n = ifelse(verbose > 0, 50, 0)
  )

  train_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  list(
    model = model,
    feature_cols = feature_cols,
    xgb_params = xgb_params,
    best_iteration = model$best_iteration,
    train_time_sec = train_time,
    n_train = nrow(train_data),
    n_val = if (!is.null(val_data)) nrow(val_data) else 0
  )
}

# PREDICTION

#' Predict ranking scores using a trained LTR model
#'
#' @param ltr_result Result from train_ltr() containing model
#' @param data Data frame to predict on
#' @param feature_cols Character vector (default uses model's features)
#' @param group_col Character, column name for grouping (default "date")
#' @param label_col Character, column name for labels (default "outage_3h_or_more")
#' @return Data frame with original data plus rank_score column
predict_ltr <- function(ltr_result, data, feature_cols = NULL,
                        group_col = "date",
                        label_col = "outage_3h_or_more") {

  if (is.null(feature_cols)) {
    feature_cols <- ltr_result$feature_cols
  }

  # Prepare data
  pred_rank <- prepare_rank_data(data, feature_cols, group_col, label_col)

  dpred <- xgb.DMatrix(
    data = pred_rank$X,
    label = pred_rank$y,
    group = pred_rank$group
  )

  # Predict
  pred_rank$data$rank_score <- predict(ltr_result$model, dpred)

  # Return without internal columns
  pred_rank$data %>% select(-any_of(".row_id"))
}

# TOP-K EVALUATION

#' Select top-K rows per group according to allocation policy
#'
#' @param preds Data frame with rank_score column
#' @param k Integer, number of top predictions per group
#' @param group_col Column for grouping (default "date")
#' @param state_col Column for state (for per_state/hybrid modes)
#' @param allocation "national", "per_state", "hybrid", or "hybrid_state_first"
#' @param k_min Minimum alerts per state for hybrid mode
#' @param strict_k If TRUE, drop groups with n < k (national/per_state only).
#'                 If FALSE, return up to k rows per group even if n < k.
#' @return Data frame of selected rows
select_topk_rows <- function(preds, k,
                              group_col = "date",
                              state_col = NULL,
                              allocation = "national",
                              k_min = 0,
                              strict_k = TRUE) {

  if (allocation == "national") {
    if (strict_k) {
      # Filter groups with n >= k, then take top k
      selected <- preds %>%
        group_by(across(all_of(group_col))) %>%
        filter(n() >= k) %>%
        arrange(desc(rank_score), .by_group = TRUE) %>%
        slice_head(n = k) %>%
        ungroup()
    } else {
      # Take up to k per group (may be fewer if n < k)
      selected <- preds %>%
        group_by(across(all_of(group_col))) %>%
        arrange(desc(rank_score), .by_group = TRUE) %>%
        slice_head(n = k) %>%
        ungroup()
    }

  } else if (allocation == "per_state") {
    if (is.null(state_col)) stop("state_col required for per_state allocation")

    if (strict_k) {
      # Filter (day,state) groups with n >= k, then take top k
      selected <- preds %>%
        group_by(across(all_of(c(group_col, state_col)))) %>%
        filter(n() >= k) %>%
        arrange(desc(rank_score), .by_group = TRUE) %>%
        slice_head(n = k) %>%
        ungroup()
    } else {
      # Take up to k per (day,state) group
      selected <- preds %>%
        group_by(across(all_of(c(group_col, state_col)))) %>%
        arrange(desc(rank_score), .by_group = TRUE) %>%
        slice_head(n = k) %>%
        ungroup()
    }

  } else if (allocation == "hybrid") {
    # DEPRECATED: route to hybrid_state_first
    message("allocation='hybrid' is deprecated, routing to 'hybrid_state_first'")
    return(select_topk_rows(
      preds = preds, k = k,
      group_col = group_col, state_col = state_col,
      allocation = "hybrid_state_first",
      k_min = k_min, strict_k = strict_k
    ))

  } else if (allocation == "hybrid_state_first") {
    # FEASIBLE hybrid: select top-K states by max risk, then top-1 muni per state
    # Respects K_total budget exactly
    if (is.null(state_col)) stop("state_col required for hybrid_state_first allocation")

    # Handle missing rank_score: replace NA with -Inf for stable sorting
    preds <- preds %>%
      mutate(rank_score = ifelse(is.na(rank_score), -Inf, rank_score))

    # Step 1: Compute state-level risk score (max score within state per day)
    state_risk <- preds %>%
      group_by(across(all_of(c(group_col, state_col)))) %>%
      summarise(.state_max = max(rank_score, na.rm = TRUE), .groups = "drop")

    # Step 2: Select top-K states per day
    top_states <- state_risk %>%
      group_by(across(all_of(group_col))) %>%
      arrange(desc(.state_max), .by_group = TRUE) %>%
      slice_head(n = k) %>%
      ungroup() %>%
      select(all_of(c(group_col, state_col)))

    # Step 3: Within each selected state, pick top-1 municipality
    selected <- preds %>%
      semi_join(top_states, by = c(group_col, state_col)) %>%
      group_by(across(all_of(c(group_col, state_col)))) %>%
      arrange(desc(rank_score), .by_group = TRUE) %>%
      slice_head(n = 1) %>%
      ungroup()

    # Step 4: If fewer than k states exist, fill with next-best munis nationally
    n_selected <- selected %>%
      group_by(across(all_of(group_col))) %>%
      summarise(.n_sel = n(), .groups = "drop")

    if (any(n_selected$.n_sel < k)) {
      # Find days needing fill
      need_fill <- n_selected %>% filter(.n_sel < k)

      for (i in seq_len(nrow(need_fill))) {
        day_val <- need_fill[[group_col]][i]
        n_need <- k - need_fill$.n_sel[i]

        # Get already selected GID_2s for this day
        already <- selected %>%
          filter(.data[[group_col]] == day_val) %>%
          pull(GID_2)

        # Fill from remaining
        fill_rows <- preds %>%
          filter(.data[[group_col]] == day_val, !GID_2 %in% already) %>%
          arrange(desc(rank_score)) %>%
          slice_head(n = n_need)

        selected <- bind_rows(selected, fill_rows)
      }
    }

    # HARD CHECK: verify exactly k per day (unless day has < k candidates)
    chk <- selected %>%
      group_by(across(all_of(group_col))) %>%
      summarise(.n_sel = n(), .groups = "drop")
    day_counts <- preds %>%
      group_by(across(all_of(group_col))) %>%
      summarise(.n_avail = n(), .groups = "drop")
    chk <- chk %>% left_join(day_counts, by = group_col)
    violations <- chk %>% filter(.n_sel != k & .n_avail >= k)
    if (nrow(violations) > 0) {
      stop("hybrid_state_first violated K_total on ", nrow(violations), " days")
    }

  } else {
    stop("Unknown allocation: ", allocation)
  }

  selected
}

#' Compute top-K metrics for a single K value
#'
#' @param preds Data frame with rank_score and labels
#' @param k Integer, number of top predictions per group
#' @param group_col Column for grouping
#' @param label_col Column for labels
#' @param state_col Column for state (for per_state/hybrid modes)
#' @param allocation "national", "per_state", "hybrid", or "hybrid_state_first"
#' @param k_min Minimum alerts per state for hybrid mode
#' @param strict_k If TRUE, drop groups with n < k (preserves old behavior).
#'                 If FALSE, allow groups with fewer than k observations.
#' @return Tibble with metrics
compute_topk_single <- function(preds, k,
                                 group_col = "date",
                                 label_col = "outage_3h_or_more",
                                 state_col = NULL,
                                 allocation = "national",
                                 k_min = 0,
                                 strict_k = TRUE) {

  # Select top-K rows using shared helper
selected <- select_topk_rows(preds, k, group_col, state_col,
                                allocation, k_min, strict_k)

  # Determine the evaluation universe (for base_rate and recall denominator)
  # When strict_k drops groups, metrics should reflect the filtered universe
  if (strict_k && allocation == "national") {
    days_eval <- unique(selected[[group_col]])
    preds_eval <- preds %>% filter(.data[[group_col]] %in% days_eval)
  } else if (strict_k && allocation == "per_state") {
    combos <- selected %>% distinct(across(all_of(c(group_col, state_col))))
    preds_eval <- preds %>% semi_join(combos, by = c(group_col, state_col))
  } else {
    preds_eval <- preds
  }

  # Compute metrics
  base_rate <- mean(preds_eval[[label_col]])
  total_positives <- sum(preds_eval[[label_col]])
  n_hits <- sum(selected[[label_col]])
  n_alerts <- nrow(selected)
  n_days <- n_distinct(selected[[group_col]])

  precision <- if (n_alerts > 0) n_hits / n_alerts else 0
  recall <- if (total_positives > 0) n_hits / total_positives else 0
  lift <- if (base_rate > 0) precision / base_rate else 0

  tibble(
    k = k,
    allocation = allocation,
    strict_k = strict_k,
    precision = precision,
    recall = recall,
    lift = lift,
    n_hits = n_hits,
    n_alerts = n_alerts,
    n_days = n_days,
    base_rate = base_rate
  )
}

#' Evaluate top-K metrics across multiple K values with bootstrap CIs
#'
#' @param preds Data frame with predictions (must have rank_score, label_col, group_col)
#' @param K_values Integer vector of K values to evaluate
#' @param group_col Character, column name for grouping (default "date")
#' @param label_col Character, column name for labels (default "outage_3h_or_more")
#' @param allocation Character, one of "national", "per_state", "hybrid"
#' @param K_total Integer, total alerts per day for national/hybrid modes
#' @param k_state Integer, alerts per state per day for per_state mode
#' @param k_min Integer, minimum alerts per state for hybrid mode
#' @param n_boot Integer, number of bootstrap iterations for CIs (default 1000)
#' @param ci_level Numeric, confidence level for CIs (default 0.95)
#' @param seed Integer, random seed for bootstrap (default 42)
#' @param strict_k Logical, if TRUE require k obs per group (default TRUE).
#'                 Set FALSE for LOSO on small states.
#' @return List with topk_metrics (tibble) and bootstrap_ci (tibble)
evaluate_topk <- function(preds,
                          K_values = c(1, 2, 5, 10, 20, 50),
                          group_col = "date",
                          label_col = "outage_3h_or_more",
                          allocation = "national",
                          K_total = 10,
                          k_state = 10,
                          k_min = 0,
                          n_boot = 1000,
                          ci_level = 0.95,
                          seed = 42,
                          strict_k = TRUE) {

  # Detect state column if needed
  state_col <- NULL
  if (allocation %in% c("per_state", "hybrid", "hybrid_state_first")) {
    state_col <- detect_state_col(preds)
    result <- add_state_col_if_needed(preds, state_col)
    preds <- result$df
    state_col <- result$state_col
  }

  # Determine k values based on allocation
  if (allocation == "per_state") {
    k_for_eval <- K_values  # Each is per-state k
  } else {
    k_for_eval <- K_values  # Each is total K
  }

  # Compute point estimates for all K values (pass strict_k)
  topk_metrics <- map_dfr(k_for_eval, function(k) {
    compute_topk_single(preds, k, group_col, label_col,
                        state_col, allocation, k_min, strict_k)
  })

  # Bootstrap CIs for K_total (day-level resampling)
  set.seed(seed)

  # K value for bootstrap CI
  k_boot <- if (allocation == "per_state") k_state else K_total

  # Use the SAME selection logic as point estimates via select_topk_rows()
  daily_selected <- select_topk_rows(
    preds, k_boot, group_col, state_col, allocation, k_min, strict_k
  )

  # Determine evaluation universe (same logic as compute_topk_single)
  if (strict_k && allocation == "national") {
    days_eval <- unique(daily_selected[[group_col]])
    preds_eval <- preds %>% filter(.data[[group_col]] %in% days_eval)
  } else if (strict_k && allocation == "per_state") {
    combos <- daily_selected %>% distinct(across(all_of(c(group_col, state_col))))
    preds_eval <- preds %>% semi_join(combos, by = c(group_col, state_col))
  } else {
    preds_eval <- preds
  }

  base_rate <- mean(preds_eval[[label_col]])

  # Aggregate by day for bootstrap
  daily_agg <- daily_selected %>%
    group_by(across(all_of(group_col))) %>%
    summarise(
      hits = sum(.data[[label_col]]),
      alerts = n(),
      .groups = "drop"
    )

  n_days <- nrow(daily_agg)

  if (n_days < 10) {
    # Not enough days for meaningful bootstrap
    bootstrap_ci <- tibble(
      k = k_boot,
      allocation = allocation,
      strict_k = strict_k,
      precision_mean = NA_real_,
      precision_ci_lo = NA_real_,
      precision_ci_hi = NA_real_,
      lift_mean = NA_real_,
      lift_ci_lo = NA_real_,
      lift_ci_hi = NA_real_,
      n_boot_days = n_days
    )
  } else {
    boot_results <- replicate(n_boot, {
      idx <- sample.int(n_days, size = n_days, replace = TRUE)
      sampled <- daily_agg[idx, , drop = FALSE]

      total_hits <- sum(sampled$hits)
      total_alerts <- sum(sampled$alerts)
      precision <- if (total_alerts > 0) total_hits / total_alerts else 0
      lift <- if (base_rate > 0) precision / base_rate else 0

      c(precision = precision, lift = lift)
    })

    alpha <- 1 - ci_level
    bootstrap_ci <- tibble(
      k = k_boot,
      allocation = allocation,
      strict_k = strict_k,
      precision_mean = mean(boot_results["precision", ]),
      precision_ci_lo = as.numeric(quantile(boot_results["precision", ], alpha / 2)),
      precision_ci_hi = as.numeric(quantile(boot_results["precision", ], 1 - alpha / 2)),
      lift_mean = mean(boot_results["lift", ]),
      lift_ci_lo = as.numeric(quantile(boot_results["lift", ], alpha / 2)),
      lift_ci_hi = as.numeric(quantile(boot_results["lift", ], 1 - alpha / 2)),
      n_boot_days = n_days
    )
  }

  list(
    topk_metrics = topk_metrics,
    bootstrap_ci = bootstrap_ci
  )
}

# REPORT GENERATION HELPERS

#' Generate a text evaluation report
#'
#' @param metrics_list List containing topk_metrics, bootstrap_ci, etc.
#' @param model_name Character, name of the model
#' @param run_tag Character, run identifier
#' @return Character vector of report lines
generate_eval_report <- function(metrics_list, model_name = "LTR", run_tag = "") {
  lines <- c(
    strrep("=", 72),
    sprintf("LEARNING-TO-RANK EVALUATION REPORT"),
    strrep("=", 72),
    sprintf("Model: %s", model_name),
    sprintf("Run tag: %s", run_tag),
    sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    "",
    "TOP-K METRICS",
    strrep("-", 40)
  )

  # Add topk table
  if (!is.null(metrics_list$topk_metrics)) {
    topk_str <- capture.output(
      print(metrics_list$topk_metrics %>%
              select(k, allocation, precision, lift, n_hits, n_days) %>%
              mutate(across(where(is.numeric) & !matches("k|n_"), ~round(., 4))))
    )
    lines <- c(lines, topk_str, "")
  }

  # Add bootstrap CIs
  if (!is.null(metrics_list$bootstrap_ci)) {
    lines <- c(lines,
               "BOOTSTRAP 95% CI (k=10)",
               strrep("-", 40))
    ci_str <- capture.output(
      print(metrics_list$bootstrap_ci %>%
              mutate(across(where(is.numeric), ~round(., 4))))
    )
    lines <- c(lines, ci_str, "")
  }

  lines <- c(lines, strrep("=", 72))

  lines
}

# NO MAIN EXECUTION BLOCK
# This file is pure utilities - it should never run training when source()d

message("ltr_utils.R loaded successfully (functions only, no execution)")
