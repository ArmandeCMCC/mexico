# Last Update: Probability + Calibration Pipeline for Mexico Outage Prediction

**Date:** March 2026
**Project:** Predicting Municipal Nighttime Outages (>=3h) in Mexico, 2017-2021

---

## 1. Recap: What the Project Does

We want to answer a simple question: **can satellite imagery of nighttime lights predict power outages?**

More precisely: for each of Mexico's ~2,457 municipalities, on each night between 2017 and 2021, we want to predict whether a power outage lasting 3 hours or more will occur. We do this using data that is freely and globally available — satellite nighttime light (NTL) imagery from NASA, supplemented by weather data — without requiring access to grid telemetry (SCADA) or utility company real-time feeds.

**Why this matters:** In many countries, granular outage data is unavailable, delayed, or unreliable. If satellite data can predict outages, it provides an independent monitoring tool for energy access, disaster response, and infrastructure planning — especially in data-scarce environments.

**What we use:**

| Data Source | What It Provides | Role |
|-------------|-----------------|------|
| **NASA Black Marble VNP46A2** | Daily corrected nighttime radiance at 500m resolution | Core predictive signal — brightness changes indicate outages |
| **GHS-BUILT-S** (EU JRC) | Built-up surface fraction per pixel | Weights NTL extraction toward populated areas |
| **Weather / ERA5** | Temperature, precipitation, wind, humidity, pressure | Complementary signal — weather causes outages |
| **GADM v4.1** | Municipal boundaries (~2,457 polygons) | Spatial unit for aggregation |
| **CFE outage records** | Outage start/end times, municipality, (recently) cause | Ground truth labels |

**The panel:** We construct a daily municipality-level panel: ~2,457 municipalities x ~1,817 days = **4,464,369 rows**. Each row represents one municipality on one night, with NTL features, weather, and a binary label: did a >= 3h outage happen? Only about **0.5%** of rows are positive — this is an extremely imbalanced rare-event prediction problem.

### Feature Architecture (82 features in full model)

| Family | Count | Description | Examples |
|--------|-------|-------------|---------|
| NTL lags | ~12 | Yesterday's, last week's, 2-week's brightness (3 weighting schemes x 4 horizons) | `ntl_sum_built_lag1`, `ntl_mean_built_lag7`, `ntl_sd_built_lag14` |
| NTL rolling stats | ~8 | Rolling means, SDs, day-to-day diffs | `ntl_sum_built_roll7_mean`, `ntl_sum_built_diff1` |
| NTL anomaly | ~4 | Z-scores from rolling baselines | `anomaly_sum_built_z_lag1` |
| NTL drops | ~4 | Binary flags for sudden brightness drops (z < -2) | `drop_sum_built_z2_lag1` |
| Spatial neighbors | ~6 | Neighbor-municipality averaged NTL (queen contiguity) | `nb_mean_ntl_sum_built_lag1` |
| Weather (same-day) | ~16 | Contemporaneous weather (nowcast only, excluded in forecast mode) | `rain`, `min_temp`, `wind_u` |
| Weather (lagged) | ~16 | Yesterday's/last week's weather (forecast-safe) | `atm_lag1`, `min_temp_lag7`, `rain_lag7` |
| Calendar/time | ~8 | Day of year (sin/cos encoded), holidays, weekends, month | `doy_sin`, `is_holiday`, `month_cos` |
| Quality/coverage | ~4 | NTL data availability indicators | `wsum_built_lag1`, `built_share_mask` |

### Temporal Split (Strict, No Leakage)

| Split | Period | Rows | Days | Prevalence |
|-------|--------|------|------|------------|
| **Train** | 2017-01-01 to 2019-12-31 | 2,683,044 | 1,096 | 0.59% |
| **Validation** | 2020-01-01 to 2020-06-30 | 447,174 | 182 | 0.38% |
| **Test** | 2020-07-01 to 2021-12-31 | 1,334,151 | 543 | 0.40% |

The validation set is further split 50/50 for **strict calibration-threshold separation** (see Section 3, Step 4).

### Model Configuration

- **Algorithm:** XGBoost (gradient-boosted trees)
- **Trees:** 500, max depth 6, learning rate 0.05
- **Subsampling:** 80% of rows, 80% of features per tree
- **Class imbalance:** `scale_pos_weight` = 167.2 (N_neg / N_pos)
- **Objective:** binary:logistic (probability output)
- **Training time:** ~3.4 minutes

---

## 2. Why We Moved from Top-K / LTR to Probability + Calibration

### What we had before: Top-K and Learning-to-Rank

At the last meeting, the main pipeline was:

1. **Anomaly detection** on NTL time series (z-scores, rolling baselines)
2. **Embeddings** (PCA, autoencoders) to compress NTL history
3. **XGBoost classifier** producing raw scores
4. **Learning-to-Rank (LTR)** trained with LambdaMART to directly optimize daily rankings
5. Evaluation via **Recall@K**: each day, pick the top-K municipalities and check how many actual outages we caught

This framing assumes a **fixed daily budget**: every day, a decision-maker inspects exactly K municipalities. LTR is well-suited for this because it optimizes ranking quality within each day group.

### Why we shifted

The literature on rare-event prediction and operational decision-making motivated a reframing. The key insight from the literature review is:

**When operations do not impose a fixed K** (e.g., some days you send 5 alerts, other days 20, depending on risk), **a calibrated probabilistic forecast is the better primary model** because it supports:

- **Flexible thresholds:** The decision-maker can choose any risk cutoff, not just a fixed K. A municipality is flagged when its predicted probability exceeds a threshold chosen by policy.
- **Probability-based communication:** A statement like "this municipality has a 5% chance of outage tonight" is interpretable by anyone. A ranking score of 0.73 is not.
- **Decision curves and utility analysis:** If we know the relative cost of missing an outage vs. sending a false alert, we can derive the optimal threshold directly from the calibrated probabilities.
- **Policy-based evaluation:** Instead of evaluating "did we catch X out of K?", we evaluate "under a policy of maximum 10 alerts/day, what fraction of outages do we catch, and at what precision?"

**When is Top-K / LTR still relevant?** When a fixed budget constraint is genuinely imposed — e.g., a utility can only dispatch K inspection teams per day. In that case, LTR is better aligned because it directly optimizes within-day ranking. We keep LTR as a **secondary benchmark** for this scenario.

**Key point from the literature:** Good ranking (high ROC-AUC) does not imply good probabilities. Models like XGBoost and SVMs can rank well but produce severely distorted probability estimates. Post-hoc calibration (Platt, Beta, Isotonic) is needed to turn raw scores into honest probabilities. This is why calibration is central to the new pipeline.

### Summary: two approaches, two use cases

| | Calibrated Binary Classifier (Primary) | LTR / Top-K (Secondary) |
|---|---|---|
| **Question answered** | "What is the probability of outage tonight?" | "Which K municipalities are most at risk today?" |
| **Operational assumption** | Flexible alert budget | Fixed daily budget |
| **Output** | Calibrated probability [0, 1] | Ranking score (no probability interpretation) |
| **Threshold** | Policy parameter (precision floor, alerts/day cap) | Fixed K |
| **Evaluation** | PR-AUC, calibration metrics, decision curves | Recall@K, NDCG |
| **Strengths** | Flexible, interpretable, composable | Directly optimizes ranking within daily groups |
| **When to use** | Default — unless a fixed K is imposed | When inspecting exactly K units/day |

---

## 3. How the Probability Pipeline Works (Step by Step)

### Step 1: Feature engineering

For each municipality-night, we compute features from the recent history of nighttime lights and weather:

- **NTL lags:** How bright was this municipality last night? Last week? Two weeks ago? We use the total radiance in built-up areas (`ntl_sum_built`), weighted by how built-up each pixel is, so we focus on populated/electrified zones, not forests or deserts.
- **NTL anomalies:** Is tonight's brightness abnormally low compared to the municipality's recent history? A z-score below -2 is a "brightness drop" — potentially an outage signature.
- **Spatial neighbors:** Are neighboring municipalities also dimming? A regional pattern suggests a widespread event (storm, grid failure), not a local artifact.
- **Weather:** Temperature extremes, rainfall, wind, humidity — both same-day (for diagnostic/nowcast mode) and lagged (for strict next-night forecast mode).
- **Calendar:** Day of year (seasonal patterns), holidays, weekends.

**Strict forecast rule:** In our main ("forecast") mode, we only use features available *before* the night we're predicting. This means: NTL from previous nights (lags), yesterday's weather, calendar features. We never use same-day NTL or same-day weather, because in a real operational setting, the satellite hasn't overflown yet and weather observations for tonight aren't available at decision time. This is enforced by code — the pipeline explicitly removes any feature that could cause temporal leakage.

### Step 2: Train the XGBoost classifier

We train a binary XGBoost model that takes the features and predicts P(outage >= 3h tonight).

- **Temporal split:** Train on 2017-2019, validate on first half of 2020, test on second half of 2020 + all of 2021. The model never sees future data during training. This is critical for time-series data: random cross-validation would overestimate performance because nearby days are correlated.
- **Class imbalance handling:** With only 0.5% positives, the model would naturally learn to predict "no outage" for everything. We use `scale_pos_weight = 167` (ratio of negatives to positives) to tell the model that missing a positive is 167x worse than a false alarm, ensuring it actually learns to detect outages.

### Step 3: Raw output — scores, not probabilities

XGBoost outputs a number between 0 and 1 for each municipality-night. But **this number is not a probability** — it's an internal confidence score. Empirically, XGBoost's raw scores are severely miscalibrated: when it says "10% risk", the true frequency might be 0.3% or 25%. The scores are useful for *ranking* (higher score = more likely outage) but not for *decision-making* (where you need to know actual risk levels).

This is a known property of boosted trees and SVMs in the ML literature: good ranking (ROC-AUC) does not imply good probability estimates.

### Step 4: Post-hoc calibration — turning scores into probabilities

We apply **post-hoc calibration methods** to transform raw scores into well-calibrated probabilities. "Well-calibrated" means: among all municipality-nights where the model says "5% risk", approximately 5% actually experience outages.

We test three calibration methods:

| Method | How it works | Pros | Cons |
|--------|-------------|------|------|
| **Platt scaling** (default) | Fits a logistic regression on the raw scores | Simple (2 parameters), stable, robust | Assumes sigmoid shape |
| **Beta calibration** (sensitivity) | Fits a Beta distribution transformation | More flexible than Platt, handles asymmetry | 3 parameters — slightly more risk of overfit |
| **Isotonic regression** (diagnostic) | Non-parametric monotone fit | Corrects any monotonic distortion | Can overfit with small calibration sets — use for diagnostics only |

**Strict calibration-threshold separation:** A subtle but important methodological point. If we use the same data to (a) fit the calibrator and (b) choose the operating threshold, we double-dip and may overfit. We split the validation period 50/50:
- **Jan-Mar 2020 (91 days):** Fit calibration parameters
- **Apr-Jun 2020 (91 days):** Select thresholds

This ensures honest estimates of deployment performance.

### Step 5: Threshold selection — the policy decision

A calibrated probability by itself doesn't make a binary alert decision. We need a **threshold**: "alert if probability > T." The choice of T is a **policy decision**, not a statistical optimization:

- **"We can handle at most 10 alerts per day"** → Pick T so that, on average, 10 municipalities per day exceed it. This yields T = 0.11 in our case.
- **"We need at least 10% of alerts to be true outages"** → Pick T to achieve >= 10% precision. This yields T = 0.079.
- **"Maximize overall balance between precision and recall"** → Pick T to maximize F1. This yields T = 0.057, with more alerts but more catches.

**Key insight:** The threshold is not "the model's decision" — it's the decision-maker's choice. The model provides calibrated risk estimates; the threshold converts them into alerts based on operational constraints. This is fundamentally different from top-K, where the decision rule (pick K per day) is baked into the evaluation.

### Step 6: Evaluate on held-out test period

All metrics are computed on the test set (July 2020 - December 2021), which the model never saw during training, calibration, or threshold selection.

---

## 4. Understanding the Evaluation Metrics

For a non-ML audience, here is what each metric tells us:

### Discrimination metrics: can the model tell outages apart from non-outages?

**ROC-AUC (0.879):** If we pick one random municipality-night that had an outage and one that didn't, the model assigns a higher risk to the outage night 87.9% of the time. Perfect discrimination = 1.0, random guessing = 0.5. Our 0.879 is strong — the model clearly contains real predictive signal.

*Caveat:* ROC-AUC can be misleadingly high when the positive class is very rare (0.5% here). A model that is good at ranking the 99.5% of negatives correctly can achieve high ROC-AUC even if it struggles with the 0.5% of positives.

**PR-AUC (0.056):** This is the area under the Precision-Recall curve, which focuses specifically on how well the model retrieves the rare positive class. A random classifier would achieve PR-AUC = 0.004 (equal to the prevalence). Our model achieves **0.056 — 14 times the random baseline**. This is the more informative metric for rare events, as recommended in the literature.

*Why it looks "low":* 0.056 sounds small, but in a setting where only 1 in 200 municipality-nights has an outage, it represents substantial signal. Achieving PR-AUC close to 1.0 would require near-perfect detection, which is unrealistic with satellite data alone.

### Calibration metrics: are the probabilities honest?

**Brier Score:** Mean squared error between predicted probability and actual outcome (0 or 1). Lower = better. Perfect calibration = prevalence x (1 - prevalence) ≈ 0.004. Our Platt-calibrated Brier = **0.0039**, essentially at the theoretical minimum.

**Log-loss:** Like Brier, but penalizes confident wrong predictions more harshly. Raw XGBoost: **0.30**. After Platt calibration: **0.021**. This 93% improvement shows the raw model was making very confident but wrong probability statements.

**ECE (Expected Calibration Error):** We bin municipality-nights by predicted probability, and check whether the observed outage rate matches the predicted probability in each bin. ECE = 0.001 after Platt calibration means the predictions are essentially perfectly calibrated.

**Calibration slope and intercept:** After calibration, the ideal reliability plot has intercept = 0 and slope = 1. Our Platt-calibrated model achieves intercept = -0.05, slope = 1.06 — near-perfect.

### Operational metrics: what happens when we use the model?

**Precision (12.5% at default policy):** Among all municipalities the model flags as alerts, 12.5% actually have outages. 1 in 8 alerts is a true positive. This is modest, but in a setting where outages are 0.5% of all nights, 12.5% precision represents a **25x concentration of risk** compared to random alerting.

**Recall (9.7% at default policy):** The model catches about 1 in 10 actual outages. This is limited by the alert budget (10/day) — if we allow more alerts, recall increases (up to 23.6% at ~30 alerts/day).

**Alerts per day (7.6 at default policy):** The average daily alert volume. This is the operational cost of the system.

**F1 (10.9%):** Harmonic mean of precision and recall. Useful as a single summary, but the policy-based metrics (precision at a given recall, or alerts/day) are more operationally meaningful.

---

## 5. Main Results: Strict Forecast Benchmark

**Run ID:** `20260223_125430_forecast_strict`
**Task mode:** Forecast (strict t-1 only — no same-day features at decision time)

### 5.1 Discrimination

| Metric | Validation | Test |
|--------|------------|------|
| **ROC-AUC** | 0.886 | **0.879** |
| **PR-AUC** | 0.047 | **0.056** |

The small validation-to-test gap (0.007 ROC-AUC) confirms the model is not overfitting. Test PR-AUC is actually higher than validation PR-AUC, likely because the test period (Jul 2020 - Dec 2021) has slightly higher outage prevalence and more seasonal variation (includes two hurricane seasons).

### 5.2 Calibration

| Method | Log-loss | Brier Score | ECE | Cal. Slope |
|--------|----------|-------------|-----|------------|
| **Raw XGBoost** | 0.2998 | 0.0933 | 0.196 | — |
| **Platt** (default) | **0.0212** | **0.0039** | **0.001** | 1.06 |
| **Beta** (sensitivity) | 0.0212 | 0.0039 | 0.001 | 1.10 |
| **Isotonic** (diagnostic) | 0.0316 | 0.0040 | 0.001 | 0.33* |

(*Isotonic's low slope reflects its step-function nature — it's not a meaningful slope in the same way.)

**Key insight:** Calibration does not create predictive signal. The ROC-AUC and PR-AUC are identical before and after calibration (calibration is a monotone transformation, so ranking is preserved). What calibration does is make the **probability values honest** — turning a scoring system into a forecasting system.

**Why Platt is the default:** Platt scaling is a simple 2-parameter logistic fit. It is stable, interpretable, and robust to small calibration sets. Beta calibration is a sensitivity check (more flexible, 3 parameters). Isotonic is non-parametric and can overfit in sparse-data settings — we use it as a diagnostic only.

### 5.3 Operating Points (Policy-Based Thresholds)

Using Platt-calibrated probabilities:

| Policy | Threshold | Precision | Recall | F1 | Alerts/day |
|--------|-----------|-----------|--------|----|------------|
| **Alerts/day cap 10** | 0.110 | 12.5% | 9.7% | 10.9% | 7.6 |
| Precision floor >= 10% | 0.079 | 9.8% | 16.3% | 12.2% | 16.4 |
| Alerts/day cap 20 | 0.077 | 9.7% | 17.0% | 12.3% | 17.3 |
| F1 maximizing | 0.057 | 7.6% | 23.6% | 11.5% | 30.5 |

**How to read this table:** The threshold is a policy knob. Moving it:
- **Higher** (stricter) → Fewer alerts, higher precision, lower recall. You catch fewer outages but almost every alert is real.
- **Lower** (more permissive) → More alerts, lower precision, higher recall. You catch more outages but generate more false alarms.

There is no "correct" threshold — it depends on the relative cost of missing an outage vs. acting on a false alert. The table shows the trade-off explicitly.

**Default deployment policy** (alerts/day cap = 10):
- Bootstrap 95% CI: Precision [11.5%, 13.6%], Recall [8.9%, 10.5%]
- Deployment readiness: **PASS** (precision stays above 5% floor, F1 temporal CV < 65%)

### 5.4 Top Feature Importances

| Rank | Feature | Gain | What it means |
|------|---------|------|---------------|
| 1 | `ntl_sum_built_lag1` | 12.6% | Total brightness in built areas, last night |
| 2 | `ntl_sum_built_lag7` | 9.1% | Same, one week ago |
| 3 | `ntl_sum_built_lag14` | 7.2% | Same, two weeks ago |
| 4 | `ntl_sum_built_lag2` | 5.1% | Same, two nights ago |
| 5 | `rain` | 4.6% | Rainfall |
| 6 | `nb_mean_ntl_sum_built_lag1` | 3.6% | Neighbor municipalities' brightness, last night |
| 7 | `max_dew` | 3.4% | Maximum dew point temperature |
| 8 | `min_temp` | 3.2% | Minimum temperature |

**The top 4 features are all NTL lags**, confirming that brightness dynamics over the past 1-14 days are the core predictive signal. Weather variables (rain, dew point, temperature) appear next, confirming a complementary role. The spatial neighbor feature (#6) shows that regional brightness patterns carry information beyond the individual municipality.

---

## 6. Ablation Study: What Features Actually Matter?

### Why run ablations?

Knowing that "the model works" is not enough for a scientific contribution. We need to understand **which features drive performance** and how much each adds. An ablation study trains the same model architecture with different feature subsets, keeping everything else identical (same split, same calibration, same evaluation protocol).

### Results

**Batch ID:** `20260302_122455` — 7 ablations, all strict forecast mode.

| Ablation | What's included | # Features | ROC-AUC | PR-AUC | Recall@Policy |
|----------|----------------|------------|---------|--------|---------------|
| **ntl_plus_weather** | NTL + lagged weather + calendar | 62 | 0.872 | 0.052 | **18.0%** |
| **full_model** | Everything (incl. spatial, drops) | 66 | 0.877 | 0.053 | 17.3% |
| ntl_only | NTL features only | 33 | 0.845 | 0.042 | 9.9% |
| ntl_plus_spatial | NTL + neighbor features | 30 | 0.845 | 0.042 | 9.1% |
| ntl_core | Basic NTL lags only (no rolling/anomalies) | 26 | 0.835 | 0.033 | 5.5% |
| ntl_plus_drops | NTL + brightness drop flags | 28 | 0.834 | 0.033 | 5.0% |
| weather_lagged_only | Only weather + calendar (no NTL) | 47 | 0.804 | 0.035 | 4.1% |

### Interpretation

1. **NTL is the core signal.** NTL-only (ROC-AUC 0.845) achieves ~96% of the full model's discrimination. Even the most basic NTL lag features alone (ntl_core, 26 features) achieve ROC-AUC 0.835. This validates the fundamental premise: satellite nighttime lights carry substantial information about outage risk.

2. **Weather alone is weak but complementary.** Weather-only achieves ROC-AUC 0.804 (the weakest model) and only 4.1% recall. But adding lagged weather to NTL nearly **doubles recall** (9.9% → 18.0%). Weather doesn't predict outages well on its own, but it helps the NTL model distinguish weather-driven dimming from other causes of brightness variation.

3. **Spatial neighbors add negligibly** to NTL alone. This is somewhat surprising — one might expect regional patterns to help. The likely explanation is that NTL lags already encode some spatial correlation (neighboring municipalities tend to dim together during storms), and explicit neighbor features are redundant.

4. **NTL anomaly/drop features add little** in strict next-night forecast mode. These features detect same-day brightness anomalies; in forecast mode (where we only have t-1 data), their lagged versions carry less marginal information above standard lags.

5. **ntl_plus_weather is the operational winner** (highest recall), while **full_model wins on discrimination** (highest AUC). The 4 extra features in the full model improve ranking but don't translate into more alerts caught at the default threshold.

---

## 7. Nowcast vs Forecast

### What's the difference?

- **Forecast mode** (primary): Uses only information from yesterday and earlier. This is what you'd use in a real operational setting where the satellite overpass hasn't happened yet.
- **Nowcast mode** (secondary): Also uses same-day NTL features. This is useful for detection ("did an outage happen tonight?") rather than prediction.

### Results

| Mode | Features | ROC-AUC | PR-AUC | Recall |
|------|----------|---------|--------|--------|
| **Forecast** (strict t-1) | 66 | 0.877 | 0.053 | 17.3% |
| **Nowcast** (same-day) | 83 | 0.891 | 0.058 | 18.3% |

Same-day NTL features provide a **modest uplift** (+1.4pp ROC-AUC, +1pp recall). The forecast model captures most of the signal without requiring same-day data. The nowcast serves as an **upper bound** on achievable performance with real-time satellite data.

**Implication:** The predictive signal is mostly in the *history* of brightness, not in tonight's observation. This is a positive finding for operational deployment, since it means we can issue forecasts before the night begins.

---

## 8. Robustness and Sensitivity Tests

### 8.1 Texas Cold Snap (February 15-17, 2021)

**What happened:** In February 2021, a severe winter storm (Uri) caused catastrophic power failures in Texas and cascading outages in northern Mexico (Chihuahua, Nuevo León, Tamaulipas). This was an extreme, unprecedented event.

**Concern:** Does this extreme event drive our test-period results? If removing 3 days dramatically changes the metrics, our conclusions would rest on a single shock.

**Test:** We re-evaluate the model on the full test period minus Feb 15-17, 2021 (removing 7,371 rows, 0.55% of the test set).

| Model | Full Test ROC-AUC | Excl. Texas ROC-AUC | Change |
|-------|-------------------|---------------------|--------|
| ntl_plus_weather | 0.872 | 0.879 | **+0.6pp** |
| full_model | 0.877 | 0.883 | **+0.6pp** |

**Conclusion:** The model actually performs **slightly better** without the Texas storm dates. This means the extreme event is a small perturbation — it adds noise (difficult-to-predict unprecedented conditions), but is not inflating performance. Main conclusions and model rankings are unaffected.

### 8.2 Spatial Holdout: Southern Mexico (Guerrero, Oaxaca, Chiapas)

**What is a spatial holdout?** A standard temporal split tests generalization to future time periods. A spatial holdout tests generalization to **unseen geographic regions** — municipalities the model has never trained on.

**Why Guerrero, Oaxaca, Chiapas?** These three states in southern Mexico were chosen because they represent a **challenging test case**:
- They are among the most rural and poorest states in Mexico
- They have different grid infrastructure and outage patterns compared to the industrial north
- They include areas prone to hurricanes (Pacific coast), earthquakes (Oaxaca), and limited electrification
- Together they represent a substantial geographic block (~417,000 test rows)
- Excluding them from training is a meaningful test — the model must generalize from the rest of Mexico to a region with different characteristics

**Test design:** The model is trained and calibrated on all of Mexico *except* these three states. The test set consists *only* of these three states.

| Metric | Full Test (all states) | Spatial Holdout (South only) | Retention |
|--------|-----------|-----------------|-----------|
| ROC-AUC | 0.872 | 0.859 | **98.5%** |
| PR-AUC | 0.052 | 0.016 | 31.3% |
| Recall@Policy | 18.0% | 1.8% | 9.7% |
| Prevalence | 0.40% | 0.23% | — |

**Interpretation (nuanced — do not overclaim):**

- **ROC-AUC retention is strong (98.5%):** The model's ability to *rank* municipality-nights by risk transfers well to unseen geography. If you ask "which municipality in the South is most at risk tonight?", the model's ranking is nearly as good as for the national set.

- **PR-AUC and recall drop substantially.** This is driven by **multiple interacting factors**:
  - **Lower prevalence** in the South (0.23% vs 0.40%) mechanically depresses PR-AUC even at equal discrimination
  - **Calibration mismatch:** The threshold of 0.11 was calibrated on national data; the South may have different risk levels, making this threshold too strict
  - **Genuinely harder prediction:** Rural municipalities with different grid infrastructure, different weather patterns, and sparser NTL data may be inherently harder to predict
  - **Distribution shift:** The model has never seen these municipalities' NTL baselines, so it cannot learn municipality-specific patterns

- **This is a genuine limitation, not an artifact.** The model generalizes its ranking ability but not its calibrated probability levels to unseen regions. In deployment, one would need to re-calibrate for new regions, or accept lower precision/recall in under-represented areas. This is consistent with the literature on spatial generalization in remote sensing models.

---

## 9. Cause-Specific Analysis (New Extension)

### 9.1 Why Study Causes?

Until now, we predicted "does an outage happen?" without distinguishing *why*. But different causes have different signatures:

- **Environmental outages** (storms, cold fronts, trees falling on lines, earthquakes) should be more predictable from NTL + weather, because the same weather that causes the outage also dims lights and shows up in our features.
- **Technical outages** (equipment failure, human error, animals on transformers, corrosion) are driven by infrastructure aging and random failures — harder to see from space.
- **Planned outages** (scheduled maintenance) follow utility decisions, not environmental patterns.

If the model predicts environmental outages better than technical ones, this is **scientifically coherent** and informative about what satellite data can and cannot observe.

### 9.2 How Causes Enter the Panel

Cause labels come from a detailed CFE file (`night_outages_3hrs_with_locations_clean_by_reason.csv`) where each outage event has a cause. Since our modeling unit is municipality-night (not individual outage events), we need to aggregate:

- **85.8%** of positive municipality-nights have a single outage → cause is unambiguous
- **12.5%** have multiple outages of the same cause category → still unambiguous
- **1.7%** have multiple outages with different cause categories → we assign the **most frequent** cause

Given the very low ambiguity rate (1.7%), this dominant-cause assignment is defensible for a first-pass analysis. A mixed-cause sensitivity check (excluding the ambiguous 1.7%) will be run once the updated panel is rebuilt.

### 9.3 Evaluation Design

**The model is NOT retrained.** We use the exact same XGBoost model and Platt-calibrated scores from the strict benchmark. The question is: **does the same model's risk score separate one cause type from "no outage" better or worse than another?**

This is a **one-vs-rest** evaluation: for each cause C, we define y=1 if (outage AND cause==C), y=0 otherwise (including outages from other causes treated as negatives). This is conservative — it tests whether the model can specifically detect C-type outages, not just any outage.

### 9.4 Results

**Cause distribution in test period:**

| Cause | Positive nights | Share |
|-------|----------------|-------|
| **Environmental** (storms, cold, trees, earthquakes) | 4,150 | 77.8% |
| **Technical** (equipment, human error, animals) | 1,055 | 19.8% |
| Planned (maintenance) | 98 | 1.8% |
| Other (crime, unknown) | 31 | 0.6% |

**Performance by cause (Platt-calibrated, alerts/day cap = 10 policy):**

| Cause | Support | ROC-AUC | PR-AUC | Precision | Recall | Priority |
|-------|---------|---------|--------|-----------|--------|----------|
| **Environmental** | 4,150 | **0.883** | 0.041 | 9.8% | 7.7% | Primary |
| **Technical** | 1,055 | **0.819** | 0.013 | 3.1% | 9.5% | Primary |
| Planned | 98 | 0.889 | 0.001 | 0.2% | 5.1% | Exploratory |
| Other | 31 | 0.835 | 0.0003 | 0.1% | 6.5% | Exploratory |

### 9.5 Environmental vs Technical (Bootstrapped Comparison)

| Metric | Technical − Environmental | 95% CI | Significant? |
|--------|--------------------------|--------|-------------|
| **ROC-AUC** | **−6.4pp** | [−7.8, −5.1] | **Yes** |
| **PR-AUC** | **−2.8pp** | [−3.3, −2.2] | **Yes** |
| **Precision** | **−6.7pp** | [−7.9, −5.5] | **Yes** |
| Recall | +1.8pp | [−0.04, +3.7] | No |

### 9.6 Interpretation

**The model predicts environmental outages significantly better than technical ones.** The 6.4pp ROC-AUC gap is large, and the confidence interval excludes zero.

This makes scientific sense:
- **Environmental outages are weather-driven.** The model uses weather features *and* weather-correlated NTL dimming. Both channels carry information about environmental causes.
- **Technical outages are driven by equipment state, human factors, and random failures.** These mechanisms leave no observable signature in satellite brightness or weather data. The model can still partially predict them (ROC-AUC 0.819 > 0.5) because technical outages correlate with infrastructure characteristics that are spatially persistent and partially captured by NTL levels — but discrimination is significantly weaker.

**Planned outages** show a deceptively high ROC-AUC (0.889) but essentially zero precision/recall (0.2%, 5.1%). This means planned outages happen in municipalities that the model flags as generally high-risk (probably urban, high-load areas) — but the model cannot specifically predict *when* maintenance is scheduled, so it cannot retrieve them at any useful rate. With only 98 events, this is exploratory.

**Other** has too few events (31) for meaningful inference.

**The recall difference is not statistically significant** (+1.8pp, CI includes zero). This means the model catches a similar *fraction* of both environmental and technical outages at the default threshold — but the alerts for technical outages are much less precise (3.1% vs 9.8%), meaning most technical "catches" are accidental (the model flagged the municipality for other reasons).

---

## 10. Summary: What the Model Can and Cannot Do

### What it can do:
- Produce **calibrated municipality-night outage risk probabilities** using only freely available satellite and weather data
- Achieve **strong discrimination** (ROC-AUC 0.879) for next-night outage forecasting — no grid telemetry required
- **Concentrate risk** — at 10 alerts/day, 12.5% of alerts are true outages (25x the base rate)
- Predict **environmental outages significantly better** than technical ones, confirming that the signal is weather-and-light-driven
- **Maintain ranking ability across unseen geographic regions** (98.5% ROC-AUC retention)
- Provide **honest probability estimates** after Platt calibration (ECE ≈ 0.001)
- Support **flexible decision-making** through policy-based thresholds, not a fixed K

### What it cannot do:
- Achieve high precision at low alert volumes (best: ~12.5% at 10 alerts/day)
- Predict technical outages (equipment failure, human error) with the same confidence as environmental ones
- Guarantee that calibrated probabilities transfer to geographically unseen regions without re-calibration
- Detect planned maintenance or crime-related outages
- Replace real-time grid monitoring (SCADA) or operational utility systems

### Current thesis structure:
1. **Primary contribution:** Calibrated probabilistic forecasting with NTL as core signal
2. **Feature story:** NTL + weather is the dominant combination; spatial/anomaly/drop features add little in strict forecast mode
3. **Calibration story:** Post-hoc calibration (Platt) is essential — raw XGBoost scores are useless as probabilities
4. **Cause-specific story:** Environmental outages are significantly more predictable, confirming the NTL+weather signal pathway
5. **Robustness:** Temporally stable, not driven by Texas shock, ranking generalizes spatially (but calibration does not)
6. **Secondary benchmark:** LTR / top-K retained as comparator for fixed-budget scenarios

---

## 11. Comparison: Old Pipeline (Top-K/LTR) vs New Pipeline (Probability + Calibration)

| Aspect | Old Pipeline (Top-K/LTR) | New Pipeline (Prob + Calib) |
|--------|--------------------------|---------------------------|
| **Core model** | XGBoost + LambdaMART ranking | XGBoost + Platt calibration |
| **Additional models tried** | Anomaly detection, PCA embeddings, autoencoder embeddings | — (kept simple) |
| **Output** | Daily ranked list of K municipalities | Calibrated probability per municipality-night |
| **Evaluation** | Recall@K, NDCG | PR-AUC, ROC-AUC, Brier, ECE, decision curves |
| **Threshold** | Fixed K per day | Flexible policy-based (alerts/day cap, precision floor, etc.) |
| **Scientific framing** | "Which municipalities should we inspect?" | "What is the probability of outage?" |
| **Ablations** | Limited | Systematic 7-model ablation with strict protocol |
| **Calibration** | Not studied | Central contribution — 3 methods compared |
| **Cause analysis** | Not done | Environmental vs Technical, bootstrapped CIs |
| **Robustness** | LOSO | Texas sensitivity, spatial holdout, temporal stability |
| **Direct comparison** | — | Benchmark table: binary vs LTR on same test split (Section 12) |

**Why the simplification is a strength:** The old pipeline had many moving parts (anomaly detection, embeddings, LTR, re-ranking). Each added complexity but not necessarily performance. The new pipeline is simpler — one XGBoost model + calibration — but evaluated much more rigorously. In the literature, simple well-evaluated models are preferred over complex under-evaluated ones.

The anomaly detection and embedding approaches remain valid ideas (they are supported by the NTL literature on disaster response), but our ablations show that standard NTL lag features already capture most of the same signal. Adding embeddings or anomaly scores on top does not meaningfully improve performance in this setting.

---

## 12. Direct Comparison: Binary Classifier vs LTR (Benchmark Table)

### Can we compare the two approaches?

Yes — with caveats. The benchmark table (`benchmark_table_same_run.csv`) compares the binary classifier (raw, Platt, Beta) and the LTR model on the **same test split**. But the comparison requires care because the two approaches answer different questions:

- **Binary classifier + threshold:** "Alert all municipalities above probability T." The number of alerts varies by day (some days have more risk than others).
- **LTR + top-K:** "Alert exactly the top K municipalities per day." Every day produces exactly K alerts, regardless of risk level.

These are **different operational policies**, so precision/recall numbers are not directly comparable at face value. The benchmark table aligns them as closely as possible by matching the average alert budget:

### Benchmark Results (same test period, alerts/day ≈ 10)

| Model | Method | Precision | Recall | Alerts/day | Calibration |
|-------|--------|-----------|--------|------------|-------------|
| **Binary** | **Platt** (default) | **12.5%** | 9.7% | 7.6 | Log-loss 0.021, ECE 0.001 |
| **Binary** | Beta (sensitivity) | 12.2% | 10.7% | 8.7 | Log-loss 0.021, ECE 0.001 |
| **Binary** | Raw (uncalibrated) | 9.9% | 16.2% | 16.1 | Log-loss 0.300, ECE 0.196 |
| **LTR** | rank:pairwise | 10.2% | 10.4% | 10.0 (fixed) | Not applicable |

*Source: `benchmark_table_same_run.csv` — run `20260223_125430_forecast_strict` (binary) vs run `20260220_114434` (LTR). Same test split, same time period.*

**LTR Lift@10:** 25.5 — meaning the top-10 daily shortlist catches outages at 25.5x the base rate. This is strong ranking performance.

### How to read this comparison

1. **At similar alert volumes (~8-10/day), calibrated binary and LTR achieve similar recall** (~10%). Neither dominates the other in raw detection power. This is expected: both use the same underlying features and XGBoost architecture.

2. **Calibrated binary achieves higher precision** (12.5% Platt vs 10.2% LTR) at slightly fewer alerts (7.6 vs 10.0/day). The calibrated model is more selective — it only alerts on high-confidence nights, whereas top-K=10 forces exactly 10 alerts even on low-risk days.

3. **LTR achieves slightly higher recall** (10.4% vs 9.7%) by alerting on exactly 10 municipalities every day. On quiet days, some of these are wasted; on high-risk days, 10 may not be enough. The binary model adapts: more alerts on risky days, fewer on quiet days.

4. **Raw (uncalibrated) binary is misleading:** It appears to have the highest recall (16.2%), but at 16 alerts/day — not a fair comparison to K=10. And its probability estimates are useless (log-loss 0.30 vs 0.02).

5. **Calibration is the key differentiator.** LTR scores cannot be interpreted as probabilities. Platt-calibrated scores can. This matters for:
   - Risk communication ("5% chance" vs "ranked 7th today")
   - Decision curves and cost-benefit analysis
   - Downstream modeling (count, duration — where P(outage) is an input)
   - Flexible policies (different thresholds for different regions or seasons)

### When each approach wins

| Scenario | Better approach | Why |
|----------|----------------|-----|
| "Inspect exactly 10 municipalities per day, every day" | **LTR** | Directly optimizes within-day ranking for fixed K |
| "Alert when risk exceeds a threshold; some days 5, others 20" | **Binary + calibration** | Flexible, probability-based, adapts to daily risk |
| "Communicate risk levels to decision-makers" | **Binary + calibration** | Probabilities are interpretable; ranking scores are not |
| "Feed into a downstream severity model" | **Binary + calibration** | P(outage) is a meaningful input; rank position is not |
| "Compare model performance to a fixed-budget comparator" | **Both** | LTR provides the budget-constrained reference |

### Bottom line

The two approaches are complementary, not competing. The calibrated binary classifier is the **primary model** because it is more general, interpretable, and composable. LTR is a **secondary benchmark** that validates performance under a specific (fixed-K) operational constraint. Both confirm that the underlying NTL + weather features carry substantial predictive signal.

---

## 13. Next Steps

1. **Rebuild panel** with additional label columns (total duration, cause-specific counts, mixed-cause flag) — code is ready, awaiting panel rebuild with NTL data
2. **Mixed-cause sensitivity** for cause-specific evaluation (expected minimal impact given 1.7% ambiguity rate)
3. **Count modeling** (n_outages | outage occurred): Hurdle model — Poisson/NB on positives only
4. **Duration/severity modeling** (total_length_min | outage occurred): Log-normal regression on positives
5. **Decomposed expected burden:** E[total disruption] = P(outage) x E[duration | outage]

---

## Appendix A: File Locations

| Output | Path |
|--------|------|
| Main strict run | `data/baselines/binary_threshold/20260223_125430_forecast_strict/` |
| Ablation batch | `data/baselines/binary_threshold/ablation_batches/20260302_122455/` |
| Nowcast batch | `data/baselines/binary_threshold/ablation_batches/20260302_135016/` |
| Spatial holdout | `data/baselines/binary_threshold/ablation_batches/20260302_153151/` |
| Texas sensitivity | `data/baselines/binary_threshold/sensitivity_texas/` |
| Cause-specific eval | `data/baselines/binary_threshold/20260223_125430_forecast_strict/cause_specific_eval/` |
| Panel building | `ML_Mexico_final.R` |
| Main eval script | `scripts/05b_binary_threshold_eval_main.R` |
| Ablation orchestrator | `scripts/07b_probcal_ablations.R` |
| Cause eval script | `scripts/13_cause_specific_eval.R` |
| Benchmark table (binary vs LTR) | `data/baselines/binary_threshold/20260223_125430_forecast_strict/benchmark_table_same_run.csv` |
| LTR run | `runs/ltr/20260220_114434/` |
| Literature review | `ML_outages_litreview.pdf` |

## Appendix B: Glossary for Non-ML Readers

| Term | Plain-language meaning |
|------|----------------------|
| **XGBoost** | A machine learning algorithm that builds many small decision trees sequentially, each correcting the errors of the previous ones. Fast, accurate, widely used. |
| **Calibration** | Adjusting a model's raw scores so they correspond to real probabilities. "10% predicted risk" should mean ~10% of those cases actually have outages. |
| **Platt scaling** | A specific calibration method: fit a logistic curve to the raw scores. Simple, stable, 2 parameters. |
| **ROC-AUC** | "How well does the model rank outage-nights above non-outage-nights?" 1.0 = perfect, 0.5 = random. |
| **PR-AUC** | "How well does the model retrieve rare outage events?" More informative than ROC-AUC when events are rare. |
| **Brier score** | Mean squared error of probability predictions. Lower = better calibrated. |
| **ECE** | Expected Calibration Error — average gap between predicted probabilities and observed frequencies. 0 = perfect. |
| **Threshold** | The risk cutoff above which the model issues an alert. A policy choice, not a model parameter. |
| **Precision** | Among alerts, what fraction are true outages? |
| **Recall** | Among actual outages, what fraction did the model catch? |
| **Ablation** | Removing groups of features to see which ones actually matter. |
| **Temporal split** | Training on past data, testing on future data — prevents the model from "seeing the future." |
| **Spatial holdout** | Training on some regions, testing on others — tests geographic generalization. |
| **NTL** | Nighttime lights — satellite-measured brightness of the Earth's surface at night. |
| **Prevalence** | The fraction of positive cases (outage-nights) in the dataset. Here: ~0.5%. |
| **scale_pos_weight** | Tells the model how much more important it is to correctly detect a positive (outage) than a negative (no outage). |
| **LTR (Learning-to-Rank)** | A model trained to rank items within groups (here: municipalities within each day) rather than predicting probabilities. Optimizes ranking metrics like NDCG. |
| **Top-K** | A decision rule: each day, alert the K highest-ranked municipalities. Assumes a fixed daily budget. |
| **Lift@K** | How much better the top-K shortlist concentrates outages compared to random selection. Lift@10 = 25 means 25x the base rate. |
| **Hurdle model** | A two-part model: first predict whether an event occurs (binary), then predict its magnitude (count/duration) conditional on occurrence. |
| **Decision curve** | A plot showing the "net benefit" of using the model at each threshold, accounting for the relative cost of false positives vs missed outages. |