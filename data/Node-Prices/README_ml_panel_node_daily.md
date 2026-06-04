# ml_panel_node_daily.rds — Daily Substation × Day ML Panel

## Source
Built from three inputs via `03_scripts/build_ml_panel.R` (CMCC job 92811):
1. `marginal_node_prices.rds` — CENACE hourly prices
2. `outages.rds` — CENACE outage events (576K events, 2017–2021)
3. CMCC weather: `02_gen/02_weather/agem_daily/weighted/daily_agem_w.rds`

## Coverage
- **Dates:** 2017-01-01 to 2021-12-31 (1,826 days)
- **Substations:** 1,570 (`ss_node`)
- **Rows:** 2,866,820 (perfectly balanced panel: 1,570 × 1,826)
- **Columns:** 53

## Unit of Observation
Substation × day (`ss_node` × `date`)

## Variable Groups

### Identifiers
| Variable | Description |
|---|---|
| `ss_node` | Substation code (3-character CENACE identifier) |
| `agem` | Municipality code (linked via substation location) |
| `date` | Calendar date |

### Outage outcomes
| Variable | Description |
|---|---|
| `is_outage` | 1 if any outage on this ss_node-day |
| `n_outages` | Number of outage events |
| `total_outage_min` | Total outage duration (minutes) |
| `mean_dur_min` | Mean event duration (minutes) |
| `max_dur_min` | Max event duration (minutes) |
| `n_tech` | Technical-cause events |
| `n_env` | Environmental-cause events |
| `n_planned` | Planned outage events |

### Price features (from hourly data, aggregated to daily)
| Variable | Description |
|---|---|
| `price_mean` | Daily mean LMP (MXN/MWh) |
| `price_median` | Daily median LMP |
| `price_sd` | Daily std dev of LMP |
| `price_max` | Daily max LMP |
| `price_min` | Daily min LMP |
| `price_range` | Daily range (max − min) |
| `price_peak_mean` | Mean LMP during peak hours (8–22) |
| `price_offpeak_mean` | Mean LMP during off-peak hours |
| `peak_offpeak_ratio` | `price_peak_mean / price_offpeak_mean` |
| `congestion_mean` | Mean congestion component |
| `congestion_max` | Max congestion component |
| `congestion_share` | `congestion_mean / price_mean` |
| `energy_mean` | Mean energy component |
| `losses_mean` | Mean losses component |
| `n_hours_obs` | Hours with price data (max 24) |
| `n_nodes_in_ss` | Distinct nodes observed in substation |
| `price_roll7` | 7-day rolling mean of `price_mean` |
| `price_dev_roll7` | `price_mean − price_roll7` (deviation from trend) |

### Price lags
`price_mean_lag1`, `price_mean_lag7`, `price_sd_lag1`, `price_sd_lag7`,
`price_max_lag1`, `price_max_lag7`, `price_range_lag1`, `price_range_lag7`,
`congestion_mean_lag1`, `congestion_mean_lag7`, `congestion_share_lag1`, `congestion_share_lag7`

### Weather (municipality-level, population-weighted)
| Variable | Description | Units |
|---|---|---|
| `temp` | Mean daily temperature | °C |
| `temp_min` | Daily minimum temperature | °C |
| `temp_max` | Daily maximum temperature | °C |
| `rain` | Daily precipitation | mm |
| `rh` | Relative humidity | % |
| `dew` | Dew point | °C |
| `wsp` | Wind speed | m/s |

### Calendar
`year`, `month`, `dow` (day of week), `is_weekend`, `yday` (day of year)

## Missing Data
- Price variables: ~4% NA (substations without CENACE price nodes for that date)
- Price data starts 2017-01-27 (market launch); no price data for Jan 1–26, 2017
- Weather: 0% NA (full coverage)
- Outage variables: 0% NA (zeros imputed for days with no events)
- Lag variables: ~4–4.3% NA (inherits from base price missingness)

## File Info
- Format: R RDS (gzip compressed)
- Size: 469 MB
- Location on CMCC: `/work/cmcc/ls01122/blackouts/02_gen/04_outages/ml_panel_node_daily.rds`
