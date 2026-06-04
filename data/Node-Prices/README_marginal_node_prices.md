# marginal_node_prices.rds — Hourly Marginal Node Prices

## Source
CENACE SW-PML (Sistema de Información de Precios Marginales Locales), ex-post prices.
Downloaded via `03_scripts/000_download_node_prices.R` from the CENACE public API.

## Coverage
- **Dates:** 2017-01-27 to present (gaps in July 2025; API returns no data for 2025-07-01 to 2025-08-11)
- **Nodes:** ~2,414 NodosP from the SIN (Sistema Interconectado Nacional)
- **Frequency:** Hourly (hours 1–24 per day per node)

## Unit of Observation
Node × hour (`node_id` × `date` × `hour`)

## Variables
| Variable | Description | Units |
|---|---|---|
| `date` | Calendar date | Date |
| `hour` | Hour of day (1–24) | Integer |
| `node_id` | CENACE node identifier | Character |
| `ss_node` | Substation code (characters 3–5 of `node_id`) | Character |
| `marginal_price` | Local marginal price | MXN/MWh |
| `energy` | Energy component of LMP | MXN/MWh |
| `losses` | Losses component of LMP | MXN/MWh |
| `congestion` | Congestion component of LMP | MXN/MWh |

## Notes
- `marginal_price = energy + losses + congestion`
- Congestion can be negative (counter-flow relief)
- The market (and thus price data) launched on 2017-01-27; no data exists before that date
- One CSV per calendar day in `01_data/01_outages/node_prices/YYYY_QN/`
- Processed on CMCC (job 73955) using `03_scripts/process_node_prices.R`

## File Info
- Format: R RDS (compressed)
- Size: ~1.3 GB
- Location on CMCC: `/work/cmcc/ls01122/blackouts/02_gen/04_outages/marginal_node_prices.rds`
