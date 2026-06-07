# Polymarket Whale Comment Analysis

**Do large-position holders ("whales") move prices on Polymarket through their public comments?**

QSS 20 Final Project — Charlie Fisher, Dartmouth College

---

## Key Findings

- Whale comments are followed by **2.2× larger absolute price moves** than retail (mean |Δp| = 0.21 vs. 0.10)
- The gap **grows with market volume** — opposite of what thin-order-book manipulation predicts
- OLS whale dummy: **β = +0.020** (p < 0.001); whale × log-volume interaction: **p = 0.85** (not significant)
- Position-consistency rate: **52.8%** (barely above the 50% random benchmark)
- Results are consistent with **event-driven confounding**: whales comment when news breaks

---

## Repository Structure

```
polymarket-whale-analysis/
├── code/               ← numbered pipeline scripts (run in order 00 → 07)
│   └── utils/          ← api.py, sentiment.py (shared helpers)
├── data/
│   ├── raw/            ← one subdirectory per market (from 00_collect / 06_collect_sample)
│   └── processed/      ← per-market events.csv + pooled_events.csv
├── output/             ← figures and tables referenced in paper
└── requirements.txt
```

Full raw dataset (470 markets, ~2.7M comment events) is available at:
**[Dropbox data folder](https://www.dropbox.com/scl/fo/ob6plako47aqmfp0cdp7e/ALMGJtxsBfMfWoXQf1ogfJs?rlkey=zkfe6gztjlmlc6wis0zj7e7vo&st=qwf3esbj&dl=0)**

---

## Pipeline Scripts

Run the main pipeline in order: `00` → `01` → `02` → `03` → `04` → `05`

| Script | Inputs | What it does | Outputs |
|---|---|---|---|
| [`code/00_collect.py`](code/00_collect.py) | `--market <id>` `--event-id <id>` (Polymarket public API) | Fetches market metadata, top-200 holders, all comments (tagged with position size), and hourly YES-token price history for one market | `data/raw/<market_id>/market.json`, `holders.json`, `comments_tagged.json`, `price_history.json` |
| [`code/01_collect_whale_trades.py`](code/01_collect_whale_trades.py) | `data/raw/<market_id>/holders.json` (Polymarket CLOB API) | Fetches on-chain trade history for every whale wallet in a market; used for pump/dump trade-window analysis | `data/raw/<market_id>/whale_trades.json` |
| [`code/02_process.py`](code/02_process.py) | `data/raw/<market_id>/` | For each comment, finds nearest YES-token price before and after (±7-day nearest-neighbor window), computes Δp, scores VADER sentiment with custom prediction-market lexicon, classifies whale (≥$5,000) vs. retail. Adds trade windows if `whale_trades.json` exists | `data/processed/<market_id>/events.csv`, `prices.csv`, `comments.csv`, `holders.csv` |
| [`code/03_pool.py`](code/03_pool.py) | `data/processed/<market_id>/events.csv` (all markets) | Concatenates all per-market event files; adds `volume_tier` labels; re-applies whale threshold; prints per-market diagnostics (volume, whale count, price-matched count) | `data/processed/pooled_events.csv` |
| [`code/04_analyze.py`](code/04_analyze.py) | `data/processed/pooled_events.csv` | Applies convergence filter (price_before outside 0.10–0.90); runs four pooled OLS models (M1–M4) with HC3 robust SE; prints coefficient table; runs position-direction summary | `output/regression_table.tex`, `output/summary_stats.csv` |
| [`code/05_generate_figures.py`](code/05_generate_figures.py) | `data/processed/pooled_events.csv`, `data/raw/*/market.json`, `data/raw/*/holders.json` | Applies convergence + Fed-market filters; generates all paper figures (Figs 1–9) | `output/fig1_price_trajectories.png` through `output/fig9_scatter_position_impact.png` |

### Batch collection (alternative to running 00 + 02 per-market)

| Script | Inputs | What it does | Outputs |
|---|---|---|---|
| [`code/06_collect_sample.py`](code/06_collect_sample.py) | Sample JSON from market enumeration; Polymarket APIs | Batch-collects and processes a stratified sample of markets; internally calls the same logic as `00_collect.py` + `02_process.py`; reuses comment pools shared across markets | `data/raw/<market_id>/` and `data/processed/<market_id>/events.csv` for each market |
| [`code/07_market_level_regression.py`](code/07_market_level_regression.py) | `data/processed/pooled_events.csv` | Computes per-market whale effect (mean |Δp| whale − retail) and runs cross-market OLS: whale_effect ~ log(volume) | `output/market_level_effects.csv` |

### Utility / one-time scripts

| Script | Purpose |
|---|---|
| [`code/utils/api.py`](code/utils/api.py) | Polymarket API wrappers: pagination, rate limiting, market/event/comment/price/holder fetching |
| [`code/utils/sentiment.py`](code/utils/sentiment.py) | VADER scorer with custom prediction-market lexicon (`moon`, `bullish`, `rekt`, etc.) |
| [`code/refetch_prices_pmxt.py`](code/refetch_prices_pmxt.py) | One-time: replaced sparse CLOB price histories with dense hourly candles via pmxt API; re-ran `02_process.py` + `03_pool.py` after |
| [`code/enrich_positions.py`](code/enrich_positions.py) | One-time: filled missing `position_usd` values by querying the Data API for commenter wallets not in the top-200 holder list |
| [`code/collect_live_markets.py`](code/collect_live_markets.py) | Discovers and collects live markets; safe to interrupt and restart |

---

## Output Files

| File | Description |
|---|---|
| `output/fig1_price_trajectories.png` | YES-token price trajectories for 6 representative markets; red dots = whale events |
| `output/fig3_sentiment_analysis.png` | Mean \|Δp\| by market volume quintile, whale vs. retail (±1 SE) |
| `output/fig4_pump_dump.png` | 3×2 heatmap: VADER sentiment × holder direction for 4,910 whale events |
| `output/fig9_scatter_position_impact.png` | Position size vs. \|Δp\| with decile bin means (±1 SE) |
| `output/regression_table.tex` | LaTeX booktabs table for Models M1–M4 |
| `output/summary_stats.csv` | Summary statistics for key variables |
| `output/market_level_effects.csv` | Per-market whale effect estimates |

---

## Reproducing the Analysis

```bash
pip install -r requirements.txt

# Collect one market (needs market ID and event ID from Polymarket)
python code/00_collect.py --market 253591 --event-id 903193

# Or batch-collect the full sample
python code/06_collect_sample.py

# Process each market (run for each market_id)
python code/02_process.py --market 253591

# Pool all markets
python code/03_pool.py

# Run regressions
python code/04_analyze.py

# Generate figures
python code/05_generate_figures.py
```

No API key required — all Polymarket endpoints used are public.
All paths are relative to the repository root; no hardcoded absolute paths.
