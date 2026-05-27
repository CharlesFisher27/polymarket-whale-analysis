# Polymarket Whale Comment Analysis

**Do large-position holders ("whales") move prices on Polymarket through their public comments?**

QSS 20 Final Project — Charlie Fisher, Dartmouth College

---

## Overview

This project tests whether whale commenters (users with ≥$5,000 positions) cause larger price movements than retail commenters on Polymarket prediction markets. Using an event-study design, I measure the absolute price change in the 2 hours following each comment and compare whales vs. retail across 200+ markets.

---

## Code

| Script | Purpose |
|--------|---------|
| `code/00_collect.py` | Pulls raw data from Polymarket APIs (comments, price history, holder positions) |
| `code/01_collect_whale_trades.py` | Fetches on-chain trade data for whale wallets |
| `code/02_process.py` | Processes raw data into event windows with price changes and sentiment scores |
| `code/03_pool.py` | Pools processed events across all markets into a single analysis file |
| `code/04_analyze.py` | Runs pooled OLS regressions and generates Figures 1–4 |
| `code/05_collect_sample.py` | Collects data for a stratified sample of 200+ markets |
| `code/06_market_level_regression.py` | Cross-market OLS: whale effect vs. log(market volume); generates Figure 5 |

---

## Output

| File | Description |
|------|-------------|
| `output/fig1_price_trajectories.png` | Price trajectories around whale vs. retail comment events |
| `output/fig2_event_study_by_market.png` | Main result: whale vs. retail price impact per market |
| `output/fig3_sentiment_analysis.png` | Sentiment analysis of whale vs. retail comments |
| `output/fig4_pump_dump.png` | Position-direction analysis and pump/dump signal detection |
| `output/fig5_cross_market_regression.png` | Cross-market regression: whale effect vs. market volume |
| `output/market_level_effects.csv` | Market-level summary table (200+ markets) |
| `output/summary_stats.csv` | Pooled summary statistics |

---

## Data

Raw data (price histories, comments, holder snapshots) is too large to store in full on GitHub. The `sample_data/` folder contains a small sample of the processed data files.

Full dataset available on request.

---

## Requirements

```bash
pip install pandas numpy scipy statsmodels matplotlib vaderSentiment requests
