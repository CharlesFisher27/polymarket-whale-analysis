# Polymarket Whale Comment Analysis

**Do large-position holders ("whales") move prices on Polymarket through their public comments?**

QSS 20 Final Project — Charlie Fisher, Dartmouth College

---

## Overview

This project tests whether whale commenters (users with ≥$5,000 positions) cause larger price movements than retail commenters on Polymarket prediction markets. Using an event-study design, I measure the absolute price change in the 2 hours following each comment and compare whales vs. retail across 200+ markets.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `code/00_collect.py` | Pulls raw data from Polymarket APIs (comments, price history, holders) |
| `code/04_analyze.py` | Runs event-study analysis and generates figures 1–4 |
| `code/06_market_level_regression.py` | Cross-market OLS: whale effect vs. log(market volume); generates Figure 5 |

---

## Output

| File | Description |
|------|-------------|
| `output/fig2_event_study_by_market.png` | Main result: whale vs. retail price impact per market |
| `output/fig5_cross_market_regression.png` | Cross-market regression: whale effect vs. market volume |
| `output/market_level_effects.csv` | Market-level summary table (200+ markets) |

---

## Data

Raw data (price histories, comments, holder snapshots) is too large for GitHub.

**[Download data from Google Drive](https://drive.google.com/placeholder)** ← update with real link

Data is collected from:
- [Polymarket CLOB API](https://clob.polymarket.com) — price history
- [Polymarket Gamma API](https://gamma-api.polymarket.com) — market metadata
- [Polymarket Data API](https://data-api.polymarket.com) — holder positions

---

## Requirements

```bash
pip install pandas numpy scipy statsmodels matplotlib vaderSentiment requests
```
