"""
Pool processed events.csv files across multiple markets into one DataFrame
for cross-market regression analysis.

Usage:
    python code/03_pool.py
    python code/03_pool.py --whale-threshold 5000

Outputs:
    data/processed/pooled_events.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Market manifest: id → human label for plots
#
# Event comment pools (comments are per-event, not per-market):
#   Event 903193: Trump Win + Harris Win share the same ~113K comment pool
#   Event 903216: Trump Popular Vote + Harris Popular Vote share the same ~4.4K pool
#   Event 903215: Biden Dem Nominee — independent pool (~2.5K comments)
#   Event 903219: Ethereum ETF — independent pool (~805 comments)
#
# All markets pass the convergence filter (price spent meaningful time in
# the 0.10–0.90 uncertainty band). Ethereum ETF has 0 post-filter whale
# events (all whale comments occurred after approval probability converged
# to ~1) but contributes retail observations and broadens domain diversity.
MARKETS = {
    "253591": {"label": "Trump 2024 Win",       "tier": "Very High (>$1B)"},
    "253597": {"label": "Harris 2024 Win",      "tier": "Very High (>$1B)"},
    "253706": {"label": "Trump Popular Vote",   "tier": "High ($100M–$1B)"},
    "253727": {"label": "Harris Popular Vote",  "tier": "High ($100M–$1B)"},
    "253697": {"label": "Biden Dem Nominee",    "tier": "Medium ($10M–$100M)"},
    "253750": {"label": "Ethereum ETF",         "tier": "Medium ($10M–$100M)"},
}


def pool(whale_threshold: float = 5000) -> pd.DataFrame:
    frames = []
    for market_id, meta in MARKETS.items():
        path = PROCESSED_DIR / market_id / "events.csv"
        if not path.exists():
            print(f"  SKIP {market_id} ({meta['label']}) — events.csv not found")
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df["market_label"] = meta["label"]
        df["volume_tier"]  = meta["tier"]
        # re-apply whale threshold in case process was run with a different value
        df["is_whale"] = df["position_usd"] >= whale_threshold
        frames.append(df)
        n_whale = df["is_whale"].sum()
        n_price = df["price_change"].notna().sum()
        vol = df["market_volume"].iloc[0] if "market_volume" in df.columns else 0
        print(f"  {meta['label']:<30}  ${vol:>14,.0f}  whale_cmts={n_whale:>5}  with_price={n_price:>6}")

    if not frames:
        raise RuntimeError("No markets loaded. Run collect.py + process.py for each market first.")

    pooled = pd.concat(frames, ignore_index=True)

    # add log position for regression
    pooled["log_position"] = np.where(
        pooled["position_usd"] > 0,
        np.log10(pooled["position_usd"]),
        np.nan
    )

    out_path = PROCESSED_DIR / "pooled_events.csv"
    pooled.to_csv(out_path, index=False)
    print(f"\nPooled {len(pooled):,} events from {len(frames)} markets → {out_path}")
    return pooled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--whale-threshold", type=float, default=5000)
    args = parser.parse_args()
    print("Pooling markets:")
    pool(args.whale_threshold)
