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
def volume_tier(vol: float) -> str:
    if vol >= 1_000_000_000: return "Very High (>$1B)"
    if vol >= 100_000_000:   return "High ($100M–$1B)"
    if vol >= 10_000_000:    return "Medium ($10M–$100M)"
    if vol >= 1_000_000:     return "Low ($1M–$10M)"
    return "Very Low (<$1M)"


def pool(whale_threshold: float = 5000) -> pd.DataFrame:
    frames = []
    market_dirs = sorted([p for p in PROCESSED_DIR.iterdir()
                          if p.is_dir() and (p / "events.csv").exists()])
    print(f"Found {len(market_dirs)} markets with events.csv")

    for p in market_dirs:
        path = p / "events.csv"
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
        except Exception as e:
            print(f"  SKIP {p.name} — read error: {e}")
            continue
        if len(df) < 5:
            continue
        vol = float(df["market_volume"].iloc[0]) if "market_volume" in df.columns else 0
        lbl = df["market_label"].iloc[0] if "market_label" in df.columns else p.name
        df["market_label"] = lbl
        df["volume_tier"]  = volume_tier(vol)
        df["is_whale"]     = df["position_usd"] >= whale_threshold
        frames.append(df)
        n_whale = df["is_whale"].sum()
        n_price = df["price_change"].notna().sum()
        print(f"  {str(lbl)[:35]:<35}  ${vol:>14,.0f}  whales={n_whale:>4}  with_price={n_price:>6}")

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
