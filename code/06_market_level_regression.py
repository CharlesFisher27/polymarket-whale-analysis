"""
06_market_level_regression.py — Cross-market regression: whale effect vs market volume.

Unit of analysis: market m
    DV:  whale_effect_m = mean|Δp|(whale events) − mean|Δp|(retail events)
    IV:  log10(market_volume_m)

This avoids the multicollinearity that prevents estimating the volume coefficient
in pooled OLS with market fixed effects.

Two whale definitions are compared:
  1. Absolute threshold ($5,000 USD position)  — same as pooled analysis
  2. Relative threshold (top-10 commenters by position in each market) — better
     for cross-market comparison where thin markets may have no absolute whales

Usage:
    python code/06_market_level_regression.py
    python code/06_market_level_regression.py --rebuild   # force rebuild of effects table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT     = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR  = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

CONVERGENCE_LO = 0.10
CONVERGENCE_HI = 0.90
MIN_EVENTS     = 10    # minimum non-convergence events for a market to be included
MIN_WHALE      = 1     # minimum whale events to compute whale_effect


def build_effects_from_scratch(top_n_relative: int = 10) -> pd.DataFrame:
    """
    Build market_level_effects.csv from individual events.csv files.

    For each market, computes:
      - whale_effect:    mean|Δp|(abs-whale) − mean|Δp|(retail)  [fixed $5k threshold]
      - rel_whale_effect: mean|Δp|(top-N commenters) − mean|Δp|(others)  [relative]
    """
    rows = []
    for p in PROC_DIR.iterdir():
        if not p.is_dir():
            continue
        evpath = p / "events.csv"
        if not evpath.exists():
            continue
        try:
            ev = pd.read_csv(evpath, parse_dates=["timestamp"])
        except Exception:
            continue
        nc = ev[ev["price_change"].notna() & ~ev["is_convergence_period"]]
        if len(nc) < MIN_EVENTS:
            continue
        vol = float(ev["market_volume"].iloc[0]) if "market_volume" in ev.columns else 0
        lbl = ev["market_label"].iloc[0]         if "market_label"   in ev.columns else p.name

        # Absolute whale
        wh = nc[nc["is_whale"]]["abs_price_change"].dropna()
        rt = nc[~nc["is_whale"]]["abs_price_change"].dropna()
        if len(rt) < 5:
            continue

        # Relative whale: top-N commenters by position_usd in this market
        if "position_usd" in nc.columns and len(nc) >= top_n_relative:
            top_n_cutoff = nc["position_usd"].nlargest(top_n_relative).min()
            rel_wh = nc[nc["position_usd"] >= top_n_cutoff]["abs_price_change"].dropna()
            rel_rt = nc[nc["position_usd"] <  top_n_cutoff]["abs_price_change"].dropna()
            rel_effect = float(rel_wh.mean() - rel_rt.mean()) if (len(rel_wh) >= 1 and len(rel_rt) >= 5) else np.nan
        else:
            rel_effect = np.nan

        rows.append({
            "market_id":      p.name,
            "market_label":   lbl,
            "volume_usd":    vol,
            "log_volume":    np.log10(vol) if vol > 0 else np.nan,
            "n_events":      len(nc),
            "n_whale":       len(wh),
            "n_retail":      len(rt),
            "whale_abs_dp":  wh.mean() if len(wh) >= 1 else np.nan,
            "retail_abs_dp": rt.mean(),
            "whale_effect":  wh.mean() - rt.mean() if len(wh) >= MIN_WHALE else np.nan,
            "rel_whale_effect": rel_effect,
        })
    df = pd.DataFrame(rows).sort_values("volume_usd", ascending=False)
    return df.reset_index(drop=True)


def load_market_effects(rebuild: bool = False) -> pd.DataFrame:
    """Load or (re-)build market_level_effects.csv."""
    cache = PROC_DIR / "market_level_effects.csv"
    if cache.exists() and not rebuild:
        df = pd.read_csv(cache)
        # Add rel_whale_effect column if missing (old cache)
        if "rel_whale_effect" not in df.columns:
            df = build_effects_from_scratch()
            df.to_csv(cache, index=False)
        return df
    df = build_effects_from_scratch()
    df.to_csv(cache, index=False)
    return df


def run_cross_market_regression(df: pd.DataFrame) -> None:
    """OLS: whale_effect ~ log_volume (markets with ≥1 whale event)."""
    fit_df = df[df["whale_effect"].notna() & df["log_volume"].notna()].copy()
    print(f"\n=== CROSS-MARKET OLS (n = {len(fit_df)} markets with whale events) ===")

    if len(fit_df) < 5:
        print("  Insufficient markets — collect more data first.")
        return

    res = smf.ols("whale_effect ~ log_volume", data=fit_df).fit(cov_type="HC3")
    print(res.summary2().tables[1].to_string())
    print(f"\nR² = {res.rsquared:.3f}")

    # Pearson correlation
    r, p = stats.pearsonr(fit_df["log_volume"], fit_df["whale_effect"])
    print(f"Pearson r = {r:+.3f}  p = {p:.4f}")
    print("\nInterpretation:")
    if p < 0.05:
        if r < 0:
            print("  NEGATIVE relationship: whale effect LARGER in thin markets (Barberis BSV supported)")
        else:
            print("  POSITIVE relationship: whale effect LARGER in liquid markets (BSV not supported)")
    else:
        print("  No significant relationship between market volume and whale effect (p >= 0.05)")

    return res


def make_cross_market_figure(df: pd.DataFrame) -> None:
    """Figure: scatter of whale_effect vs log_volume across all markets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: whale_effect (mean|Δp| diff) vs log volume
    ax = axes[0]
    fit_df = df[df["whale_effect"].notna() & df["log_volume"].notna()].copy()
    no_whale = df[df["whale_effect"].isna() & df["log_volume"].notna()]

    ax.scatter(fit_df["log_volume"], fit_df["whale_effect"],
               s=np.clip(np.sqrt(fit_df["n_events"]) * 3, 20, 200),
               alpha=0.6, color="steelblue", edgecolors="white", lw=0.5,
               label=f"≥1 whale event (n={len(fit_df)})")
    ax.scatter(no_whale["log_volume"], [0] * len(no_whale),
               s=20, alpha=0.3, color="gray", marker="x",
               label=f"0 whale events (n={len(no_whale)})")

    if len(fit_df) >= 5:
        slope, intercept, r, p, se = stats.linregress(
            fit_df["log_volume"], fit_df["whale_effect"])
        xs = np.linspace(fit_df["log_volume"].min(), fit_df["log_volume"].max(), 100)
        ax.plot(xs, intercept + slope * xs, "k--", lw=1.5,
                label=f"OLS slope={slope:+.4f}, r={r:+.2f}, p={p:.3f}")

    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("log₁₀(Market Volume USD)")
    ax.set_ylabel("Whale Effect\n(mean|Δp| whale − mean|Δp| retail)")
    ax.set_title("Cross-Market: Whale Comment Effect\nvs Market Liquidity")
    ax.legend(fontsize=8)

    # Right: whale_abs_dp and retail_abs_dp vs log_volume (two series)
    ax2 = axes[1]
    has_both = df[df["whale_abs_dp"].notna() & df["log_volume"].notna()]
    ax2.scatter(has_both["log_volume"], has_both["whale_abs_dp"],
                alpha=0.5, s=30, color="steelblue", label="Whale mean|Δp|")
    ax2.scatter(df[df["log_volume"].notna()]["log_volume"],
                df[df["log_volume"].notna()]["retail_abs_dp"],
                alpha=0.3, s=20, color="gray", label="Retail mean|Δp|")
    ax2.set_xlabel("log₁₀(Market Volume USD)")
    ax2.set_ylabel("Mean |Δp| (2-hour window)")
    ax2.set_title("Whale vs Retail |Δp|\nAcross Markets")
    ax2.legend(fontsize=8)

    plt.suptitle(
        f"Market-Level Analysis: {len(df)} Markets  |  Whale threshold = $5,000",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out = OUT_DIR / "fig5_cross_market_regression.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nFigure saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-events", type=int, default=MIN_EVENTS)
    parser.add_argument("--min-whale",  type=int, default=MIN_WHALE)
    parser.add_argument("--rebuild",    action="store_true",
                        help="Force rebuild of market_level_effects.csv from raw events.csv files")
    args = parser.parse_args()

    print("Loading market-level effects...")
    df = load_market_effects(rebuild=args.rebuild)
    print(f"Markets loaded: {len(df)}")
    print(f"  With abs-whale events (≥$5k): {df['whale_effect'].notna().sum()}")
    print(f"  With rel-whale events (top-10): {df['rel_whale_effect'].notna().sum() if 'rel_whale_effect' in df else 'N/A'}")
    print(f"  Volume range: ${df['volume_usd'].min():,.0f} – ${df['volume_usd'].max():,.0f}")
    print(f"  log(vol) range: {df['log_volume'].min():.2f} – {df['log_volume'].max():.2f}")

    print("\nTop 20 markets by volume:")
    cols = ["market_id","volume_usd","n_events","n_whale","whale_effect","log_volume"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(20).to_string(index=False))

    # Absolute-whale regression
    print("\n" + "="*60)
    print("REGRESSION 1: Absolute whale (≥$5,000 position)")
    res1 = run_cross_market_regression(df)

    # Relative-whale regression
    if "rel_whale_effect" in df.columns and df["rel_whale_effect"].notna().sum() >= 5:
        print("\n" + "="*60)
        print("REGRESSION 2: Relative whale (top-10 commenters by position)")
        rel_df = df.rename(columns={"rel_whale_effect": "whale_effect"})
        rel_df = rel_df.drop(columns=["whale_effect"], errors="ignore") if "whale_effect" in rel_df.columns else rel_df
        rel_df["whale_effect"] = df["rel_whale_effect"]
        run_cross_market_regression(rel_df)

    make_cross_market_figure(df)

    # Save market-level table
    df.to_csv(OUT_DIR / "market_level_effects.csv", index=False)
    df.to_csv(PROC_DIR / "market_level_effects.csv", index=False)
    print(f"\nTable saved to {OUT_DIR}/market_level_effects.csv")


if __name__ == "__main__":
    main()
