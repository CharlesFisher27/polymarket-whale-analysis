"""
05_sample_markets.py — Enumerate and stratified-sample closed binary markets,
then collect the minimum data (comments, prices, holders) needed for a
market-level whale-effect analysis.

Market-level analysis rationale
--------------------------------
The professor asked for the *market* as the unit of analysis so that we can
regress the within-market whale comment effect on log(market volume) without
the perfect-multicollinearity that arises when market FEs and log-volume are
both included in a pooled OLS.

For each sampled market we compute:
    whale_effect_m  = mean |Δp| (whale events) − mean |Δp| (retail events)
    log_volume_m    = log10(market_volume_usd)

Then the cross-market regression is:
    whale_effect_m ~ β0 + β1 * log_volume_m + ε_m

Sampling strategy (stratified by log-volume bin):
    Tier 1  >$1M          include all (≈12 markets)
    Tier 2  $100k–$1M     include all (≈30)
    Tier 3  $10k–$100k    include up to 120
    Tier 4  $1k–$10k      include up to 80
    Tier 5  <$1k          include up to 40
    ─────────────────────────────────────────
    Target  250–300 markets (plus the 6 already processed)

Usage:
    python code/05_sample_markets.py                    # run everything
    python code/05_sample_markets.py --list-only        # print market list, don't collect
    python code/05_sample_markets.py --skip-existing    # don't re-collect already-processed

Outputs (written into the standard data/ tree):
    data/raw/<market_id>/comments_tagged.json
    data/raw/<market_id>/price_history.json
    data/raw/<market_id>/holders.json
    data/raw/<market_id>/market.json
    data/processed/<market_id>/events.csv   (via process.py)
    data/processed/market_level_effects.csv  (market-level summary)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
SRC_DIR    = ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from utils.api import (
    _get,
    get_all_comments,
    get_full_price_history,
    GAMMA_BASE,
    DATA_BASE,
)
import process as proc            # src/process.py

# ── sampling parameters ────────────────────────────────────────────────────
WHALE_THRESHOLD   = 5_000          # USD — consistent with existing analysis
CONVERGENCE_LO    = 0.10
CONVERGENCE_HI    = 0.90
MAX_COMMENT_FETCH = 3_000          # skip markets with more comments than this
                                   # (already-processed markets exceed this)
TIER_CAPS = {
    ">1M":       None,    # all
    "100k-1M":   None,    # all
    "10k-100k":  120,
    "1k-10k":    80,
    "<1k":       40,
}


def volume_tier(vol: float) -> str:
    if vol >= 1_000_000:  return ">1M"
    if vol >= 100_000:    return "100k-1M"
    if vol >= 10_000:     return "10k-100k"
    if vol >= 1_000:      return "1k-10k"
    return "<1k"


def _parse_outcomes(m: dict) -> list:
    o = m.get("outcomes", "[]")
    if isinstance(o, list):
        return o
    try:
        return json.loads(o)
    except Exception:
        return []


def fetch_market_list(page_limit: int = 12) -> list[dict]:
    """Fetch all closed binary markets from the Gamma API."""
    markets: list[dict] = []
    offset = 0
    limit  = 500
    for _ in range(page_limit):
        batch = _get(f"{GAMMA_BASE}/markets", params={
            "limit": limit, "offset": offset, "closed": "true",
        })
        if not batch:
            break
        markets.extend(batch)
        offset += limit
        time.sleep(0.3)
    binary = [
        m for m in markets
        if _parse_outcomes(m) == ["Yes", "No"]
        and not m.get("negRiskOther", False)
        and float(m.get("volumeNum") or 0) > 0
    ]
    return binary


def stratified_sample(markets: list[dict], skip_ids: set[str]) -> list[dict]:
    """Stratified sample with tier caps, excluding already-processed IDs."""
    tier_buckets: dict[str, list[dict]] = {t: [] for t in TIER_CAPS}
    for m in markets:
        mid = str(m["id"])
        if mid in skip_ids:
            continue
        vol = float(m.get("volumeNum") or 0)
        tier_buckets[volume_tier(vol)].append(m)

    # Sort each tier by volume descending (prefer larger within tier)
    selected: list[dict] = []
    for tier, cap in TIER_CAPS.items():
        bucket = sorted(tier_buckets[tier],
                        key=lambda x: float(x.get("volumeNum") or 0),
                        reverse=True)
        selected.extend(bucket[:cap])

    return selected


# ── data-collection helpers ────────────────────────────────────────────────

def collect_market(market: dict, whale_threshold: float) -> bool:
    """
    Collect raw data for a single market and run process.py.
    Returns True on success, False if skipped or errored.
    """
    mid        = str(market["id"])
    condition  = market.get("conditionId", "")
    vol        = float(market.get("volumeNum") or 0)
    out_raw    = RAW_DIR / mid
    out_raw.mkdir(parents=True, exist_ok=True)

    # ── 1. market metadata ──────────────────────────────────────────────
    (out_raw / "market.json").write_text(json.dumps(market))

    # ── 2. comments ─────────────────────────────────────────────────────
    events = market.get("events") or []
    event_id = None
    if events:
        e0 = events[0]
        event_id = e0.get("id") if isinstance(e0, dict) else e0
    if event_id is None:
        print(f"  [{mid}] SKIP — no event ID")
        return False

    try:
        raw_comments = get_all_comments(event_id)
    except Exception as exc:
        print(f"  [{mid}] SKIP — comment fetch failed: {exc}")
        return False

    if len(raw_comments) > MAX_COMMENT_FETCH:
        print(f"  [{mid}] SKIP — {len(raw_comments):,} comments exceeds cap "
              f"(market already processed separately)")
        return False

    if not raw_comments:
        print(f"  [{mid}] SKIP — 0 comments")
        return False

    # ── 3. holders ──────────────────────────────────────────────────────
    if not condition:
        print(f"  [{mid}] SKIP — no conditionId")
        return False
    try:
        holders_raw = _get(f"{DATA_BASE}/holders",
                           params={"market": condition, "limit": 100})
        if not holders_raw:
            holders_raw = []
    except Exception as exc:
        print(f"  [{mid}] WARN — holder fetch failed: {exc}; using empty")
        holders_raw = []

    # Tag comments with position USD using holder map
    wallet_pos: dict[str, float] = {}
    for token_entry in (holders_raw or []):
        for h in (token_entry.get("holders") or []):
            w   = h.get("proxyWallet", "")
            amt = float(h.get("amount") or 0)
            wallet_pos[w] = wallet_pos.get(w, 0) + amt

    tagged = []
    for c in raw_comments:
        proxy = (c.get("profile") or {}).get("proxyWallet", "")
        addr  = c.get("userAddress", "")
        pos   = wallet_pos.get(proxy) or wallet_pos.get(addr) or 0.0
        c["_position_usd"] = pos
        tagged.append(c)

    (out_raw / "comments_tagged.json").write_text(json.dumps(tagged))
    (out_raw / "holders.json").write_text(json.dumps(holders_raw))

    # ── 4. price history ────────────────────────────────────────────────
    token_ids = market.get("clobTokenIds")
    if isinstance(token_ids, str):
        try:
            token_ids = json.loads(token_ids)
        except Exception:
            token_ids = []
    if not token_ids:
        print(f"  [{mid}] SKIP — no clobTokenIds")
        return False

    # Estimate market lifespan from comments timestamps
    ts_vals = [c.get("createdAt") for c in tagged if c.get("createdAt")]
    if ts_vals:
        import re
        unix_ts = []
        for ts in ts_vals:
            try:
                unix_ts.append(int(pd.Timestamp(ts).timestamp()))
            except Exception:
                pass
        start_ts = min(unix_ts) - 7 * 86400 if unix_ts else int(time.time()) - 365 * 86400
        end_ts   = max(unix_ts) + 7 * 86400 if unix_ts else int(time.time())
    else:
        start_ts = int(time.time()) - 365 * 86400
        end_ts   = int(time.time())

    price_hist: dict[str, list] = {}
    for tid in token_ids[:2]:   # YES and NO tokens
        try:
            pts = get_full_price_history(tid, start_ts=start_ts, end_ts=end_ts)
            price_hist[tid] = pts
        except Exception as exc:
            print(f"  [{mid}] WARN — price history failed for {tid[:12]}: {exc}")
            price_hist[tid] = []

    (out_raw / "price_history.json").write_text(json.dumps(price_hist))

    # ── 5. process ──────────────────────────────────────────────────────
    try:
        proc.process_market(mid, whale_threshold_usd=whale_threshold)
    except Exception as exc:
        print(f"  [{mid}] WARN — process failed: {exc}")
        return False

    n_wh = sum(1 for c in tagged if c.get("_position_usd", 0) >= whale_threshold)
    print(f"  [{mid}] OK  vol=${vol:>12,.0f}  "
          f"comments={len(tagged):>5}  whales={n_wh:>3}")
    return True


# ── market-level summary ───────────────────────────────────────────────────

def build_market_level_effects(market_ids: list[str]) -> pd.DataFrame:
    """
    For each processed market, compute the within-market whale effect and
    return a market-level DataFrame.
    """
    rows = []
    for mid in market_ids:
        evpath = PROC_DIR / mid / "events.csv"
        if not evpath.exists():
            continue
        ev = pd.read_csv(evpath, parse_dates=["timestamp"])
        nc = ev[
            ev["price_change"].notna() &
            ~ev["is_convergence_period"]
        ]
        if len(nc) < 10:
            continue
        vol  = float(ev["market_volume"].iloc[0]) if "market_volume" in ev.columns else 0
        lbl  = ev["market_label"].iloc[0]   if "market_label"   in ev.columns else mid

        wh = nc[nc["is_whale"]]["abs_price_change"].dropna()
        rt = nc[~nc["is_whale"]]["abs_price_change"].dropna()
        if len(rt) < 5:
            continue

        whale_effect  = wh.mean() - rt.mean() if len(wh) >= 1 else np.nan
        retail_mean   = rt.mean()
        whale_mean    = wh.mean() if len(wh) >= 1 else np.nan

        rows.append({
            "market_id":     mid,
            "market_label":  lbl,
            "volume_usd":    vol,
            "log_volume":    np.log10(vol) if vol > 0 else np.nan,
            "n_events":      len(nc),
            "n_whale":       len(wh),
            "n_retail":      len(rt),
            "whale_abs_dp":  whale_mean,
            "retail_abs_dp": retail_mean,
            "whale_effect":  whale_effect,   # main DV for cross-market regression
        })

    df = pd.DataFrame(rows).sort_values("volume_usd", ascending=False)
    return df.reset_index(drop=True)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-only",    action="store_true",
                        help="Print the sampled market list without collecting data")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip markets already in data/processed/")
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false")
    parser.add_argument("--whale-threshold", type=float, default=WHALE_THRESHOLD)
    parser.add_argument("--max-comments",    type=int,   default=MAX_COMMENT_FETCH)
    args = parser.parse_args()

    # Update module-level comment cap before calling collect_market
    import sys as _sys
    _sys.modules[__name__].MAX_COMMENT_FETCH = args.max_comments

    # 1. Already-processed market IDs
    skip_ids: set[str] = set()
    if args.skip_existing:
        skip_ids = {
            p.name for p in PROC_DIR.iterdir()
            if p.is_dir() and (p / "events.csv").exists()
        }
        print(f"Skipping {len(skip_ids)} already-processed markets")

    # 2. Fetch full market list
    print("Fetching closed binary markets from Gamma API...")
    all_binary = fetch_market_list()
    print(f"Found {len(all_binary)} closed binary markets with volume > 0")

    # 3. Stratified sample
    sampled = stratified_sample(all_binary, skip_ids)
    print(f"\nStratified sample: {len(sampled)} markets to collect")
    tier_counts: dict[str, int] = {}
    for m in sampled:
        t = volume_tier(float(m.get("volumeNum") or 0))
        tier_counts[t] = tier_counts.get(t, 0) + 1
    for tier, n in sorted(tier_counts.items()):
        print(f"  {tier}: {n}")

    if args.list_only:
        print("\n--- Market list ---")
        for m in sampled:
            print(f"  {m['id']:>8}  ${float(m.get('volumeNum',0)):>12,.0f}  {m.get('question','')[:60]}")
        return

    # 4. Collect
    print("\nCollecting data...")
    n_ok = 0
    for i, market in enumerate(sampled, 1):
        mid = str(market["id"])
        vol = float(market.get("volumeNum") or 0)
        print(f"\n[{i}/{len(sampled)}] market {mid}  ${vol:,.0f}  {market.get('question','')[:55]}")
        ok = collect_market(market, args.whale_threshold)
        if ok:
            n_ok += 1
        time.sleep(0.5)   # gentle pacing

    print(f"\nCollection complete: {n_ok}/{len(sampled)} markets OK")

    # 5. Build market-level effects table (all processed markets)
    print("\nBuilding market-level effects table...")
    all_market_ids = [p.name for p in PROC_DIR.iterdir() if p.is_dir() and (p / "events.csv").exists()]
    mdf = build_market_level_effects(all_market_ids)
    out = PROC_DIR / "market_level_effects.csv"
    mdf.to_csv(out, index=False)
    print(f"Market-level effects: {len(mdf)} markets → {out}")
    print(mdf[["market_id","volume_usd","n_whale","whale_effect","log_volume"]].to_string(index=False))


if __name__ == "__main__":
    main()
