"""
05_collect_sample.py — Collect data for the stratified market sample and
build the market-level effects table for cross-market analysis.

This script is more efficient than 05_sample_markets.py because it:
1. Uses the pre-built sample from the event scan (see build_sample.py output)
2. Reuses fetched event comments when multiple markets share an event
3. Runs process.py for each market to produce events.csv

Usage:
    python code/05_collect_sample.py
    python code/05_collect_sample.py --sample /tmp/final_sample.json
    python code/05_collect_sample.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
SRC_DIR  = ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from utils.api import (
    _get, get_all_comments, get_full_price_history,
    GAMMA_BASE, DATA_BASE,
)
import process as proc

WHALE_THRESHOLD = 5_000


def tag_comments(raw_comments: list, wallet_pos: dict) -> list:
    tagged = []
    for c in raw_comments:
        proxy = (c.get("profile") or {}).get("proxyWallet", "")
        addr  = c.get("userAddress", "")
        pos   = wallet_pos.get(proxy) or wallet_pos.get(addr) or 0.0
        c["_position_usd"] = pos
        tagged.append(c)
    return tagged


def collect_market(market: dict, event_comments_cache: dict,
                   skip_existing: bool, whale_threshold: float) -> bool:
    mid       = str(market["market_id"])
    ev_id     = market["event_id"]
    condition = market.get("condition_id", "")
    vol       = market.get("volume_usd", 0)
    clob_ids  = market.get("clob_token_ids") or []
    if isinstance(clob_ids, str):
        try:   clob_ids = json.loads(clob_ids)
        except: clob_ids = []

    out_raw  = RAW_DIR / mid
    out_proc = PROC_DIR / mid

    if skip_existing and (out_proc / "events.csv").exists():
        print(f"  [{mid}] SKIP (already processed)")
        return True

    # ── price pre-filter ─────────────────────────────────────────────────
    # Fetch a quick sample of the price history to check if this market
    # ever spent time in the uncertain zone (0.10, 0.90).  Markets that
    # converged immediately (long-shot or near-certainty from the start)
    # produce 0 non-convergence events and are useless for the analysis.
    if clob_ids:
        tid0 = str(clob_ids[0])
        try:
            quick = _get(f"https://clob.polymarket.com/prices-history",
                         params={"market": tid0, "interval": "max", "fidelity": 86400})
            hist  = quick.get("history", quick) if isinstance(quick, dict) else quick
            if isinstance(hist, list) and hist:
                prices = [float(pt.get("p", pt.get("price", 0.5))) for pt in hist]
                uncertain = [p for p in prices if 0.10 < p < 0.90]
                if len(uncertain) < 3:
                    print(f"  [{mid}] SKIP — price never uncertain (< 3 pts in 0.10–0.90 band)")
                    return False
        except Exception:
            pass   # If pre-check fails, continue anyway

    out_raw.mkdir(parents=True, exist_ok=True)

    # Fake market.json so process.py can read volumeNum
    mkt_meta = {
        "id": mid,
        "conditionId": condition,
        "volumeNum": vol,
        "volume": vol,
        "question": market.get("question", ""),
        "clobTokenIds": clob_ids,
    }
    (out_raw / "market.json").write_text(json.dumps(mkt_meta))

    # ── comments (cached per event) ─────────────────────────────────────
    if ev_id not in event_comments_cache:
        try:
            event_comments_cache[ev_id] = get_all_comments(ev_id)
        except Exception as exc:
            print(f"  [{mid}] SKIP — comment fetch failed: {exc}")
            return False
    raw_comments = event_comments_cache[ev_id]
    if not raw_comments:
        print(f"  [{mid}] SKIP — 0 comments in event {ev_id}")
        return False

    # ── holders ─────────────────────────────────────────────────────────
    holders_raw = []
    if condition:
        try:
            holders_raw = _get(f"{DATA_BASE}/holders",
                               params={"market": condition, "limit": 100}) or []
        except Exception as exc:
            print(f"  [{mid}] WARN — holders failed: {exc}")

    wallet_pos: dict[str, float] = {}
    for token_entry in holders_raw:
        for h in (token_entry.get("holders") or []):
            w   = h.get("proxyWallet", "")
            amt = float(h.get("amount") or 0)
            wallet_pos[w] = wallet_pos.get(w, 0) + amt

    tagged = tag_comments(raw_comments, wallet_pos)
    (out_raw / "comments_tagged.json").write_text(json.dumps(tagged))
    (out_raw / "holders.json").write_text(json.dumps(holders_raw))

    # ── price history ────────────────────────────────────────────────────
    if not clob_ids:
        print(f"  [{mid}] SKIP — no clobTokenIds")
        return False

    ts_vals = []
    for c in tagged:
        try:
            ts_vals.append(int(pd.Timestamp(c["createdAt"]).timestamp()))
        except Exception:
            pass

    start_ts = (min(ts_vals) - 7 * 86400) if ts_vals else int(time.time()) - 365 * 86400
    end_ts   = (max(ts_vals) + 7 * 86400) if ts_vals else int(time.time())

    price_hist: dict[str, list] = {}
    for tid in clob_ids[:2]:
        try:
            pts = get_full_price_history(str(tid), start_ts=start_ts, end_ts=end_ts)
            price_hist[str(tid)] = pts
        except Exception as exc:
            print(f"  [{mid}] WARN — price history failed for {str(tid)[:12]}: {exc}")
            price_hist[str(tid)] = []

    (out_raw / "price_history.json").write_text(json.dumps(price_hist))

    # ── process ─────────────────────────────────────────────────────────
    try:
        proc.process_market(mid, whale_threshold_usd=whale_threshold)
    except Exception as exc:
        print(f"  [{mid}] WARN — process failed: {exc}")
        return False

    n_wh = sum(1 for c in tagged if c.get("_position_usd", 0) >= whale_threshold)
    print(f"  [{mid}] OK  vol=${vol:>12,.0f}  comments={len(tagged):>5}  whales={n_wh:>3}")
    return True


def build_market_effects_table() -> pd.DataFrame:
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
        if len(nc) < 5:
            continue
        vol = float(ev["market_volume"].iloc[0]) if "market_volume" in ev.columns else 0
        lbl = ev["market_label"].iloc[0]         if "market_label"   in ev.columns else p.name
        wh  = nc[nc["is_whale"]]["abs_price_change"].dropna()
        rt  = nc[~nc["is_whale"]]["abs_price_change"].dropna()
        if len(rt) < 5:
            continue
        rows.append({
            "market_id":     p.name,
            "market_label":  lbl,
            "volume_usd":    vol,
            "log_volume":    np.log10(vol) if vol > 0 else np.nan,
            "n_events":      len(nc),
            "n_whale":       int(len(wh)),
            "n_retail":      int(len(rt)),
            "whale_abs_dp":  float(wh.mean()) if len(wh) >= 1 else np.nan,
            "retail_abs_dp": float(rt.mean()),
            "whale_effect":  float(wh.mean() - rt.mean()) if len(wh) >= 1 else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("volume_usd", ascending=False)
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="/tmp/final_sample.json",
                        help="Path to sampled market JSON list")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--whale-threshold", type=float, default=WHALE_THRESHOLD)
    parser.add_argument("--max-markets", type=int, default=None,
                        help="Cap on markets to process (for testing)")
    args = parser.parse_args()

    sample_path = Path(args.sample)
    if not sample_path.exists():
        print(f"Sample file not found: {sample_path}")
        print("Run code/05_sample_markets.py first, or point --sample at your JSON file.")
        sys.exit(1)

    markets = json.loads(sample_path.read_text())
    if args.max_markets:
        markets = markets[: args.max_markets]

    print(f"Processing {len(markets)} markets")

    # Group by event_id so we can cache comments per event
    event_to_markets: dict[int, list] = defaultdict(list)
    for m in markets:
        event_to_markets[m["event_id"]].append(m)

    event_comments_cache: dict[int, list] = {}
    n_ok = n_skip = n_fail = 0

    for i, market in enumerate(markets, 1):
        mid = str(market["market_id"])
        vol = market.get("volume_usd", 0)
        print(f"\n[{i}/{len(markets)}] market {mid}  ${vol:,.0f}  {market.get('question','')[:55]}")

        ok = collect_market(market, event_comments_cache, args.skip_existing,
                            args.whale_threshold)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
        time.sleep(0.3)

    print(f"\nDone: {n_ok} OK  {n_fail} failed  {n_skip} skipped")

    print("\nBuilding market-level effects table...")
    mdf = build_market_effects_table()
    out = PROC_DIR / "market_level_effects.csv"
    mdf.to_csv(out, index=False)
    print(f"  {len(mdf)} markets → {out}")

    n_with_whale = mdf["whale_effect"].notna().sum()
    print(f"  Markets with ≥1 whale event: {n_with_whale}")
    if n_with_whale >= 5:
        import statsmodels.formula.api as smf
        res = smf.ols("whale_effect ~ log_volume",
                      data=mdf[mdf["whale_effect"].notna() & mdf["log_volume"].notna()]
                      ).fit(cov_type="HC3")
        print("\nCross-market OLS: whale_effect ~ log_volume")
        print(res.summary2().tables[1])


if __name__ == "__main__":
    main()
