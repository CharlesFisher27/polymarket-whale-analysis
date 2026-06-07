"""
enrich_positions.py — Fill in missing position_usd values by querying the
Polymarket Data API for every commenter wallet that currently has position_usd=0.

SAFE TO INTERRUPT AND RESTART: all API responses are cached to
  data/processed/position_cache.json
before anything is written to the output CSV. A restart skips already-queried
wallets and picks up where it left off.

Output: data/processed/pooled_events_enriched.csv
  Same schema as pooled_events.csv but with position_usd and is_whale
  updated for every wallet/market pair we could resolve.

Usage:
    python code/enrich_positions.py                     # full run
    python code/enrich_positions.py --max-wallets 500   # partial run / test
    python code/enrich_positions.py --apply-only        # skip API, just apply cache
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
PROC       = ROOT / "data" / "processed"
RAW        = ROOT / "data" / "raw"
CACHE_FILE = PROC / "position_cache.json"
INPUT_CSV  = PROC / "pooled_events.csv"
OUTPUT_CSV = PROC / "pooled_events_enriched.csv"

WHALE_THRESHOLD = 5_000
RATE_SLEEP      = 0.25   # seconds between API calls
SAVE_EVERY      = 100    # flush cache to disk every N wallets


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            print("  Warning: cache file corrupt, starting fresh.")
    return {}


def save_cache(cache: dict) -> None:
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(CACHE_FILE)   # atomic on most filesystems


def fetch_positions(wallet: str) -> list:
    """
    Hit the Data API for a wallet's positions. Returns list of position dicts,
    or [] on any error. Uses curl so the User-Agent header matches what works.
    """
    url = f"https://data-api.polymarket.com/positions?user={wallet}&limit=500"
    try:
        result = subprocess.run(
            ["curl", "-s", "-A", "Mozilla/5.0", "--max-time", "10", url],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        data = json.loads(result.stdout)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def build_condition_map() -> tuple[dict, dict]:
    """
    Returns:
        mkt_to_cid : {market_label_str -> conditionId_lower}
        cid_to_mkt : {conditionId_lower -> market_label_str}
    """
    mkt_to_cid: dict[str, str] = {}
    for p in RAW.iterdir():
        mj = p / "market.json"
        if not mj.exists():
            continue
        try:
            m   = json.loads(mj.read_text())
            cid = m.get("conditionId", "").lower()
            if cid:
                mkt_to_cid[p.name] = cid
        except Exception:
            pass
    cid_to_mkt = {v: k for k, v in mkt_to_cid.items()}
    return mkt_to_cid, cid_to_mkt


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-wallets", type=int, default=None,
                        help="Stop after querying this many wallets (for testing)")
    parser.add_argument("--apply-only", action="store_true",
                        help="Skip API calls; just apply existing cache to CSV")
    args = parser.parse_args()

    # 1. Load data
    print("Loading pooled_events.csv...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df["market_label"] = df["market_label"].astype(str)
    print(f"  {len(df):,} rows  |  {df['market_label'].nunique()} markets  "
          f"|  {df['proxy_wallet'].nunique():,} unique wallets")

    # 2. Build market → conditionId map
    mkt_to_cid, _ = build_condition_map()
    print(f"  conditionIds loaded for {len(mkt_to_cid)} markets")

    # 3. Load cache
    cache = load_cache()
    print(f"  Cache: {len(cache):,} wallets already resolved")

    # 4. Determine which wallets need querying
    if not args.apply_only:
        zero_mask    = df["position_usd"].fillna(0) == 0
        zero_wallets = df.loc[zero_mask, "proxy_wallet"].dropna().unique().tolist()
        to_query     = [w for w in zero_wallets if w not in cache]

        if args.max_wallets is not None:
            to_query = to_query[: args.max_wallets]

        print(f"\n  Wallets with position_usd=0 : {len(zero_wallets):,}")
        print(f"  Not yet in cache            : {len(to_query):,}")
        if args.max_wallets:
            print(f"  (capped at --max-wallets={args.max_wallets})")
        print()

        # 5. Query and cache
        found_count = 0
        for i, wallet in enumerate(to_query):
            positions = fetch_positions(wallet)

            wallet_map: dict[str, dict] = {}
            for pos in positions:
                cid = pos.get("conditionId", "").lower()
                if not cid:
                    continue
                wallet_map[cid] = {
                    "position_usd":  float(pos.get("currentValue") or 0),
                    "outcome":       pos.get("outcome", ""),
                    "outcomeIndex":  pos.get("outcomeIndex", -1),
                }
            cache[wallet] = wallet_map
            if wallet_map:
                found_count += 1

            # Save periodically
            if (i + 1) % SAVE_EVERY == 0:
                save_cache(cache)
                pct = (i + 1) / len(to_query) * 100
                print(f"  [{i+1:>{len(str(len(to_query)))}}/{len(to_query)}]  "
                      f"{pct:.1f}%  |  wallets with any position: {found_count}  "
                      f"|  cache saved ✓")

            time.sleep(RATE_SLEEP)

        # Final save
        save_cache(cache)
        print(f"\n  Done querying. {found_count:,} of {len(to_query):,} "
              f"wallets had at least one position.")
    else:
        print("  --apply-only: skipping API calls, using existing cache.\n")

    # 6. Apply cache to DataFrame
    print("\nApplying cache to dataset...")
    df["_market_cid"] = df["market_label"].map(mkt_to_cid).str.lower()

    new_pos = df["position_usd"].copy().fillna(0)
    enriched_rows = 0

    for idx, row in df.iterrows():
        if float(new_pos.at[idx] or 0) > 0:
            continue  # already had position data — keep original
        wallet = str(row.get("proxy_wallet") or "")
        cid    = row.get("_market_cid") or ""
        if not wallet or not cid:
            continue
        wallet_map = cache.get(wallet, {})
        pos_info   = wallet_map.get(cid, {})
        usd        = float(pos_info.get("position_usd") or 0)
        if usd > 0:
            new_pos.at[idx] = usd
            enriched_rows  += 1

    df["position_usd"] = new_pos
    df["is_whale"]     = df["position_usd"] >= WHALE_THRESHOLD
    df = df.drop(columns=["_market_cid"], errors="ignore")

    # 7. Save output
    df.to_csv(OUTPUT_CSV, index=False)

    # 8. Report
    print(f"\n{'='*55}")
    print(f"  Rows enriched with new position data : {enriched_rows:,}")
    print(f"  Total whale events                   : {df['is_whale'].sum():,}")
    n_whale_markets = df[df["is_whale"]]["market_label"].nunique()
    print(f"  Markets with whale events             : {n_whale_markets}")
    print(f"  Output → {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"  Cache  → {CACHE_FILE.relative_to(ROOT)}  "
          f"({len(cache):,} wallets)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
