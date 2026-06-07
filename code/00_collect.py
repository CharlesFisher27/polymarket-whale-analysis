"""
Data collection pipeline for a single Polymarket market.

Strategy:
  1. Fetch top holders (whales defined by on-chain position size)
  2. Fetch all comments for the event
  3. Cross-reference: which whales also left comments?
  4. Fetch price history for the event-study analysis

This is fast because we only look up positions for top-N holders, not all 10k+ commenters.

Usage:
    python code/00_collect.py --market <market_id> --event-id <event_id>
    python code/00_collect.py --market 253591 --event-id 903193 --top-holders 100
"""

import json
import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "utils"))
import api

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def collect_market(market_id: str, event_id: int = None, top_holders: int = 200):
    out_dir = RAW_DIR / market_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Market metadata ----
    print("[1/4] Fetching market metadata...")
    market = api.get_market(market_id)
    with open(out_dir / "market.json", "w") as f:
        json.dump(market, f, indent=2)
    print(f"      {market.get('question', market_id)}")
    print(f"      conditionId: {market.get('conditionId','')[:20]}...")

    # ---- 2. Top holders (defines our whale universe) ----
    print(f"\n[2/4] Fetching top {top_holders} holders...")
    condition_id = market.get("conditionId", "")
    holders_raw = api.get_holders(condition_id, limit=top_holders) or []
    with open(out_dir / "holders.json", "w") as f:
        json.dump(holders_raw, f, indent=2)

    # Build wallet -> position_usd mapping
    # Each entry is {token, holders: [{proxyWallet, amount, outcomeIndex, ...}]}
    whale_positions: dict[str, float] = {}
    for token_entry in holders_raw:
        for h in (token_entry.get("holders") or []):
            wallet = h.get("proxyWallet", "")
            amt = float(h.get("amount") or 0)
            whale_positions[wallet] = whale_positions.get(wallet, 0) + amt

    # Sort descending by position size
    whale_positions = dict(sorted(whale_positions.items(), key=lambda x: x[1], reverse=True))
    print(f"      {len(whale_positions)} unique whale wallets")
    if whale_positions:
        top3 = list(whale_positions.items())[:3]
        for w, amt in top3:
            print(f"      {w[:16]}... ${amt:,.0f}")

    # ---- 3. Comments ----
    print(f"\n[3/4] Fetching comments (event {event_id})...")
    if event_id is None:
        event_id = api.get_event_id_for_market(market)
    if not event_id:
        print("      Warning: no event ID — skipping comments. Use --event-id.")
        comments = []
    else:
        comments = api.get_all_comments(event_id)
    with open(out_dir / "comments_raw.json", "w") as f:
        json.dump(comments, f, indent=2)
    print(f"      {len(comments)} total comments")

    # Tag each comment with whale status using the profile proxyWallet
    for c in comments:
        proxy = (c.get("profile") or {}).get("proxyWallet", "")
        base  = c.get("userAddress", "")
        c["_position_usd"] = whale_positions.get(proxy, whale_positions.get(base, 0))
        c["_is_whale"] = c["_position_usd"] > 0  # any holder is "whale" for now
    whale_comments = [c for c in comments if c["_is_whale"]]
    print(f"      {len(whale_comments)} comments from top-{top_holders} holders")

    with open(out_dir / "comments_tagged.json", "w") as f:
        json.dump(comments, f, indent=2)

    # ---- 4. Price history ----
    # Use chunked weekly fetcher so closed (historical) markets are covered too.
    print(f"\n[4/4] Fetching price history (hourly)...")
    clob_token_ids = _extract_token_ids(market)

    # Determine market time range from metadata
    import datetime
    start_str = market.get("startDate") or market.get("startDateIso") or "2024-01-01T00:00:00Z"
    end_str   = market.get("endDate")   or market.get("endDateIso")   or "2025-01-01T00:00:00Z"
    def _to_ts(s):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return int(datetime.datetime.strptime(s[:26], fmt[:len(fmt)]).timestamp())
            except Exception:
                pass
        return int(datetime.datetime.fromisoformat(s.replace("Z","+00:00")).timestamp())

    start_ts = _to_ts(start_str)
    end_ts   = _to_ts(end_str) + 86400  # +1 day buffer

    price_histories = {}
    for tid in clob_token_ids:
        pts = api.get_full_price_history(tid, start_ts=start_ts, end_ts=end_ts, fidelity=3600)
        price_histories[tid] = pts
        print(f"      token ...{tid[-8:]}: {len(pts)} hourly price points")
    with open(out_dir / "price_history.json", "w") as f:
        json.dump(price_histories, f, indent=2)

    print(f"\n Done. Raw data in {out_dir}/")
    print(f"  Next: python code/02_process.py --market {market_id}")


def _extract_token_ids(market: dict) -> list[str]:
    for field in ("clobTokenIds", "clob_token_ids", "tokenIds"):
        val = market.get(field)
        if val:
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    pass
            if isinstance(val, list):
                return val
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Polymarket data for whale-comment analysis")
    parser.add_argument("--market", required=True, help="Polymarket market ID (from browse_markets.py)")
    parser.add_argument("--event-id", type=int, default=None,
                        help="Event ID for fetching comments (shown in browse_markets.py)")
    parser.add_argument("--top-holders", type=int, default=200,
                        help="Number of top holders to define as whale universe (default: 200)")
    args = parser.parse_args()
    collect_market(args.market, event_id=args.event_id, top_holders=args.top_holders)
