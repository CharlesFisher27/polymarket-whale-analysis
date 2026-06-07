"""
collect_live_markets.py — Discover and collect ~100 live Polymarket markets,
then re-process and re-pool everything into an updated pooled_events.csv.

SAFE TO INTERRUPT AND RESTART: progress is saved to
  data/processed/live_collection_progress.json
after every market. A restart skips already-collected markets.

Pipeline per market:
  1. Discover live markets via Gamma API (sorted by volume desc)
  2. collect_market()  → data/raw/{market_id}/
  3. process_market()  → data/processed/{market_id}/
  4. Re-pool all markets → data/processed/pooled_events.csv

Usage:
    python code/collect_live_markets.py               # full run (~100 markets)
    python code/collect_live_markets.py --target 20   # quick test
    python code/collect_live_markets.py --pool-only   # skip collection, just re-pool
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import requests

ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROGRESS = PROC_DIR / "live_collection_progress.json"

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

import api as polyapi
from collect_live_markets_helpers import collect_market_safe, process_market_safe, pool_all


# ── Progress helpers ───────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text())
        except Exception:
            pass
    return {"done": [], "failed": [], "skipped": []}


def save_progress(prog: dict) -> None:
    tmp = PROGRESS.with_suffix(".tmp")
    tmp.write_text(json.dumps(prog, indent=2))
    tmp.replace(PROGRESS)


# ── Market discovery ───────────────────────────────────────────────────────────

def discover_live_markets(target: int = 120) -> list[dict]:
    """
    Fetch live markets from the Gamma API sorted by volume descending.
    Filters for markets with volume > $50k and an event_id (needed for comments).
    Returns list of {market_id, event_id, question, volume} dicts.
    """
    print("Discovering live markets...")
    candidates = []
    offset = 0
    limit  = 100

    while len(candidates) < target * 2:   # over-fetch so we have room to filter
        batch = polyapi.get_markets(limit=limit, offset=offset, closed=False)
        if not batch:
            break
        for m in batch:
            vol = float(m.get("volumeNum") or m.get("volume") or 0)
            if vol < 50_000:
                continue
            mid = str(m.get("id", ""))
            if not mid:
                continue
            event_id = None
            events = m.get("events", [])
            if events:
                first = events[0]
                event_id = first.get("id") if isinstance(first, dict) else first
            if not event_id:
                continue
            candidates.append({
                "market_id": mid,
                "event_id":  int(event_id),
                "question":  m.get("question", "")[:80],
                "volume":    vol,
            })
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.3)

    # Sort by volume descending, deduplicate on market_id
    seen = set()
    unique = []
    for c in sorted(candidates, key=lambda x: x["volume"], reverse=True):
        if c["market_id"] not in seen:
            seen.add(c["market_id"])
            unique.append(c)

    print(f"  Found {len(unique)} live markets with volume > $50k")
    return unique[:target]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",    type=int, default=100,
                        help="Number of live markets to collect (default: 100)")
    parser.add_argument("--top-holders", type=int, default=200,
                        help="Holders to fetch per market (default: 200)")
    parser.add_argument("--pool-only", action="store_true",
                        help="Skip collection, just re-pool existing markets")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip markets already in data/raw/ (default: True)")
    args = parser.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    prog = load_progress()
    already_done = set(prog["done"]) | set(prog["skipped"])

    if not args.pool_only:
        # 1. Discover
        markets = discover_live_markets(target=args.target + 20)  # buffer for skips/failures

        # 2. Filter already-done and already-in-raw
        existing_raw = {p.name for p in RAW_DIR.iterdir() if p.is_dir()}
        to_collect = []
        for m in markets:
            mid = m["market_id"]
            if mid in already_done:
                continue
            if args.skip_existing and mid in existing_raw:
                prog["skipped"].append(mid)
                already_done.add(mid)
                continue
            to_collect.append(m)
            if len(to_collect) >= args.target:
                break

        save_progress(prog)
        print(f"\n  Already collected (this run): {len(prog['done'])}")
        print(f"  Skipped (already in raw/):   {len(prog['skipped'])}")
        print(f"  To collect now:              {len(to_collect)}")
        print()

        # 3. Collect + process each market
        for i, m in enumerate(to_collect):
            mid      = m["market_id"]
            event_id = m["event_id"]
            vol      = m["volume"]
            q        = m["question"]

            print(f"\n{'─'*60}")
            print(f"[{i+1}/{len(to_collect)}]  Market {mid}  (${vol:,.0f})")
            print(f"  {q}")

            try:
                # Collect
                collect_market_safe(mid, event_id, top_holders=args.top_holders)
                # Process
                process_market_safe(mid)
                prog["done"].append(mid)
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                traceback.print_exc()
                prog["failed"].append({"market_id": mid, "error": str(e)})

            save_progress(prog)
            time.sleep(0.5)   # be polite between markets

        print(f"\n{'='*60}")
        print(f"Collection complete.")
        print(f"  Done:   {len(prog['done'])}")
        print(f"  Failed: {len(prog['failed'])}")

    # 4. Re-pool everything
    print("\nRe-pooling all markets into pooled_events.csv...")
    pool_all()
    print("Done. Run 05_generate_figures.py to update figures.")


if __name__ == "__main__":
    main()
