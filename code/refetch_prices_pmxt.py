"""
refetch_prices_pmxt.py — Replace sparse CLOB price histories with true hourly
candles fetched via pmxt's fetch_ohlcv (14-day chunks, 1h resolution).

SAFE TO INTERRUPT AND RESTART: progress saved to
  data/processed/price_refetch_progress.json
Completed markets are skipped on restart.

After completion, re-runs process.py + pool.py so pooled_events.csv
reflects the new dense price data.

Usage:
    python code/refetch_prices_pmxt.py               # all original markets
    python code/refetch_prices_pmxt.py --dry-run      # show plan, no API calls
    python code/refetch_prices_pmxt.py --pool-only    # skip fetch, just reprocess
"""

import argparse
import datetime
import json
import time
from pathlib import Path

import os
import pmxt

ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROGRESS = PROC_DIR / "price_refetch_progress.json"

CHUNK_DAYS  = 14      # max window pmxt accepts for hourly data
SLEEP_CALL  = 0.3     # seconds between API calls
SLEEP_MKT   = 0.5     # seconds between markets


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def parse_date(s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def fetch_token_hourly(poly, token_id: str, start: datetime.datetime,
                       end: datetime.datetime) -> list:
    """
    Fetch complete hourly candle history for one token by chunking into
    14-day windows. Returns list of {t, p} dicts (Unix seconds, close price).
    """
    all_pts = []
    chunk_start = start
    chunk_delta = datetime.timedelta(days=CHUNK_DAYS)

    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_delta, end)
        try:
            candles = poly.fetch_ohlcv(
                token_id,
                resolution="1h",
                start=chunk_start,
                end=chunk_end,
            )
            for c in candles:
                ts_sec = int(c.timestamp / 1000)  # ms → seconds
                all_pts.append({"t": ts_sec, "p": float(c.close)})
        except Exception as e:
            print(f"      Warning: chunk {chunk_start.date()}→{chunk_end.date()} failed: {e}")
        chunk_start = chunk_end
        time.sleep(SLEEP_CALL)

    # Deduplicate and sort
    seen = set()
    unique = []
    for pt in sorted(all_pts, key=lambda x: x["t"]):
        if pt["t"] not in seen:
            seen.add(pt["t"])
            unique.append(pt)
    return unique


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show plan without making API calls")
    parser.add_argument("--pool-only", action="store_true",
                        help="Skip fetch, just reprocess + repool")
    args = parser.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    prog = load_progress()
    already_done = set(prog["done"]) | set(prog["skipped"])

    # Exclude newly collected live markets
    live_prog_path = PROC_DIR / "live_collection_progress.json"
    live_new = set()
    if live_prog_path.exists():
        lp = json.loads(live_prog_path.read_text())
        live_new = set(lp.get("done", [])) | set(lp.get("skipped", []))

    # Gather markets to process
    markets_to_fetch = []
    for p in sorted(RAW_DIR.iterdir()):
        mid = p.name
        if mid in live_new:
            continue
        mj = p / "market.json"
        ph = p / "price_history.json"
        if not mj.exists():
            continue
        try:
            m    = json.loads(mj.read_text())
            raw_tids = m.get("clobTokenIds", [])
            tids = raw_tids if isinstance(raw_tids, list) else json.loads(raw_tids)
            if not tids:
                continue
            start_str = m.get("startDate") or m.get("startDateIso") or ""
            end_str   = m.get("endDate")   or m.get("endDateIso")   or ""
            if not start_str or not end_str:
                continue
            start = parse_date(start_str)
            end   = parse_date(end_str) + datetime.timedelta(days=1)
            markets_to_fetch.append({
                "market_id": mid,
                "token_ids": tids,
                "start": start,
                "end":   end,
                "question": m.get("question", "")[:60],
            })
        except Exception as e:
            print(f"  Skipping {mid}: {e}")

    to_process = [m for m in markets_to_fetch if m["market_id"] not in already_done]

    print(f"Markets with token IDs : {len(markets_to_fetch)}")
    print(f"Already done           : {len(already_done)}")
    print(f"To fetch now           : {len(to_process)}")

    # Estimate time
    total_chunks = sum(
        max((m["end"] - m["start"]).days, 1) // CHUNK_DAYS * len(m["token_ids"])
        for m in to_process
    )
    est_mins = total_chunks * (SLEEP_CALL + 0.8) / 60
    print(f"Estimated API chunks   : {total_chunks:,}")
    print(f"Estimated time         : ~{est_mins:.0f} minutes")

    if args.dry_run:
        print("\n--dry-run: exiting before API calls.")
        return

    if not args.pool_only:
        pmxt_key = os.environ.get("PMXT_API_KEY")
        poly = pmxt.Polymarket(pmxt_api_key=pmxt_key)
        print(f"  pmxt mode: {'hosted (API key set)' if pmxt_key else 'local sidecar (no API key)'}")

        for i, m in enumerate(to_process):
            mid   = m["market_id"]
            tids  = m["token_ids"]
            start = m["start"]
            end   = m["end"]

            print(f"\n[{i+1}/{len(to_process)}] {mid} — {m['question']}")
            print(f"  {start.date()} → {end.date()}  ({(end-start).days} days, {len(tids)} tokens)")

            try:
                price_histories = {}
                for tid in tids:
                    pts = fetch_token_hourly(poly, tid, start, end)
                    price_histories[tid] = pts
                    print(f"  token ...{tid[-6:]}: {len(pts)} hourly pts")

                # Only save if we actually got data — never overwrite with empty
                has_data = any(len(pts) > 0 for pts in price_histories.values())
                if not has_data:
                    print(f"  ⚠ No data returned from pmxt — keeping original price_history.json")
                    prog["skipped"].append(mid)
                    save_progress(prog)
                    time.sleep(SLEEP_MKT)
                    continue

                ph_path = RAW_DIR / mid / "price_history.json"
                ph_path.write_text(json.dumps(price_histories))

                prog["done"].append(mid)

            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                prog["failed"].append({"market_id": mid, "error": str(e)})

            save_progress(prog)
            time.sleep(SLEEP_MKT)

        print(f"\n{'='*55}")
        print(f"Fetch complete: {len(prog['done'])} done, {len(prog['failed'])} failed")

    # Reprocess each market and repool
    print("\nReprocessing all markets...")
    import sys
    import importlib.util

    code_dir = Path(__file__).parent

    def process_market(mid):
        spec = importlib.util.spec_from_file_location("process02", code_dir / "02_process.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.process_market(str(mid))

    failed_proc = []
    all_market_ids = [m["market_id"] for m in markets_to_fetch]
    for i, mid in enumerate(all_market_ids):
        events_path = PROC_DIR / mid / "events.csv"
        # Only reprocess if we just updated the price history
        ph_path = RAW_DIR / mid / "price_history.json"
        if not ph_path.exists():
            continue
        try:
            process_market(mid)
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(all_market_ids)} markets...")
        except Exception as e:
            failed_proc.append(mid)

    print(f"Processing complete. Failed: {len(failed_proc)}")

    # Repool
    print("\nRepooling all markets...")
    spec = importlib.util.spec_from_file_location("pool03", code_dir / "03_pool.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = mod.pool()
    print(f"\nDone. Run 05_generate_figures.py to update figures.")


if __name__ == "__main__":
    main()
