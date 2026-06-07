"""
Fetch per-wallet trade histories for every whale who commented in each market.

This enables two analyses that hourly price candles cannot support:
  1. Pre-comment trading: did the whale build or exit their position BEFORE commenting?
  2. Post-comment trading: did the whale sell into any price impact (pump/dump)?

Usage:
    python code/01_collect_whale_trades.py                  # all four markets
    python code/01_collect_whale_trades.py --market 253591  # single market

Output per market:
    data/raw/<market_id>/whale_trades.json
        {
          "<proxy_wallet>": [
            {"timestamp": 1730..., "side": "BUY"|"SELL",
             "size": 94.37, "price": 0.998, "asset": "<token_id>", ...},
            ...
          ],
          ...
        }
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "utils"))
import api

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

MARKETS = {
    "253591": "Trump 2024 Win",
    "511754": "Trump Inauguration",
    "512340": "Romanian Election",
    "559687": "Oprah 2028 Dem Nom",
}


def _get_whale_wallets(market_id: str, whale_threshold: float = 5000) -> dict[str, float]:
    """
    Return {proxy_wallet: position_usd} for every commenter above the threshold.
    Falls back to userAddress if proxyWallet is missing.
    """
    path = RAW_DIR / market_id / "comments_tagged.json"
    comments = json.loads(path.read_text())
    wallets: dict[str, float] = {}
    for c in comments:
        pos = float(c.get("_position_usd") or 0)
        if pos < whale_threshold:
            continue
        proxy = (c.get("profile") or {}).get("proxyWallet", "")
        addr  = c.get("userAddress", "")
        wallet = proxy or addr
        if wallet:
            wallets[wallet] = max(wallets.get(wallet, 0), pos)
    return wallets


def collect_whale_trades(market_id: str, whale_threshold: float = 5000) -> dict:
    label = MARKETS.get(market_id, market_id)
    market_meta = json.loads((RAW_DIR / market_id / "market.json").read_text())
    condition_id = market_meta.get("conditionId", "")
    if not condition_id:
        print(f"  [{label}] No conditionId — skipping")
        return {}

    wallets = _get_whale_wallets(market_id, whale_threshold)
    print(f"\n[{label}]  conditionId={condition_id[:20]}...")
    print(f"  {len(wallets)} unique whale wallets to fetch")

    result: dict[str, list] = {}
    for i, (wallet, pos) in enumerate(
        sorted(wallets.items(), key=lambda kv: -kv[1])
    ):
        trades = api.get_all_user_market_trades(wallet, condition_id)
        result[wallet] = trades
        sides = [t.get("side","?") for t in trades]
        buys  = sides.count("BUY")
        sells = sides.count("SELL")
        print(f"  [{i+1:>2}/{len(wallets)}] {wallet[:22]}...  "
              f"pos=${pos:>10,.0f}  trades={len(trades)}  "
              f"(B:{buys} S:{sells})")
        time.sleep(0.2)

    out_path = RAW_DIR / market_id / "whale_trades.json"
    out_path.write_text(json.dumps(result, indent=2))
    total_trades = sum(len(v) for v in result.values())
    print(f"  → Saved {total_trades} trades for {len(result)} wallets → {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", help="Single market ID (default: all)")
    parser.add_argument("--whale-threshold", type=float, default=5000)
    args = parser.parse_args()

    targets = [args.market] if args.market else list(MARKETS.keys())
    for mid in targets:
        if not (RAW_DIR / mid / "comments_tagged.json").exists():
            print(f"SKIP {mid} — no comments_tagged.json")
            continue
        collect_whale_trades(mid, args.whale_threshold)
