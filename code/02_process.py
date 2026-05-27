"""
Processing pipeline: joins comments, positions, prices, and whale trades into
analysis-ready DataFrames.

Usage:
    python code/02_process.py --market <market_id> --whale-threshold 5000
"""

import json
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "utils"))
import sentiment as sent

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Event-study window sizes (price series)
PRE_MINUTES  = 60   # look-back before comment
POST_MINUTES = 120  # look-forward after comment

# Convergence filter: exclude comments where the market probability is already
# near the resolution boundary — the price is converging to 0/1 and any
# measured change is noise, not a response to the comment.
CONVERGENCE_LO = 0.10   # exclude price_before < 0.10
CONVERGENCE_HI = 0.90   # exclude price_before > 0.90

# Trade-window sizes for the pump/dump analysis
PRE_TRADE_HOURS  = 24   # look back this many hours before comment
POST_TRADE_HOURS = 24   # look forward this many hours after comment


def process_market(market_id: str, whale_threshold_usd: float = 5000):
    in_dir  = RAW_DIR / market_id
    out_dir = PROCESSED_DIR / market_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load raw files ----
    comments_tagged = json.loads((in_dir / "comments_tagged.json").read_text())
    price_history   = json.loads((in_dir / "price_history.json").read_text())
    holders_raw     = json.loads((in_dir / "holders.json").read_text())
    market_meta     = json.loads((in_dir / "market.json").read_text())
    market_volume   = float(market_meta.get("volumeNum") or market_meta.get("volume") or 0)

    # ---- optional whale trades ----
    whale_trades_path = in_dir / "whale_trades.json"
    whale_trades: dict = {}
    if whale_trades_path.exists():
        whale_trades = json.loads(whale_trades_path.read_text())
        print(f"Whale trades: loaded for {len(whale_trades)} wallets")

    # ---- build prices DataFrame (YES token = first token) ----
    prices_df = _build_prices_df(price_history)
    prices_df.to_csv(out_dir / "prices.csv", index=False)
    print(f"Prices: {len(prices_df)} hourly observations "
          f"({prices_df['timestamp'].min().date()} → {prices_df['timestamp'].max().date()})")

    # ---- build comments DataFrame ----
    comments_df = _build_comments_df(comments_tagged, whale_threshold_usd)
    comments_df.to_csv(out_dir / "comments.csv", index=False)
    total    = len(comments_df)
    n_whale  = comments_df["is_whale"].sum()
    n_retail = total - n_whale
    print(f"Comments: {total} total | {n_whale} whale (≥${whale_threshold_usd:,.0f}) "
          f"| {n_retail} retail")

    # ---- build top-holders summary ----
    holders_df = _build_holders_df(holders_raw)
    holders_df.to_csv(out_dir / "holders.csv", index=False)
    print(f"Holders: {len(holders_df)} unique wallets")

    # ---- event study (price windows) ----
    events_df = _build_event_windows(comments_df, prices_df)
    events_df["market_id"]           = market_id
    events_df["market_volume"]       = market_volume
    events_df["log_market_volume"]   = np.log10(market_volume) if market_volume > 0 else np.nan
    events_df["is_convergence_period"] = (
        (events_df["price_before"] > CONVERGENCE_HI) |
        (events_df["price_before"] < CONVERGENCE_LO)
    )

    # ---- trade windows (pump/dump analysis) ----
    if whale_trades:
        events_df = _build_trade_windows(events_df, whale_trades)

    events_df.to_csv(out_dir / "events.csv", index=False)

    # ---- summary ----
    has_price = events_df["price_change"].notna()
    non_conv  = ~events_df["is_convergence_period"]
    n_with_prices  = has_price.sum()
    n_clean        = (has_price & non_conv).sum()
    print(f"Event windows: {len(events_df)} comments | "
          f"{n_with_prices} with price data | "
          f"{n_clean} non-convergence with price data")

    whale_ev  = events_df[events_df["is_whale"] & has_price & non_conv]
    retail_ev = events_df[~events_df["is_whale"] & has_price & non_conv]
    if len(whale_ev) + len(retail_ev) > 0:
        print(f"\n--- Quick Summary (non-convergence only) ---")
        print(f"Whale  comments: {len(whale_ev):>6}  avg Δp = {whale_ev['price_change'].mean():+.4f}")
        print(f"Retail comments: {len(retail_ev):>6}  avg Δp = {retail_ev['price_change'].mean():+.4f}")

    if "post_net_usd" in events_df.columns:
        pump = events_df[
            events_df["is_whale"] &
            (events_df["sentiment_direction"] > 0) &
            (events_df["post_net_usd"] < 0)
        ]
        print(f"\nPotential pump events (bullish comment + net-sold after): {len(pump)}")

    print(f"\nProcessed files saved to {out_dir}/")
    return comments_df, prices_df, holders_df, events_df


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _build_prices_df(price_history: dict) -> pd.DataFrame:
    """
    Use the first (YES) token as the market probability proxy.
    price_history is {token_id: [{t, p}, ...]}
    """
    frames = []
    for i, (token_id, pts) in enumerate(price_history.items()):
        if not pts:
            continue
        if isinstance(pts, dict):
            pts = pts.get("history", [])
        df = pd.DataFrame(pts)
        if df.empty or "t" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df["price"]     = pd.to_numeric(df.get("p", df.get("price")), errors="coerce")
        df["token_id"]  = token_id
        df["outcome"]   = "YES" if i == 0 else "NO"
        frames.append(df[["timestamp", "price", "token_id", "outcome"]])

    if not frames:
        return pd.DataFrame(columns=["timestamp", "price", "token_id", "outcome"])

    combined = pd.concat(frames).dropna(subset=["price"])
    return combined.sort_values("timestamp").reset_index(drop=True)


def _build_comments_df(comments_tagged: list, whale_threshold_usd: float) -> pd.DataFrame:
    rows = []
    for c in comments_tagged:
        addr  = c.get("userAddress", "")
        proxy = (c.get("profile") or {}).get("proxyWallet", "")
        try:
            ts = pd.to_datetime(c["createdAt"], utc=True)
        except Exception:
            continue

        body         = (c.get("body") or "").strip()
        position_usd = float(c.get("_position_usd") or 0)
        scores       = sent.score(body)

        rows.append({
            "comment_id":          c.get("id", ""),
            "address":             addr,
            "proxy_wallet":        proxy,
            "timestamp":           ts,
            "body":                body,
            "position_usd":        position_usd,
            "is_whale":            position_usd >= whale_threshold_usd,
            "reaction_count":      int(c.get("reactionCount") or 0),
            # sentiment fields
            "sentiment_compound":  scores["compound"],
            "sentiment_label":     scores["sentiment_label"],
            "sentiment_direction": scores["sentiment_direction"],
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def _build_holders_df(holders_raw: list) -> pd.DataFrame:
    rows = []
    for token_entry in holders_raw:
        for h in (token_entry.get("holders") or []):
            rows.append({
                "token_id":      token_entry.get("token", ""),
                "proxy_wallet":  h.get("proxyWallet", ""),
                "name":          h.get("name", ""),
                "amount":        float(h.get("amount") or 0),
                "outcome_index": h.get("outcomeIndex"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = (df.groupby("proxy_wallet")["amount"].sum()
             .reset_index()
             .rename(columns={"amount": "total_position_usd"})
             .sort_values("total_position_usd", ascending=False))
    return agg.reset_index(drop=True)


def _build_event_windows(comments_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each comment, compute price change in the post-window vs pre-window.

    Uses an adaptive window based on price data density:
    - Dense markets (≥1 price point/day): forward-filled hourly series, ±1h/2h window.
    - Sparse markets (<1 price point/day): nearest observed price before/after, ±7 day max.
    """
    if prices_df.empty:
        return comments_df.assign(
            price_before=np.nan, price_after=np.nan,
            price_change=np.nan, abs_price_change=np.nan, sentiment_aligned=np.nan,
        )

    yes_raw = (prices_df[prices_df["outcome"] == "YES"]
               .set_index("timestamp")["price"]
               .sort_index())

    # Compute data density
    if len(yes_raw) >= 2:
        span_days   = (yes_raw.index[-1] - yes_raw.index[0]).total_seconds() / 86400
        pts_per_day = len(yes_raw) / max(span_days, 1)
    else:
        pts_per_day = 0

    if pts_per_day >= 1.0:
        series  = yes_raw.resample("1h").last().ffill().dropna()
        pre_td  = pd.Timedelta(hours=PRE_MINUTES // 60)
        post_td = pd.Timedelta(hours=POST_MINUTES // 60)
        def get_before(t):
            s = series[(series.index >= t - pre_td) & (series.index < t)]
            return float(s.iloc[-1]) if not s.empty else np.nan
        def get_after(t):
            s = series[(series.index >= t) & (series.index <= t + post_td)]
            return float(s.iloc[-1]) if not s.empty else np.nan
    else:
        series = yes_raw
        max_td = pd.Timedelta(days=7)
        def get_before(t):
            s = series[(series.index < t) & (series.index >= t - max_td)]
            return float(s.iloc[-1]) if not s.empty else np.nan
        def get_after(t):
            s = series[(series.index >= t) & (series.index <= t + max_td)]
            return float(s.iloc[0]) if not s.empty else np.nan

    rows = []
    for _, comment in comments_df.iterrows():
        t            = comment["timestamp"]
        price_before = get_before(t)
        price_after  = get_after(t)
        price_change = (price_after - price_before
                        if not (pd.isna(price_before) or pd.isna(price_after))
                        else np.nan)

        direction = comment.get("sentiment_direction", 0)
        sentiment_aligned = (
            (1 if direction * price_change > 0 else -1)
            if (not pd.isna(price_change) and direction != 0) else np.nan
        )

        row = comment.to_dict()
        row.update({
            "price_before":      price_before,
            "price_after":       price_after,
            "price_change":      price_change,
            "abs_price_change":  abs(price_change) if not pd.isna(price_change) else np.nan,
            "sentiment_aligned": sentiment_aligned,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def _build_trade_windows(events_df: pd.DataFrame,
                         whale_trades: dict[str, list]) -> pd.DataFrame:
    """
    For each whale comment, look up the commenter's trades in the windows
    [t − PRE_TRADE_HOURS, t] and [t, t + POST_TRADE_HOURS].

    Adds columns:
        pre_buy_usd    – USD bought in the pre-window
        pre_sell_usd   – USD sold in the pre-window
        pre_net_usd    – pre_buy_usd − pre_sell_usd  (>0 = net accumulating)
        pre_n_trades   – number of trades in the pre-window
        post_buy_usd   – USD bought in the post-window
        post_sell_usd  – USD sold in the post-window
        post_net_usd   – post_buy_usd − post_sell_usd  (>0 = net buying / <0 = distributing)
        post_n_trades  – number of trades in the post-window
        pump_signal    – True if bullish comment AND net-sold in post-window
        dump_signal    – True if bearish comment AND net-bought in post-window
    """
    pre_td  = pd.Timedelta(hours=PRE_TRADE_HOURS)
    post_td = pd.Timedelta(hours=POST_TRADE_HOURS)

    # Preprocess: build per-wallet DataFrame of trades
    wallet_dfs: dict[str, pd.DataFrame] = {}
    for wallet, trades in whale_trades.items():
        if not trades:
            continue
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["size"]      = pd.to_numeric(df["size"],  errors="coerce").fillna(0)
        df["price"]     = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        wallet_dfs[wallet] = df

    def _window_stats(wallet: str, t_comment, lo_td, hi_td):
        """Aggregate buy/sell USD in [t_comment − lo_td, t_comment + hi_td]."""
        df = wallet_dfs.get(wallet)
        if df is None or df.empty:
            return 0.0, 0.0, 0
        window = df[(df["timestamp"] >= t_comment - lo_td) &
                    (df["timestamp"] <  t_comment + hi_td)]
        buy_usd  = window[window["side"] == "BUY"]["size"].sum()
        sell_usd = window[window["side"] == "SELL"]["size"].sum()
        return float(buy_usd), float(sell_usd), len(window)

    out = events_df.copy()
    (pre_buy, pre_sell, pre_n,
     post_buy, post_sell, post_n) = [], [], [], [], [], []

    for _, row in events_df.iterrows():
        wallet = row.get("proxy_wallet") or row.get("address", "")
        t      = row["timestamp"]
        if not pd.isna(t):
            t = pd.Timestamp(t)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")

        if row.get("is_whale") and wallet:
            b0, s0, n0 = _window_stats(wallet, t, pre_td,  pd.Timedelta(0))
            b1, s1, n1 = _window_stats(wallet, t, pd.Timedelta(0), post_td)
        else:
            b0, s0, n0 = 0.0, 0.0, 0
            b1, s1, n1 = 0.0, 0.0, 0

        pre_buy.append(b0);  pre_sell.append(s0);  pre_n.append(n0)
        post_buy.append(b1); post_sell.append(s1); post_n.append(n1)

    out["pre_buy_usd"]  = pre_buy
    out["pre_sell_usd"] = pre_sell
    out["pre_net_usd"]  = np.array(pre_buy)  - np.array(pre_sell)
    out["pre_n_trades"] = pre_n
    out["post_buy_usd"]  = post_buy
    out["post_sell_usd"] = post_sell
    out["post_net_usd"]  = np.array(post_buy) - np.array(post_sell)
    out["post_n_trades"] = post_n

    # Pump signal: commented bullishly, then NET-sold in the following window
    out["pump_signal"] = (
        (out["is_whale"]) &
        (out["sentiment_direction"] > 0) &
        (out["post_net_usd"] < 0)
    )
    # Dump signal: commented bearishly, then NET-bought (accumulating at lower prices)
    out["dump_signal"] = (
        (out["is_whale"]) &
        (out["sentiment_direction"] < 0) &
        (out["post_net_usd"] > 0)
    )

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True, help="Polymarket market ID")
    parser.add_argument("--whale-threshold", type=float, default=5000,
                        help="Min USD position to count as whale (default: 5000)")
    args = parser.parse_args()
    process_market(args.market, args.whale_threshold)
