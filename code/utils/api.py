"""
Polymarket API client.
Wraps the three public APIs: Gamma (markets/comments), CLOB (prices/trades), Data (positions).
"""

import time
import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "polymarket-whale-analysis/1.0"})


def _get(url, params=None, retries=3, backoff=2.0):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429 and attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(backoff)
            else:
                raise


# ---------- Gamma API ----------

def get_markets(limit=100, offset=0, closed=False):
    """List markets with metadata."""
    return _get(f"{GAMMA_BASE}/markets", params={
        "limit": limit,
        "offset": offset,
        "closed": str(closed).lower(),
    })


def get_market(market_id):
    """Single market by ID."""
    return _get(f"{GAMMA_BASE}/markets/{market_id}")


def get_comments(event_id, limit=100, offset=0):
    """Comments for an event. Comments are attached to Events, not markets."""
    return _get(f"{GAMMA_BASE}/comments", params={
        "parent_entity_id": event_id,
        "parent_entity_type": "Event",
        "limit": limit,
        "offset": offset,
    })


def get_all_comments(event_id):
    """Paginate through all comments for an event."""
    comments = []
    offset = 0
    limit = 100
    while True:
        batch = get_comments(event_id, limit=limit, offset=offset)
        if not batch:
            break
        comments.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.1)
    return comments


def get_event_id_for_market(market: dict) -> int | None:
    """Extract the event ID from a market metadata dict."""
    events = market.get("events", [])
    if not events:
        return None
    first = events[0]
    return first.get("id") if isinstance(first, dict) else first


# ---------- CLOB API ----------

def get_price_history(token_id, interval="max", start_ts=None, end_ts=None, fidelity=3600):
    """
    Historical prices for an outcome token.
    interval: 1d, 1w, max  (ignored when start_ts/end_ts provided)
    fidelity: resolution in seconds (3600 = hourly)
    Note: CLOB API uses 'market' as the param name for the token/asset ID.
    Max explicit window: ~1 week (API rejects longer ranges).
    """
    params = {"market": token_id, "fidelity": fidelity}
    if start_ts or end_ts:
        if start_ts:
            params["startTs"] = int(start_ts)
        if end_ts:
            params["endTs"] = int(end_ts)
    else:
        params["interval"] = interval
    return _get(f"{CLOB_BASE}/prices-history", params=params)


def get_full_price_history(token_id, start_ts: int, end_ts: int, fidelity: int = 3600) -> list:
    """
    Fetch complete price history for a closed market by chunking into weekly windows.
    Returns a flat list of {t, p} dicts sorted by timestamp.
    """
    WEEK = 7 * 24 * 3600
    all_points = []
    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + WEEK, end_ts)
        result = get_price_history(token_id, start_ts=chunk_start, end_ts=chunk_end, fidelity=fidelity)
        history = result.get("history", result) if isinstance(result, dict) else result
        if isinstance(history, list):
            all_points.extend(history)
        chunk_start = chunk_end
        time.sleep(0.3)
    # deduplicate and sort
    seen = set()
    unique = []
    for pt in sorted(all_points, key=lambda x: x.get("t", 0)):
        t = pt.get("t")
        if t not in seen:
            seen.add(t)
            unique.append(pt)
    return unique


def get_user_activity(user_address, limit=500, offset=0):
    """
    Full activity timeline for a wallet (trades, redemptions, etc.) from the Data API.
    This is the reliable per-user trade history — the bulk /trades endpoint ignores market filters.
    """
    return _get(f"{DATA_BASE}/activity", params={
        "user": user_address,
        "limit": limit,
        "offset": offset,
    })


def get_all_user_market_trades(user_address: str, condition_id: str) -> list:
    """
    Fetch ALL trades for a single wallet in a specific market (conditionId).

    The Data API /trades endpoint accepts both 'user' and 'market' filters
    simultaneously and supports offset-based pagination.  Whale commenters
    typically have O(10–100) trades per market, so this rarely needs more
    than 1–2 pages.

    Returns a flat list of trade dicts with keys:
        proxyWallet, side, asset, conditionId, size, price, timestamp, title
    """
    all_trades: list = []
    offset = 0
    PAGE = 500
    while True:
        batch = _get(f"{DATA_BASE}/trades", params={
            "user": user_address,
            "market": condition_id,
            "limit": PAGE,
            "offset": offset,
        })
        if not isinstance(batch, list) or not batch:
            break
        all_trades.extend(batch)
        if len(batch) < PAGE:
            break
        offset += PAGE
        time.sleep(0.2)
    # sort oldest → newest
    all_trades.sort(key=lambda t: t.get("timestamp", 0))
    return all_trades


# ---------- Data API ----------

def get_positions(user_address, market=None, limit=100, offset=0):
    """Positions held by a wallet address."""
    params = {"user": user_address, "limit": limit, "offset": offset}
    if market:
        params["market"] = market
    return _get(f"{DATA_BASE}/positions", params=params)


def get_user_trades(user_address, market=None, limit=500, offset=0):
    """Trade history for a wallet address."""
    params = {"user": user_address, "limit": limit, "offset": offset}
    if market:
        params["market"] = market
    return _get(f"{DATA_BASE}/trades", params=params)


def get_holders(condition_id, limit=20):
    """
    Top holders across all outcome tokens for a market.
    Requires the market's conditionId (0x...), not the numeric market ID.
    Returns a list of {token, holders: [...]} dicts.
    """
    return _get(f"{DATA_BASE}/holders", params={"market": condition_id, "limit": limit})
