"""
Thin wrappers around the existing collect / process / pool scripts.
Imported by collect_live_markets.py.
"""

import sys
from pathlib import Path

ROOT      = Path(__file__).parent.parent
CODE_DIR  = Path(__file__).parent

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "utils"))


def collect_market_safe(market_id: str, event_id: int, top_holders: int = 200):
    """Run the collect pipeline for one market."""
    import importlib
    import importlib.util

    spec = importlib.util.spec_from_file_location("collect00", CODE_DIR / "00_collect.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.collect_market(str(market_id), event_id=event_id, top_holders=top_holders)


def process_market_safe(market_id: str, whale_threshold: float = 5000):
    """Run the process pipeline for one market."""
    import importlib
    import importlib.util

    spec = importlib.util.spec_from_file_location("process02", CODE_DIR / "02_process.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.process_market(str(market_id), whale_threshold_usd=whale_threshold)


def pool_all(whale_threshold: float = 5000):
    """Re-pool all markets into pooled_events.csv."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pool03", CODE_DIR / "03_pool.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df   = mod.pool(whale_threshold=whale_threshold)
    return df
