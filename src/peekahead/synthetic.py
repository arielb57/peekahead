"""Deterministic synthetic OHLCV tables, so checks never need real market data."""

from __future__ import annotations

import datetime as _dt
import math
import random
from typing import Any, Dict, List

EPS_EVERY = 5


def ohlcv(rows: int = 250, seed: int = 0, eps_every: int = EPS_EVERY) -> Dict[str, List[Any]]:
    """Daily bars from a geometric random walk.

    Columns: ``ts`` (ISO date), ``open``, ``high``, ``low``, ``close``,
    ``volume`` (int) and ``eps``, a sparse fundamentals column that is only
    reported every ``eps_every`` rows (``nan`` in between), which is what
    backward fills and as-of joins operate on.
    """
    if rows < 0:
        raise ValueError("rows must be >= 0")
    if eps_every < 1:
        raise ValueError("eps_every must be >= 1")
    rng = random.Random(seed)
    start = _dt.date(2020, 1, 1)
    table: Dict[str, List[Any]] = {k: [] for k in ("ts", "open", "high", "low", "close", "volume", "eps")}
    price = 100.0
    eps = 1.0
    for i in range(rows):
        open_ = price * math.exp(rng.gauss(0, 0.002))
        close = open_ * math.exp(rng.gauss(0, 0.01))
        high = max(open_, close) * math.exp(abs(rng.gauss(0, 0.004)))
        low = min(open_, close) * math.exp(-abs(rng.gauss(0, 0.004)))
        table["ts"].append((start + _dt.timedelta(days=i)).isoformat())
        table["open"].append(round(open_, 4))
        table["high"].append(round(high, 4))
        table["low"].append(round(low, 4))
        table["close"].append(round(close, 4))
        table["volume"].append(int(rng.lognormvariate(13, 0.4)))
        if i % eps_every == 0:
            eps = round(eps * (1 + rng.gauss(0.01, 0.05)), 4)
            table["eps"].append(eps)
        else:
            table["eps"].append(float("nan"))
        price = close
    return table
