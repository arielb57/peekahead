"""What the leak is worth.

Knowing a feature reads two rows ahead is a bug report. Knowing that removing
the leak takes the strategy's Sharpe from 1.8 to 0.1 is a decision. This module
answers the second question, because the first one on its own rarely gets a
backtest thrown away.

The method is the only honest one available: recompute the feature causally.
For every row ``t``, call the feature on the table truncated at ``t`` and keep
the value it produces for ``t``. That is, by construction, the number the
feature would have produced in real time, whatever it does internally — the
same black-box stance the detector takes. It costs one call per row, so the
whole thing is quadratic in the number of rows; this is a diagnostic to run on
a few hundred rows, not a backtest engine.

The trading rule has to be causal too, or it would introduce a second leak and
muddy the comparison. The default rule uses the expanding median of the signal
so far as its threshold: it needs no scale assumption, works for a feature that
is a price level as readily as one that is a z-score, and reads nothing it
could not have read at the time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from .core import Table, normalize_output, values_equal

#: Bars per year for annualising. Daily bars by default.
DEFAULT_PERIODS = 252


@dataclass(frozen=True)
class Series:
    """One leg of the comparison."""

    name: str
    #: Feature value per row, ``None`` where the feature produced nothing.
    values: List[Optional[float]]
    #: Realised per-bar return of the rule trading this feature.
    returns: List[float]

    @property
    def total_return(self) -> float:
        out = 1.0
        for r in self.returns:
            out *= 1.0 + r
        return out - 1.0

    def sharpe(self, periods: int = DEFAULT_PERIODS) -> float:
        """Annualised Sharpe of the per-bar returns. Zero variance gives 0.0."""
        n = len(self.returns)
        if n < 2:
            return 0.0
        mean = sum(self.returns) / n
        var = sum((r - mean) ** 2 for r in self.returns) / (n - 1)
        if var <= 0.0:
            return 0.0
        return mean / math.sqrt(var) * math.sqrt(periods)


@dataclass(frozen=True)
class CostReport:
    """What the leak was worth, and how much of the feature it touched."""

    reported: Series
    causal: Series
    #: Rows where the causal value differs from the reported one.
    diverging_rows: Tuple[int, ...]
    #: Rows the comparison could use at all (both legs produced a number).
    compared_rows: int
    #: Feature calls made, for the cost of running this.
    calls: int
    periods: int

    @property
    def leaked(self) -> bool:
        return len(self.diverging_rows) > 0

    @property
    def divergence_rate(self) -> float:
        return len(self.diverging_rows) / self.compared_rows if self.compared_rows else 0.0

    @property
    def sharpe_gap(self) -> float:
        """How much Sharpe the leak was adding. Negative means it was costing."""
        return self.reported.sharpe(self.periods) - self.causal.sharpe(self.periods)

    def summary(self) -> str:
        if self.compared_rows == 0:
            return "no rows could be compared: the feature produced nothing on truncated tables"
        head = (
            f"{len(self.diverging_rows)} of {self.compared_rows} rows differ once the feature is "
            f"recomputed causally ({self.divergence_rate:.1%})"
        )
        if not self.leaked:
            return f"{head}\nthe feature is causal, so there is nothing to attribute"
        return (
            f"{head}\n"
            f"Sharpe   as reported {self.reported.sharpe(self.periods):+.2f}"
            f"   causal {self.causal.sharpe(self.periods):+.2f}"
            f"   the leak was worth {self.sharpe_gap:+.2f}\n"
            f"return   as reported {self.reported.total_return:+.1%}"
            f"   causal {self.causal.total_return:+.1%}\n"
            f"first divergence at row {self.diverging_rows[0]}"
        )


def expanding_median_rule(values: Sequence[Optional[float]]) -> List[int]:
    """Position per row: long above the expanding median of the signal, short below.

    Causal by construction — row ``t`` sees ``values[:t + 1]`` and nothing
    else. Rows where the feature produced no number are flat.
    """
    positions: List[int] = []
    seen: List[float] = []
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            positions.append(0)
            continue
        seen.append(value)
        ordered = sorted(seen)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        positions.append(1 if value > median else -1 if value < median else 0)
    return positions


def _as_number(value: Any) -> Optional[float]:
    """A single float from a feature output row, or None if there is not one."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    if isinstance(value, Mapping) and len(value) == 1:
        return _as_number(next(iter(value.values())))
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _as_number(value[0])
    return None


def _truncate(table: Table, upto: int) -> Table:
    return {key: list(column[: upto + 1]) for key, column in table.items()}


def causal_values(
    fn: Callable[[Table], Any],
    table: Mapping[Any, Sequence[Any]],
    *,
    rows: Optional[int] = None,
) -> Tuple[List[Optional[float]], int]:
    """The value the feature would have produced at each row, in real time.

    Returns the values and the number of feature calls made. A call that raises
    — a rolling window that has not warmed up, say — yields ``None`` for that
    row rather than failing the whole run.
    """
    columns = {key: list(values) for key, values in table.items()}
    n = rows if rows is not None else len(next(iter(columns.values())))
    out: List[Optional[float]] = []
    calls = 0
    for t in range(n):
        calls += 1
        try:
            produced = normalize_output(fn(_truncate(columns, t)))
        except Exception:  # noqa: BLE001 - a warm-up failure is a missing value, not a crash
            out.append(None)
            continue
        out.append(_as_number(produced.get(t)))
    return out, calls


def measure(
    fn: Callable[[Table], Any],
    table: Mapping[Any, Sequence[Any]],
    *,
    price: Any = "close",
    rule: Callable[[Sequence[Optional[float]]], List[int]] = expanding_median_rule,
    periods: int = DEFAULT_PERIODS,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> CostReport:
    """Compare the feature as written against the same feature computed causally.

    ``price`` names the column the rule trades. Each row's position is held for
    one bar and earns that bar's return, so the last row earns nothing and is
    dropped.
    """
    columns: Table = {key: list(values) for key, values in table.items()}
    if price not in columns:
        raise KeyError(f"price column {price!r} is not in the table; columns are {sorted(map(str, columns))}")
    prices = [float(p) for p in columns[price]]
    n = len(prices)

    reported_rows = normalize_output(fn(columns))
    reported = [_as_number(reported_rows.get(t)) for t in range(n)]
    causal, calls = causal_values(fn, columns, rows=n)

    diverging = tuple(
        t
        for t in range(n)
        if reported[t] is not None and causal[t] is not None and not values_equal(reported[t], causal[t], rtol, atol)
    )
    compared = sum(1 for t in range(n) if reported[t] is not None and causal[t] is not None)

    bar_returns = [prices[t + 1] / prices[t] - 1.0 for t in range(n - 1)]
    legs = []
    for name, values in (("reported", reported), ("causal", causal)):
        positions = rule(values)
        legs.append(Series(name, list(values), [positions[t] * bar_returns[t] for t in range(n - 1)]))

    return CostReport(
        reported=legs[0],
        causal=legs[1],
        diverging_rows=diverging,
        compared_rows=compared,
        calls=calls + 1,
        periods=periods,
    )
