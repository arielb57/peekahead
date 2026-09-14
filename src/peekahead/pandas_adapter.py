"""Run :func:`peekahead.check` on functions that take and return pandas objects.

pandas is imported lazily; the core package never needs it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable

from .core import LeakReport, check


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise ImportError("peekahead.pandas_adapter needs pandas: pip install 'peekahead[pandas]'") from exc
    return pd


def check_frame(
    fn: Callable[[Any], Any],
    df: Any,
    *,
    freeze: Iterable[Any] = (),
    **kwargs: Any,
) -> LeakReport:
    """Check a DataFrame-in, Series-or-DataFrame-out feature function.

    Rows must be in time order (a monotonic increasing index, typically a
    DatetimeIndex). The index is never perturbed; neither are datetime
    columns, which are almost always join keys. Outputs are aligned to input
    rows by index label, so a function that drops warm-up rows or returns a
    shorter Series is compared row-for-row, not by position.

    Keyword arguments are passed to :func:`peekahead.check`.
    """
    pd = _pd()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("check_frame expects a pandas DataFrame")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted in time order")
    if df.columns.has_duplicates:
        raise ValueError("DataFrame has duplicate column names")
    index = df.index
    dtypes = df.dtypes
    columns = list(df.columns)
    frozen = set(freeze) | {c for c in columns if pd.api.types.is_datetime64_any_dtype(dtypes[c])}
    table: Dict[Any, list] = {c: df[c].tolist() for c in columns}

    def to_frame(tab: Dict[Any, list]) -> Any:
        rows = len(tab[columns[0]]) if columns else 0
        frame = pd.DataFrame({c: tab[c] for c in columns}, index=index[:rows], columns=df.columns)
        for c in columns:
            if frame[c].dtype != dtypes[c]:
                try:
                    frame[c] = frame[c].astype(dtypes[c])
                except (TypeError, ValueError):
                    pass
        return frame

    def wrapped(tab: Dict[Any, list]) -> Any:
        return _align(pd, fn(to_frame(tab)), index)

    if not columns:
        raise ValueError("DataFrame has no columns to perturb")
    return check(wrapped, table, freeze=frozen, **kwargs)


def _align(pd: Any, out: Any, index: Any) -> Any:
    """Key each output row by the position of the last input row at or before its label."""
    if isinstance(out, pd.Series):
        labels, values = out.index, out.tolist()
        cols = None
    elif isinstance(out, pd.DataFrame):
        labels = out.index
        cols = tuple(out.columns)
        values = [tuple(row) for row in zip(*(out[c].tolist() for c in out.columns))] if cols else [()] * len(out)
    else:
        return out
    try:
        positions = index.searchsorted(labels, side="right") - 1
    except (TypeError, ValueError) as exc:
        raise TypeError("output index cannot be compared with the input index") from exc
    rows: Dict[int, list] = {}
    for pos, label, value in zip(positions.tolist(), labels.tolist(), values):
        entry = (label, value) if cols is None else (label, tuple(zip(cols, value)))
        rows.setdefault(max(int(pos), 0), []).append(entry)
    return {pos: tuple(entries) for pos, entries in rows.items()}


def frame_from_table(table: Dict[str, list], index_column: str = "ts") -> Any:
    """Build a DataFrame with a DatetimeIndex from a synthetic dict-of-lists table."""
    pd = _pd()
    data = {k: v for k, v in table.items() if k != index_column}
    return pd.DataFrame(data, index=pd.DatetimeIndex(pd.to_datetime(table[index_column]), name=index_column))
