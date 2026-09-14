"""Reference features: intentionally leaky ones with their known leak, and clean ones.

Every feature exists twice: a pure-Python version over dict-of-lists tables
(``fn``) used by the core test-suite, and the pandas code a researcher would
actually write (``snippet``, defining ``feature(df)``), which the pandas
adapter tests execute. Both versions carry the same expected answer.

The tables come from :func:`peekahead.synthetic.ohlcv`: ``eps`` is reported
every 5 rows, which fixes the horizon of the fills and joins below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .core import WHOLE_SAMPLE

NAN = float("nan")


@dataclass(frozen=True)
class Feature:
    name: str
    leak_class: Optional[str]  # None for clean features
    fn: Callable[[Dict[str, List[Any]]], Any]
    columns: Tuple[str, ...]
    horizon: Any
    snippet: str

    @property
    def leaky(self) -> bool:
        return self.leak_class is not None


# -- small pure-Python equivalents of the pandas idioms -----------------------


def _nan(x: float) -> bool:
    return x != x


def _div(a: float, b: float) -> float:
    if _nan(a) or _nan(b) or b == 0:
        return NAN
    return a / b


def _shift(xs: Sequence[float], k: int) -> List[float]:
    n = len(xs)
    return [xs[i - k] if 0 <= i - k < n else NAN for i in range(n)]


def _mean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if not _nan(x)]
    return math.fsum(vals) / len(vals) if vals else NAN


def _std(xs: Sequence[float]) -> float:
    vals = [x for x in xs if not _nan(x)]
    if len(vals) < 2:
        return NAN
    m = math.fsum(vals) / len(vals)
    return math.sqrt(math.fsum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def _rolling(xs: Sequence[float], window: int, agg: Callable, center: bool = False) -> List[float]:
    n = len(xs)
    out = []
    for i in range(n):
        hi = i + (window // 2 if center else 0)
        lo = hi - window + 1
        if lo < 0 or hi >= n:
            out.append(NAN)
            continue
        win = xs[lo : hi + 1]
        out.append(NAN if any(_nan(x) for x in win) else agg(win))
    return out


def _bfill(xs: Sequence[float]) -> List[float]:
    out = list(xs)
    nxt = NAN
    for i in range(len(out) - 1, -1, -1):
        if _nan(out[i]):
            out[i] = nxt
        else:
            nxt = out[i]
    return out


def _ffill(xs: Sequence[float]) -> List[float]:
    out = list(xs)
    prev = NAN
    for i, x in enumerate(out):
        if _nan(x):
            out[i] = prev
        else:
            prev = x
    return out


def _ewm(xs: Sequence[float], alpha: float) -> List[float]:
    out = []
    state = NAN
    for x in xs:
        if not _nan(x):
            state = x if _nan(state) else (1 - alpha) * state + alpha * x
        out.append(state)
    return out


def _diff(xs: Sequence[float]) -> List[float]:
    return [NAN] + [xs[i] - xs[i - 1] for i in range(1, len(xs))]


def _pct_change(xs: Sequence[float]) -> List[float]:
    return [NAN] + [_div(xs[i], xs[i - 1]) - 1 for i in range(1, len(xs))]


def _rank_pct(xs: Sequence[float]) -> List[float]:
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg / n
        i = j + 1
    return ranks


# -- leaky features ------------------------------------------------------------


def shift_minus_1(t):
    return _shift(t["close"], -1)


def shift_minus_3(t):
    return _shift(t["close"], -3)


def shift_minus_10(t):
    return _shift(t["close"], -10)


def next_bar_return(t):
    c = t["close"]
    return [_div(c[i + 1], c[i]) - 1 if i + 1 < len(c) else NAN for i in range(len(c))]


def next_bar_label_joined(t):
    c, ts = t["close"], t["ts"]
    labels = {ts[i]: float(c[i + 1] > c[i]) for i in range(len(c) - 1)}
    return [labels.get(stamp, NAN) for stamp in ts]


def next_bar_label_dropped(t):
    c = t["close"]
    return [float(c[i + 1] > c[i]) for i in range(len(c) - 1)]


def centred_mean_5(t):
    return _rolling(t["close"], 5, _mean, center=True)


def centred_mean_21(t):
    return _rolling(t["close"], 21, _mean, center=True)


def centred_volume_mean_11(t):
    return _rolling(t["volume"], 11, _mean, center=True)


def zscore_full_sample(t):
    c = t["close"]
    m, s = _mean(c), _std(c)
    return [_div(x - m, s) for x in c]


def minmax_full_sample(t):
    c = t["close"]
    lo, hi = min(c), max(c)
    return [_div(x - lo, hi - lo) for x in c]


def demeaned_volume(t):
    m = _mean(t["volume"])
    return [v - m for v in t["volume"]]


def rank_pct_full_sample(t):
    return _rank_pct(t["close"])


def scaler_fit_on_full_sample(t):
    r = _pct_change(t["close"])
    m, s = _mean(r), _std(r)
    return [_div(x - m, s) for x in r]


def future_peak_drawdown(t):
    c = t["close"]
    out = [NAN] * len(c)
    peak = -math.inf
    for i in range(len(c) - 1, -1, -1):
        peak = max(peak, c[i])
        out[i] = c[i] / peak - 1
    return out


def full_sample_drawdown(t):
    peak = max(t["close"])
    return [x / peak - 1 for x in t["close"]]


def bfill_eps(t):
    return _bfill(t["eps"])


def merge_asof_forward_eps(t):
    return _bfill(t["eps"])


def next_report_strict(t):
    eps = t["eps"]
    out = [NAN] * len(eps)
    nxt = NAN
    for i in range(len(eps) - 1, -1, -1):
        out[i] = nxt
        if not _nan(eps[i]):
            nxt = eps[i]
    return out


def interpolated_eps(t):
    eps = list(t["eps"])
    known = [i for i, x in enumerate(eps) if not _nan(x)]
    out = list(eps)
    for a, b in zip(known, known[1:]):
        for i in range(a + 1, b):
            out[i] = eps[a] + (eps[b] - eps[a]) * (i - a) / (b - a)
    if known:
        for i in range(known[-1] + 1, len(eps)):
            out[i] = eps[known[-1]]
    return out


def future_max_high_5(t):
    return _shift(_rolling(t["high"], 5, max), -5)


def breakout_next_5(t):
    fut = _shift(_rolling(t["high"], 5, max), -5)
    return [float(f > 1.05 * c) for f, c in zip(fut, t["close"])]


def row_count_progress(t):
    n = len(t["close"])
    return [i / max(n - 1, 1) for i in range(n)]


def full_sample_vwap(t):
    c, v = t["close"], t["volume"]
    vwap = math.fsum(a * b for a, b in zip(c, v)) / math.fsum(v)
    return [vwap] * len(c)


# -- clean features ------------------------------------------------------------


def lag_1(t):
    return _shift(t["close"], 1)


def lag_5(t):
    return _shift(t["close"], 5)


def trailing_mean_20(t):
    return _rolling(t["close"], 20, _mean)


def trailing_std_20(t):
    return _rolling(t["close"], 20, _std)


def expanding_mean(t):
    out, total = [], 0.0
    for i, x in enumerate(t["close"]):
        total += x
        out.append(total / (i + 1))
    return out


def expanding_zscore(t):
    out = []
    c = t["close"]
    for i in range(len(c)):
        out.append(_div(c[i] - _mean(c[: i + 1]), _std(c[: i + 1])))
    return out


def ewm_span_10(t):
    return _ewm(t["close"], 2 / 11)


def ffill_eps(t):
    return _ffill(t["eps"])


def merge_asof_backward_eps(t):
    return _ffill(t["eps"])


def pct_change_1(t):
    return _pct_change(t["close"])


def log_return(t):
    return _diff([math.log(x) for x in t["close"]])


def trailing_max_high_20(t):
    return _rolling(t["high"], 20, max)


def running_drawdown(t):
    out, peak = [], -math.inf
    for x in t["close"]:
        peak = max(peak, x)
        out.append(x / peak - 1)
    return out


def rsi_14(t):
    d = _diff(t["close"])
    gain = _ewm([x if _nan(x) else max(x, 0.0) for x in d], 1 / 14)
    loss = _ewm([x if _nan(x) else max(-x, 0.0) for x in d], 1 / 14)
    return [100 - 100 / (1 + _div(g, lo)) if not _nan(_div(g, lo)) else NAN for g, lo in zip(gain, loss)]


def true_range(t):
    h, lo, c = t["high"], t["low"], t["close"]
    out = []
    for i in range(len(c)):
        if i == 0:
            out.append(h[0] - lo[0])
        else:
            out.append(max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])))
    return out


def label_shifted_back(t):
    return _shift(next_bar_label_dropped(t) + [0.0], 1)


def centred_then_lagged(t):
    return _shift(_rolling(t["close"], 5, _mean, center=True), 2)


def cumulative_volume(t):
    out, total = [], 0
    for v in t["volume"]:
        total += v
        out.append(total)
    return out


def rolling_vwap_20(t):
    pv = _rolling([a * b for a, b in zip(t["close"], t["volume"])], 20, math.fsum)
    vol = _rolling(t["volume"], 20, math.fsum)
    return [_div(a, b) for a, b in zip(pv, vol)]


def _snip(body: str) -> str:
    lines = body.strip("\n").splitlines()
    return "def feature(df):\n" + "\n".join("    " + line for line in lines) + "\n"


NEG_SHIFT = "negative shift"
LABEL = "next-bar label joined back"
CENTRED = "centred rolling window"
FULL = "full-sample statistic"
PEAK = "future extremum"
FILL = "backward fill / forward as-of join"
COUNT = "row-count dependence"

LEAKY: List[Feature] = [
    Feature("shift_minus_1", NEG_SHIFT, shift_minus_1, ("close",), 1, _snip('return df["close"].shift(-1)')),
    Feature("shift_minus_3", NEG_SHIFT, shift_minus_3, ("close",), 3, _snip('return df["close"].shift(-3)')),
    Feature("shift_minus_10", NEG_SHIFT, shift_minus_10, ("close",), 10, _snip('return df["close"].shift(-10)')),
    Feature(
        "next_bar_return", NEG_SHIFT, next_bar_return, ("close",), 1, _snip('return df["close"].pct_change().shift(-1)')
    ),
    Feature(
        "next_bar_label_joined",
        LABEL,
        next_bar_label_joined,
        ("close",),
        1,
        _snip(
            'y = (df["close"].shift(-1) > df["close"]).astype(float).iloc[:-1].rename("y")\n'
            'return df[["close"]].join(y)["y"]'
        ),
    ),
    Feature(
        "next_bar_label_dropped",
        LABEL,
        next_bar_label_dropped,
        ("close",),
        1,
        _snip('return (df["close"].shift(-1) > df["close"]).astype(float).iloc[:-1]'),
    ),
    Feature(
        "centred_mean_5",
        CENTRED,
        centred_mean_5,
        ("close",),
        2,
        _snip('return df["close"].rolling(5, center=True).mean()'),
    ),
    Feature(
        "centred_mean_21",
        CENTRED,
        centred_mean_21,
        ("close",),
        10,
        _snip('return df["close"].rolling(21, center=True).mean()'),
    ),
    Feature(
        "centred_volume_mean_11",
        CENTRED,
        centred_volume_mean_11,
        ("volume",),
        5,
        _snip('return df["volume"].rolling(11, center=True).mean()'),
    ),
    Feature(
        "zscore_full_sample",
        FULL,
        zscore_full_sample,
        ("close",),
        WHOLE_SAMPLE,
        _snip('return (df["close"] - df["close"].mean()) / df["close"].std()'),
    ),
    Feature(
        "minmax_full_sample",
        FULL,
        minmax_full_sample,
        ("close",),
        WHOLE_SAMPLE,
        _snip('c = df["close"]\nreturn (c - c.min()) / (c.max() - c.min())'),
    ),
    Feature(
        "demeaned_volume",
        FULL,
        demeaned_volume,
        ("volume",),
        WHOLE_SAMPLE,
        _snip('return df["volume"] - df["volume"].mean()'),
    ),
    Feature(
        "rank_pct_full_sample",
        FULL,
        rank_pct_full_sample,
        ("close",),
        WHOLE_SAMPLE,
        _snip('return df["close"].rank(pct=True)'),
    ),
    Feature(
        "scaler_fit_on_full_sample",
        FULL,
        scaler_fit_on_full_sample,
        ("close",),
        WHOLE_SAMPLE,
        _snip('r = df["close"].pct_change()\nreturn (r - r.mean()) / r.std()  # scaler fit before the split'),
    ),
    Feature(
        "full_sample_vwap",
        FULL,
        full_sample_vwap,
        ("close", "volume"),
        WHOLE_SAMPLE,
        _snip(
            'vwap = (df["close"] * df["volume"]).sum() / df["volume"].sum()\n' "return pd.Series(vwap, index=df.index)"
        ),
    ),
    Feature(
        "future_peak_drawdown",
        PEAK,
        future_peak_drawdown,
        ("close",),
        WHOLE_SAMPLE,
        _snip('c = df["close"]\nreturn c / c[::-1].cummax()[::-1] - 1'),
    ),
    Feature(
        "full_sample_drawdown",
        PEAK,
        full_sample_drawdown,
        ("close",),
        WHOLE_SAMPLE,
        _snip('return df["close"] / df["close"].max() - 1'),
    ),
    Feature(
        "future_max_high_5",
        PEAK,
        future_max_high_5,
        ("high",),
        5,
        _snip('return df["high"].rolling(5).max().shift(-5)'),
    ),
    Feature(
        "breakout_next_5",
        PEAK,
        breakout_next_5,
        ("high",),
        5,
        _snip(
            'future_high = df["high"].rolling(5).max().shift(-5)\nreturn (future_high > 1.05 * df["close"]).astype(float)'
        ),
    ),
    Feature("bfill_eps", FILL, bfill_eps, ("eps",), 4, _snip('return df["eps"].bfill()')),
    Feature(
        "merge_asof_forward_eps",
        FILL,
        merge_asof_forward_eps,
        ("eps",),
        4,
        _snip(
            'reports = df.loc[df["eps"].notna(), ["eps"]]\n'
            'joined = pd.merge_asof(df[[]], reports, left_index=True, right_index=True, direction="forward")\n'
            'return joined["eps"]'
        ),
    ),
    Feature(
        "next_report_strict",
        FILL,
        next_report_strict,
        ("eps",),
        5,
        _snip(
            'reports = df.loc[df["eps"].notna(), ["eps"]]\n'
            "joined = pd.merge_asof(df[[]], reports, left_index=True, right_index=True,\n"
            '                       direction="forward", allow_exact_matches=False)\n'
            'return joined["eps"]'
        ),
    ),
    Feature("interpolated_eps", FILL, interpolated_eps, ("eps",), 4, _snip('return df["eps"].interpolate()')),
    Feature(
        "row_count_progress",
        COUNT,
        row_count_progress,
        (),
        WHOLE_SAMPLE,
        _snip("return pd.Series(range(len(df)), index=df.index, dtype=float) / max(len(df) - 1, 1)"),
    ),
]

CLEAN: List[Feature] = [
    Feature("lag_1", None, lag_1, (), None, _snip('return df["close"].shift(1)')),
    Feature("lag_5", None, lag_5, (), None, _snip('return df["close"].shift(5)')),
    Feature("trailing_mean_20", None, trailing_mean_20, (), None, _snip('return df["close"].rolling(20).mean()')),
    Feature("trailing_std_20", None, trailing_std_20, (), None, _snip('return df["close"].rolling(20).std()')),
    Feature("expanding_mean", None, expanding_mean, (), None, _snip('return df["close"].expanding().mean()')),
    Feature(
        "expanding_zscore",
        None,
        expanding_zscore,
        (),
        None,
        _snip('c = df["close"]\nreturn (c - c.expanding().mean()) / c.expanding().std()'),
    ),
    Feature("ewm_span_10", None, ewm_span_10, (), None, _snip('return df["close"].ewm(span=10, adjust=False).mean()')),
    Feature("ffill_eps", None, ffill_eps, (), None, _snip('return df["eps"].ffill()')),
    Feature(
        "merge_asof_backward_eps",
        None,
        merge_asof_backward_eps,
        (),
        None,
        _snip(
            'reports = df.loc[df["eps"].notna(), ["eps"]]\n'
            'joined = pd.merge_asof(df[[]], reports, left_index=True, right_index=True, direction="backward")\n'
            'return joined["eps"]'
        ),
    ),
    Feature("pct_change_1", None, pct_change_1, (), None, _snip('return df["close"].pct_change()')),
    Feature("log_return", None, log_return, (), None, _snip('return np.log(df["close"]).diff()')),
    Feature("trailing_max_high_20", None, trailing_max_high_20, (), None, _snip('return df["high"].rolling(20).max()')),
    Feature(
        "running_drawdown", None, running_drawdown, (), None, _snip('return df["close"] / df["close"].cummax() - 1')
    ),
    Feature(
        "rsi_14",
        None,
        rsi_14,
        (),
        None,
        _snip(
            'delta = df["close"].diff()\n'
            "gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()\n"
            "loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()\n"
            "return 100 - 100 / (1 + gain / loss)"
        ),
    ),
    Feature(
        "true_range",
        None,
        true_range,
        (),
        None,
        _snip(
            'prev = df["close"].shift(1)\n'
            'parts = [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()]\n'
            "return pd.concat(parts, axis=1).max(axis=1)"
        ),
    ),
    Feature(
        "label_shifted_back",
        None,
        label_shifted_back,
        (),
        None,
        _snip(
            'y = (df["close"].shift(-1) > df["close"]).astype(float)\nreturn y.shift(1)  # label moved back to when it is known'
        ),
    ),
    Feature(
        "centred_then_lagged",
        None,
        centred_then_lagged,
        (),
        None,
        _snip('return df["close"].rolling(5, center=True).mean().shift(2)'),
    ),
    Feature("cumulative_volume", None, cumulative_volume, (), None, _snip('return df["volume"].cumsum()')),
    Feature(
        "rolling_vwap_20",
        None,
        rolling_vwap_20,
        (),
        None,
        _snip('pv = (df["close"] * df["volume"]).rolling(20).sum()\nreturn pv / df["volume"].rolling(20).sum()'),
    ),
]

ALL: List[Feature] = LEAKY + CLEAN


def by_name(name: str) -> Feature:
    for feature in ALL:
        if feature.name == name:
            return feature
    raise KeyError(name)
