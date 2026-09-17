#!/usr/bin/env python3
"""Reproduce the cost table in the README.

Detection and cost are different questions: this prints a leak worth 19 Sharpe,
one worth 2, one worth 0.1, and a clean feature, all on the same data.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peekahead import measure, ohlcv  # noqa: E402

TABLE = ohlcv(150, seed=3)


def close(t):
    return list(t["close"])


def next_return(t):
    c = close(t)
    return [c[min(i + 1, len(c) - 1)] / c[i] - 1 for i in range(len(c))]


def centred_mean_7(t):
    c = close(t)
    return [statistics.fmean(c[max(0, i - 3) : i + 4]) for i in range(len(c))]


def whole_sample_zscore(t):
    c = close(t)
    mean, sd = statistics.fmean(c), statistics.pstdev(c)
    return [(x - mean) / (sd + 1e-9) for x in c]


def rolling_mean_5(t):
    c = close(t)
    return [None if i < 4 else statistics.fmean(c[i - 4 : i + 1]) for i in range(len(c))]


FEATURES = [
    ("close.shift(-1) return", next_return),
    ("rolling(7, center=True).mean()", centred_mean_7),
    ("whole-sample z-score", whole_sample_zscore),
    ("rolling(5).mean()", rolling_mean_5),
]


def main() -> int:
    print(f"peekahead cost table — ohlcv(150, seed=3), {len(TABLE['close'])} rows\n")
    print("| feature | rows differing | Sharpe as reported | Sharpe causal | the leak was worth |")
    print("|---|---:|---:|---:|---:|")
    for name, fn in FEATURES:
        r = measure(fn, TABLE)
        if not r.leaked:
            print(f"| `{name}` | 0.0% | — | — | nothing to attribute |")
            continue
        print(
            f"| `{name}` | {r.divergence_rate:.1%} | {r.reported.sharpe():+.2f} "
            f"| {r.causal.sharpe():+.2f} | **{r.sharpe_gap:+.2f}** |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
