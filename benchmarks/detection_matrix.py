"""Detection matrix over the corpus, against a regex linter baseline.

    python benchmarks/detection_matrix.py            # full run
    python benchmarks/detection_matrix.py --quick    # fewer seeds, for CI

Prints markdown tables: recall per leak class (flagged / exact column / exact
horizon), false positives on clean features, function calls per check and
how calls scale with table length.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import re
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from peekahead import check, ohlcv  # noqa: E402
from peekahead.corpus import CLEAN, LEAKY, by_name  # noqa: E402

REGEX_BASELINE = re.compile(r"shift\(\s*-|center\s*=\s*True|bfill|method\s*=\s*['\"]bfill")


def regex_flags(feature) -> bool:
    return bool(REGEX_BASELINE.search(feature.snippet))


def run(seeds: int, clean_seeds: int, rows: int) -> str:
    out = []
    started = time.perf_counter()
    per_class = defaultdict(lambda: {"features": 0, "checks": 0, "flagged": 0, "column": 0, "horizon": 0, "regex": 0})
    leak_calls = []
    for f in LEAKY:
        stats = per_class[f.leak_class]
        stats["features"] += 1
        stats["regex"] += regex_flags(f)
        for s in range(seeds):
            r = check(f.fn, ohlcv(rows, seed=s), freeze=["ts"], seed=s)
            stats["checks"] += 1
            stats["flagged"] += r.leaked
            stats["column"] += r.leaked and set(r.columns) == set(f.columns)
            stats["horizon"] += r.leaked and r.horizon == f.horizon
            leak_calls.append(r.calls)

    out.append(f"### Leaky features ({len(LEAKY)} features x {seeds} seeds, {rows} rows)\n")
    out.append("| leak class | features | flagged | exact column | exact horizon | regex baseline |")
    out.append("|---|---:|---:|---:|---:|---:|")
    totals = defaultdict(int)
    for cls, st in per_class.items():
        for k, v in st.items():
            totals[k] += v
        out.append(
            f"| {cls} | {st['features']} | {st['flagged']}/{st['checks']} | {st['column']}/{st['checks']} "
            f"| {st['horizon']}/{st['checks']} | {st['regex']}/{st['features']} |"
        )
    out.append(
        f"| **all** | {totals['features']} | {totals['flagged']}/{totals['checks']} | "
        f"{totals['column']}/{totals['checks']} | {totals['horizon']}/{totals['checks']} | "
        f"{totals['regex']}/{totals['features']} |"
    )

    clean_checks = clean_fp = 0
    clean_calls = []
    regex_fp = [f.name for f in CLEAN if regex_flags(f)]
    for f in CLEAN:
        for s in range(clean_seeds):
            r = check(f.fn, ohlcv(rows, seed=s), freeze=["ts"], seed=s)
            clean_checks += 1
            clean_fp += r.leaked
            clean_calls.append(r.calls)
    out.append(f"\n### Clean features ({len(CLEAN)} features x {clean_seeds} seeds)\n")
    out.append("| detector | false positives |")
    out.append("|---|---:|")
    out.append(f"| peekahead | {clean_fp}/{clean_checks} checks |")
    out.append(f"| regex baseline | {len(regex_fp)}/{len(CLEAN)} features ({', '.join(regex_fp) or 'none'}) |")

    out.append("\n### Function calls per check\n")
    out.append("| outcome | mean | median | max |")
    out.append("|---|---:|---:|---:|")
    for label, calls in (("leak found (detect + columns + horizon)", leak_calls), ("clean", clean_calls)):
        out.append(f"| {label} | {statistics.mean(calls):.0f} | {statistics.median(calls):.0f} | {max(calls)} |")

    out.append("\n### Scaling with table length (leak starting at 70% of the table, no random cuts)\n")
    out.append("| rows | cuts probed | detection-stage calls | log2(rows) |")
    out.append("|---:|---:|---:|---:|")
    for n in (100, 1_000, 10_000, 100_000):
        start = int(n * 0.7)

        def late_leak(t, start=start):
            xs = t["x"]
            return [xs[i + 1] if start <= i < len(xs) - 1 else xs[i] for i in range(len(xs))]

        r = check(late_leak, {"x": [float(i % 97) for i in range(n)]}, random_cuts=0)
        assert r.earliest_row == start
        out.append(f"| {n} | {len(r.cuts_checked)} | {r.perturbations} | {math.log2(n):.1f} |")

    horizon_calls = check(by_name("zscore_full_sample").fn, ohlcv(rows, seed=0), freeze=["ts"]).calls
    elapsed = time.perf_counter() - started
    out.append(
        f"\nFull-sample z-score check on {rows} rows: {horizon_calls} calls. "
        f"Total wall time {elapsed:.1f}s on {platform.machine()} / Python {platform.python_version()}."
    )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="3 seeds instead of 10 / 50")
    parser.add_argument("--rows", type=int, default=200)
    args = parser.parse_args()
    seeds, clean_seeds = (3, 3) if args.quick else (10, 50)
    print(run(seeds, clean_seeds, args.rows))


if __name__ == "__main__":
    main()
