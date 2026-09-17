"""peekahead MODULE:FUNCTION --rows N

Runs a feature function against a synthetic OHLCV table and reports look-ahead.
Exit status: 0 clean, 1 leak found, 2 usage or import error.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from typing import Any, Callable, List, Optional

from . import __version__
from .core import check
from .cost import DEFAULT_PERIODS, measure
from .synthetic import ohlcv


def load_function(spec: str) -> Callable[..., Any]:
    """Resolve ``package.module:function`` or ``path/to/file.py:function``."""
    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr:
        raise ValueError(f"expected MODULE:FUNCTION, got {spec!r}")
    if target.endswith(".py") or os.sep in target:
        path = os.path.abspath(target)
        if not os.path.isfile(path):
            raise ValueError(f"no such file: {target}")
        module_spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        module = importlib.import_module(target)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            raise ValueError(f"{target} has no attribute {attr!r}")
    if not callable(obj):
        raise ValueError(f"{spec} is not callable")
    return obj


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="peekahead",
        description="Detect look-ahead bias in a feature function by perturbing the future of a synthetic OHLCV table.",
    )
    parser.add_argument("function", help="MODULE:FUNCTION or path/to/file.py:FUNCTION")
    parser.add_argument("--rows", type=int, default=250, help="rows in the synthetic table (default 250)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the table and the perturbations")
    parser.add_argument(
        "--pandas",
        action="store_true",
        help="pass a DataFrame with a DatetimeIndex instead of a dict of lists",
    )
    parser.add_argument("--random-cuts", type=int, default=8, help="random cut points on top of bisection")
    parser.add_argument("--window", type=int, default=16, help="source rows probed one at a time for the horizon")
    parser.add_argument(
        "--cost",
        action="store_true",
        help="also recompute the feature causally and report what the leak was worth in Sharpe",
    )
    parser.add_argument(
        "--periods", type=int, default=DEFAULT_PERIODS, help="bars per year for annualising (default 252)"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--version", action="version", version=f"peekahead {__version__}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rows < 0:
        print("peekahead: --rows must be >= 0", file=sys.stderr)
        return 2
    try:
        fn = load_function(args.function)
    except (ValueError, ImportError) as exc:
        print(f"peekahead: {exc}", file=sys.stderr)
        return 2
    table = ohlcv(args.rows, seed=args.seed)
    options = dict(seed=args.seed, random_cuts=args.random_cuts, horizon_window=args.window)
    try:
        if args.pandas:
            from .pandas_adapter import check_frame, frame_from_table

            report = check_frame(fn, frame_from_table(table), **options)
        else:
            report = check(fn, table, freeze=["ts"], **options)
    except ImportError as exc:
        print(f"peekahead: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface the feature's own failure without a traceback wall
        print(
            f"peekahead: {args.function} failed on the unmodified table: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 2
    cost = None
    if args.cost:
        try:
            cost = measure(fn, table, periods=args.periods)
        except Exception as exc:  # noqa: BLE001 - the detector's verdict still stands without this
            print(f"peekahead: could not measure the cost: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.json:
        payload = report.to_dict()
        if cost is not None:
            payload["cost"] = {
                "diverging_rows": len(cost.diverging_rows),
                "compared_rows": cost.compared_rows,
                "sharpe_reported": cost.reported.sharpe(cost.periods),
                "sharpe_causal": cost.causal.sharpe(cost.periods),
                "sharpe_gap": cost.sharpe_gap,
                "return_reported": cost.reported.total_return,
                "return_causal": cost.causal.total_return,
            }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"{args.function} on {args.rows} synthetic OHLCV rows (seed {args.seed})")
        print(report.summary())
        if cost is not None:
            print()
            print(cost.summary())
    return 1 if report.leaked else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
