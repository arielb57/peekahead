"""Black-box causality checks for feature functions over dict-of-lists tables."""

from __future__ import annotations

import math
import numbers
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

WHOLE_SAMPLE = "whole-sample"

METHODS = ("removal", "noise", "extreme", "permute")

Table = Dict[Any, List[Any]]
Rows = Dict[int, Any]
Horizon = Any  # int, WHOLE_SAMPLE, or None


@dataclass(frozen=True)
class Evidence:
    """One observed violation: perturbing the future changed a past output row."""

    stage: str  # "detect", "column" or "horizon"
    method: str  # removal, noise, spike, dip, permute
    cut: int  # rows > cut were modified (for stage "horizon": source_row - 1)
    row: int  # earliest output row that changed
    expected: Any
    observed: Any
    column: Any = None
    source_row: Optional[int] = None

    def describe(self) -> str:
        where = f"row {self.row}: {_short(self.expected)} -> {_short(self.observed)}"
        if self.stage == "horizon":
            return (
                f"{self.method:<8} {self.column!r}[{self.source_row}] alone changed {where}"
                f" ({self.source_row - self.row} rows ahead)"
            )
        scope = "all columns" if self.column is None else repr(self.column)
        return f"{self.method:<8} {scope}, rows > {self.cut} changed {where}"


@dataclass
class LeakReport:
    """Result of :func:`check`. ``earliest_row is None`` means no leak was observed."""

    earliest_row: Optional[int]
    columns: Tuple[Any, ...]
    horizon: Horizon
    evidence: Tuple[Evidence, ...]
    rows: int
    calls: int
    perturbations: int
    cuts_checked: Tuple[int, ...]
    column_horizons: Dict[Any, Horizon] = field(default_factory=dict)
    errors: int = 0

    @property
    def leaked(self) -> bool:
        return self.earliest_row is not None

    def detection_bound(self, p: float) -> float:
        """Lower bound on the probability that this check detects a leak which
        independently changes a past output with probability ``p`` on each
        perturbed call. Only meaningful for a clean report, where every
        detection-stage perturbation ran."""
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        return 1.0 - (1.0 - p) ** self.perturbations

    def min_detectable_rate(self, confidence: float = 0.95) -> float:
        """Smallest per-perturbation firing rate that would have been caught
        with at least ``confidence`` probability."""
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if self.perturbations == 0:
            return 1.0
        return 1.0 - (1.0 - confidence) ** (1.0 / self.perturbations)

    def horizon_text(self) -> str:
        return _horizon_text(self.horizon)

    def summary(self) -> str:
        if not self.leaked:
            lines = [f"clean: no look-ahead observed over {self.rows} rows"]
            if self.perturbations:
                lines.append(
                    f"  {self.perturbations} perturbations; a data-dependent leak firing on >= "
                    f"{self.min_detectable_rate():.1%} of perturbations would be caught with 95% probability"
                )
            else:
                lines.append("  table has fewer than 2 rows, nothing to perturb")
        else:
            cols = ", ".join(repr(c) for c in self.columns) if self.columns else "none (depends on row count)"
            lines = [
                f"LEAK: earliest leaking row {self.earliest_row}",
                f"  columns: {cols}",
                f"  horizon: {self.horizon_text()}",
            ]
            for col, h in self.column_horizons.items():
                lines.append(f"    {col!r}: {_horizon_text(h)}")
            lines.append("  evidence:")
            lines.extend("    " + e.describe() for e in self.evidence)
        lines.append(f"  function calls: {self.calls}  errors on perturbed input: {self.errors}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "leaked": self.leaked,
            "earliest_row": self.earliest_row,
            "columns": [str(c) for c in self.columns],
            "horizon": self.horizon,
            "column_horizons": {str(k): v for k, v in self.column_horizons.items()},
            "rows": self.rows,
            "calls": self.calls,
            "perturbations": self.perturbations,
            "errors": self.errors,
            "cuts_checked": list(self.cuts_checked),
            "evidence": [
                {
                    "stage": e.stage,
                    "method": e.method,
                    "cut": e.cut,
                    "row": e.row,
                    "column": None if e.column is None else str(e.column),
                    "source_row": e.source_row,
                    "expected": repr(e.expected),
                    "observed": repr(e.observed),
                }
                for e in self.evidence
            ],
        }


def check(
    fn: Callable[[Table], Any],
    table: Mapping[Any, Sequence[Any]],
    *,
    freeze: Iterable[Any] = (),
    seed: int = 0,
    random_cuts: int = 8,
    noise_trials: int = 2,
    horizon_window: int = 16,
    methods: Iterable[str] = METHODS,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> LeakReport:
    """Test whether ``fn`` lets rows after a cut point influence outputs at or before it.

    ``table`` maps column names to equal-length lists, rows in time order.
    ``fn`` must return per-row output: a list (row i is input row i; a shorter
    list is taken to have dropped trailing rows), a dict of equal-length lists
    (one per output column), or a dict mapping input row positions to values
    (for outputs that skip rows, e.g. dropped warm-up rows).

    Columns named in ``freeze`` (timestamps, keys) are never perturbed.
    """
    return _Checker(
        fn,
        table,
        freeze=freeze,
        seed=seed,
        random_cuts=random_cuts,
        noise_trials=noise_trials,
        horizon_window=horizon_window,
        methods=methods,
        rtol=rtol,
        atol=atol,
    ).run()


# ---------------------------------------------------------------------------
# value helpers


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(x != x)
    except (TypeError, ValueError):
        return False


def _is_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and type(x).__name__ != "bool_"


def values_equal(a: Any, b: Any, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    """Equality for feature outputs: missing == missing (NaN, None, NaT), floats
    within tolerance, containers element-wise."""
    if a is b:
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y, rtol, atol) for x, y in zip(a, b))
    ma, mb = is_missing(a), is_missing(b)
    if ma or mb:
        return ma and mb
    if _is_number(a) and _is_number(b):
        if a == b:
            return True
        if math.isinf(a) or math.isinf(b):
            return False
        return abs(a - b) <= atol + rtol * max(abs(a), abs(b))
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def normalize_output(out: Any) -> Rows:
    """Convert a feature output into {input row position: row value}."""
    if hasattr(out, "tolist") and not isinstance(out, (list, tuple, dict)):
        out = out.tolist()
    if isinstance(out, (list, tuple)):
        return dict(enumerate(out))
    if isinstance(out, Mapping):
        if not out:
            return {}
        keys = list(out.keys())
        if all(isinstance(k, int) and not isinstance(k, bool) for k in keys):
            return dict(out)
        columns = []
        for k in keys:
            col = out[k]
            if hasattr(col, "tolist"):
                col = col.tolist()
            if not isinstance(col, (list, tuple)):
                raise TypeError(
                    f"output column {k!r} is {type(col).__name__}; expected a list per column "
                    "or a dict keyed by int row positions"
                )
            columns.append(col)
        length = len(columns[0])
        if any(len(c) != length for c in columns):
            raise TypeError("output columns have different lengths")
        return {i: tuple(zip(keys, (c[i] for c in columns))) for i in range(length)}
    raise TypeError(
        f"feature returned {type(out).__name__}; expected a per-row list, a dict of lists, "
        "or a dict keyed by row position"
    )


def _short(x: Any) -> str:
    text = repr(x)
    return text if len(text) <= 48 else text[:45] + "..."


def _horizon_text(h: Horizon) -> str:
    if h == WHOLE_SAMPLE:
        return "whole sample (the last row reaches back to the earliest leaking row)"
    if h is None:
        return "undetermined (no single-row perturbation reproduced the leak)"
    return f"reads {h} row{'s' if h != 1 else ''} ahead"


@dataclass
class _ColumnStats:
    kind: str  # "bool", "int", "float", "other", "empty"
    lo: float = 0.0
    hi: float = 0.0
    span: float = 1.0
    nonneg: bool = False
    values: Tuple[Any, ...] = ()

    @classmethod
    def of(cls, col: Sequence[Any]) -> "_ColumnStats":
        present = [v for v in col if not is_missing(v)]
        if not present:
            return cls("empty")
        if all(isinstance(v, bool) or type(v).__name__ == "bool_" for v in present):
            return cls("bool")
        if all(_is_number(v) for v in present):
            finite = [float(v) for v in present if not math.isinf(v)]
            if not finite:
                return cls("other", values=tuple(present))
            lo, hi = min(finite), max(finite)
            # A constant column still needs a noise scale, or value
            # perturbations could never move it.
            span = hi - lo if hi > lo else max(abs(lo), 1.0)
            kind = "int" if all(isinstance(v, numbers.Integral) for v in present) else "float"
            return cls(kind, lo, hi, span, lo >= 0)
        return cls("other", values=tuple(present))

    def _cast(self, x: float) -> Any:
        return int(round(x)) if self.kind == "int" else x

    def noise(self, rng: random.Random, current: Any) -> Any:
        if self.kind == "bool":
            return rng.random() < 0.5
        if self.kind in ("int", "float"):
            low = self.lo - self.span / 2
            if self.nonneg:
                low = max(low, self.lo / 2)
            return self._cast(rng.uniform(low, self.hi + self.span / 2))
        if self.kind == "other":
            return rng.choice(self.values)
        return current

    def spike(self, current: Any) -> Any:
        if self.kind == "bool":
            return not current
        if self.kind in ("int", "float"):
            return self._cast(self.hi + 10 * self.span + 1)
        return self._different(current)

    def dip(self, current: Any) -> Any:
        if self.kind == "bool":
            return not current
        if self.kind in ("int", "float"):
            if self.nonneg:
                return self._cast(self.lo * 0.1)
            return self._cast(self.lo - 10 * self.span - 1)
        return self._different(current, reverse=True)

    def _different(self, current: Any, reverse: bool = False) -> Any:
        pool = reversed(self.values) if reverse else self.values
        for v in pool:
            if not values_equal(v, current):
                return v
        return current


class _Checker:
    BRACKET = 3

    def __init__(self, fn, table, *, freeze, seed, random_cuts, noise_trials, horizon_window, methods, rtol, atol):
        if not callable(fn):
            raise TypeError("fn must be callable")
        if not isinstance(table, Mapping):
            raise TypeError("table must be a mapping of column name -> list of values")
        self.table: Table = {k: list(v) for k, v in table.items()}
        lengths = {len(v) for v in self.table.values()}
        if len(lengths) > 1:
            raise ValueError(f"columns have different lengths: {sorted(lengths)}")
        self.n = lengths.pop() if lengths else 0
        self.fn = fn
        frozen = set(freeze)
        unknown = frozen - set(self.table)
        if unknown:
            raise ValueError(f"freeze names unknown columns: {sorted(map(str, unknown))}")
        self.columns = [c for c in self.table if c not in frozen]
        self.methods = tuple(methods)
        bad = set(self.methods) - set(METHODS)
        if bad or not self.methods:
            raise ValueError(f"methods must be a non-empty subset of {METHODS}")
        if random_cuts < 0 or noise_trials < 0 or horizon_window < 1:
            raise ValueError("random_cuts and noise_trials must be >= 0, horizon_window >= 1")
        self.rng = random.Random(seed)
        self.random_cuts = random_cuts
        self.noise_trials = noise_trials
        self.window = horizon_window
        self.rtol, self.atol = rtol, atol
        self.stats = {c: _ColumnStats.of(self.table[c]) for c in self.columns}
        self.calls = 0
        self.perturbations = 0
        self.errors = 0
        self.base: Rows = {}
        self.base_keys: List[int] = []

    # -- plumbing -----------------------------------------------------------

    def _call(self, table: Table, *, is_base: bool = False) -> Optional[Rows]:
        self.calls += 1
        fresh = {k: list(v) for k, v in table.items()}
        if is_base:
            return normalize_output(self.fn(fresh))
        try:
            return normalize_output(self.fn(fresh))
        except Exception:  # noqa: BLE001 - a perturbed input may be outside what fn accepts
            self.errors += 1
            return None

    def _first_diff(self, other: Rows, upto: int) -> Optional[Tuple[int, Any, Any]]:
        """Earliest position <= upto where ``other`` disagrees with the base output."""
        best: Optional[Tuple[int, Any, Any]] = None
        for pos in self.base_keys:
            if pos > upto:
                break
            if pos not in other:
                best = (pos, self.base[pos], "<row missing>")
                break
            if not values_equal(self.base[pos], other[pos], self.rtol, self.atol):
                best = (pos, self.base[pos], other[pos])
                break
        for pos in other:
            if pos <= upto and pos not in self.base and (best is None or pos < best[0]):
                best = (pos, "<row missing>", other[pos])
        return best

    def _replace_future(self, cut: int, cols: Sequence[Any], make: Callable[[Any, Any], Any]) -> Optional[Table]:
        new = dict(self.table)
        changed = False
        for c in cols:
            old = self.table[c]
            tail = [make(c, v) for v in old[cut + 1 :]]
            if tail != old[cut + 1 :]:
                changed = True
            new[c] = old[: cut + 1] + tail
        return new if changed else None

    def _permute_future(self, cut: int, cols: Sequence[Any]) -> Optional[Table]:
        order = list(range(cut + 1, self.n))
        self.rng.shuffle(order)
        new = dict(self.table)
        changed = False
        for c in cols:
            old = self.table[c]
            tail = [old[i] for i in order]
            if tail != old[cut + 1 :]:
                changed = True
            new[c] = old[: cut + 1] + tail
        return new if changed else None

    def _future_variants(self, cut: int, cols: Sequence[Any], removal: bool) -> Iterator[Tuple[str, Table]]:
        if removal and "removal" in self.methods:
            yield "removal", {k: v[: cut + 1] for k, v in self.table.items()}
        if not cols:
            return
        if "extreme" in self.methods:
            for name in ("spike", "dip"):
                t = self._replace_future(cut, cols, lambda c, v, name=name: getattr(self.stats[c], name)(v))
                if t is not None:
                    yield name, t
        if "noise" in self.methods:
            for _ in range(self.noise_trials):
                t = self._replace_future(cut, cols, lambda c, v: self.stats[c].noise(self.rng, v))
                if t is not None:
                    yield "noise", t
        if "permute" in self.methods:
            t = self._permute_future(cut, cols)
            if t is not None:
                yield "permute", t

    def _single_row_variants(self, col: Any, row: int) -> Iterator[Tuple[str, Table]]:
        stats = self.stats[col]
        current = self.table[col][row]
        candidates = [("spike", stats.spike(current)), ("dip", stats.dip(current))]
        candidates += [("noise", stats.noise(self.rng, current)) for _ in range(max(self.noise_trials, 1))]
        for name, value in candidates:
            if values_equal(value, current, 0.0, 0.0):
                continue
            new = dict(self.table)
            column = list(self.table[col])
            column[row] = value
            new[col] = column
            yield name, new

    # -- stages -------------------------------------------------------------

    def _probe_cut(self, cut: int) -> Optional[Evidence]:
        for method, table in self._future_variants(cut, self.columns, removal=True):
            self.perturbations += 1
            out = self._call(table)
            if out is None:
                continue
            diff = self._first_diff(out, cut)
            if diff is not None:
                return Evidence("detect", method, cut, diff[0], diff[1], diff[2])
        return None

    def _detect(self) -> Tuple[Optional[int], List[Evidence], List[int]]:
        last_cut = self.n - 2
        fixed = {0, last_cut // 4, last_cut // 2, (3 * last_cut) // 4, last_cut}
        randoms = [self.rng.randint(0, last_cut) for _ in range(self.random_cuts)]
        cuts: List[int] = []
        for c in sorted(fixed) + randoms:
            if c not in cuts:
                cuts.append(c)
        checked: List[int] = []
        hits: List[Evidence] = []
        for c in cuts:
            checked.append(c)
            ev = self._probe_cut(c)
            if ev is not None:
                hits.append(ev)
        if not hits:
            return None, [], checked
        # Bisection assumes that once some row leaks, later cuts keep leaking;
        # that holds for "every row reads k ahead" leaks, and the random cuts
        # above are what catch leaks confined to a stretch of the table.
        # Each step probes a bracket of consecutive cuts because fills over
        # sparse data do not leak at the cut that lands on a report row.
        earliest = min(e.row for e in hits)
        lo, hi = 0, earliest
        while lo < hi:
            mid = (lo + hi) // 2
            ev = None
            for cut in range(mid, min(mid + self.BRACKET, hi)):
                checked.append(cut)
                ev = self._probe_cut(cut)
                if ev is not None:
                    break
            if ev is None:
                lo = mid + 1
            else:
                hits.append(ev)
                hi = ev.row
        return hi, hits, checked

    def _attribute_columns(self, leak_cuts: Sequence[int]) -> Tuple[List[Any], List[Evidence]]:
        found: List[Any] = []
        evidence: List[Evidence] = []
        for col in self.columns:
            hit = None
            for cut in leak_cuts:
                for method, table in self._future_variants(cut, [col], removal=False):
                    out = self._call(table)
                    if out is None:
                        continue
                    diff = self._first_diff(out, cut)
                    if diff is not None:
                        hit = Evidence("column", method, cut, diff[0], diff[1], diff[2], column=col)
                        break
                if hit is not None:
                    break
            if hit is not None:
                found.append(col)
                evidence.append(hit)
        return found, evidence

    def _column_horizon(self, col: Any, earliest: int) -> Tuple[Horizon, Optional[Evidence]]:
        last = self.n - 1
        mid = (earliest + last) // 2
        sources = sorted(set(range(earliest + 1, min(earliest + self.window, last) + 1)) | {mid, last})
        best: Optional[Evidence] = None
        reach: Dict[int, int] = {}
        for s in sources:
            if s <= earliest:
                continue
            for method, table in self._single_row_variants(col, s):
                out = self._call(table)
                if out is None:
                    continue
                diff = self._first_diff(out, s - 1)
                if diff is None:
                    continue
                reach[s] = min(reach.get(s, s), diff[0])
                ev = Evidence("horizon", method, s - 1, diff[0], diff[1], diff[2], column=col, source_row=s)
                if best is None or (s - diff[0]) > (best.source_row - best.row):
                    best = ev
        if best is None:
            return None, None
        if mid > earliest and reach.get(last, last) <= earliest and reach.get(mid, mid) <= earliest:
            return WHOLE_SAMPLE, best
        return best.source_row - best.row, best

    def _removal_horizon(self, cut: int) -> Horizon:
        """Smallest k such that keeping rows <= cut + k leaves outputs <= cut unchanged."""
        full_k = self.n - 1 - cut
        lo, hi = 1, full_k
        while lo < hi:
            mid = (lo + hi) // 2
            out = self._call({k: v[: cut + mid + 1] for k, v in self.table.items()})
            if out is not None and self._first_diff(out, cut) is None:
                hi = mid
            else:
                lo = mid + 1
        return WHOLE_SAMPLE if lo == full_k else lo

    def run(self) -> LeakReport:
        if self.n < 2:
            return LeakReport(None, (), None, (), self.n, 0, 0, ())
        self.base = self._call(self.table, is_base=True)
        self.base_keys = sorted(self.base)
        earliest, hits, checked = self._detect()
        if earliest is None:
            return LeakReport(
                None, (), None, (), self.n, self.calls, self.perturbations, tuple(checked), errors=self.errors
            )
        leak_cuts: List[int] = []
        for ev in sorted(hits, key=lambda e: (e.row, e.cut)):
            if ev.cut not in leak_cuts:
                leak_cuts.append(ev.cut)
        leak_cuts = leak_cuts[:4]
        columns, col_evidence = self._attribute_columns(leak_cuts)
        column_horizons: Dict[Any, Horizon] = {}
        horizon_evidence: List[Evidence] = []
        for col in columns:
            h, ev = self._column_horizon(col, earliest)
            column_horizons[col] = h
            if ev is not None:
                horizon_evidence.append(ev)
        known = [h for h in column_horizons.values() if h is not None]
        if WHOLE_SAMPLE in known:
            horizon: Horizon = WHOLE_SAMPLE
        elif known:
            horizon = max(known)
        else:
            horizon = self._removal_horizon(earliest)
        first = min(hits, key=lambda e: (e.row, e.cut))
        return LeakReport(
            earliest_row=earliest,
            columns=tuple(columns),
            horizon=horizon,
            evidence=(first, *col_evidence, *horizon_evidence),
            rows=self.n,
            calls=self.calls,
            perturbations=self.perturbations,
            cuts_checked=tuple(checked),
            column_horizons=column_horizons,
            errors=self.errors,
        )
