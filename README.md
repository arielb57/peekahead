# peekahead

Black-box look-ahead bias detector for backtest features: it names the leaking column, how many rows ahead the feature reads, and how much of your backtest's performance was the leak.

## The problem

Research code leaks the future in ways no regex catches: a z-score normalised
over the whole sample, a centred rolling window, `bfill`,
`merge_asof(direction="forward")`, a scaler fitted before the train/test
split. The backtest looks great and the bug shows up only in live trading.
Linters flag the obvious `shift(-1)` and miss the rest. Code review misses them
too, because the leaking line looks harmless. On this project's own corpus of
24 leaky features, a regex linter looking for `shift(-`, `center=True` and
`bfill` catches 12, and it flags 2 of 19 clean features.

## How it works

peekahead never reads your source. It treats the feature function `f` as a
black box that maps a time-ordered table to per-row outputs, and tests one
invariant:

> For a cut point `t`, output rows `<= t` must be identical whether rows `> t`
> are left alone, **removed**, replaced with **noise**, replaced with
> **extreme** values (a spike above the column's range, a dip below it), or
> **permuted**.

Removal and value perturbation catch different bugs, so peekahead runs both.
Value perturbation cannot see a feature that depends only on how many rows
exist (`i / len(df)`), and removal cannot see `x - x.mean()` on a constant
column. The test suite has a case for each.

A check runs in three stages:

1. **Detect (where).** Probe five fixed cuts (0, the quartiles, the last row)
   and `random_cuts` random ones. Each probe tries removal, spike, dip,
   `noise_trials` noise draws and a permutation, and stops at the first output
   row that changes. If anything leaks, **bisect** over `[0, earliest changed row]`
   to find the earliest leaking row, in O(log n) cuts. Each bisection step
   probes up to 3 consecutive cuts. Backward fills over sparse data do not leak
   at a cut that lands on a report row, and plain bisection steps right over
   them. (That bug was reporting row 31 instead of row 1 for `bfill`.)
2. **Attribute (which column).** At the leaking cuts, perturb the future of one
   column at a time. A leak that only removal reproduces is reported as
   row-count dependence, with no column named.
3. **Horizon (how far).** For each leaking column, perturb a *single* source
   row `s` (the `horizon_window` rows after the earliest leak, plus the middle
   and last rows). A changed output row `r < s` shows the feature read `s - r`
   rows ahead, and the horizon is the largest such distance. If both the
   middle row and the last row reach back to the earliest leaking row, the
   horizon is reported as the whole sample.

Worked example, `df["close"].rolling(5, center=True).mean()`:

```
cut t=2, rows > 2 removed      -> output row 2 goes from 98.04 to NaN     leak, earliest row 2
future of 'close' only, spiked -> row 2 changes                          column 'close'
'close'[4] alone spiked        -> row 2 changes, 4 - 2 = 2                reads 2 rows ahead
```

Some leaks fire only for some data, for example only when a future value
crosses a threshold. A clean report states its power against them: with `R`
effective perturbations, a leak that changes a past output with probability `p`
on each perturbation is caught with probability at least `1 - (1 - p)^R`.
`report.detection_bound(p)` and `report.min_detectable_rate(0.95)` expose this,
and a test checks it empirically over 300 runs.

Outputs are compared with NaN equal to NaN (None and NaT count as missing too).
Floats are compared within `rtol=1e-9, atol=1e-12`. Tuples and lists are
compared element by element.

### What the leak is worth

"This feature reads two rows ahead" is a bug report. "Removing the leak takes the Sharpe from 1.8 to 0.1" is a decision. A leak that nobody prices tends not to get fixed, so `measure` answers the second question.

There is only one honest way to do it, and it is the same black-box stance as the detector: **recompute the feature causally.** For every row `t`, call the feature on the table truncated at `t` and keep the value it produces for `t`. That is by construction what the feature would have produced in real time, whatever it does internally. Then trade both versions with the same rule and compare.

The rule has to be causal too, or it introduces a second leak and muddies the comparison. The default is long above the expanding median of the signal so far, short below: no scale assumption, so it works on a feature that is a price level as readily as one that is a z-score, and it reads nothing it could not have read at the time.

```
$ peekahead features.py:next_return --rows 120 --cost
...
119 of 120 rows differ once the feature is recomputed causally (99.2%)
Sharpe   as reported +22.54   causal +0.00   the leak was worth +22.54
return   as reported +158.6%   causal +0.0%
first divergence at row 0
```

Three features on the same table (`ohlcv(150, seed=3)`), to show the range:

| feature | rows differing | Sharpe as reported | Sharpe causal | the leak was worth |
|---|---:|---:|---:|---:|
| `close.shift(-1)` return | 99.3% | +18.95 | +0.00 | **+18.95** |
| `rolling(7, center=True).mean()` | 99.3% | +0.91 | −1.11 | **+2.01** |
| whole-sample z-score | 99.3% | −1.07 | −1.17 | **+0.10** |
| `rolling(5).mean()` | 0.0% | — | — | nothing to attribute |

Reproduce with `python benchmarks/cost_table.py`.

The third row is the one worth reading twice. A whole-sample z-score is a genuine leak — every row of it changes under causal recomputation — and it is worth almost nothing here, because a monotone rescaling barely moves a rule that only compares against a median. The centred mean is the opposite: a smaller-looking leak that turns a losing strategy into a winning-looking one. Detection and cost are different questions, and a tool that only answers the first invites both panic and complacency.

This costs one feature call per row, so it is quadratic in rows. It is a diagnostic for a few hundred rows, not a backtest engine.

## Install and usage

Python 3.9+. The core has no runtime dependencies; pandas is optional.

```sh
git clone <this repo> && cd peekahead
python -m venv .venv && . .venv/bin/activate
pip install -e .              # core, stdlib only
pip install -e ".[pandas]"    # optional: DataFrame adapter
pip install pytest && pytest  # test suite (pandas tests are skipped without pandas)
```

### CLI

`peekahead MODULE:FUNCTION` (or `path/to/file.py:FUNCTION`) runs the function
on a self-generated synthetic OHLCV table. The table has `ts`, `open`, `high`,
`low`, `close`, `volume`, and a sparse `eps` column reported every 5 rows.
Exit status is 0 for clean, 1 for a leak, and 2 for a usage error.

```
$ peekahead peekahead.corpus:centred_mean_5 --rows 250
peekahead.corpus:centred_mean_5 on 250 synthetic OHLCV rows (seed 0)
LEAK: earliest leaking row 2
  columns: 'close'
  horizon: reads 2 rows ahead
    'close': reads 2 rows ahead
  evidence:
    removal  all columns, rows > 2 changed row 2: 98.03926 -> nan
    spike    'close', rows > 2 changed row 2: 98.03926 -> 173.15932000000004
    spike    'close'[4] alone changed row 2: 98.03926 -> 135.3779 (2 rows ahead)
  function calls: 206  errors on perturbed input: 0
```

```
$ peekahead peekahead.corpus:zscore_full_sample --rows 250
peekahead.corpus:zscore_full_sample on 250 synthetic OHLCV rows (seed 0)
LEAK: earliest leaking row 0
  columns: 'close'
  horizon: whole sample (the last row reaches back to the earliest leaking row)
    'close': whole sample (the last row reaches back to the earliest leaking row)
  evidence:
    removal  all columns, rows > 0 changed row 0: -0.3893569139967232 -> nan
    spike    'close', rows > 0 changed row 0: -0.3893569139967232 -> -15.748142747638525
    spike    'close'[249] alone changed row 0: -0.3893569139967232 -> -0.16037671646955665 (249 rows ahead)
  function calls: 187  errors on perturbed input: 0
```

```
$ peekahead peekahead.corpus:ewm_span_10 --rows 250
peekahead.corpus:ewm_span_10 on 250 synthetic OHLCV rows (seed 0)
clean: no look-ahead observed over 250 rows
  76 perturbations; a data-dependent leak firing on >= 3.9% of perturbations would be caught with 95% probability
  function calls: 77  errors on perturbed input: 0
```

Other flags are `--pandas` (pass a DataFrame with a DatetimeIndex), `--seed`,
`--random-cuts`, `--window` and `--json`. Your own pandas feature:

```sh
echo 'def feat(df): return (df["close"] - df["close"].mean()) / df["close"].std()' > features.py
peekahead features.py:feat --pandas
```

### Library

```python
from peekahead import check, ohlcv

def next_diff(t):                       # t is {column: [values]}
    c = t["close"]
    return [c[i + 1] - c[i] for i in range(len(c) - 1)]   # one row short

report = check(next_diff, ohlcv(250), freeze=["ts"])
report.leaked          # True
report.earliest_row    # 0
report.columns         # ('close',)
report.horizon         # 1   (or peekahead.WHOLE_SAMPLE)
report.evidence        # tuple of Evidence(stage, method, cut, row, expected, observed, column, source_row)
```

`fn` may return a list (row `i` is input row `i`; a shorter list is taken to
have dropped trailing rows), a dict of equal-length lists (several output
columns), or a dict keyed by input row position (for outputs that drop warm-up
rows). Columns in `freeze` (timestamps, join keys) are never perturbed. If `fn`
raises on a perturbed table, the call is counted in `report.errors` and is not
treated as evidence.

```python
from peekahead.pandas_adapter import check_frame, frame_from_table

report = check_frame(lambda df: df["eps"].bfill(), frame_from_table(ohlcv(250)))
report.columns, report.horizon     # (('eps',), 4)
```

The adapter aligns outputs to inputs **by index label**, so `.dropna()` or a
shorter Series is compared row for row. It never perturbs the index or
datetime columns.

### Leak classes

Every row below is a corpus feature. The snippet is the pandas code, which the
adapter tests execute. The report columns come from `check` on a 250-row table
with seed 0.

| leak class | minimal leaking snippet | earliest row | column | horizon |
|---|---|---:|---|---|
| negative shift | `df["close"].shift(-1)` | 0 | close | 1 |
| negative shift | `df["close"].shift(-3)` | 0 | close | 3 |
| negative shift | `df["close"].shift(-10)` | 0 | close | 10 |
| negative shift | `df["close"].pct_change().shift(-1)` | 0 | close | 1 |
| next-bar label joined back | `y = (df["close"].shift(-1) > df["close"]).astype(float).iloc[:-1]; df[["close"]].join(y.rename("y"))["y"]` | 0 | close | 1 |
| next-bar label joined back | `(df["close"].shift(-1) > df["close"]).astype(float).iloc[:-1]` (one row short) | 0 | close | 1 |
| centred rolling window | `df["close"].rolling(5, center=True).mean()` | 2 | close | 2 |
| centred rolling window | `df["close"].rolling(21, center=True).mean()` | 10 | close | 10 |
| centred rolling window | `df["volume"].rolling(11, center=True).mean()` | 5 | volume | 5 |
| full-sample statistic | `(df["close"] - df["close"].mean()) / df["close"].std()` | 0 | close | whole sample |
| full-sample statistic | `(c - c.min()) / (c.max() - c.min())` | 0 | close | whole sample |
| full-sample statistic | `df["volume"] - df["volume"].mean()` | 0 | volume | whole sample |
| full-sample statistic | `df["close"].rank(pct=True)` | 0 | close | whole sample |
| full-sample statistic | `r = df["close"].pct_change(); (r - r.mean()) / r.std()` (scaler fit before the split) | 1 | close | whole sample |
| full-sample statistic | `(df["close"] * df["volume"]).sum() / df["volume"].sum()` broadcast to every row | 0 | close, volume | whole sample |
| future extremum | `c / c[::-1].cummax()[::-1] - 1` (drawdown from the future peak) | 0 | close | whole sample |
| future extremum | `df["close"] / df["close"].max() - 1` | 0 | close | whole sample |
| future extremum | `df["high"].rolling(5).max().shift(-5)` | 0 | high | 5 |
| future extremum | `(df["high"].rolling(5).max().shift(-5) > 1.05 * df["close"]).astype(float)` (fires only on a threshold crossing) | 0 | high | 5 |
| backward fill / forward as-of join | `df["eps"].bfill()` | 1 | eps | 4 |
| backward fill / forward as-of join | `pd.merge_asof(df[[]], reports, left_index=True, right_index=True, direction="forward")` | 1 | eps | 4 |
| backward fill / forward as-of join | same with `allow_exact_matches=False` | 0 | eps | 5 |
| backward fill / forward as-of join | `df["eps"].interpolate()` | 1 | eps | 4 |
| row-count dependence | `pd.Series(range(len(df)), index=df.index) / (len(df) - 1)` | 1 | none (row count) | whole sample |

The fills and joins read 4 or 5 rows ahead because `eps` is reported every 5
rows. On your data, the horizon is the longest reporting gap.

The clean reference features all produce zero findings over 50 seeds:
`shift(1)`, `shift(5)`, trailing rolling mean, std and max, expanding mean and
z-score, causal EWM, RSI, true range, `ffill`, backward `merge_asof`, returns,
running drawdown, cumulative volume, rolling VWAP, and two that a regex linter
flags but that are causal: `rolling(5, center=True).mean().shift(2)` and a
next-bar label shifted back by one.

## Results

Measured with `python benchmarks/detection_matrix.py`, which is deterministic
and takes about 50 s. Tables have 200 rows. Every leaky feature ran with 10
seeds and every clean feature with 50; the table and the perturbations both
change with the seed. Hardware: Apple Silicon (arm64) laptop, CPython 3.13.0,
pure-Python features.

| leak class | features | flagged | exact column | exact horizon | regex baseline |
|---|---:|---:|---:|---:|---:|
| negative shift | 4 | 40/40 | 40/40 | 40/40 | 4/4 |
| next-bar label joined back | 2 | 20/20 | 20/20 | 20/20 | 2/2 |
| centred rolling window | 3 | 30/30 | 30/30 | 30/30 | 3/3 |
| full-sample statistic | 6 | 60/60 | 60/60 | 60/60 | 0/6 |
| future extremum | 4 | 40/40 | 40/40 | 40/40 | 2/4 |
| backward fill / forward as-of join | 4 | 40/40 | 40/40 | 40/40 | 1/4 |
| row-count dependence | 1 | 10/10 | 10/10 | 10/10 | 0/1 |
| **all** | 24 | 240/240 | 240/240 | 240/240 | 12/24 |

| detector | false positives on clean features |
|---|---:|
| peekahead | 0/950 checks |
| regex baseline | 2/19 features (`label_shifted_back`, `centred_then_lagged`) |

| function calls per check | mean | median | max |
|---|---:|---:|---:|
| leak found (detect + columns + horizon) | 196 | 188 | 245 |
| clean | 76 | 78 | 78 |

Scaling of the detection stage, for a leak that starts at 70% of the table,
with no random cuts:

| rows | cuts probed | detection-stage calls | log2(rows) |
|---:|---:|---:|---:|
| 100 | 20 | 105 | 6.6 |
| 1000 | 26 | 136 | 10.0 |
| 10000 | 38 | 203 | 13.3 |
| 100000 | 44 | 234 | 16.6 |

The corpus is small and was written alongside the detector. Treat these
numbers as proof that the corpus is handled exactly, not as recall on
arbitrary research code. The regex column is scored on the pandas snippets.

## Design notes

**Horizon from single-row perturbations, not from moving the cut.** The obvious
way to measure a horizon is to fix a cut `t`, perturb row `t + k` for
`k = 1..K`, and ask whether output `t` changes. That answer depends on where
`t` falls. For `bfill` over data reported every 5 rows, the measured horizon
cycles through 4, 3, 2, 1, 0 as `t` moves. Instead, peekahead perturbs one
source row `s` and looks at *every* earlier output row. The earliest row that
moves gives the lookahead into `s` directly, and `horizon_window` consecutive
sources cover every phase of a periodic gap shorter than the window. The cost is `(2 + noise_trials) × (horizon_window + 2)` calls per
leaking column, spent only after a leak is confirmed. A clean check stays at
about 76 calls.

**Extreme values alongside noise, and errors are not evidence.** Uniform noise
in the column's range almost never moves a full-sample max, and never trips
`future_high > 1.05 * close` on a calm random walk. A spike above the range and
a dip below it do both on every call, so each probe includes them, clamped to
stay non-negative for columns that are never negative (so `log(price)` still
works). The flip side is that perturbed inputs can be implausible, and a
feature may reject them. peekahead counts those exceptions but does not report
them as leaks. The choice trades a missed "validates the whole table" leak for
no false positives on features that are simply strict about their input.

## Limitations

- **It only sees what the table exercises.** A leak that fires on a pattern
  the synthetic data never produces, and that no spike, dip or noise draw
  triggers, is missed. The detection bound quantifies this only for leaks that
  fire independently per perturbation.
- **Bisection assumes leaks persist.** A leak confined to a short stretch of
  rows is found only if a random cut lands in it (see
  `test_intermittent_leak_needs_random_cuts_and_bisects_to_its_start`). Raise
  `random_cuts` for such features. The earliest row reported is the earliest
  row *observed*.
- **Horizon edge cases.** A bounded horizon longer than half the table can look
  like "whole sample", so use a table several times longer than the longest
  window you expect. A leak that no single-row perturbation reproduces (one
  that needs several future rows to change together) gets
  `horizon=None` for that column.
- **The cost depends on the rule that trades the feature.** The default
  expanding-median rule is causal and scale-free, but it is one rule, and
  "worth +2.01 Sharpe" means worth that much *to it*. A leak that is
  worthless to a median-crossing rule can be worth a great deal to one that
  trades the size of the signal. Pass your own `rule` when it matters, and
  read the divergence rate — which depends on nothing but the feature — as
  the scale-free number.
- **Causal recomputation reruns the feature per row.** That is quadratic, and
  it assumes the feature is a pure function of the table it is handed. One
  that caches, reads a global, or hits the network will report nonsense.
- **A feature must reduce to one number per row to be priced.** `measure`
  takes a single float per row; a feature returning several columns is
  detected as leaking but not priced.
- **Positional outputs must drop rows at the end, not the start.** A plain
  list shorter than the input is aligned from row 0. A function that drops
  warm-up rows must return `{row: value}`, or go through the pandas adapter,
  which aligns by label. Otherwise it will be flagged.
- **Tolerance hides tiny leaks.** Differences under `rtol=1e-9` are ignored,
  so a reverse EWM with a long half-life can decay below tolerance within the
  table. Pass `rtol=0, atol=0` for bitwise comparison.
- **Features must be deterministic.** A function with unseeded randomness
  will be reported as leaking.
- **Cost is measured in calls.** A check makes about 80 to 250 calls to the
  feature function, which is fine for vectorised features and slow for one
  that trains a model per call.
- Only single tables are handled. A feature that joins a second table has to
  read it from the same table (as the `eps` column does here) or close over
  it, and a closed-over table is not perturbed.

## License

MIT, see [LICENSE](LICENSE).
