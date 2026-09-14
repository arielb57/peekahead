import hashlib
import math

import pytest

from peekahead import WHOLE_SAMPLE, check, normalize_output, ohlcv, values_equal
from peekahead.corpus import by_name


def ramp(n, name="x"):
    return {name: [float(i) for i in range(n)]}


def shift_ahead(k, col="x"):
    def fn(t):
        xs = t[col]
        return [xs[i + k] if i + k < len(xs) else math.nan for i in range(len(xs))]

    return fn


# -- edge cases -------------------------------------------------------------------


def test_empty_table_is_clean_and_never_calls_fn():
    def explode(t):
        raise AssertionError("must not be called")

    report = check(explode, {"x": []})
    assert not report.leaked
    assert report.calls == 0 and report.rows == 0
    assert check(explode, {}).rows == 0


def test_single_row_table_has_nothing_to_perturb():
    report = check(shift_ahead(1), {"x": [1.0]})
    assert not report.leaked
    assert report.calls == 0
    assert "fewer than 2 rows" in report.summary()


def test_two_row_table_still_finds_a_one_step_leak():
    report = check(shift_ahead(1), {"x": [1.0, 2.0]})
    assert report.leaked
    assert report.earliest_row == 0
    assert report.horizon == 1


def test_constant_column_leak_is_only_found_by_value_perturbation():
    def demean(t):
        m = sum(t["x"]) / len(t["x"])
        return [x - m for x in t["x"]]

    table = {"x": [5.0] * 80}
    report = check(demean, table)
    assert report.leaked
    assert report.evidence[0].method in ("spike", "dip", "noise")
    assert report.columns == ("x",)
    assert report.horizon == WHOLE_SAMPLE
    # removal keeps the mean at 5.0 and permuting a constant column is a no-op
    blind = check(demean, table, methods=("removal", "permute"))
    assert not blind.leaked


def test_noise_alone_finds_constant_column_leak():
    def demean(t):
        m = sum(t["x"]) / len(t["x"])
        return [x - m for x in t["x"]]

    report = check(demean, {"x": [5.0] * 80}, methods=("noise",))
    assert report.leaked
    assert report.evidence[0].method == "noise"


def test_length_only_leak_needs_removal():
    fn = by_name("row_count_progress").fn
    table = ohlcv(60, seed=1)
    assert check(fn, table, freeze=["ts"]).leaked
    assert not check(fn, table, freeze=["ts"], methods=("noise", "extreme", "permute")).leaked


def test_nan_warm_up_rows_compare_equal():
    def trailing_mean(t):
        xs = t["x"]
        return [float("nan") if i < 9 else sum(xs[i - 9 : i + 1]) / 10 for i in range(len(xs))]

    report = check(trailing_mean, ramp(60))
    assert not report.leaked, report.summary()


def test_none_and_nan_are_both_missing():
    def mixed(t):
        return [None if i % 2 else math.nan for i in range(len(t["x"]))]

    assert not check(mixed, ramp(30)).leaked
    assert values_equal(None, math.nan)
    assert not values_equal(None, 0.0)


def test_float_outputs_within_tolerance_are_equal():
    # A length-dependent wobble of one part in 1e13 is rounding, not signal.
    def wobble(t):
        n = len(t["x"])
        return [x * (1 + 1e-13 * (n % 2)) for x in t["x"]]

    table = {"x": [float(i + 1) for i in range(50)]}
    assert not check(wobble, table).leaked
    strict = check(wobble, table, rtol=0.0, atol=0.0)
    assert strict.leaked
    assert strict.evidence[0].method == "removal"


def test_values_equal_rules():
    assert values_equal(1.0, 1.0 + 1e-12)
    assert not values_equal(1.0, 1.0001)
    assert values_equal(math.inf, math.inf)
    assert not values_equal(math.inf, 1e308)
    assert values_equal((1, math.nan), [1, math.nan])
    assert not values_equal((1, 2), (1, 2, 3))
    assert values_equal("a", "a") and not values_equal("a", "b")


def test_fewer_rows_returned_positionally_are_compared_by_prefix():
    def next_diff(t):
        xs = t["x"]
        return [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]

    report = check(next_diff, {"x": [float(i * i) for i in range(40)]})
    assert report.leaked
    assert report.horizon == 1


def test_row_keyed_output_skips_warm_up_without_false_positives():
    def trailing_sum_dropped(t):
        xs = t["x"]
        return {i: sum(xs[i - 4 : i + 1]) for i in range(4, len(xs))}

    def leading_sum_dropped(t):
        xs = t["x"]
        return {i: sum(xs[i : i + 3]) for i in range(len(xs) - 2)}

    assert not check(trailing_sum_dropped, ramp(50)).leaked
    report = check(leading_sum_dropped, ramp(50))
    assert report.leaked and report.horizon == 2


def test_dict_of_lists_output_attributes_the_right_column():
    def features(t):
        c, v = t["close"], t["volume"]
        n = len(c)
        return {
            "lagged_close": [c[i - 1] if i else math.nan for i in range(n)],
            "volume_lead3": [v[i + 3] if i + 3 < n else math.nan for i in range(n)],
        }

    report = check(features, ohlcv(120, seed=2), freeze=["ts"])
    assert report.columns == ("volume",)
    assert report.horizon == 3


def test_intermittent_leak_needs_random_cuts_and_bisects_to_its_start():
    # Rows 60..70 read two rows ahead; the fixed quartile cuts all miss that stretch.
    def local_leak(t):
        xs = t["x"]
        return [xs[i + 2] if 60 <= i <= 70 and i + 2 < len(xs) else xs[i] for i in range(len(xs))]

    table = {"x": [float((i * 7919) % 101) for i in range(200)]}
    assert not check(local_leak, table, random_cuts=0).leaked
    found = [check(local_leak, table, seed=s, random_cuts=64) for s in range(20)]
    hits = [r for r in found if r.leaked]
    assert len(hits) >= 15
    assert all(r.earliest_row == 60 and r.horizon == 2 for r in hits)


def test_bisection_uses_logarithmically_many_cuts():
    def leaks_from_row_7000(t):
        xs = t["x"]
        return [xs[i + 1] if 7000 <= i < len(xs) - 1 else xs[i] for i in range(len(xs))]

    report = check(leaks_from_row_7000, ramp(10_000), random_cuts=0)
    assert report.earliest_row == 7000
    # five fixed cuts, then bisection over [0, first hit] probing up to 3 cuts per step
    assert len(report.cuts_checked) <= 5 + 3 * (math.ceil(math.log2(10_000)) + 1)


def test_bisection_is_not_fooled_by_non_leaking_cuts_on_report_rows():
    # bfill over a column reported every 5 rows: cuts on report rows (0, 5, 10, ...)
    # do not leak, yet row 1 already reads row 5.
    fn = by_name("bfill_eps").fn
    for seed in range(10):
        report = check(fn, ohlcv(250, seed=seed), freeze=["ts"], seed=seed)
        assert report.earliest_row == 1, report.summary()


def test_frozen_columns_are_never_perturbed():
    def needs_sorted_ts(t):
        ts = t["ts"]
        if any(a >= b for a, b in zip(ts, ts[1:])):
            raise ValueError("unsorted")
        return list(t["close"])

    table = ohlcv(60, seed=3)
    frozen = check(needs_sorted_ts, table, freeze=["ts"])
    assert not frozen.leaked and frozen.errors == 0
    unfrozen = check(needs_sorted_ts, table)
    assert not unfrozen.leaked and unfrozen.errors > 0


def test_errors_on_perturbed_input_are_counted_not_reported_as_leaks():
    def rejects_big_values(t):
        if max(t["x"]) > 100:
            raise ValueError("out of range")
        return list(t["x"])

    report = check(rejects_big_values, ramp(40))
    assert not report.leaked
    assert report.errors > 0


def test_base_call_errors_propagate():
    def broken(t):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        check(broken, ramp(10))


def test_same_seed_gives_identical_report():
    fn = by_name("bfill_eps").fn
    a = check(fn, ohlcv(150, seed=4), freeze=["ts"], seed=9)
    b = check(fn, ohlcv(150, seed=4), freeze=["ts"], seed=9)
    assert a.to_dict() == b.to_dict()


def test_function_that_mutates_its_input_cannot_corrupt_the_check():
    def mutating_lag(t):
        xs = t["x"]
        out = [math.nan] + xs[:-1]
        xs[:] = [0.0] * len(xs)
        return out

    assert not check(mutating_lag, ramp(40)).leaked


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        check(list, {"a": [1, 2], "b": [1]})
    with pytest.raises(ValueError):
        check(lambda t: t["a"], {"a": [1, 2]}, freeze=["nope"])
    with pytest.raises(ValueError):
        check(lambda t: t["a"], {"a": [1, 2]}, methods=("telepathy",))
    with pytest.raises(TypeError):
        check(lambda t: 3.0, {"a": [1.0, 2.0]})
    with pytest.raises(TypeError):
        check("not callable", {"a": [1.0, 2.0]})


def test_normalize_output_shapes():
    assert normalize_output([1, 2]) == {0: 1, 1: 2}
    assert normalize_output({3: "a"}) == {3: "a"}
    assert normalize_output({"f": [1, 2], "g": [3, 4]}) == {0: (("f", 1), ("g", 3)), 1: (("f", 2), ("g", 4))}
    with pytest.raises(TypeError):
        normalize_output({"f": [1, 2], "g": [3]})


def test_summary_and_dict_describe_the_leak():
    report = check(by_name("centred_mean_5").fn, ohlcv(120, seed=5), freeze=["ts"])
    text = report.summary()
    assert "LEAK" in text and "'close'" in text and "reads 2 rows ahead" in text
    d = report.to_dict()
    assert d["leaked"] and d["columns"] == ["close"] and d["horizon"] == 2
    assert d["evidence"][0]["stage"] == "detect"


# -- data-dependent leaks: the detection bound ------------------------------------


def coin_leak(p):
    """Row 0 flips a pseudo-random coin keyed on the whole table: every distinct
    perturbation independently changes it with probability about p."""

    def fn(t):
        digest = hashlib.sha256(repr(sorted(t.items())).encode()).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        out = [0.0] * len(t["x"])
        out[0] = 1.0 if u < p else 0.0
        return out

    return fn


@pytest.mark.parametrize("p", [0.01, 0.03])
def test_threshold_leak_is_detected_at_least_as_often_as_the_stated_bound(p):
    runs = 300
    detected = 0
    bounds = []
    for s in range(runs):
        table = {"x": [float(v) for v in ohlcv(60, seed=s)["close"]]}
        clean = check(lambda t: list(t["x"]), table, seed=s)
        bounds.append(clean.detection_bound(p))
        detected += check(coin_leak(p), table, seed=s).leaked
    bound = sum(bounds) / runs
    assert 0.2 < bound < 0.95, "the bound must be informative for this test to mean anything"
    slack = 3 * math.sqrt(bound * (1 - bound) / runs)
    assert detected / runs >= bound - slack, (detected / runs, bound)


def test_detection_bound_arithmetic():
    report = check(lambda t: list(t["x"]), ramp(60))
    r = report.perturbations
    assert r > 0
    assert report.detection_bound(0.0) == 0.0
    assert report.detection_bound(1.0) == 1.0
    assert report.detection_bound(0.02) == pytest.approx(1 - 0.98**r)
    assert report.detection_bound(report.min_detectable_rate(0.9)) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        report.detection_bound(1.5)
    with pytest.raises(ValueError):
        report.min_detectable_rate(1.0)
