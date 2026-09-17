"""What the leak is worth: causal recomputation and the rule that trades it."""

from __future__ import annotations

import math
import statistics

import pytest

from peekahead import ohlcv
from peekahead.cost import causal_values, expanding_median_rule, measure


def close(table):
    return list(table["close"])


def next_return(table):
    """shift(-1): perfect foresight of the very next bar."""
    c = close(table)
    return [c[min(i + 1, len(c) - 1)] / c[i] - 1 for i in range(len(c))]


def centred_mean_7(table):
    c = close(table)
    return [statistics.fmean(c[max(0, i - 3) : i + 4]) for i in range(len(c))]


def past_return(table):
    c = close(table)
    return [None if i == 0 else c[i] / c[i - 1] - 1 for i in range(len(c))]


def rolling_mean_5(table):
    c = close(table)
    return [None if i < 4 else statistics.fmean(c[i - 4 : i + 1]) for i in range(len(c))]


TABLE = ohlcv(120, seed=11)


class TestCausalRecomputation:
    def test_a_causal_feature_is_unchanged_by_recomputation(self):
        for feature in (past_return, rolling_mean_5):
            report = measure(feature, TABLE)
            assert report.diverging_rows == (), feature.__name__
            assert not report.leaked
            assert report.compared_rows > 100

    def test_a_leaking_feature_diverges_almost_everywhere(self):
        report = measure(next_return, TABLE)
        assert report.leaked
        assert report.divergence_rate > 0.9

    def test_recomputation_matches_a_hand_written_causal_version(self):
        # The point of the module is that it needs no access to the source, so
        # the check is against a version written out by hand.
        values, calls = causal_values(centred_mean_7, TABLE)
        c = close(TABLE)
        expected = [statistics.fmean(c[max(0, i - 3) : i + 1]) for i in range(len(c))]
        assert calls == len(c)
        for got, want in zip(values, expected):
            assert got == pytest.approx(want)

    def test_a_warm_up_failure_is_a_missing_value_not_a_crash(self):
        def needs_ten_rows(table):
            c = close(table)
            if len(c) < 10:
                raise ValueError("not enough history")
            return [statistics.fmean(c[max(0, i - 9) : i + 1]) for i in range(len(c))]

        values, _ = causal_values(needs_ten_rows, TABLE)
        assert values[:9] == [None] * 9
        assert values[9] is not None


class TestAttribution:
    def test_perfect_foresight_is_worth_a_large_sharpe(self):
        report = measure(next_return, TABLE)
        # Trading tomorrow's return known today is the textbook impossible
        # backtest; the causal leg has no information at all.
        assert report.reported.sharpe() > 5
        assert report.causal.sharpe() == pytest.approx(0.0, abs=1e-9)
        assert report.sharpe_gap == pytest.approx(report.reported.sharpe())

    def test_a_clean_feature_attributes_nothing(self):
        report = measure(rolling_mean_5, TABLE)
        assert report.sharpe_gap == pytest.approx(0.0, abs=1e-12)
        assert "nothing to attribute" in report.summary()

    def test_the_gap_can_be_small_even_when_the_leak_is_real(self):
        """A leak is not automatically worth anything, and saying so is the point."""

        def whole_sample_zscore(table):
            c = close(table)
            mean, sd = statistics.fmean(c), statistics.pstdev(c)
            return [(x - mean) / (sd + 1e-9) for x in c]

        report = measure(whole_sample_zscore, TABLE)
        assert report.leaked
        # A monotone rescaling barely moves a rule that only looks at rank.
        assert abs(report.sharpe_gap) < abs(report.reported.sharpe()) + 1.0

    def test_missing_price_column_is_named(self):
        with pytest.raises(KeyError, match="mid"):
            measure(rolling_mean_5, TABLE, price="mid")


class TestRule:
    def test_the_rule_reads_only_the_past(self):
        values = [3.0, 1.0, 2.0, 5.0, 0.0]
        positions = expanding_median_rule(values)
        # Row 0 is its own median, so it is flat; each later row compares
        # against the median of everything up to and including itself.
        assert positions[0] == 0
        for cut in range(1, len(values)):
            assert expanding_median_rule(values[: cut + 1]) == positions[: cut + 1]

    def test_missing_values_are_flat(self):
        assert expanding_median_rule([None, float("nan"), 1.0, 2.0]) == [0, 0, 0, 1]

    def test_sharpe_of_a_constant_series_is_zero_not_infinite(self):
        report = measure(lambda t: [1.0] * len(t["close"]), TABLE)
        assert report.reported.sharpe() == 0.0
        assert math.isfinite(report.sharpe_gap)
