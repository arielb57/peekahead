"""The corpus is the ground truth: exact column and horizon for every leak,
zero findings for every clean feature across 50 seeds."""

import pytest

from peekahead import WHOLE_SAMPLE, check, ohlcv
from peekahead.corpus import ALL, CLEAN, LEAKY, by_name


def test_corpus_size_meets_the_spec():
    assert len(LEAKY) >= 20
    assert len(CLEAN) >= 15
    assert len({f.name for f in ALL}) == len(ALL)


@pytest.mark.parametrize("feature", LEAKY, ids=lambda f: f.name)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_leaky_feature_is_flagged_with_exact_column_and_horizon(feature, seed):
    report = check(feature.fn, ohlcv(200, seed=seed), freeze=["ts"], seed=seed)
    assert report.leaked, report.summary()
    assert set(report.columns) == set(feature.columns), report.summary()
    assert report.horizon == feature.horizon, report.summary()
    assert report.evidence, "a leak must come with evidence"


@pytest.mark.parametrize("feature", CLEAN, ids=lambda f: f.name)
def test_clean_features_have_zero_findings_across_50_seeds(feature):
    for seed in range(50):
        report = check(feature.fn, ohlcv(120, seed=seed), freeze=["ts"], seed=seed)
        assert not report.leaked, f"seed {seed}:\n{report.summary()}"
        assert report.errors == 0


def test_earliest_row_accounts_for_warm_up():
    table = ohlcv(200, seed=4)
    assert check(by_name("shift_minus_3").fn, table, freeze=["ts"]).earliest_row == 0
    # rolling(21, center=True) is NaN until row 10, which then reads row 20
    assert check(by_name("centred_mean_21").fn, table, freeze=["ts"]).earliest_row == 10
    # pct_change is NaN on row 0, so the full-sample scaler first leaks on row 1
    assert check(by_name("scaler_fit_on_full_sample").fn, table, freeze=["ts"]).earliest_row == 1


def test_multi_column_leak_names_both_columns():
    report = check(by_name("full_sample_vwap").fn, ohlcv(150, seed=5), freeze=["ts"])
    assert report.columns == ("close", "volume")
    assert report.column_horizons == {"close": WHOLE_SAMPLE, "volume": WHOLE_SAMPLE}


def test_row_count_leak_is_found_by_removal_and_names_no_column():
    report = check(by_name("row_count_progress").fn, ohlcv(100, seed=6), freeze=["ts"])
    assert report.leaked
    assert report.columns == ()
    assert report.evidence[0].method == "removal"
    assert report.horizon == WHOLE_SAMPLE


def test_horizon_evidence_points_at_the_source_row():
    report = check(by_name("shift_minus_10").fn, ohlcv(200, seed=7), freeze=["ts"])
    horizon_ev = [e for e in report.evidence if e.stage == "horizon"]
    assert horizon_ev
    assert all(e.source_row - e.row == 10 for e in horizon_ev)
    assert horizon_ev[0].column == "close"
