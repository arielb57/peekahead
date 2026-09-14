"""The pandas snippets from the corpus (the code shown in the README) run through
the adapter and must get the same verdict as the pure-Python versions."""

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from peekahead import WHOLE_SAMPLE  # noqa: E402
from peekahead.corpus import CLEAN, LEAKY  # noqa: E402
from peekahead.pandas_adapter import check_frame, frame_from_table  # noqa: E402
from peekahead.synthetic import ohlcv  # noqa: E402


def compile_snippet(feature):
    namespace = {"pd": pd, "np": np}
    exec(feature.snippet, namespace)
    return namespace["feature"]


@pytest.mark.parametrize("feature", LEAKY, ids=lambda f: f.name)
def test_pandas_leaky_snippets_report_exact_column_and_horizon(feature):
    df = frame_from_table(ohlcv(200, seed=11))
    report = check_frame(compile_snippet(feature), df, seed=11)
    assert report.leaked, report.summary()
    assert set(report.columns) == set(feature.columns), report.summary()
    assert report.horizon == feature.horizon, report.summary()


@pytest.mark.parametrize("feature", CLEAN, ids=lambda f: f.name)
def test_pandas_clean_snippets_have_no_findings(feature):
    fn = compile_snippet(feature)
    for seed in range(5):
        report = check_frame(fn, frame_from_table(ohlcv(120, seed=seed)), seed=seed)
        assert not report.leaked, f"seed {seed}:\n{report.summary()}"


def test_dropped_warm_up_rows_are_aligned_by_label_not_position():
    df = frame_from_table(ohlcv(120, seed=3))

    def trailing_mean_dropna(frame):
        return frame["close"].rolling(10).mean().dropna()

    def centred_mean_dropna(frame):
        return frame["close"].rolling(7, center=True).mean().dropna()

    assert not check_frame(trailing_mean_dropna, df).leaked
    report = check_frame(centred_mean_dropna, df)
    assert report.columns == ("close",)
    assert report.horizon == 3
    assert report.earliest_row == 3


def test_dataframe_output_and_datetime_column_are_handled():
    df = frame_from_table(ohlcv(120, seed=8))
    df["report_date"] = df.index

    def two_features(frame):
        return pd.DataFrame(
            {"lag": frame["close"].shift(1), "lead": frame["volume"].shift(-2)},
            index=frame.index,
        )

    report = check_frame(two_features, df)
    assert report.columns == ("volume",)
    assert report.horizon == 2
    assert "report_date" not in report.column_horizons


def test_whole_sample_scaler_on_dataframe_output():
    df = frame_from_table(ohlcv(100, seed=9))

    def scaled(frame):
        cols = frame[["close", "volume"]]
        return (cols - cols.mean()) / cols.std()

    report = check_frame(scaled, df)
    assert set(report.columns) == {"close", "volume"}
    assert report.horizon == WHOLE_SAMPLE


def test_rejects_unsorted_index_and_non_frames():
    df = frame_from_table(ohlcv(20, seed=1))
    with pytest.raises(ValueError):
        check_frame(lambda f: f["close"], df.iloc[::-1])
    with pytest.raises(TypeError):
        check_frame(lambda f: f, {"close": [1.0, 2.0]})
