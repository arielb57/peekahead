import json
import subprocess
import sys
import textwrap

import pytest

from peekahead.cli import load_function, main


def test_leaky_corpus_feature_exits_1_and_names_column(capsys):
    assert main(["peekahead.corpus:shift_minus_3", "--rows", "120"]) == 1
    out = capsys.readouterr().out
    assert "LEAK" in out
    assert "'close'" in out
    assert "reads 3 rows ahead" in out


def test_clean_feature_exits_0(capsys):
    assert main(["peekahead.corpus:ewm_span_10", "--rows", "120"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("peekahead.corpus:ewm_span_10 on 120 synthetic OHLCV rows")
    assert "clean" in out


def test_json_output_is_machine_readable(capsys):
    assert main(["peekahead.corpus:bfill_eps", "--rows", "100", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["columns"] == ["eps"]
    assert data["horizon"] == 4
    assert data["calls"] > 0


def test_file_path_target(tmp_path, capsys):
    path = tmp_path / "features.py"
    path.write_text(
        textwrap.dedent(
            """
            def zscore(t):
                c = t["close"]
                m = sum(c) / len(c)
                return [x - m for x in c]
            """
        )
    )
    assert main([f"{path}:zscore", "--rows", "80"]) == 1
    assert "whole sample" in capsys.readouterr().out


def test_pandas_target(tmp_path, capsys):
    pytest.importorskip("pandas")
    path = tmp_path / "pdfeatures.py"
    path.write_text("def centred(df):\n    return df['close'].rolling(9, center=True).mean()\n")
    assert main([f"{path}:centred", "--rows", "120", "--pandas"]) == 1
    assert "reads 4 rows ahead" in capsys.readouterr().out


@pytest.mark.parametrize(
    "spec",
    ["no_colon_here", "peekahead.corpus:does_not_exist", "not_a_module_xyz:fn", "missing_file.py:fn"],
)
def test_bad_targets_exit_2(spec, capsys):
    assert main([spec]) == 2
    assert "peekahead:" in capsys.readouterr().err


def test_feature_crashing_on_real_data_exits_2(tmp_path, capsys):
    path = tmp_path / "broken.py"
    path.write_text("def f(t):\n    return t['no_such_column']\n")
    assert main([f"{path}:f", "--rows", "20"]) == 2
    assert "KeyError" in capsys.readouterr().err


def test_load_function_rejects_non_callables():
    with pytest.raises(ValueError):
        load_function("peekahead.corpus:NAN")


def test_module_entry_point_runs():
    result = subprocess.run(
        [sys.executable, "-m", "peekahead", "peekahead.corpus:centred_mean_5", "--rows", "60"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "reads 2 rows ahead" in result.stdout
