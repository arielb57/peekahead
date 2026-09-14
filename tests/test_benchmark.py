import importlib.util
import pathlib

from peekahead.corpus import CLEAN, LEAKY

PATH = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "detection_matrix.py"


def load_benchmark():
    spec = importlib.util.spec_from_file_location("detection_matrix", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regex_baseline_misses_what_it_cannot_see_and_flags_clean_code():
    bench = load_benchmark()
    missed = {f.leak_class for f in LEAKY if not bench.regex_flags(f)}
    assert "full-sample statistic" in missed
    assert "row-count dependence" in missed
    assert {f.name for f in CLEAN if bench.regex_flags(f)} == {"label_shifted_back", "centred_then_lagged"}


def test_benchmark_runs_and_reports_perfect_recall_on_one_seed():
    text = load_benchmark().run(seeds=1, clean_seeds=1, rows=120)
    assert f"| **all** | {len(LEAKY)} | {len(LEAKY)}/{len(LEAKY)} | {len(LEAKY)}/{len(LEAKY)} |" in text
    assert f"| peekahead | 0/{len(CLEAN)} checks |" in text
    assert "| 100000 |" in text
