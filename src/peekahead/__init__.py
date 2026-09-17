"""peekahead: black-box look-ahead bias detection for backtest features."""

from .core import METHODS, WHOLE_SAMPLE, Evidence, LeakReport, check, normalize_output, values_equal
from .cost import CostReport, Series, causal_values, expanding_median_rule, measure
from .synthetic import ohlcv

__all__ = [
    "METHODS",
    "WHOLE_SAMPLE",
    "CostReport",
    "Evidence",
    "LeakReport",
    "Series",
    "causal_values",
    "check",
    "expanding_median_rule",
    "measure",
    "normalize_output",
    "ohlcv",
    "values_equal",
]

__version__ = "0.1.0"
