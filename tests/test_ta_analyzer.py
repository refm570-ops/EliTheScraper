"""Tests for the vendored-TA adapter (skills/ta/analyzer.py).

Skipped automatically if pandas/numpy (the `ta` extra) aren't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

from skills.ta.analyzer import TAAnalyzer


def _series(mult_seq):
    p = 1.0
    out = []
    for m in mult_seq:
        p *= m
        out.append({"price_usd": p, "volume": 1000})
    return out


def test_uptrend_is_bullish():
    ta = TAAnalyzer(min_candles=15)
    pole = [1.08] * 10
    flag = [0.99, 1.005] * 6
    sig = ta.analyze(_series(pole + flag))
    assert sig["available"] is True
    assert sig["bias"] == "bullish"
    assert sig["ta_score"] > 0


def test_downtrend_is_bearish():
    ta = TAAnalyzer(min_candles=15)
    sig = ta.analyze(_series([0.95] * 22))
    assert sig["available"] is True
    assert sig["bias"] == "bearish"
    assert sig["ta_score"] < 0


def test_choppy_is_neutral():
    ta = TAAnalyzer(min_candles=15)
    sig = ta.analyze(_series([1.01, 0.99] * 11))
    assert sig["available"] is True
    assert sig["bias"] == "neutral"


def test_too_few_candles_unavailable():
    ta = TAAnalyzer(min_candles=15)
    sig = ta.analyze([{"price_usd": 1.0}] * 5)
    assert sig["available"] is False
    assert sig["bias"] == "neutral"


def test_empty_candles_unavailable():
    ta = TAAnalyzer()
    assert ta.analyze([])["available"] is False
