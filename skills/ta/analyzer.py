"""TAAnalyzer — turn OHLCV candles into a compact technical-analysis signal.

Wraps the vendored tradingbot detectors. Every detector call is isolated in
try/except so a single failure degrades that dimension to neutral rather than
killing the whole signal. Returns a plain dict (JSON-serializable) so it drops
straight into the evaluator payload and the position store.

Input candles: list of dicts, chronological (oldest first), each with at least
`price_usd` (or `close`); optional `high`, `low`, `volume`, `ts` (unix seconds).
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()


def _neutral(reason: str) -> dict[str, Any]:
    return {
        "available": False, "reason": reason, "bias": "neutral", "ta_score": 0,
        "patterns": [], "support": None, "resistance": None,
        "position_in_range": None, "breakout_status": None, "volume_signal": None,
    }


class TAAnalyzer:
    def __init__(self, min_candles: int = 15) -> None:
        self._min_candles = min_candles

    def analyze(self, candles: list[dict[str, Any]]) -> dict[str, Any]:
        if not candles or len(candles) < self._min_candles:
            return _neutral(f"need >= {self._min_candles} candles, got {len(candles or [])}")
        try:
            import numpy as np  # noqa: F401
            import pandas as pd
        except ImportError:
            return _neutral("pandas/numpy not installed (pip install '.[ta]')")

        try:
            df = self._to_frame(candles, pd)
        except Exception as e:  # noqa: BLE001
            return _neutral(f"frame build failed: {e}")

        patterns: list[dict[str, Any]] = []
        bull = self._safe_patterns("bull_flag", df, bullish=True)
        bear = self._safe_patterns("bear_flag", df, bullish=False)
        patterns.extend(bull)
        patterns.extend(bear)

        sr = self._safe_sr(df)
        vol = self._safe_volume(df)

        closes = [float(c.get("price_usd", c.get("close"))) for c in candles]
        trend_pct = (closes[-1] - closes[0]) / closes[0] if closes and closes[0] else 0.0

        bstatus = (sr or {}).get("breakout_status")
        pos = bstatus.get("position_in_range") if isinstance(bstatus, dict) else None
        if isinstance(bstatus, dict):
            if bstatus.get("above_resistance"):
                status_str = "above_resistance"
            elif bstatus.get("below_support"):
                status_str = "below_support"
            else:
                status_str = "in_range"
        else:
            status_str = None

        bias, score = self._aggregate(bull, bear, bstatus, pos, vol, trend_pct)
        return {
            "available": True,
            "bias": bias,
            "ta_score": score,
            "trend_pct": round(trend_pct, 4),
            "patterns": patterns,
            "support": (sr or {}).get("nearest_support"),
            "resistance": (sr or {}).get("nearest_resistance"),
            "position_in_range": pos,
            "breakout_status": status_str,
            "volume_signal": (vol or {}).get("signal"),
            "volume_divergence": (vol or {}).get("divergence"),
        }

    # ---- frame ------------------------------------------------------------
    def _to_frame(self, candles: list[dict[str, Any]], pd):
        closes = [float(c.get("price_usd", c.get("close"))) for c in candles]
        highs = [float(c.get("high", c.get("price_usd", c.get("close")))) for c in candles]
        lows = [float(c.get("low", c.get("price_usd", c.get("close")))) for c in candles]
        vols = [float(c.get("volume", 0.0) or 0.0) for c in candles]
        ts = [c.get("ts") for c in candles]
        if all(t is not None for t in ts):
            idx = pd.to_datetime([int(t) for t in ts], unit="s")
        else:
            idx = pd.date_range(end=pd.Timestamp.now("UTC"), periods=len(closes), freq="h")
        return pd.DataFrame({"High": highs, "Low": lows, "Close": closes, "Volume": vols}, index=idx)

    # ---- detectors (isolated) --------------------------------------------
    def _safe_patterns(self, kind: str, df, bullish: bool) -> list[dict[str, Any]]:
        try:
            if kind == "bull_flag":
                from skills.ta.vendor.bull_flag_detector import BullFlagDetector
                found = BullFlagDetector().detect(df)
            else:
                from skills.ta.vendor.bear_flag_detector import BearFlagDetector
                found = BearFlagDetector().detect(df)
            out = []
            for p in found or []:
                out.append({"name": kind, "bullish": bullish,
                            "confidence": round(float(p.get("confidence", 0.0)), 3)})
            return out
        except Exception:  # noqa: BLE001
            log.debug("ta.pattern_error", kind=kind, exc_info=True)
            return []

    def _safe_sr(self, df) -> dict[str, Any] | None:
        try:
            from skills.ta.vendor.support_resistance_detector import SupportResistanceDetector
            return SupportResistanceDetector().detect(df)
        except Exception:  # noqa: BLE001
            log.debug("ta.sr_error", exc_info=True)
            return None

    def _safe_volume(self, df) -> dict[str, Any] | None:
        try:
            from skills.ta.vendor.volume_analyzer import VolumeAnalyzer
            return VolumeAnalyzer().analyze("token", hist_data=df)
        except Exception:  # noqa: BLE001
            log.debug("ta.volume_error", exc_info=True)
            return None

    # ---- aggregation ------------------------------------------------------
    def _aggregate(self, bull, bear, bstatus, pos, vol, trend_pct) -> tuple[str, int]:
        score = 0.0
        # Pattern signals.
        for p in bull:
            score += 40 * p["confidence"]
        for p in bear:
            score -= 40 * p["confidence"]

        # Structure: breakouts and where price sits in the range.
        if isinstance(bstatus, dict):
            if bstatus.get("above_resistance"):
                score += 20
            if bstatus.get("below_support"):
                score -= 30
        if isinstance(pos, (int, float)):
            # Low in range = more upside room; high = extended. Modest weight —
            # trend below decides whether "low" is a dip or a knife.
            score += (0.5 - float(pos)) * 15

        # Trend is the dominant term (a downtrend is not a "dip to buy").
        score += max(-30.0, min(30.0, float(trend_pct) * 60.0))

        # Volume confirmation.
        vol = vol or {}
        vsig = str(vol.get("signal") or "").lower()
        if vsig in ("buy", "bullish", "accumulation"):
            score += 15
        elif vsig in ("sell", "bearish", "distribution"):
            score -= 15
        if vol.get("divergence"):
            score -= 10

        score = max(-100.0, min(100.0, score))
        if score >= 20:
            bias = "bullish"
        elif score <= -20:
            bias = "bearish"
        else:
            bias = "neutral"
        return bias, int(round(score))
