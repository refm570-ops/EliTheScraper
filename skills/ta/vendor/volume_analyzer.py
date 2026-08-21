"""
Volume Analyzer
===============
Analyzes volume patterns to detect unusual activity, price-volume divergences,
and volume trends. Extracted from backend_server.py.
"""

from typing import Dict, Any, Optional
import numpy as np


class VolumeAnalyzer:
    """Analyzes volume patterns for trading signals."""

    def analyze(self, ticker: str, hist_data=None) -> Dict[str, Any]:
        """
        Analyze volume patterns for a given ticker.

        Args:
            ticker: Stock ticker symbol
            hist_data: Pre-fetched historical DataFrame with OHLCV columns.
                       If None, fetches 3mo of data via yfinance.

        Returns:
            Dict with volume metrics, signal, confidence, and analysis text.
        """
        if hist_data is None:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period="3mo")

        if hist_data is None or len(hist_data) < 20:
            return self._empty_result(ticker)

        today_vol = float(hist_data['Volume'].iloc[-1])
        avg_20_vol = float(hist_data['Volume'].iloc[-20:].mean())
        avg_5_vol = float(hist_data['Volume'].iloc[-5:].mean())

        relative_vol = today_vol / avg_20_vol if avg_20_vol > 0 else 1.0
        vol_trend_ratio = avg_5_vol / avg_20_vol if avg_20_vol > 0 else 1.0

        price_change = float(hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]) if len(hist_data) >= 2 else 0
        price_up = price_change >= 0
        unusual_volume = relative_vol > 1.5

        # Divergence detection
        divergence = False
        if len(hist_data) >= 10:
            recent_closes = hist_data['Close'].iloc[-10:].values
            recent_vols = hist_data['Volume'].iloc[-10:].values
            price_trending_up = recent_closes[-1] > recent_closes[0]
            vol_trending_down = float(np.mean(recent_vols[-5:])) < float(np.mean(recent_vols[:5]))
            vol_trending_up = float(np.mean(recent_vols[-5:])) > float(np.mean(recent_vols[:5]))

            if price_trending_up and vol_trending_down:
                divergence = True  # Bearish divergence
            elif not price_trending_up and vol_trending_up:
                divergence = True  # Bullish divergence (potential bounce)

        # Signal logic
        signal = self._determine_signal(relative_vol, price_up, divergence, hist_data)

        # Confidence
        confidence = self._calculate_confidence(unusual_volume, divergence, vol_trend_ratio)

        # Analysis text
        analysis_text = self._build_analysis_text(
            ticker, relative_vol, vol_trend_ratio, price_up, unusual_volume, divergence, hist_data
        )

        return {
            "relative_volume": round(relative_vol, 2),
            "vol_trend_ratio": round(vol_trend_ratio, 2),
            "unusual_volume": unusual_volume,
            "price_up": price_up,
            "divergence": divergence,
            "signal": signal,
            "confidence": round(confidence, 2),
            "analysis_text": analysis_text,
        }

    def _determine_signal(self, relative_vol: float, price_up: bool, divergence: bool, hist_data) -> str:
        """Determine trading signal from volume data."""
        if relative_vol > 1.5 and price_up:
            return "buy"
        elif relative_vol > 1.5 and not price_up:
            return "sell"
        elif divergence:
            # Price up + vol down = weakening rally -> hold
            # Price down + vol up = climax selling -> potential buy
            if len(hist_data) >= 10:
                price_trending_up = hist_data['Close'].iloc[-1] > hist_data['Close'].iloc[-10]
                if not price_trending_up:
                    return "buy"  # Climax selling, potential bounce
            return "hold"
        return "hold"

    def _calculate_confidence(self, unusual_volume: bool, divergence: bool, vol_trend_ratio: float) -> float:
        """Calculate confidence score with wider range."""
        confidence = 0.30  # Lower base — volume alone is confirming, not primary
        if unusual_volume:
            confidence += 0.25  # Unusual volume is a strong signal
        if divergence:
            confidence += 0.20  # Divergence is highly actionable
        if vol_trend_ratio > 1.3:
            confidence += 0.10  # Strong volume trend
        elif vol_trend_ratio > 1.1 or vol_trend_ratio < 0.8:
            confidence += 0.05
        return round(min(max(confidence, 0.15), 0.95), 2)

    def _build_analysis_text(
        self, ticker: str, relative_vol: float, vol_trend_ratio: float,
        price_up: bool, unusual_volume: bool, divergence: bool, hist_data
    ) -> str:
        """Build thesis-driven analysis text."""
        parts = []
        parts.append(f"{ticker.upper()} today's trading activity is {relative_vol:.1f}x the 20-day average.")

        if relative_vol > 1.5 and price_up:
            parts.append("Thesis: Much more trading than usual on a green day — big players are buying in. This kind of heavy buying backs up the move higher and makes it more likely to continue.")
        elif relative_vol > 1.5 and not price_up:
            parts.append("Thesis: Much more trading than usual on a red day — heavy selling is happening. Big players may be getting out. This adds weight to the downside move.")
        elif relative_vol < 0.7 and price_up:
            parts.append("Thesis: The stock rose, but very few people are actually trading it. Moves on low activity are unreliable and often reverse. Don't trust this rally until more buyers show up.")
        elif relative_vol < 0.7 and not price_up:
            parts.append("Thesis: The stock fell, but on very light trading. The selling pressure is weak. This dip may not have much follow-through.")
        else:
            parts.append("Trading activity is normal — nothing unusual to report.")

        if vol_trend_ratio > 1.2:
            parts.append(f"Over the past 5 days, trading activity has been picking up ({vol_trend_ratio:.1f}x normal) — growing interest in this stock.")
        elif vol_trend_ratio < 0.8:
            parts.append(f"Over the past 5 days, trading activity has been dropping off ({vol_trend_ratio:.1f}x normal) — the stock is getting quiet, which often happens before a big move.")

        if divergence and len(hist_data) >= 10:
            recent_closes = hist_data['Close'].iloc[-10:].values
            price_trending_up = recent_closes[-1] > recent_closes[0]
            if price_trending_up:
                parts.append("Warning: The stock is going up but fewer people are buying each day. This disconnect suggests the rally is losing steam and could reverse.")
            else:
                parts.append("Note: The stock is falling but trading is picking up. This can mean the selling is reaching a climax — a bounce may be coming soon.")

        return " ".join(parts)

    def _empty_result(self, ticker: str) -> Dict[str, Any]:
        """Return empty result when insufficient data."""
        return {
            "relative_volume": 1.0,
            "vol_trend_ratio": 1.0,
            "unusual_volume": False,
            "price_up": False,
            "divergence": False,
            "signal": "hold",
            "confidence": 0.0,
            "analysis_text": f"Insufficient data for {ticker} volume analysis.",
        }
