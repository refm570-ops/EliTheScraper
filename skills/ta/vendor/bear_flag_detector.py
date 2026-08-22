"""
Bear Flag Pattern Detector
Identifies bear flag continuation patterns: sharp decline (pole) + upward consolidation (flag)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime


class BearFlagDetector:
    """
    Detects bear flag continuation patterns in price data.

    A bear flag consists of:
    1. Pole: Sharp downward move (10%+ decline in 5-10 candles)
    2. Flag: Upward or sideways consolidation (counter-trend bounce)
    3. Breakdown: Price breaks below flag low with volume
    """

    def __init__(
        self,
        min_pole_decline: float = 0.10,      # Minimum 10% decline for pole
        max_flag_range: float = 0.6,         # Max flag range (60% of pole height)
        min_flag_range: float = 0.2,         # Min flag range (20% of pole height)
        min_candles: int = 15,               # Minimum candles needed
        require_volume_surge: bool = False   # Require volume on breakdown
    ):
        """
        Initialize bear flag detector.

        Args:
            min_pole_decline: Minimum pole decline percentage (default 0.10 = 10%)
            max_flag_range: Maximum flag range as % of pole (default 0.6 = 60%)
            min_flag_range: Minimum flag range as % of pole (default 0.2 = 20%)
            min_candles: Minimum candles required (default 15)
            require_volume_surge: Require volume surge for confirmation (default False)
        """
        self.min_pole_decline = min_pole_decline
        self.max_flag_range = max_flag_range
        self.min_flag_range = min_flag_range
        self.min_candles = min_candles
        self.require_volume_surge = require_volume_surge

    def detect(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect bear flag patterns in price data.

        Args:
            df: DataFrame with OHLCV data (DatetimeIndex, columns: High, Low, Close, Volume)

        Returns:
            List of detected patterns (empty list if none found)
        """
        patterns = []

        # Validate data
        if df is None or len(df) < self.min_candles:
            return patterns

        # Ensure required columns exist
        required_cols = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns: {required_cols}")

        has_volume = 'Volume' in df.columns

        # Scan for potential poles (strong downward moves)
        for i in range(len(df) - self.min_candles):
            # Look at 10-candle window for pole
            pole_window = df.iloc[i:i+10]

            if len(pole_window) < 5:
                continue

            # Check for pole: strong downward move
            pole_start_price = pole_window['Close'].iloc[0]
            pole_end_price = pole_window['Close'].iloc[-1]
            pole_decline = (pole_start_price - pole_end_price) / pole_start_price

            # Require minimum decline
            if pole_decline < self.min_pole_decline:
                continue

            pole_start_idx = i
            pole_end_idx = i + len(pole_window) - 1
            pole_height = pole_start_price - pole_end_price

            # Check for flag consolidation after pole
            flag_start_idx = pole_end_idx + 1
            flag_end_idx = min(flag_start_idx + 15, len(df) - 1)

            if flag_end_idx >= len(df):
                break

            flag_window = df.iloc[flag_start_idx:flag_end_idx]

            # Need at least 5 candles for flag
            if len(flag_window) < 5:
                continue

            # Flag characteristics
            flag_high = flag_window['High'].max()
            flag_low = flag_window['Low'].min()
            flag_range = flag_high - flag_low

            # Flag range should be 20-60% of pole height
            if flag_range > pole_height * self.max_flag_range:
                continue  # Too wide
            if flag_range < pole_height * self.min_flag_range:
                continue  # Too narrow

            # Check for upward or sideways slope (characteristic of bear flag)
            flag_closes = flag_window['Close'].values

            if len(flag_closes) >= 3:
                x = np.arange(len(flag_closes))
                slope = np.polyfit(x, flag_closes, 1)[0]

                # Should be upward or sideways (not strongly bearish)
                if slope < -0.01 * flag_closes[0]:  # Too bearish
                    continue

                # Classify slope
                if slope > 0.005 * flag_closes[0]:
                    flag_slope = "upward"
                elif slope < -0.005 * flag_closes[0]:
                    flag_slope = "slight_down"
                else:
                    flag_slope = "sideways"
            else:
                flag_slope = "unknown"

            # Breakdown level
            breakout_level = flag_low * 0.995  # 0.5% below flag low
            target = flag_low - pole_height     # Project pole height downward

            # Check current status
            current_price = df['Close'].iloc[-1]

            # Determine if pattern is confirmed or forming
            if flag_end_idx == len(df) - 1:
                # Pattern is at the end of data
                status = "confirmed" if current_price <= breakout_level else "forming"
            else:
                # Check if breakdown happened after flag
                breakout_window_end = min(flag_end_idx + 10, len(df) - 1)
                prices_after_flag = df['Close'].iloc[flag_end_idx:breakout_window_end+1]

                # Check for breakdown
                if prices_after_flag.min() <= breakout_level:
                    status = "confirmed"
                else:
                    status = "forming"

            # Volume confirmation (if available and confirmed)
            volume_confirmed = False
            if has_volume and status == "confirmed":
                # Check for volume surge on breakdown
                avg_volume = df['Volume'].iloc[max(0, flag_end_idx-20):flag_end_idx].mean()

                if flag_end_idx < len(df) - 1:
                    breakdown_volume = df['Volume'].iloc[flag_end_idx:min(flag_end_idx + 3, len(df))].max()
                    volume_confirmed = breakdown_volume > avg_volume * 1.5
                else:
                    current_volume = df['Volume'].iloc[-1]
                    volume_confirmed = current_volume > avg_volume * 1.5

            # Skip if volume surge required but not present
            if self.require_volume_surge and status == "confirmed" and not volume_confirmed:
                continue

            # Pattern quality scoring
            quality_score = 0.70  # Base confidence

            # Pole strength bonus
            if pole_decline >= 0.15:
                quality_score += 0.10  # Strong pole

            # Flag slope bonus (upward is better for bear flags)
            if flag_slope == "upward":
                quality_score += 0.10
            elif flag_slope == "sideways":
                quality_score += 0.05

            # Flag range bonus (tighter is better)
            flag_range_ratio = flag_range / pole_height
            if flag_range_ratio <= 0.40:
                quality_score += 0.05  # Tight flag

            # Volume bonus
            if volume_confirmed:
                quality_score += 0.10

            # Status penalty (forming patterns less confident)
            if status == "forming":
                quality_score *= 0.85

            quality_score = min(quality_score, 0.90)

            # Create pattern result
            pattern = {
                "detected": True,
                "pattern": "bear_flag",
                "confidence": round(quality_score, 2),
                "status": status,
                "coordinates": {
                    "pole_start": {
                        "time": df.index[pole_start_idx].strftime('%Y-%m-%d'),
                        "price": float(pole_start_price)
                    },
                    "pole_end": {
                        "time": df.index[pole_end_idx].strftime('%Y-%m-%d'),
                        "price": float(pole_end_price)
                    },
                    "flag_start": df.index[flag_start_idx].strftime('%Y-%m-%d'),
                    "flag_end": df.index[flag_end_idx].strftime('%Y-%m-%d'),
                    "flag_top": float(flag_high),
                    "flag_bottom": float(flag_low),
                    "breakout_level": float(breakout_level),
                    "target": float(target)
                },
                "description": f"Bear flag with {pole_decline*100:.1f}% decline. Target: ${target:.2f} ({status})",
                "pole_decline_percent": float(round(pole_decline * 100, 1)),
                "flag_slope": flag_slope,  # Already a string
                "volume_confirmed": bool(volume_confirmed)
            }

            patterns.append(pattern)

        return patterns

    def detect_from_ticker(self, ticker: str, period: str = "6mo") -> List[Dict[str, Any]]:
        """
        Convenience method: detect bear flags from ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g., "TSLA")
            period: Data period (default "6mo")

        Returns:
            List of detected patterns
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required for ticker input. Install with: pip install yfinance")

        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            return []

        return self.detect(df)

    def get_latest_pattern(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Get the most recent bear flag pattern.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Most recent pattern or None
        """
        patterns = self.detect(df)

        if not patterns:
            return None

        # Return pattern with most recent flag_start date
        return max(patterns, key=lambda p: p['coordinates']['flag_start'])

    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze ticker for bear flags (compatible with agent interface).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Analysis result with signal
        """
        patterns = self.detect_from_ticker(ticker, period="6mo")

        if patterns:
            # Filter for confirmed patterns first
            confirmed = [p for p in patterns if p['status'] == 'confirmed']

            if confirmed:
                pattern = confirmed[0]
                return {
                    "signal": "sell",
                    "confidence": pattern['confidence'],
                    "target": pattern['coordinates']['target'],
                    "pattern_detected": True,
                    "status": "confirmed",
                    "details": pattern
                }
            else:
                # Return forming pattern
                pattern = patterns[0]
                return {
                    "signal": "watch",
                    "confidence": pattern['confidence'] * 0.8,  # Lower confidence
                    "target": pattern['coordinates']['target'],
                    "pattern_detected": True,
                    "status": "forming",
                    "details": pattern
                }
        else:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "target": None,
                "pattern_detected": False,
                "status": None,
                "details": None
            }
