"""
Bull Flag Pattern Detector
Identifies bull flag continuation patterns: strong upward move (pole) followed by consolidation (flag)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime


class BullFlagDetector:
    """
    Detects bull flag continuation patterns in price data.

    A bull flag consists of:
    1. Pole: Strong upward move (10%+ in 5-10 candles)
    2. Flag: Consolidation with slight downward slope (5-15 candles)
    3. Breakout: Price breaks above flag resistance
    """

    def __init__(
        self,
        min_pole_gain: float = 0.10,     # Minimum 10% gain for pole
        max_flag_range: float = 1.0,     # Flag max 100% of pole height
        min_flag_range: float = 0.2,     # Flag min 20% of pole height
        min_candles: int = 15            # Minimum candles needed
    ):
        """
        Initialize bull flag detector.

        Args:
            min_pole_gain: Minimum pole rise percentage (default 0.10 = 10%)
            max_flag_range: Maximum flag height as % of pole (default 1.0)
            min_flag_range: Minimum flag height as % of pole (default 0.2)
            min_candles: Minimum candles required (default 15)
        """
        self.min_pole_gain = min_pole_gain
        self.max_flag_range = max_flag_range
        self.min_flag_range = min_flag_range
        self.min_candles = min_candles

    def detect(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect bull flag patterns in price data.

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

        # Scan for potential poles (strong upward moves)
        for i in range(len(df) - self.min_candles):
            window = df.iloc[i:i+10]

            # Check for pole: strong upward move
            pole_gain = (window['Close'].iloc[-1] - window['Close'].iloc[0]) / window['Close'].iloc[0]

            if pole_gain < self.min_pole_gain:
                continue

            pole_start_idx = i
            pole_end_idx = i + len(window) - 1
            pole_height = window['Close'].iloc[-1] - window['Close'].iloc[0]

            # Check for flag consolidation in next 5-15 candles
            flag_start_idx = pole_end_idx + 1
            flag_end_idx = min(flag_start_idx + 15, len(df) - 1)

            if flag_end_idx >= len(df):
                break

            flag_window = df.iloc[flag_start_idx:flag_end_idx]

            if len(flag_window) < 5:
                continue

            # Flag characteristics: consolidation with narrowing range
            flag_high = flag_window['High'].max()
            flag_low = flag_window['Low'].min()
            flag_range = flag_high - flag_low

            # Flag should be 20-100% of pole height
            if flag_range > pole_height * self.max_flag_range or flag_range < pole_height * self.min_flag_range:
                continue

            # Check for downward sloping consolidation (characteristic of bull flag)
            flag_closes = flag_window['Close'].values
            if len(flag_closes) >= 3:
                # Simple linear regression slope
                x = np.arange(len(flag_closes))
                slope = np.polyfit(x, flag_closes, 1)[0]

                # Should be slightly downward or sideways
                if slope > 0.01 * flag_closes[0]:  # Too bullish, not a flag
                    continue

            # Calculate breakout level and target
            breakout_level = flag_high * 1.005  # 0.5% above flag high
            target = flag_high + pole_height    # Project pole height from breakout

            # Check current status - ONLY CONFIRMED BREAKOUTS
            # Skip forming patterns (no decision yet)
            if flag_end_idx == len(df) - 1:
                continue  # Pattern is still forming

            # Check if breakout happened after flag completed
            # Look at price action in next 10 candles after flag
            breakout_window_end = min(flag_end_idx + 10, len(df) - 1)
            prices_after_flag = df['Close'].iloc[flag_end_idx:breakout_window_end+1]

            # Skip if no breakout occurred
            if prices_after_flag.max() <= breakout_level:
                continue  # No breakout yet

            # Pattern confirmed!
            status = "confirmed"

            # Volume confirmation on breakout
            has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
            volume_confirmed = False
            if has_volume:
                avg_volume = df['Volume'].iloc[max(0, flag_end_idx-20):flag_end_idx].mean()
                breakout_volume = df['Volume'].iloc[flag_end_idx:min(flag_end_idx + 3, len(df))].max()
                volume_confirmed = breakout_volume > avg_volume * 1.5

            # Quality-based confidence scoring
            confidence = 0.65  # Base for confirmed pattern

            # Pole strength bonus
            if pole_gain >= 0.15:
                confidence += 0.10  # Strong pole

            # Volume confirmation bonus/penalty
            if volume_confirmed:
                confidence += 0.15  # High-volume breakout
            else:
                confidence -= 0.10  # Low-volume breakout is suspect

            # Calculate trendline coordinates for visualization
            # Upper trendline (resistance)
            flag_highs = flag_window['High'].values
            upper_trendline = []
            for idx, (date_idx, high_val) in enumerate(zip(range(flag_start_idx, flag_end_idx + 1), flag_highs)):
                if high_val >= flag_high * 0.95:  # Near the top
                    upper_trendline.append({
                        "time": df.index[date_idx].strftime('%Y-%m-%d'),
                        "price": float(high_val)
                    })

            # Lower trendline (support)
            flag_lows = flag_window['Low'].values
            lower_trendline = []
            for idx, (date_idx, low_val) in enumerate(zip(range(flag_start_idx, flag_end_idx + 1), flag_lows)):
                if low_val <= flag_low * 1.05:  # Near the bottom
                    lower_trendline.append({
                        "time": df.index[date_idx].strftime('%Y-%m-%d'),
                        "price": float(low_val)
                    })

            # Create synthetic boundary points if not enough natural points
            if len(upper_trendline) < 2:
                upper_trendline = [
                    {"time": df.index[flag_start_idx].strftime('%Y-%m-%d'), "price": float(flag_high)},
                    {"time": df.index[flag_end_idx].strftime('%Y-%m-%d'), "price": float(flag_high)}
                ]
            if len(lower_trendline) < 2:
                lower_trendline = [
                    {"time": df.index[flag_start_idx].strftime('%Y-%m-%d'), "price": float(flag_low)},
                    {"time": df.index[flag_end_idx].strftime('%Y-%m-%d'), "price": float(flag_low)}
                ]

            # Calculate flag duration
            flag_duration = (df.index[flag_end_idx] - df.index[flag_start_idx]).days

            # Create pattern object
            pattern = {
                "detected": True,
                "pattern": "bull_flag",
                "confidence": confidence,
                "status": status,
                "coordinates": {
                    "pole_start": {
                        "time": df.index[pole_start_idx].strftime('%Y-%m-%d'),
                        "price": float(df['Close'].iloc[pole_start_idx])
                    },
                    "pole_end": {
                        "time": df.index[pole_end_idx].strftime('%Y-%m-%d'),
                        "price": float(df['Close'].iloc[pole_end_idx])
                    },
                    "flag_start": df.index[flag_start_idx].strftime('%Y-%m-%d'),
                    "flag_end": df.index[flag_end_idx].strftime('%Y-%m-%d'),
                    "upper_trendline": upper_trendline,
                    "lower_trendline": lower_trendline,
                    "breakout_level": float(breakout_level),
                    "target": float(target)
                },
                "description": f"Bull flag pattern with {pole_gain*100:.1f}% pole{' (volume confirmed)' if volume_confirmed else ' (low volume)'}. Target: ${target:.2f}",
                "pole_gain_percent": float(round(pole_gain * 100, 1)),
                "pole_height": float(round(pole_height, 2)),
                "flag_duration_days": int(flag_duration),
                "volume_confirmed": volume_confirmed
            }

            patterns.append(pattern)

        return patterns

    def detect_from_ticker(self, ticker: str, period: str = "6mo") -> List[Dict[str, Any]]:
        """
        Convenience method: detect bull flags from ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
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
        Get the most recent bull flag pattern.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Most recent pattern or None
        """
        patterns = self.detect(df)

        if not patterns:
            return None

        # Return pattern with most recent flag_end date
        return max(patterns, key=lambda p: p['coordinates']['flag_end'])

    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze ticker for bull flag (compatible with agent interface).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Analysis result with signal
        """
        patterns = self.detect_from_ticker(ticker)

        if patterns:
            pattern = patterns[0]
            return {
                "signal": "buy",
                "confidence": pattern['confidence'],
                "target": pattern['coordinates']['target'],
                "pattern_detected": True,
                "details": pattern
            }
        else:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "target": None,
                "pattern_detected": False,
                "details": None
            }
