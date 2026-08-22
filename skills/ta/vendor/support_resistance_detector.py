"""
Support and Resistance Detector
Identifies key support and resistance levels using swing point analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime


class SupportResistanceDetector:
    """
    Detects support and resistance levels in price data.

    Support levels indicate where buying pressure prevented further decline.
    Resistance levels show where selling pressure capped upward movement.
    """

    def __init__(
        self,
        lookback: int = 5,              # Pivot lookback period
        history_days: int = 90,          # Days of history to analyze
        tolerance: float = 0.02          # Price clustering tolerance (2%)
    ):
        """
        Initialize support/resistance detector.

        Args:
            lookback: Number of candles to look back/forward for pivots (default 5)
            history_days: Days of historical data to analyze (default 90)
            tolerance: Tolerance for level clustering as % (default 0.02 = 2%)
        """
        self.lookback = lookback
        self.history_days = history_days
        self.tolerance = tolerance

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect support and resistance levels in price data.

        Args:
            df: DataFrame with OHLCV data (DatetimeIndex, columns: High, Low, Close)

        Returns:
            Dictionary with support/resistance levels and analysis
        """
        # Validate data
        if df is None or len(df) < self.lookback * 2 + 1:
            return self._empty_result()

        # Ensure required columns exist
        required_cols = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns: {required_cols}")

        # Use recent history
        recent_data = df.tail(min(self.history_days, len(df)))

        # Current price
        current_price = float(df['Close'].iloc[-1])

        # Find swing points
        swing_highs_raw = self._find_swing_highs(recent_data)
        swing_lows_raw = self._find_swing_lows(recent_data)

        # Cluster and strengthen levels
        resistance_levels = self._process_levels(
            swing_highs_raw, recent_data, current_price, level_type="resistance"
        )
        support_levels = self._process_levels(
            swing_lows_raw, recent_data, current_price, level_type="support"
        )

        # Find nearest levels
        nearest_resistance = self._find_nearest_resistance(
            resistance_levels, current_price, df['High'].max()
        )
        nearest_support = self._find_nearest_support(
            support_levels, current_price, df['Low'].min()
        )

        # Trading range detection
        in_trading_range = self._detect_trading_range(
            nearest_support, nearest_resistance, current_price,
            support_levels, resistance_levels
        )

        # Breakout status
        breakout_status = self._analyze_breakout(
            current_price, nearest_support, nearest_resistance
        )

        result = {
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "nearest_support": float(nearest_support),
            "nearest_resistance": float(nearest_resistance),
            "current_price": float(current_price),
            "in_trading_range": in_trading_range,
            "breakout_status": breakout_status
        }

        # Convert all numpy types to native Python types
        return self._convert_to_native(result)

    def _find_swing_highs(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find swing highs (potential resistance)"""
        swing_highs = []
        lookback = self.lookback

        for i in range(lookback, len(df) - lookback):
            # Swing high: Higher than lookback candles on both sides
            if (df['High'].iloc[i] > df['High'].iloc[i-lookback:i].max() and
                df['High'].iloc[i] > df['High'].iloc[i+1:i+lookback+1].max()):
                swing_highs.append({
                    'price': float(df['High'].iloc[i]),
                    'time': df.index[i].strftime('%Y-%m-%d'),
                    'index': i
                })

        return swing_highs

    def _find_swing_lows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find swing lows (potential support)"""
        swing_lows = []
        lookback = self.lookback

        for i in range(lookback, len(df) - lookback):
            # Swing low: Lower than lookback candles on both sides
            if (df['Low'].iloc[i] < df['Low'].iloc[i-lookback:i].min() and
                df['Low'].iloc[i] < df['Low'].iloc[i+1:i+lookback+1].min()):
                swing_lows.append({
                    'price': float(df['Low'].iloc[i]),
                    'time': df.index[i].strftime('%Y-%m-%d'),
                    'index': i
                })

        return swing_lows

    def _process_levels(
        self,
        raw_levels: List[Dict[str, Any]],
        df: pd.DataFrame,
        current_price: float,
        level_type: str
    ) -> List[Dict[str, Any]]:
        """
        Process raw swing points into levels with strength ratings.

        Args:
            raw_levels: Raw swing points
            df: Price data
            current_price: Current price for distance calculation
            level_type: "support" or "resistance"

        Returns:
            List of processed levels with strength
        """
        if not raw_levels:
            return []

        # Cluster nearby levels
        clustered = self._cluster_levels(raw_levels)

        # Calculate strength (touches) for each level
        processed = []
        for cluster_price in clustered:
            # Count how many times price tested this level
            strength = self._calculate_strength(cluster_price, df, level_type)

            # Find last touch
            last_touch = self._find_last_touch(cluster_price, df, level_type)

            # Calculate distance from current price
            distance_pct = abs(cluster_price - current_price) / current_price * 100

            processed.append({
                'price': float(cluster_price),
                'strength': strength,
                'distance_pct': round(distance_pct, 2),
                'type': level_type,
                'last_touch': last_touch
            })

        # Sort by strength (descending) and distance (ascending)
        processed.sort(key=lambda x: (-x['strength'], x['distance_pct']))

        return processed

    def _cluster_levels(self, levels: List[Dict[str, Any]]) -> List[float]:
        """
        Cluster nearby levels within tolerance.

        Args:
            levels: Raw swing points

        Returns:
            List of clustered price levels
        """
        if not levels:
            return []

        prices = sorted([l['price'] for l in levels])
        clusters = []
        current_cluster = [prices[0]]

        for price in prices[1:]:
            # If within tolerance of current cluster, add to it
            if (price - current_cluster[0]) / current_cluster[0] <= self.tolerance:
                current_cluster.append(price)
            else:
                # Finalize current cluster (use average)
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [price]

        # Don't forget last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))

        return clusters

    def _calculate_strength(self, level: float, df: pd.DataFrame, level_type: str) -> int:
        """
        Calculate strength of a level (number of touches).

        Args:
            level: Price level
            df: Price data
            level_type: "support" or "resistance"

        Returns:
            Number of touches
        """
        touches = 0

        if level_type == "support":
            # Count how many lows came close to this level
            for low in df['Low']:
                if abs(low - level) / level < self.tolerance:
                    touches += 1
        else:  # resistance
            # Count how many highs came close to this level
            for high in df['High']:
                if abs(high - level) / level < self.tolerance:
                    touches += 1

        return min(touches, 10)  # Cap at 10 for display

    def _find_last_touch(self, level: float, df: pd.DataFrame, level_type: str) -> str:
        """Find the date of the last touch of this level"""
        last_touch_date = None

        if level_type == "support":
            for i in range(len(df) - 1, -1, -1):
                if abs(df['Low'].iloc[i] - level) / level < self.tolerance:
                    last_touch_date = df.index[i]
                    break
        else:  # resistance
            for i in range(len(df) - 1, -1, -1):
                if abs(df['High'].iloc[i] - level) / level < self.tolerance:
                    last_touch_date = df.index[i]
                    break

        return last_touch_date.strftime('%Y-%m-%d') if last_touch_date is not None else "unknown"

    def _find_nearest_resistance(
        self,
        resistance_levels: List[Dict[str, Any]],
        current_price: float,
        fallback: float
    ) -> float:
        """Find nearest resistance above current price"""
        above_price = [r['price'] for r in resistance_levels if r['price'] > current_price]
        return min(above_price) if above_price else fallback

    def _find_nearest_support(
        self,
        support_levels: List[Dict[str, Any]],
        current_price: float,
        fallback: float
    ) -> float:
        """Find nearest support below current price"""
        below_price = [s['price'] for s in support_levels if s['price'] < current_price]
        return max(below_price) if below_price else fallback

    def _detect_trading_range(
        self,
        support: float,
        resistance: float,
        current_price: float,
        support_levels: List[Dict[str, Any]],
        resistance_levels: List[Dict[str, Any]]
    ) -> bool:
        """
        Detect if price is in a defined trading range.

        Criteria:
        - At least 2 touches of support
        - At least 2 touches of resistance
        - Range is at least 5% wide
        """
        # Find touches for nearest levels
        support_touches = next((s['strength'] for s in support_levels if s['price'] == support), 1)
        resistance_touches = next((r['strength'] for r in resistance_levels if r['price'] == resistance), 1)

        # Check criteria
        range_width_pct = (resistance - support) / support

        return bool(
            support_touches >= 2 and
            resistance_touches >= 2 and
            range_width_pct >= 0.05  # At least 5% range
        )

    def _analyze_breakout(
        self,
        current_price: float,
        support: float,
        resistance: float
    ) -> Dict[str, Any]:
        """Analyze breakout status"""
        # Position in range (0.0 = at support, 1.0 = at resistance)
        if resistance > support:
            position = (current_price - support) / (resistance - support)
            position = max(0.0, min(1.0, position))  # Clamp to 0-1
        else:
            position = 0.5

        return {
            "above_resistance": bool(current_price > resistance),
            "below_support": bool(current_price < support),
            "position_in_range": round(position, 3)
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result when detection fails"""
        return {
            "support_levels": [],
            "resistance_levels": [],
            "nearest_support": None,
            "nearest_resistance": None,
            "current_price": None,
            "in_trading_range": False,
            "breakout_status": {
                "above_resistance": False,
                "below_support": False,
                "position_in_range": 0.5
            }
        }

    def detect_from_ticker(self, ticker: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Convenience method: detect support/resistance from ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            period: Data period (default "6mo")

        Returns:
            Detection result dictionary
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required for ticker input. Install with: pip install yfinance")

        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            return self._empty_result()

        return self.detect(df)

    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze ticker for support/resistance (compatible with agent interface).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Analysis result with signal
        """
        result = self.detect_from_ticker(ticker, period="6mo")

        if result['current_price'] is None:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "key_levels": {
                    "support": None,
                    "resistance": None
                },
                "details": None
            }

        # Determine signal based on position
        position = result['breakout_status']['position_in_range']
        current = result['current_price']
        support = result['nearest_support']
        resistance = result['nearest_resistance']

        # Distance to levels
        distance_to_support = abs(current - support) / support
        distance_to_resistance = abs(current - resistance) / resistance

        # Signal logic
        if position < 0.25 and distance_to_support < 0.03:  # Near support
            signal = "buy"
            confidence = 0.75
        elif position > 0.75 and distance_to_resistance < 0.03:  # Near resistance
            signal = "sell"
            confidence = 0.75
        elif result['breakout_status']['above_resistance']:
            signal = "bullish"
            confidence = 0.70
        elif result['breakout_status']['below_support']:
            signal = "bearish"
            confidence = 0.70
        else:
            signal = "hold"
            confidence = 0.50

        return {
            "signal": signal,
            "confidence": confidence,
            "key_levels": {
                "support": support,
                "resistance": resistance
            },
            "details": result
        }

    def _convert_to_native(self, obj):
        """Recursively convert numpy/pandas types to Python native types."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native(item) for item in obj]
        return obj
