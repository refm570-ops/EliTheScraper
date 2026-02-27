from __future__ import annotations

from typing import Any

from storage.ticker_store import TickerStore


class SocialMetricsCounter:
    """Thin wrapper over TickerStore for social signal queries."""

    def __init__(self, ticker_store: TickerStore) -> None:
        self._store = ticker_store

    async def get_metrics(
        self, ticker: str, window_hours: float = 6.0
    ) -> dict[str, Any]:
        """Get social metrics for a ticker within a time window."""
        return await self._store.get_social_metrics(ticker, window_hours)

    def meets_group_threshold(
        self, metrics: dict[str, Any], min_groups: int
    ) -> bool:
        """Check if the ticker has been mentioned in enough unique groups."""
        return metrics.get("unique_groups", 0) >= min_groups

    @staticmethod
    def has_multi_platform(metrics: dict[str, Any]) -> bool:
        """Check if the ticker has mentions from multiple platforms."""
        return metrics.get("unique_sources", 0) >= 2
