from __future__ import annotations

import time

import structlog

log = structlog.get_logger()


class RateLimiter:
    """Per-ticker cooldown and global rate limiting for alerts."""

    def __init__(
        self,
        ticker_cooldown_seconds: float = 1800.0,  # 30 minutes
        global_max_per_minute: int = 20,
    ) -> None:
        self._ticker_cooldown = ticker_cooldown_seconds
        self._global_max = global_max_per_minute
        self._ticker_last_sent: dict[str, float] = {}
        self._global_timestamps: list[float] = []

    def can_send_ticker(self, ticker: str) -> bool:
        """Check if we can send an alert for this ticker (per-ticker cooldown)."""
        now = time.monotonic()
        last = self._ticker_last_sent.get(ticker)
        if last is not None and (now - last) < self._ticker_cooldown:
            remaining = self._ticker_cooldown - (now - last)
            log.debug(
                "rate_limiter.ticker_cooldown",
                ticker=ticker,
                remaining_seconds=round(remaining),
            )
            return False
        return True

    def can_send_global(self) -> bool:
        """Check global rate limit (max N messages per minute)."""
        now = time.monotonic()
        # Prune old timestamps
        self._global_timestamps = [
            t for t in self._global_timestamps if (now - t) < 60.0
        ]
        return len(self._global_timestamps) < self._global_max

    def record_send(self, ticker: str) -> None:
        """Record that an alert was sent for this ticker."""
        now = time.monotonic()
        self._ticker_last_sent[ticker] = now
        self._global_timestamps.append(now)
