from __future__ import annotations

import asyncio
import random
import time
from functools import wraps
from typing import Any, Callable, Coroutine

import structlog
from telethon.errors import FloodWaitError

log = structlog.get_logger()


class FloodGuard:
    """Exponential backoff and jitter for Telegram API safety."""

    def __init__(
        self,
        initial_backoff: float = 60.0,
        max_backoff: float = 900.0,  # 15 minutes
        jitter_min: float = 1.0,
        jitter_max: float = 3.0,
    ) -> None:
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._jitter_min = jitter_min
        self._jitter_max = jitter_max
        self._current_backoff = initial_backoff
        self._consecutive_floods = 0

    def reset(self) -> None:
        self._current_backoff = self._initial_backoff
        self._consecutive_floods = 0

    async def jitter(self) -> None:
        """Add random delay before any proactive API call."""
        delay = random.uniform(self._jitter_min, self._jitter_max)
        log.debug("flood_guard.jitter", delay_seconds=round(delay, 2))
        await asyncio.sleep(delay)

    async def handle_flood_wait(self, error: FloodWaitError) -> None:
        """Handle FloodWaitError with the server-mandated wait + extra backoff."""
        telegram_wait = error.seconds
        extra = self._current_backoff
        total_wait = telegram_wait + extra

        self._consecutive_floods += 1
        log.warning(
            "flood_guard.flood_wait",
            telegram_wait=telegram_wait,
            extra_backoff=extra,
            total_wait=total_wait,
            consecutive_floods=self._consecutive_floods,
        )

        await asyncio.sleep(total_wait)

        # Exponential backoff for next flood
        self._current_backoff = min(
            self._current_backoff * 2, self._max_backoff
        )

    def safe_call(
        self, func: Callable[..., Coroutine[Any, Any, Any]]
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        """Decorator that wraps an async call with jitter and flood handling."""
        guard = self

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await guard.jitter()
            try:
                result = await func(*args, **kwargs)
                guard.reset()
                return result
            except FloodWaitError as e:
                await guard.handle_flood_wait(e)
                # Retry once after waiting
                return await func(*args, **kwargs)

        return wrapper
