from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis
import structlog

log = structlog.get_logger()

REDIS_KEY_RAW = "signals:messages:raw"


class MessageBuffer:
    """Redis-backed message buffer using RPUSH/LPOP pattern."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    async def push(self, message: dict[str, Any]) -> None:
        """Push a single message to the buffer."""
        await self._redis.rpush(REDIS_KEY_RAW, json.dumps(message))
        log.debug("buffer.pushed", message_id=message.get("message_id"))

    async def pop_batch(
        self, max_size: int = 50, max_wait_seconds: float = 300.0
    ) -> list[dict[str, Any]]:
        """Pop up to max_size messages, or whatever accumulated in max_wait_seconds.

        Uses blocking BLPOP with timeout so we wake up either when messages
        arrive or when the wait expires.
        """
        messages: list[dict[str, Any]] = []
        import asyncio

        deadline = asyncio.get_event_loop().time() + max_wait_seconds

        while len(messages) < max_size:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break

            # BLPOP with remaining timeout
            result = await self._redis.blpop(
                REDIS_KEY_RAW, timeout=int(max(remaining, 1))
            )
            if result is None:
                # Timeout reached, return what we have
                break

            _key, raw = result
            try:
                msg = json.loads(raw)
                messages.append(msg)
            except json.JSONDecodeError:
                log.error("buffer.invalid_json", raw=raw[:200])
                continue

            # Drain any remaining messages without blocking (up to max_size)
            while len(messages) < max_size:
                raw = await self._redis.lpop(REDIS_KEY_RAW)
                if raw is None:
                    break
                try:
                    messages.append(json.loads(raw))
                except json.JSONDecodeError:
                    log.error("buffer.invalid_json", raw=str(raw)[:200])

        log.info("buffer.batch_popped", count=len(messages))
        return messages

    async def length(self) -> int:
        """Current buffer length."""
        return await self._redis.llen(REDIS_KEY_RAW)
