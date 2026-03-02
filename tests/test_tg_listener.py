from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skills.tg_listener.listener import (
    MAX_MSGS_PER_GROUP,
    RATE_WINDOW_SECONDS,
    TelegramListener,
    _SENDER_CACHE_CLEANUP_INTERVAL,
    _SENDER_CACHE_TTL,
)


def _make_listener() -> TelegramListener:
    """Build a TelegramListener with mocked dependencies."""
    session = MagicMock()
    redis = AsyncMock()
    with patch.object(TelegramListener, "_load_groups", return_value={
        -1001: {"name": "TestGroup", "category": "alpha", "priority": "high"},
        -1002: {"name": "SpamGroup", "category": "degen", "priority": "low"},
    }):
        return TelegramListener(session, redis)


def _make_event(
    chat_id: int = -1001,
    sender_id: int = 42,
    sender_obj: object | None = None,
    text: str = "bullish on $SOL",
) -> MagicMock:
    """Create a mock NewMessage event."""
    event = MagicMock()
    event.chat_id = chat_id
    event.sender_id = sender_id
    event.sender = sender_obj  # Telethon's sync cached attribute
    event.id = 999
    event.raw_text = text
    event.media = None
    event.reply_to_msg_id = None
    event.get_sender = AsyncMock(return_value=sender_obj)
    return event


# ---------------------------------------------------------------------------
# Sender cache tests
# ---------------------------------------------------------------------------

class TestSenderCache:
    @pytest.mark.asyncio
    async def test_sync_sender_available_no_api_call(self) -> None:
        """When event.sender is populated, no API call or cache is needed."""
        listener = _make_listener()
        sender_obj = SimpleNamespace(username="alice", first_name="Alice", id=42)
        event = _make_event(sender_obj=sender_obj)

        name, sid = await listener._get_sender_name(event)

        assert name == "alice"
        assert sid == 42
        event.get_sender.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_api_and_caches(self) -> None:
        """When event.sender is None and cache empty, falls back to API."""
        listener = _make_listener()
        fetched = SimpleNamespace(username="bob", first_name="Bob", id=42)
        event = _make_event(sender_id=42, sender_obj=None)
        event.get_sender = AsyncMock(return_value=fetched)

        name, sid = await listener._get_sender_name(event)

        assert name == "bob"
        assert sid == 42
        event.get_sender.assert_awaited_once()
        # Second call should hit cache
        event2 = _make_event(sender_id=42, sender_obj=None)
        event2.get_sender = AsyncMock()
        name2, _ = await listener._get_sender_name(event2)
        assert name2 == "bob"
        event2.get_sender.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_ttl_expiry(self) -> None:
        """Expired cache entries trigger a new API call."""
        listener = _make_listener()
        # Pre-populate cache with an expired entry
        expired_time = time.monotonic() - _SENDER_CACHE_TTL - 1
        listener._sender_cache[42] = ("stale_name", expired_time)

        fetched = SimpleNamespace(username="fresh", first_name="Fresh", id=42)
        event = _make_event(sender_id=42, sender_obj=None)
        event.get_sender = AsyncMock(return_value=fetched)

        name, sid = await listener._get_sender_name(event)

        assert name == "fresh"
        event.get_sender.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_api_error_returns_unknown(self) -> None:
        """On API failure, return ('unknown', sender_id) without crashing."""
        listener = _make_listener()
        event = _make_event(sender_id=99, sender_obj=None)
        event.get_sender = AsyncMock(side_effect=ConnectionError("timeout"))

        name, sid = await listener._get_sender_name(event)

        assert name == "unknown"
        assert sid == 99

    @pytest.mark.asyncio
    async def test_zero_sender_id(self) -> None:
        """sender_id=0 should short-circuit to unknown."""
        listener = _make_listener()
        event = _make_event(sender_id=0, sender_obj=None)
        event.sender_id = 0

        name, sid = await listener._get_sender_name(event)

        assert name == "unknown"
        assert sid == 0
        event.get_sender.assert_not_awaited()

    def test_cache_cleanup_evicts_stale(self) -> None:
        """Periodic cleanup removes entries older than TTL."""
        listener = _make_listener()
        now = time.monotonic()
        listener._sender_cache[1] = ("fresh", now)
        listener._sender_cache[2] = ("stale", now - _SENDER_CACHE_TTL - 10)
        listener._sender_cache[3] = ("also_stale", now - _SENDER_CACHE_TTL - 100)

        # Force cleanup by setting counter just below threshold
        listener._msg_count_since_cleanup = _SENDER_CACHE_CLEANUP_INTERVAL - 1
        listener._maybe_cleanup_sender_cache()

        assert 1 in listener._sender_cache
        assert 2 not in listener._sender_cache
        assert 3 not in listener._sender_cache

    def test_cache_cleanup_skips_when_not_due(self) -> None:
        """Cleanup only runs every N messages."""
        listener = _make_listener()
        now = time.monotonic()
        listener._sender_cache[1] = ("stale", now - _SENDER_CACHE_TTL - 10)
        listener._msg_count_since_cleanup = 0

        listener._maybe_cleanup_sender_cache()

        # Stale entry should still be there — cleanup didn't run
        assert 1 in listener._sender_cache


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_within_limit_allows_messages(self) -> None:
        """Messages under the rate limit are allowed."""
        listener = _make_listener()
        for _ in range(MAX_MSGS_PER_GROUP):
            assert listener._is_rate_limited(-1001) is False

    def test_over_limit_blocks_messages(self) -> None:
        """Messages exceeding the rate limit are blocked."""
        listener = _make_listener()
        for _ in range(MAX_MSGS_PER_GROUP):
            listener._is_rate_limited(-1001)

        assert listener._is_rate_limited(-1001) is True

    def test_window_reset_allows_messages_again(self) -> None:
        """After the rate window expires, messages are allowed again."""
        listener = _make_listener()
        # Fill up the limit
        for _ in range(MAX_MSGS_PER_GROUP):
            listener._is_rate_limited(-1001)
        assert listener._is_rate_limited(-1001) is True

        # Simulate window expiry by backdating the window start
        count, _ = listener._group_msg_counts[-1001]
        listener._group_msg_counts[-1001] = (
            count,
            time.monotonic() - RATE_WINDOW_SECONDS - 1,
        )
        assert listener._is_rate_limited(-1001) is False

    def test_separate_groups_independent(self) -> None:
        """Rate limits are tracked per group."""
        listener = _make_listener()
        for _ in range(MAX_MSGS_PER_GROUP):
            listener._is_rate_limited(-1001)

        assert listener._is_rate_limited(-1001) is True
        assert listener._is_rate_limited(-1002) is False


# ---------------------------------------------------------------------------
# Integration: _handle_message
# ---------------------------------------------------------------------------

class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_message_pushed_to_redis(self) -> None:
        """A normal message gets serialized and pushed to Redis."""
        listener = _make_listener()
        sender_obj = SimpleNamespace(username="whale", first_name="Whale", id=42)
        event = _make_event(sender_obj=sender_obj)

        await listener._handle_message(event)

        listener._redis.rpush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rate_limited_message_skipped(self) -> None:
        """Rate-limited messages are dropped before Redis push."""
        listener = _make_listener()
        sender_obj = SimpleNamespace(username="spammer", first_name="Spammer", id=1)
        # Exhaust the rate limit
        for _ in range(MAX_MSGS_PER_GROUP):
            listener._is_rate_limited(-1001)

        event = _make_event(chat_id=-1001, sender_obj=sender_obj)
        await listener._handle_message(event)

        listener._redis.rpush.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handler_never_crashes(self) -> None:
        """Exceptions in the handler are caught, not propagated."""
        listener = _make_listener()
        listener._redis.rpush = AsyncMock(side_effect=RuntimeError("redis down"))
        sender_obj = SimpleNamespace(username="user", first_name="User", id=1)
        event = _make_event(sender_obj=sender_obj)

        # Should not raise
        await listener._handle_message(event)
