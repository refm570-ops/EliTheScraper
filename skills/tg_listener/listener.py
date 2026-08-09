from __future__ import annotations

import asyncio
import json
import signal
import time
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
import structlog
import yaml
from telethon import events

from skills.local_config import resolve_config_path
from skills.tg_listener.flood_guard import FloodGuard
from skills.tg_listener.session_manager import SessionManager

log = structlog.get_logger()

REDIS_KEY_RAW = "signals:messages:raw"

# Sender cache TTL in seconds (1 hour)
_SENDER_CACHE_TTL = 3600.0
# Evict stale cache entries every N messages
_SENDER_CACHE_CLEANUP_INTERVAL = 1000

# Per-group rate limiting
MAX_MSGS_PER_GROUP = 60
RATE_WINDOW_SECONDS = 60.0


class TelegramListener:
    """Passively listens to Telegram groups and pushes messages to Redis.

    Safety: ONLY uses events.NewMessage. Never calls get_messages(),
    get_participants(), or any history/member API.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        redis: aioredis.Redis,
        groups_config_path: str = "config/groups.yml",
    ) -> None:
        self._session = session_manager
        self._redis = redis
        self._groups = self._load_groups(groups_config_path)
        self._flood_guard = FloodGuard()
        self._running = False
        self._shutdown_event = asyncio.Event()
        # sender_id → (sender_name, cached_at_timestamp)
        self._sender_cache: dict[int, tuple[str, float]] = {}
        self._msg_count_since_cleanup = 0
        # group_id → (count, window_start_timestamp)
        self._group_msg_counts: dict[int, tuple[int, float]] = {}

    @staticmethod
    def _load_groups(path: str) -> dict[int, dict[str, Any]]:
        with open(resolve_config_path(path)) as f:
            config = yaml.safe_load(f)
        groups: dict[int, dict[str, Any]] = {}
        for g in config.get("groups", []):
            groups[g["id"]] = {
                "name": g.get("name", "Unknown"),
                "category": g.get("category", ""),
                "priority": g.get("priority", "medium"),
            }
        log.info("listener.groups_loaded", count=len(groups))
        return groups

    async def start(self) -> None:
        """Start listening. Blocks until shutdown signal."""
        client = await self._session.start()
        self._running = True

        group_ids = list(self._groups.keys())
        log.info("listener.registering_handler", group_ids=group_ids)

        # incoming=True: only handle messages from others, never our own.
        # Telethon's NewMessage handler does NOT send read receipts automatically;
        # that only happens via explicit client.send_read_acknowledge() calls.
        @client.on(events.NewMessage(chats=group_ids, incoming=True))
        async def on_new_message(event: events.NewMessage.Event) -> None:
            await self._handle_message(event)

        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_shutdown)

        log.info("listener.started", groups=len(group_ids))
        await self._shutdown_event.wait()

    def _request_shutdown(self) -> None:
        log.info("listener.shutdown_requested")
        self._running = False
        self._shutdown_event.set()

    def _is_rate_limited(self, group_id: int) -> bool:
        """Return True if this group has exceeded MAX_MSGS_PER_GROUP in the current window."""
        now = time.monotonic()
        count, window_start = self._group_msg_counts.get(group_id, (0, now))

        if now - window_start >= RATE_WINDOW_SECONDS:
            # Window expired, start fresh
            self._group_msg_counts[group_id] = (1, now)
            return False

        if count >= MAX_MSGS_PER_GROUP:
            return True

        self._group_msg_counts[group_id] = (count + 1, window_start)
        return False

    async def _get_sender_name(self, event: events.NewMessage.Event) -> tuple[str, int]:
        """Resolve sender name with minimal API calls.

        Priority:
        1. event.sender (Telethon's sync cached attribute — free, no API call)
        2. In-memory _sender_cache with 1-hour TTL
        3. Fallback to await event.get_sender() and cache the result
        4. On any error, return ("unknown", sender_id)
        """
        sender_id = event.sender_id or 0
        if sender_id == 0:
            return ("unknown", 0)

        # 1. Try Telethon's sync cached attribute (no API call)
        sender = event.sender
        if sender is not None:
            name = getattr(sender, "username", "") or getattr(sender, "first_name", "")
            return (name or "unknown", sender_id)

        # 2. Check in-memory cache
        now = time.monotonic()
        cached = self._sender_cache.get(sender_id)
        if cached is not None:
            cached_name, cached_at = cached
            if now - cached_at < _SENDER_CACHE_TTL:
                log.debug("sender_cache.hit", sender_id=sender_id)
                return (cached_name, sender_id)

        # 3. API call fallback
        try:
            fetched = await event.get_sender()
            if fetched:
                name = getattr(fetched, "username", "") or getattr(fetched, "first_name", "")
                name = name or "unknown"
                self._sender_cache[sender_id] = (name, now)
                log.debug("sender_cache.miss", sender_id=sender_id)
                return (name, sender_id)
        except Exception:
            log.warning("sender_cache.fetch_error", sender_id=sender_id, exc_info=True)

        return ("unknown", sender_id)

    def _maybe_cleanup_sender_cache(self) -> None:
        """Evict stale entries from _sender_cache every N messages."""
        self._msg_count_since_cleanup += 1
        if self._msg_count_since_cleanup < _SENDER_CACHE_CLEANUP_INTERVAL:
            return
        self._msg_count_since_cleanup = 0
        now = time.monotonic()
        stale_keys = [
            sid for sid, (_, cached_at) in self._sender_cache.items()
            if now - cached_at >= _SENDER_CACHE_TTL
        ]
        for sid in stale_keys:
            del self._sender_cache[sid]
        if stale_keys:
            log.info("sender_cache.cleanup", evicted=len(stale_keys), remaining=len(self._sender_cache))

    async def _handle_message(self, event: events.NewMessage.Event) -> None:
        """Serialize a new message and push to Redis."""
        try:
            chat_id = event.chat_id
            group_info = self._groups.get(chat_id, {})

            # Per-group rate limiting
            if self._is_rate_limited(chat_id):
                log.debug(
                    "listener.rate_limited",
                    group=group_info.get("name"),
                    group_id=chat_id,
                )
                return

            sender_name, sender_id = await self._get_sender_name(event)
            self._maybe_cleanup_sender_cache()

            message_data = {
                "source": "telegram",
                "group_id": chat_id,
                "group_name": group_info.get("name", "Unknown"),
                "message_id": event.id,
                "sender_id": sender_id,
                "sender_name": sender_name,
                "text": event.raw_text or "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "has_media": event.media is not None,
                "reply_to": event.reply_to_msg_id,
            }

            await self._redis.rpush(
                REDIS_KEY_RAW, json.dumps(message_data)
            )

            log.debug(
                "listener.message_received",
                group=group_info.get("name"),
                message_id=event.id,
                text_preview=message_data["text"][:80],
            )
        except Exception:
            log.error("listener.message_handler_error", exc_info=True)

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        self._shutdown_event.set()
        await self._session.stop()
        log.info("listener.stopped")
