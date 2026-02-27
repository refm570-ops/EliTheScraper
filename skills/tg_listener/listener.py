from __future__ import annotations

import asyncio
import json
import signal
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
import structlog
import yaml
from telethon import events

from skills.tg_listener.flood_guard import FloodGuard
from skills.tg_listener.session_manager import SessionManager

log = structlog.get_logger()

REDIS_KEY_RAW = "tg:messages:raw"


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

    @staticmethod
    def _load_groups(path: str) -> dict[int, dict[str, Any]]:
        with open(path) as f:
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

        @client.on(events.NewMessage(chats=group_ids))
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

    async def _handle_message(self, event: events.NewMessage.Event) -> None:
        """Serialize a new message and push to Redis."""
        try:
            chat_id = event.chat_id
            group_info = self._groups.get(chat_id, {})

            sender = await event.get_sender()
            sender_name = ""
            sender_id = 0
            if sender:
                sender_name = getattr(sender, "username", "") or getattr(
                    sender, "first_name", ""
                )
                sender_id = sender.id

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
