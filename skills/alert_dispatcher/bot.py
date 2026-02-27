from __future__ import annotations

import asyncio
from typing import Any

import structlog
from telegram import Bot

from skills.alert_dispatcher.rate_limiter import RateLimiter
from skills.alert_dispatcher.templates import format_phase1_alert

log = structlog.get_logger()


class AlertDispatcher:
    """Sends classified signals to a private Telegram bot."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str | int,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._bot = Bot(token=bot_token)
        self._chat_id = chat_id
        self._rate_limiter = rate_limiter or RateLimiter()
        self._send_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._sender_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background sender task."""
        self._sender_task = asyncio.create_task(self._send_loop())
        log.info("dispatcher.started")

    async def stop(self) -> None:
        """Stop the sender and drain the queue."""
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
        # Drain remaining
        while not self._send_queue.empty():
            text, notify_mode = self._send_queue.get_nowait()
            await self._send_message(text, notify_mode)
        log.info("dispatcher.stopped")

    async def enqueue(self, text: str, notify_mode: str = "normal") -> None:
        """Enqueue a single pre-formatted message for dispatch."""
        await self._send_queue.put((text, notify_mode))

    async def dispatch_batch(
        self,
        classifications: list[dict[str, Any]],
        raw_messages: list[dict[str, Any]],
    ) -> None:
        """Queue alerts for a batch of classified messages (Phase 1 path)."""
        # Build a lookup from message id to raw message
        raw_by_id: dict[Any, dict[str, Any]] = {}
        for msg in raw_messages:
            raw_by_id[msg.get("message_id")] = msg

        for cls in classifications:
            ticker = cls.get("ticker")
            if not ticker:
                continue

            if not self._rate_limiter.can_send_ticker(ticker):
                continue
            if not self._rate_limiter.can_send_global():
                log.warning("dispatcher.global_rate_limit_hit")
                break

            raw_msg = raw_by_id.get(cls.get("id"))
            text = format_phase1_alert(cls, raw_msg)
            await self._send_queue.put((text, "normal"))
            self._rate_limiter.record_send(ticker)

    async def _send_loop(self) -> None:
        """Background loop that sends queued messages."""
        while True:
            try:
                text, notify_mode = await self._send_queue.get()
                await self._send_message(text, notify_mode)
                # Small delay between sends to respect Telegram rate limits
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.error("dispatcher.send_loop_error", exc_info=True)

    async def _send_message(
        self, text: str, notify_mode: str = "normal"
    ) -> None:
        """Send a single message via the Telegram Bot API."""
        try:
            disable_notification = notify_mode == "silent"
            msg = await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="HTML",
                disable_notification=disable_notification,
                disable_web_page_preview=True,
            )
            # Pin message for sound_and_pin mode
            if notify_mode == "sound_and_pin" and msg:
                try:
                    await self._bot.pin_chat_message(
                        chat_id=self._chat_id,
                        message_id=msg.message_id,
                        disable_notification=False,
                    )
                except Exception:
                    log.warning("dispatcher.pin_error", exc_info=True)

            log.debug("dispatcher.message_sent", text_preview=text[:80])
        except Exception:
            log.error("dispatcher.send_error", exc_info=True)
