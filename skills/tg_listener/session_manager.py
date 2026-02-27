from __future__ import annotations

import asyncio

import structlog
from telethon import TelegramClient

from skills.tg_listener.flood_guard import FloodGuard

log = structlog.get_logger()


class SessionManager:
    """Manages Telethon client session with auto-reconnect."""

    def __init__(
        self,
        session_name: str,
        api_id: int,
        api_hash: str,
        flood_guard: FloodGuard | None = None,
        max_reconnect_attempts: int = 10,
        initial_reconnect_delay: float = 5.0,
        max_reconnect_delay: float = 300.0,
    ) -> None:
        self._session_name = session_name
        self._api_id = api_id
        self._api_hash = api_hash
        self._flood_guard = flood_guard or FloodGuard()
        self._max_reconnect_attempts = max_reconnect_attempts
        self._initial_reconnect_delay = initial_reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._client: TelegramClient | None = None

    @property
    def client(self) -> TelegramClient:
        if self._client is None:
            raise RuntimeError("Session not started. Call start() first.")
        return self._client

    async def start(self) -> TelegramClient:
        """Create and connect the Telethon client."""
        self._client = TelegramClient(
            self._session_name, self._api_id, self._api_hash
        )
        await self._client.start()
        log.info("session_manager.connected", session=self._session_name)
        return self._client

    async def stop(self) -> None:
        """Disconnect the client gracefully."""
        if self._client and self._client.is_connected():
            await self._client.disconnect()
            log.info("session_manager.disconnected", session=self._session_name)

    def is_connected(self) -> bool:
        """Health check — is the client connected?"""
        return self._client is not None and self._client.is_connected()

    async def ensure_connected(self) -> TelegramClient:
        """Reconnect with exponential backoff if disconnected."""
        if self.is_connected():
            return self.client

        delay = self._initial_reconnect_delay
        for attempt in range(1, self._max_reconnect_attempts + 1):
            log.warning(
                "session_manager.reconnecting",
                attempt=attempt,
                max_attempts=self._max_reconnect_attempts,
            )
            try:
                await self._flood_guard.jitter()
                if self._client is None:
                    await self.start()
                else:
                    await self._client.connect()
                    if not await self._client.is_user_authorized():
                        raise RuntimeError("Session expired — re-auth needed")
                log.info("session_manager.reconnected", attempt=attempt)
                return self.client
            except Exception:
                log.error(
                    "session_manager.reconnect_failed",
                    attempt=attempt,
                    next_delay=delay,
                    exc_info=True,
                )
                if attempt == self._max_reconnect_attempts:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

        raise RuntimeError("Failed to reconnect after max attempts")
