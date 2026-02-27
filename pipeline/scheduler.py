from __future__ import annotations

import asyncio
from typing import Callable, Coroutine, Any

import structlog

log = structlog.get_logger()


class PeriodicTask:
    """Simple async periodic task runner."""

    def __init__(
        self,
        name: str,
        func: Callable[[], Coroutine[Any, Any, None]],
        interval_seconds: float,
    ) -> None:
        self.name = name
        self._func = func
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        while True:
            try:
                await self._func()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.error("scheduler.task_error", task=self.name, exc_info=True)
            await asyncio.sleep(self._interval)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        log.info("scheduler.task_started", task=self.name, interval=self._interval)

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("scheduler.task_stopped", task=self.name)
