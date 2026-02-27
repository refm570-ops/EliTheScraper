from __future__ import annotations

import asyncio
import signal
import time

import structlog

from agents.classifier.agent import MessageClassifier
from pipeline.buffer import MessageBuffer
from skills.alert_dispatcher.bot import AlertDispatcher

log = structlog.get_logger()


class Orchestrator:
    """Main pipeline loop: buffer → classifier → dispatcher.

    Pulls batches from Redis every 5 minutes (or when 50 messages accumulate),
    classifies them, and dispatches alerts.
    """

    def __init__(
        self,
        buffer: MessageBuffer,
        classifier: MessageClassifier,
        dispatcher: AlertDispatcher,
        batch_size: int = 50,
        batch_interval: float = 300.0,  # 5 minutes
        heartbeat_interval: float = 60.0,
    ) -> None:
        self._buffer = buffer
        self._classifier = classifier
        self._dispatcher = dispatcher
        self._batch_size = batch_size
        self._batch_interval = batch_interval
        self._heartbeat_interval = heartbeat_interval
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._cycle_count = 0

    async def run(self) -> None:
        """Main loop — runs until shutdown signal."""
        self._running = True

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_shutdown)

        log.info(
            "orchestrator.started",
            batch_size=self._batch_size,
            interval=self._batch_interval,
        )

        # Start heartbeat
        heartbeat_task = asyncio.create_task(self._heartbeat())

        try:
            while self._running:
                await self._process_cycle()
        except asyncio.CancelledError:
            pass
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            # Drain remaining messages before exit
            await self._drain()
            log.info("orchestrator.stopped", total_cycles=self._cycle_count)

    def _request_shutdown(self) -> None:
        log.info("orchestrator.shutdown_requested")
        self._running = False
        self._shutdown_event.set()

    async def _process_cycle(self) -> None:
        """Single processing cycle: pop batch → classify → dispatch."""
        self._cycle_count += 1
        cycle_start = time.monotonic()

        try:
            batch = await self._buffer.pop_batch(
                max_size=self._batch_size,
                max_wait_seconds=self._batch_interval,
            )

            if not batch:
                log.debug("orchestrator.empty_batch", cycle=self._cycle_count)
                return

            log.info(
                "orchestrator.processing_batch",
                cycle=self._cycle_count,
                batch_size=len(batch),
            )

            classified = await self._classifier.classify_batch(batch)

            # Filter out NOISE — only dispatch actionable signals
            actionable = [
                msg for msg in classified if msg.get("intent") != "NOISE"
            ]

            if actionable:
                await self._dispatcher.dispatch_batch(actionable, batch)
                log.info(
                    "orchestrator.batch_dispatched",
                    cycle=self._cycle_count,
                    total=len(batch),
                    actionable=len(actionable),
                    noise=len(batch) - len(actionable),
                    duration=round(time.monotonic() - cycle_start, 2),
                )
            else:
                log.info(
                    "orchestrator.all_noise",
                    cycle=self._cycle_count,
                    batch_size=len(batch),
                )
        except Exception:
            log.error(
                "orchestrator.cycle_error",
                cycle=self._cycle_count,
                exc_info=True,
            )

    async def _drain(self) -> None:
        """Drain any remaining messages on shutdown."""
        log.info("orchestrator.draining")
        remaining = await self._buffer.length()
        if remaining > 0:
            batch = await self._buffer.pop_batch(
                max_size=remaining, max_wait_seconds=1.0
            )
            if batch:
                try:
                    classified = await self._classifier.classify_batch(batch)
                    actionable = [
                        m for m in classified if m.get("intent") != "NOISE"
                    ]
                    if actionable:
                        await self._dispatcher.dispatch_batch(actionable, batch)
                except Exception:
                    log.error("orchestrator.drain_error", exc_info=True)

    async def _heartbeat(self) -> None:
        """Periodic health log."""
        while True:
            buffer_len = await self._buffer.length()
            log.info(
                "orchestrator.heartbeat",
                cycles_completed=self._cycle_count,
                buffer_size=buffer_len,
                running=self._running,
            )
            await asyncio.sleep(self._heartbeat_interval)
