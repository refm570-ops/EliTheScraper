from __future__ import annotations

import time

import structlog

from storage.db import Database

log = structlog.get_logger()


class AlertLog:
    """Write-only audit log for dispatched alerts, plus restart survivability."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(
        self,
        ticker: str,
        alert_level: str,
        notify_mode: str,
        score: float | None = None,
        summary: str | None = None,
        metadata_available: bool = True,
    ) -> None:
        """Log a dispatched alert."""
        await self._db.conn.execute(
            """INSERT INTO alert_log
               (ticker, alert_level, notify_mode, score, summary,
                metadata_available, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker.upper(),
                alert_level,
                notify_mode,
                score,
                summary,
                1 if metadata_available else 0,
                time.time(),
            ),
        )
        await self._db.conn.commit()

    async def was_recently_alerted(
        self, ticker: str, cooldown_seconds: float = 1800.0
    ) -> bool:
        """Check if an alert was sent for this ticker within cooldown window.

        Survives restarts (unlike in-memory rate limiter).
        """
        cutoff = time.time() - cooldown_seconds
        cursor = await self._db.conn.execute(
            """SELECT 1 FROM alert_log
               WHERE ticker = ? AND created_at >= ?
               LIMIT 1""",
            (ticker.upper(), cutoff),
        )
        row = await cursor.fetchone()
        return row is not None
