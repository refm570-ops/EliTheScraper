"""Persistence for trading positions and the append-only trade audit log."""

from __future__ import annotations

import time
from typing import Any

import structlog

from storage.db import Database
from trading.models import (
    Chain,
    ExecutionResult,
    Position,
    PositionStatus,
    TokenVenue,
)

log = structlog.get_logger()


def _utc_midnight(now: float | None = None) -> float:
    t = now if now is not None else time.time()
    return t - (t % 86400)  # 86400s day; epoch is UTC-aligned


class TradeStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ---- writes ------------------------------------------------------------
    async def open_position(self, position: Position) -> None:
        await self._db.conn.execute(
            """INSERT INTO positions
               (id, address, ticker, chain, venue, source, entry_price, amount_sol,
                token_amount, initial_token_amount, status, is_paper, entry_tx,
                peak_price, realized_pnl_sol, take_profit_hits, opened_at, closed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                position.id, position.address, position.ticker, position.chain.value,
                position.venue.value, position.source, position.entry_price,
                position.amount_sol, position.token_amount, position.initial_token_amount,
                position.status.value, 1 if position.is_paper else 0, position.entry_tx,
                position.peak_price, position.realized_pnl_sol, position.take_profit_hits,
                position.opened_at, position.closed_at,
            ),
        )
        await self._db.conn.commit()
        log.info("trade_store.position_opened", id=position.id, ticker=position.ticker,
                 paper=position.is_paper, amount_sol=position.amount_sol)

    async def update_position(self, position: Position) -> None:
        await self._db.conn.execute(
            """UPDATE positions SET token_amount=?, status=?, peak_price=?,
               realized_pnl_sol=?, take_profit_hits=?, closed_at=? WHERE id=?""",
            (
                position.token_amount, position.status.value, position.peak_price,
                position.realized_pnl_sol, position.take_profit_hits,
                position.closed_at, position.id,
            ),
        )
        await self._db.conn.commit()

    async def record_trade(
        self, result: ExecutionResult, ticker: str, position_id: str | None
    ) -> None:
        await self._db.conn.execute(
            """INSERT INTO trades
               (position_id, address, ticker, side, sol_amount, token_amount, price,
                tx_signature, status, is_paper, error, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                position_id, result.address, ticker, result.side.value,
                result.sol_amount, result.token_amount, result.price,
                result.tx_signature, "success" if result.success else "failed",
                1 if result.is_paper else 0, result.error, time.time(),
            ),
        )
        await self._db.conn.commit()

    # ---- reads -------------------------------------------------------------
    async def open_positions(self) -> list[Position]:
        cursor = await self._db.conn.execute(
            "SELECT * FROM positions WHERE status IN ('OPEN','CLOSING')"
        )
        rows = await cursor.fetchall()
        return [self._row_to_position(r) for r in rows]

    async def count_open(self) -> int:
        cursor = await self._db.conn.execute(
            "SELECT COUNT(*) AS n FROM positions WHERE status IN ('OPEN','CLOSING')"
        )
        row = await cursor.fetchone()
        return int(row["n"]) if row else 0

    async def total_exposure_sol(self) -> float:
        cursor = await self._db.conn.execute(
            "SELECT COALESCE(SUM(amount_sol),0) AS s FROM positions "
            "WHERE status IN ('OPEN','CLOSING')"
        )
        row = await cursor.fetchone()
        return float(row["s"]) if row else 0.0

    async def source_exposure_sol(self, source: str) -> float:
        cursor = await self._db.conn.execute(
            "SELECT COALESCE(SUM(amount_sol),0) AS s FROM positions "
            "WHERE status IN ('OPEN','CLOSING') AND source = ?",
            (source,),
        )
        row = await cursor.fetchone()
        return float(row["s"]) if row else 0.0

    async def has_open_for_address(self, address: str) -> bool:
        cursor = await self._db.conn.execute(
            "SELECT 1 FROM positions WHERE address = ? AND status IN ('OPEN','CLOSING') LIMIT 1",
            (address,),
        )
        return (await cursor.fetchone()) is not None

    async def realized_pnl_today(self, now: float | None = None) -> float:
        """Sum realized PnL for positions active today (UTC).

        Circuit-breaker input. Captures same-day round trips (the common case
        for memecoins) via opened_at>=midnight, plus positions closed today.
        """
        midnight = _utc_midnight(now)
        cursor = await self._db.conn.execute(
            "SELECT COALESCE(SUM(realized_pnl_sol),0) AS s FROM positions "
            "WHERE opened_at >= ? OR (closed_at IS NOT NULL AND closed_at >= ?)",
            (midnight, midnight),
        )
        row = await cursor.fetchone()
        return float(row["s"]) if row else 0.0

    # ---- mapping -----------------------------------------------------------
    @staticmethod
    def _row_to_position(row: Any) -> Position:
        return Position(
            id=row["id"], address=row["address"], ticker=row["ticker"],
            chain=Chain(row["chain"]), venue=TokenVenue(row["venue"]),
            source=row["source"], entry_price=row["entry_price"],
            amount_sol=row["amount_sol"], token_amount=row["token_amount"],
            initial_token_amount=row["initial_token_amount"],
            status=PositionStatus(row["status"]), is_paper=bool(row["is_paper"]),
            entry_tx=row["entry_tx"], peak_price=row["peak_price"],
            realized_pnl_sol=row["realized_pnl_sol"],
            take_profit_hits=row["take_profit_hits"],
            opened_at=row["opened_at"], closed_at=row["closed_at"],
        )
