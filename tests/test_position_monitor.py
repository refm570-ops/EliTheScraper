"""Tests for trading/monitor.py — exit rule evaluation and execution."""

from __future__ import annotations

import pytest

from storage.db import Database
from storage.trade_store import TradeStore
from trading.models import (
    Chain,
    ExecutionResult,
    Position,
    PositionStatus,
    TokenVenue,
    TradeSide,
)
from trading.monitor import PositionMonitor

EXIT_CONFIG = {
    "enabled": True,
    "take_profit_ladder": [
        {"gain_pct": 100, "portion": 0.5},
        {"gain_pct": 300, "portion": 0.25},
    ],
    "stop_loss_pct": -40,
    "trailing_stop_pct": -25,
    "max_hold_hours": 24,
}

EXECUTION_CONFIG = {"max_slippage_bps": 1000}


@pytest.fixture
async def db():
    database = Database(db_path=":memory:")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def store(db):
    return TradeStore(db=db)


class FakeExecutor:
    """Duck-typed Executor: returns a fixed price and simulates sells at it."""

    is_paper = True

    def __init__(self, price: float):
        self.price = price
        self.sell_calls: list[tuple[str, float]] = []

    async def current_price_sol(self, position: Position) -> float | None:
        return self.price

    async def sell(self, position: Position, token_amount: float, max_slippage_bps: int):
        self.sell_calls.append((position.id, token_amount))
        return ExecutionResult(
            success=True,
            side=TradeSide.SELL,
            address=position.address,
            tx_signature="FAKE-SELL",
            sol_amount=token_amount * self.price,
            token_amount=token_amount,
            price=self.price,
            is_paper=True,
        )

    async def buy(self, opportunity, sol_amount, max_slippage_bps):  # pragma: no cover
        raise NotImplementedError

    async def close(self):
        pass


def make_open_position(entry_price: float = 0.001, tokens: float = 1000.0) -> Position:
    return Position(
        address="MINT1",
        ticker="$FOO",
        chain=Chain.SOLANA,
        venue=TokenVenue.AMM,
        source="telegram",
        entry_price=entry_price,
        amount_sol=entry_price * tokens,
        token_amount=tokens,
        initial_token_amount=tokens,
        status=PositionStatus.OPEN,
        is_paper=True,
    )


@pytest.mark.asyncio
async def test_price_up_past_tp_rung_triggers_partial_sell(store: TradeStore) -> None:
    pos = make_open_position(entry_price=0.001, tokens=1000.0)
    await store.open_position(pos)

    # +150% gain clears the first rung (gain_pct=100, portion=0.5).
    price = 0.001 * 2.5
    executor = FakeExecutor(price=price)
    monitor = PositionMonitor(store, executor, EXIT_CONFIG, EXECUTION_CONFIG)

    await monitor.tick()

    [updated] = await store.open_positions()
    assert updated.take_profit_hits == 1
    assert updated.token_amount == pytest.approx(500.0)
    assert updated.status == PositionStatus.OPEN
    assert updated.realized_pnl_sol > 0
    assert executor.sell_calls == [(pos.id, 500.0)]


@pytest.mark.asyncio
async def test_price_crash_below_stop_loss_closes_position(store: TradeStore) -> None:
    pos = make_open_position(entry_price=0.001, tokens=1000.0)
    await store.open_position(pos)

    # -50% gain breaches the -40% stop-loss.
    price = 0.001 * 0.5
    executor = FakeExecutor(price=price)
    monitor = PositionMonitor(store, executor, EXIT_CONFIG, EXECUTION_CONFIG)

    await monitor.tick()

    assert await store.open_positions() == []
    cursor = await store._db.conn.execute("SELECT * FROM positions WHERE id=?", (pos.id,))
    row = await cursor.fetchone()
    assert row["status"] == PositionStatus.CLOSED.value
    assert row["token_amount"] == pytest.approx(0.0)
    assert row["closed_at"] is not None
    assert executor.sell_calls == [(pos.id, 1000.0)]


@pytest.mark.asyncio
async def test_no_price_available_skips_position_untouched(store: TradeStore) -> None:
    pos = make_open_position()
    await store.open_position(pos)

    class NoPriceExecutor(FakeExecutor):
        async def current_price_sol(self, position):
            return None

    executor = NoPriceExecutor(price=0.0)
    monitor = PositionMonitor(store, executor, EXIT_CONFIG, EXECUTION_CONFIG)
    await monitor.tick()

    [updated] = await store.open_positions()
    assert updated.status == PositionStatus.OPEN
    assert updated.token_amount == pytest.approx(pos.token_amount)
    assert executor.sell_calls == []


@pytest.mark.asyncio
async def test_tick_noop_when_exit_disabled(store: TradeStore) -> None:
    pos = make_open_position()
    await store.open_position(pos)

    executor = FakeExecutor(price=0.0005)  # would otherwise trigger stop-loss
    monitor = PositionMonitor(store, executor, dict(EXIT_CONFIG, enabled=False), EXECUTION_CONFIG)
    await monitor.tick()

    [updated] = await store.open_positions()
    assert updated.status == PositionStatus.OPEN
    assert executor.sell_calls == []
