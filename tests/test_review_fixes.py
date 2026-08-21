"""Regression tests for the security fixes from the agent review pass.

Covers: buy-slippage clamp, kill-switch fail-closed on Redis error, portfolio
(realized+unrealized) daily-loss breaker in the monitor, and approval
authorization by user id (deny-by-default).
"""

from __future__ import annotations

import pytest

from storage.db import Database
from storage.trade_store import TradeStore
from trading.config import TradingConfig
from trading.models import (
    Chain,
    Conviction,
    DecisionAction,
    Opportunity,
    Position,
    PositionStatus,
    TokenVenue,
    TradeDecision,
)
from trading.risk import RiskManager
from trading.monitor import PositionMonitor
from skills.approval.gate import ApprovalGate


def _config(**over):
    base = dict(
        risk={"max_trade_sol": 0.25, "default_trade_sol": 0.1, "max_open_positions": 5,
              "max_total_exposure_sol": 1.0, "daily_loss_limit_sol": 0.5,
              "max_exposure_per_source_sol": 0.4},
        sizing={"conviction_fractions": {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.25}},
        execution={"max_slippage_bps": 1000},
    )
    base.update(over)
    return TradingConfig(**base)


async def _store():
    db = Database(":memory:")
    await db.connect()
    return db, TradeStore(db)


@pytest.mark.asyncio
async def test_buy_slippage_clamped_to_config_cap():
    db, store = await _store()
    rm = RiskManager(_config(), store, redis=None)
    opp = Opportunity("T", address="m", chain=Chain.SOLANA, venue=TokenVenue.AMM, source="s")
    # Evaluator asks for 5000 bps (50%); config cap is 1000.
    dec = TradeDecision(DecisionAction.BUY, Conviction.HIGH, 0.1, 5000, "r")
    rd = await rm.authorize(opp, dec)
    assert rd.approved
    assert rd.slippage_bps == 1000, "buy slippage must be clamped to config cap"
    await db.close()


class _BoomRedis:
    async def get(self, *a):
        raise RuntimeError("redis down")
    async def set(self, *a):
        raise RuntimeError("redis down")


@pytest.mark.asyncio
async def test_kill_switch_fails_closed_on_redis_error():
    db, store = await _store()
    rm = RiskManager(_config(), store, redis=_BoomRedis())
    assert await rm.is_killed() is True, "unreadable kill switch must block (fail closed)"
    await db.close()


class _FakeExecutor:
    is_paper = True

    def __init__(self, price):
        self._price = price

    async def current_price_sol(self, position):
        return self._price

    async def sell(self, position, token_amount, slippage):  # pragma: no cover
        raise AssertionError("should not sell in this scenario")


class _CapturingRisk:
    def __init__(self):
        self.halted = None

    async def halt_for_day(self, reason):
        self.halted = reason


@pytest.mark.asyncio
async def test_monitor_trips_daily_breaker_on_unrealized_drawdown():
    db, store = await _store()
    # entry 1.0, price 0.9 (−10%, above −40% stop so no sell), size 10 tokens →
    # unrealized −1.0 SOL, well past the 0.5 daily limit.
    pos = Position(address="m", ticker="T", chain=Chain.SOLANA, venue=TokenVenue.AMM,
                   source="s", entry_price=1.0, amount_sol=10.0, token_amount=10.0,
                   initial_token_amount=10.0, status=PositionStatus.OPEN, is_paper=True,
                   peak_price=1.0)
    await store.open_position(pos)
    risk = _CapturingRisk()
    mon = PositionMonitor(store, _FakeExecutor(0.9),
                          exit_config={"enabled": True, "stop_loss_pct": -40},
                          execution_config={"max_slippage_bps": 1000},
                          risk_manager=risk, daily_loss_limit_sol=0.5)
    await mon.tick()
    assert risk.halted is not None, "portfolio unrealized drawdown must trip the daily halt"
    await db.close()


class _FakeUser:
    def __init__(self, uid): self.id = uid


class _FakeQuery:
    def __init__(self, uid): self.from_user = _FakeUser(uid); self.message = None


def test_approval_authorizes_by_user_id_deny_by_default():
    gate = ApprovalGate("token", owner_chat_id=42, owner_user_id=123)
    assert gate._is_authorized(123, _FakeQuery(123)) is True
    assert gate._is_authorized(999, _FakeQuery(999)) is False, "non-owner must be denied"
    assert gate._is_authorized(None, _FakeQuery(None)) is False, "missing user must be denied"
