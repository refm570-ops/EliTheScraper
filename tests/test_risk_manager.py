"""Tests for trading/risk.py — code-enforced trade authorization."""

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


@pytest.fixture
async def db():
    database = Database(db_path=":memory:")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def store(db):
    return TradeStore(db=db)


def make_config(**risk_overrides) -> TradingConfig:
    risk = {
        "max_trade_sol": 0.25,
        "default_trade_sol": 0.1,
        "max_open_positions": 5,
        "max_total_exposure_sol": 1.0,
        "daily_loss_limit_sol": 0.5,
        "max_exposure_per_source_sol": 0.4,
    }
    risk.update(risk_overrides)
    return TradingConfig(
        mode="paper",
        risk=risk,
        sizing={"conviction_fractions": {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.25}},
    )


def make_opportunity(address: str = "MINT111", source: str = "telegram") -> Opportunity:
    return Opportunity(ticker="$FOO", address=address, source=source)


def make_decision(
    size_sol: float = 0.2,
    conviction: Conviction = Conviction.HIGH,
    action: DecisionAction = DecisionAction.BUY,
) -> TradeDecision:
    return TradeDecision(
        action=action,
        conviction=conviction,
        size_sol=size_sol,
        max_slippage_bps=1000,
        reasoning="test",
    )


def make_position(
    address: str = "OPEN_MINT",
    source: str = "telegram",
    amount_sol: float = 0.3,
    status: PositionStatus = PositionStatus.OPEN,
) -> Position:
    return Position(
        address=address,
        ticker="$BAR",
        chain=Chain.SOLANA,
        venue=TokenVenue.AMM,
        source=source,
        entry_price=0.001,
        amount_sol=amount_sol,
        token_amount=amount_sol / 0.001,
        initial_token_amount=amount_sol / 0.001,
        status=status,
    )


# ---------------------------------------------------------------------------
# Size clamping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_size_clamped_down_to_per_trade_cap(store: TradeStore) -> None:
    cfg = make_config(max_trade_sol=0.25)
    rm = RiskManager(cfg, store)
    # Evaluator over-requests far beyond the cap.
    decision = make_decision(size_sol=50.0, conviction=Conviction.HIGH)
    result = await rm.authorize(make_opportunity(), decision)
    assert result.approved is True
    assert result.size_sol <= 0.25


@pytest.mark.asyncio
async def test_size_clamped_by_conviction_fraction(store: TradeStore) -> None:
    cfg = make_config(max_trade_sol=0.25)
    rm = RiskManager(cfg, store)
    # MEDIUM conviction -> 0.5 fraction of the 0.25 cap = 0.125 ceiling.
    decision = make_decision(size_sol=50.0, conviction=Conviction.MEDIUM)
    result = await rm.authorize(make_opportunity(), decision)
    assert result.approved is True
    assert result.size_sol == pytest.approx(0.125, abs=1e-6)


@pytest.mark.asyncio
async def test_size_never_exceeds_hard_absolute_ceiling(store: TradeStore) -> None:
    # Even a maliciously large configured cap can't push size past the
    # hard-coded absolute ceiling (enforced via config.max_trade_sol()).
    cfg = make_config(max_trade_sol=999.0)
    rm = RiskManager(cfg, store)
    decision = make_decision(size_sol=999.0, conviction=Conviction.HIGH)
    result = await rm.authorize(make_opportunity(), decision)
    assert result.approved is True
    assert result.size_sol <= 1.0  # HARD_ABSOLUTE_MAX_TRADE_SOL


# ---------------------------------------------------------------------------
# Position count / double-buy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blocks_when_max_open_positions_reached(store: TradeStore) -> None:
    cfg = make_config(max_open_positions=1)
    rm = RiskManager(cfg, store)
    await store.open_position(make_position(address="ALREADY_OPEN"))

    result = await rm.authorize(make_opportunity(address="NEW_MINT"), make_decision())
    assert result.approved is False
    assert "max open positions" in result.reason


@pytest.mark.asyncio
async def test_blocks_double_buy_of_open_address(store: TradeStore) -> None:
    cfg = make_config(max_open_positions=5)
    rm = RiskManager(cfg, store)
    await store.open_position(make_position(address="SAME_MINT"))

    result = await rm.authorize(make_opportunity(address="SAME_MINT"), make_decision())
    assert result.approved is False
    assert "already open" in result.reason


@pytest.mark.asyncio
async def test_allows_buy_when_no_conflicting_position(store: TradeStore) -> None:
    cfg = make_config()
    rm = RiskManager(cfg, store)
    result = await rm.authorize(make_opportunity(address="FRESH_MINT"), make_decision())
    assert result.approved is True


# ---------------------------------------------------------------------------
# Daily loss circuit breaker / kill switch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_daily_loss_breach_blocks_and_engages_kill_switch(store: TradeStore) -> None:
    cfg = make_config(daily_loss_limit_sol=0.5)
    rm = RiskManager(cfg, store, redis=None)

    # Seed a closed position with a big realized loss today.
    losing = make_position(address="LOSER", amount_sol=0.5, status=PositionStatus.CLOSED)
    await store.open_position(losing)
    losing.realized_pnl_sol = -0.6
    losing.token_amount = 0.0
    await store.update_position(losing)

    assert await store.realized_pnl_today() <= -0.5

    result = await rm.authorize(make_opportunity(address="ANOTHER_MINT"), make_decision())
    assert result.approved is False
    assert "daily loss" in result.reason

    # Kill switch call must not raise even with redis=None; is_killed stays
    # False (no redis to persist it) but the breach itself still blocks.
    assert await rm.is_killed() is False


@pytest.mark.asyncio
async def test_is_killed_true_when_redis_reports_disabled(store: TradeStore) -> None:
    class FakeRedis:
        async def get(self, key):
            return "0"

        async def set(self, key, value):
            self.last_set = (key, value)

    cfg = make_config()
    rm = RiskManager(cfg, store, redis=FakeRedis())
    assert await rm.is_killed() is True

    result = await rm.authorize(make_opportunity(), make_decision())
    assert result.approved is False
    assert "kill switch" in result.reason


# ---------------------------------------------------------------------------
# Exposure caps
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_total_exposure_cap_blocks_new_buy(store: TradeStore) -> None:
    cfg = make_config(max_total_exposure_sol=0.3, max_exposure_per_source_sol=1.0)
    rm = RiskManager(cfg, store)
    await store.open_position(make_position(address="EXPOSED", amount_sol=0.3))

    result = await rm.authorize(make_opportunity(address="NEW_MINT"), make_decision())
    assert result.approved is False
    assert "total exposure" in result.reason


@pytest.mark.asyncio
async def test_total_exposure_cap_clamps_remaining_size(store: TradeStore) -> None:
    cfg = make_config(max_total_exposure_sol=0.35, max_exposure_per_source_sol=1.0,
                       max_trade_sol=0.25)
    rm = RiskManager(cfg, store)
    await store.open_position(make_position(address="EXPOSED", amount_sol=0.2, source="other"))

    # Remaining total room is 0.15, below the 0.25 per-trade cap -> clamp to 0.15.
    result = await rm.authorize(
        make_opportunity(address="NEW_MINT", source="telegram"),
        make_decision(size_sol=0.25, conviction=Conviction.HIGH),
    )
    assert result.approved is True
    assert result.size_sol == pytest.approx(0.15, abs=1e-6)


@pytest.mark.asyncio
async def test_per_source_exposure_cap_blocks_new_buy(store: TradeStore) -> None:
    cfg = make_config(max_exposure_per_source_sol=0.2, max_total_exposure_sol=1.0)
    rm = RiskManager(cfg, store)
    await store.open_position(make_position(address="SRC_A", amount_sol=0.2, source="telegram"))

    result = await rm.authorize(
        make_opportunity(address="NEW_MINT", source="telegram"), make_decision()
    )
    assert result.approved is False
    assert "per-source cap" in result.reason


@pytest.mark.asyncio
async def test_per_source_exposure_cap_ignores_other_sources(store: TradeStore) -> None:
    cfg = make_config(max_exposure_per_source_sol=0.2, max_total_exposure_sol=1.0)
    rm = RiskManager(cfg, store)
    await store.open_position(make_position(address="SRC_A", amount_sol=0.2, source="twitter"))

    # Different source, so telegram's per-source budget is untouched.
    result = await rm.authorize(
        make_opportunity(address="NEW_MINT", source="telegram"), make_decision()
    )
    assert result.approved is True


@pytest.mark.asyncio
async def test_non_buy_decision_is_rejected(store: TradeStore) -> None:
    cfg = make_config()
    rm = RiskManager(cfg, store)
    decision = make_decision(action=DecisionAction.SKIP)
    result = await rm.authorize(make_opportunity(), decision)
    assert result.approved is False
    assert result.size_sol == 0.0
