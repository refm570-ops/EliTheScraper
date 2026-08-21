"""Tests for skills/executor/paper.py — simulated fills.

Uses a fake fetcher (duck-typed async fetch_by_address) instead of the real
TokenMetadataFetcher, so no DB/HTTP client is involved.
"""

from __future__ import annotations

import pytest

from skills.executor.paper import PaperExecutor
from trading.models import Chain, Opportunity, Position, PositionStatus, TokenVenue, TradeSide


class FakeFetcher:
    def __init__(self, prices_usd: dict[str, float | None]):
        self._prices = prices_usd

    async def fetch_by_address(self, address: str):
        if address not in self._prices:
            return None
        price = self._prices[address]
        if price is None:
            return {}
        return {"price_usd": price}


def make_opportunity(address: str = "MINT1") -> Opportunity:
    return Opportunity(ticker="$FOO", address=address, chain=Chain.SOLANA)


def make_position(address: str = "MINT1", entry_price: float = 0.001) -> Position:
    return Position(
        address=address,
        ticker="$FOO",
        chain=Chain.SOLANA,
        venue=TokenVenue.AMM,
        source="telegram",
        entry_price=entry_price,
        amount_sol=0.1,
        token_amount=100.0,
        initial_token_amount=100.0,
        status=PositionStatus.OPEN,
        is_paper=True,
    )


@pytest.mark.asyncio
async def test_buy_returns_tokens_and_price() -> None:
    fetcher = FakeFetcher({"MINT1": 15.0})  # $15 -> 0.1 SOL/token at $150 ref
    executor = PaperExecutor(fetcher, sol_usd_reference=150.0, simulated_slippage_bps=0)
    result = await executor.buy(make_opportunity("MINT1"), sol_amount=1.0, max_slippage_bps=1000)
    assert result.success is True
    assert result.side is TradeSide.BUY
    assert result.token_amount is not None and result.token_amount > 0
    assert result.price is not None and result.price > 0
    assert result.is_paper is True
    assert result.sol_amount == pytest.approx(1.0)
    # price_usd 15 / sol_usd 150 = 0.1 SOL/token, no slippage.
    assert result.price == pytest.approx(0.1)
    assert result.token_amount == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_buy_applies_adverse_slippage() -> None:
    fetcher = FakeFetcher({"MINT1": 15.0})
    executor = PaperExecutor(fetcher, sol_usd_reference=150.0, simulated_slippage_bps=1000)
    result = await executor.buy(make_opportunity("MINT1"), sol_amount=1.0, max_slippage_bps=1000)
    assert result.success is True
    # eff_price = 0.1 * 1.10 = 0.11
    assert result.price == pytest.approx(0.11)
    assert result.token_amount == pytest.approx(1.0 / 0.11)


@pytest.mark.asyncio
async def test_buy_no_price_returns_failure() -> None:
    fetcher = FakeFetcher({"MINT1": None})
    executor = PaperExecutor(fetcher)
    result = await executor.buy(make_opportunity("MINT1"), sol_amount=1.0, max_slippage_bps=1000)
    assert result.success is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_buy_unknown_address_no_price_returns_failure() -> None:
    fetcher = FakeFetcher({})
    executor = PaperExecutor(fetcher)
    result = await executor.buy(make_opportunity("UNKNOWN"), sol_amount=1.0, max_slippage_bps=1000)
    assert result.success is False


@pytest.mark.asyncio
async def test_sell_returns_sol() -> None:
    fetcher = FakeFetcher({"MINT1": 30.0})  # 0.2 SOL/token
    executor = PaperExecutor(fetcher, sol_usd_reference=150.0, simulated_slippage_bps=0)
    pos = make_position("MINT1")
    result = await executor.sell(pos, token_amount=50.0, max_slippage_bps=1000)
    assert result.success is True
    assert result.side is TradeSide.SELL
    assert result.sol_amount == pytest.approx(50.0 * 0.2)
    assert result.token_amount == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_sell_no_price_returns_failure() -> None:
    fetcher = FakeFetcher({"MINT1": None})
    executor = PaperExecutor(fetcher)
    pos = make_position("MINT1")
    result = await executor.sell(pos, token_amount=50.0, max_slippage_bps=1000)
    assert result.success is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_current_price_sol_returns_price() -> None:
    fetcher = FakeFetcher({"MINT1": 45.0})  # 0.3 SOL/token
    executor = PaperExecutor(fetcher, sol_usd_reference=150.0)
    pos = make_position("MINT1")
    price = await executor.current_price_sol(pos)
    assert price == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_current_price_sol_none_when_unavailable() -> None:
    fetcher = FakeFetcher({"MINT1": None})
    executor = PaperExecutor(fetcher)
    pos = make_position("MINT1")
    price = await executor.current_price_sol(pos)
    assert price is None
