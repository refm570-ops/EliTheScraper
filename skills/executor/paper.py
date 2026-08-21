"""PaperExecutor — simulates fills, never touches a wallet or the chain.

Runs the ENTIRE trade loop (safety → evaluator → risk → "execution" → monitor →
exit) end-to-end and accrues simulated PnL, so the system can be validated for
as long as the owner wants before a single real lamport is risked. No keypair is
loaded in paper mode.

Prices are quoted in SOL per token, derived from DexScreener USD price and a
reference SOL/USD. A simulated slippage cost is applied so paper PnL is not
rosier than live.
"""

from __future__ import annotations

import structlog

from skills.executor.base import Executor
from skills.token_metadata.fetcher import TokenMetadataFetcher
from trading.models import ExecutionResult, Opportunity, Position, TradeSide

log = structlog.get_logger()


class PaperExecutor(Executor):
    is_paper = True

    def __init__(
        self,
        fetcher: TokenMetadataFetcher,
        sol_usd_reference: float = 150.0,
        simulated_slippage_bps: int = 150,
    ) -> None:
        self._fetcher = fetcher
        self._sol_usd = sol_usd_reference if sol_usd_reference > 0 else 150.0
        self._sim_slippage_bps = simulated_slippage_bps

    async def _price_sol(self, address: str | None, fallback_meta: dict | None = None) -> float | None:
        """Current price in SOL per token."""
        price_usd = None
        if address:
            meta = await self._fetcher.fetch_by_address(address)
            if meta:
                price_usd = meta.get("price_usd")
        if price_usd is None and fallback_meta:
            price_usd = fallback_meta.get("price_usd")
        if price_usd is None or price_usd <= 0:
            return None
        return price_usd / self._sol_usd

    async def buy(
        self, opportunity: Opportunity, sol_amount: float, max_slippage_bps: int
    ) -> ExecutionResult:
        price_sol = await self._price_sol(opportunity.address, opportunity.metadata)
        if price_sol is None:
            return ExecutionResult(
                success=False, side=TradeSide.BUY, address=opportunity.address,
                is_paper=True, error="paper: no price available to simulate fill",
            )
        # Apply simulated adverse slippage to the effective buy price.
        eff_price = price_sol * (1 + self._sim_slippage_bps / 10_000)
        token_amount = sol_amount / eff_price
        log.info("paper.buy", ticker=opportunity.ticker, sol=sol_amount,
                 tokens=token_amount, price_sol=eff_price)
        return ExecutionResult(
            success=True, side=TradeSide.BUY, address=opportunity.address,
            tx_signature=f"PAPER-BUY-{opportunity.ticker}", sol_amount=sol_amount,
            token_amount=token_amount, price=eff_price, is_paper=True,
        )

    async def sell(
        self, position: Position, token_amount: float, max_slippage_bps: int
    ) -> ExecutionResult:
        price_sol = await self._price_sol(position.address)
        if price_sol is None:
            return ExecutionResult(
                success=False, side=TradeSide.SELL, address=position.address,
                is_paper=True, error="paper: no price available to simulate sell",
            )
        eff_price = price_sol * (1 - self._sim_slippage_bps / 10_000)
        sol_received = token_amount * eff_price
        log.info("paper.sell", ticker=position.ticker, tokens=token_amount,
                 sol=sol_received, price_sol=eff_price)
        return ExecutionResult(
            success=True, side=TradeSide.SELL, address=position.address,
            tx_signature=f"PAPER-SELL-{position.ticker}", sol_amount=sol_received,
            token_amount=token_amount, price=eff_price, is_paper=True,
        )

    async def current_price_sol(self, position: Position) -> float | None:
        return await self._price_sol(position.address)
