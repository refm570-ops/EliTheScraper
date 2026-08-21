"""PositionMonitor — a buy agent must sell. Evaluates exit rules on every open
position and executes sells. Runs as a PeriodicTask.

Exit policy (config/trading.yml `exit`), enforced in code, not by an LLM:
  - take-profit ladder (sell portions at gain thresholds)
  - hard stop-loss (dump all at -X%)
  - trailing stop (dump all if price falls Y% from peak, once in profit)
  - time-based (force-close after N hours)
  - emergency rug-exit (liquidity collapse -> dump)
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from skills.executor.base import Executor
from skills.token_metadata.fetcher import TokenMetadataFetcher
from storage.trade_store import TradeStore
from trading.models import ExecutionResult, Position, PositionStatus, TradeSide

log = structlog.get_logger()


class PositionMonitor:
    def __init__(
        self,
        trade_store: TradeStore,
        executor: Executor,
        exit_config: dict[str, Any],
        execution_config: dict[str, Any],
        fetcher: TokenMetadataFetcher | None = None,
        min_liquidity_usd: float = 5000.0,
    ) -> None:
        self._store = trade_store
        self._executor = executor
        self._exit = exit_config or {}
        self._slippage_bps = int((execution_config or {}).get("max_slippage_bps", 1000))
        self._fetcher = fetcher
        self._min_liq = min_liquidity_usd

    async def tick(self) -> None:
        if not self._exit.get("enabled", True):
            return
        positions = await self._store.open_positions()
        for pos in positions:
            try:
                await self._evaluate(pos)
            except Exception:  # noqa: BLE001
                log.error("monitor.position_error", id=pos.id, ticker=pos.ticker, exc_info=True)

    async def _evaluate(self, pos: Position) -> None:
        price = await self._executor.current_price_sol(pos)
        if price is None or price <= 0:
            log.debug("monitor.no_price", ticker=pos.ticker)
            return

        # Track peak for trailing stop.
        if pos.peak_price is None or price > pos.peak_price:
            pos.peak_price = price
            await self._store.update_position(pos)

        gain = pos.gain_pct(price)

        # 1. Emergency rug-exit: liquidity collapse.
        if await self._liquidity_collapsed(pos):
            await self._sell_all(pos, price, reason="emergency: liquidity collapsed")
            return

        # 2. Hard stop-loss.
        stop = self._exit.get("stop_loss_pct")
        if stop is not None and gain <= float(stop):
            await self._sell_all(pos, price, reason=f"stop-loss {gain:.0f}%")
            return

        # 3. Trailing stop (only once in profit).
        trail = self._exit.get("trailing_stop_pct")
        if trail is not None and pos.peak_price and price > pos.entry_price:
            drop_from_peak = (price - pos.peak_price) / pos.peak_price * 100.0
            if drop_from_peak <= float(trail):
                await self._sell_all(pos, price, reason=f"trailing stop {drop_from_peak:.0f}% off peak")
                return

        # 4. Time-based exit.
        max_hold = self._exit.get("max_hold_hours")
        if max_hold is not None and (time.time() - pos.opened_at) >= float(max_hold) * 3600:
            await self._sell_all(pos, price, reason=f"max hold {max_hold}h")
            return

        # 5. Take-profit ladder.
        ladder = self._exit.get("take_profit_ladder", []) or []
        # Rungs are taken in order; take_profit_hits = number already taken.
        if pos.take_profit_hits < len(ladder):
            rung = ladder[pos.take_profit_hits]
            if gain >= float(rung.get("gain_pct", 1e9)):
                portion = float(rung.get("portion", 0.0))
                tokens = pos.initial_token_amount * portion
                tokens = min(tokens, pos.token_amount)
                if tokens > 0:
                    await self._sell_partial(pos, tokens, price,
                                             reason=f"TP rung {pos.take_profit_hits + 1} at +{gain:.0f}%")

    async def _liquidity_collapsed(self, pos: Position) -> bool:
        if self._fetcher is None:
            return False
        meta = await self._fetcher.fetch_by_address(pos.address)
        if not meta:
            return False
        liq = meta.get("liquidity_usd")
        return liq is not None and liq < self._min_liq * 0.5

    async def _sell_all(self, pos: Position, price: float, reason: str) -> None:
        tokens = pos.token_amount
        if tokens <= 0:
            pos.status = PositionStatus.CLOSED
            pos.closed_at = time.time()
            await self._store.update_position(pos)
            return
        result = await self._executor.sell(pos, tokens, self._slippage_bps)
        await self._store.record_trade(result, pos.ticker, pos.id)
        if not result.success:
            log.warning("monitor.sell_failed", ticker=pos.ticker, reason=reason, error=result.error)
            return
        realized = (result.price - pos.entry_price) * tokens if result.price else 0.0
        pos.realized_pnl_sol += realized
        pos.token_amount = 0.0
        pos.status = PositionStatus.CLOSED
        pos.closed_at = time.time()
        await self._store.update_position(pos)
        log.info("monitor.position_closed", ticker=pos.ticker, reason=reason,
                 realized_pnl_sol=round(pos.realized_pnl_sol, 4))

    async def _sell_partial(self, pos: Position, tokens: float, price: float, reason: str) -> None:
        result = await self._executor.sell(pos, tokens, self._slippage_bps)
        await self._store.record_trade(result, pos.ticker, pos.id)
        if not result.success:
            log.warning("monitor.partial_sell_failed", ticker=pos.ticker, error=result.error)
            return
        realized = (result.price - pos.entry_price) * tokens if result.price else 0.0
        pos.realized_pnl_sol += realized
        pos.token_amount = max(0.0, pos.token_amount - tokens)
        pos.take_profit_hits += 1
        if pos.token_amount <= 0:
            pos.status = PositionStatus.CLOSED
            pos.closed_at = time.time()
        await self._store.update_position(pos)
        log.info("monitor.partial_taken", ticker=pos.ticker, reason=reason,
                 sold_tokens=tokens, realized_pnl_sol=round(pos.realized_pnl_sol, 4))
