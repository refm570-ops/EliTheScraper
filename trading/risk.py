"""RiskManager — code-enforced trade authorization.

Runs AFTER the evaluator and BEFORE approval/execution. Every limit here is
enforced in code, never in a prompt: the evaluator's requested size is CLAMPED
down (never trusted upward), and any breached cap blocks the trade. Reads live
book state from TradeStore so limits reflect reality, not intent.

Kill switch: a Redis flag `trading:enabled` (value "0"/"false" = killed). The
daily-loss circuit breaker trips it automatically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from storage.trade_store import TradeStore
from trading.config import HARD_ABSOLUTE_MAX_TRADE_SOL, TradingConfig
from trading.models import Conviction, Opportunity, TradeDecision

log = structlog.get_logger()

KILL_SWITCH_KEY = "trading:enabled"       # manual, sticky ("0" = killed)
HALT_UNTIL_KEY = "trading:halt_until"     # daily breaker, auto-clears at expiry


def _utc_next_midnight(now: float | None = None) -> float:
    t = now if now is not None else time.time()
    return t - (t % 86400) + 86400


@dataclass
class RiskDecision:
    approved: bool
    size_sol: float
    reason: str
    slippage_bps: int = 500


class RiskManager:
    def __init__(
        self,
        config: TradingConfig,
        trade_store: TradeStore,
        redis: object | None = None,
    ) -> None:
        self._cfg = config
        self._store = trade_store
        self._redis = redis
        self._risk = config.risk or {}
        self._sizing = (config.sizing or {}).get("conviction_fractions", {})
        self._max_slippage_bps = int((config.execution or {}).get("max_slippage_bps", 1000))

    async def is_killed(self) -> bool:
        """True if the manual kill switch is engaged OR a daily halt is active.

        FAILS CLOSED: if Redis cannot be read, the kill switch is treated as
        engaged — an unverifiable emergency stop must block, not permit, an
        autonomous live buy.
        """
        if self._redis is None:
            return False
        try:
            manual = await self._redis.get(KILL_SWITCH_KEY)
            halt_until = await self._redis.get(HALT_UNTIL_KEY)
        except Exception:  # noqa: BLE001
            log.warning("risk.kill_switch_unreadable_fail_closed", exc_info=True)
            return True
        if manual is not None and str(manual).strip().lower() in ("0", "false", "off", "no"):
            return True
        if halt_until is not None:
            try:
                if time.time() < float(halt_until):
                    return True
            except (TypeError, ValueError):
                pass
        return False

    async def engage_kill_switch(self, reason: str) -> None:
        """Permanent manual kill (stays until an operator clears it)."""
        log.error("risk.kill_switch_engaged", reason=reason)
        if self._redis is not None:
            try:
                await self._redis.set(KILL_SWITCH_KEY, "0")
            except Exception:  # noqa: BLE001
                log.warning("risk.kill_switch_set_failed", exc_info=True)

    async def halt_for_day(self, reason: str) -> None:
        """Trip the daily circuit breaker; auto-clears at the next UTC midnight."""
        until = _utc_next_midnight()
        log.error("risk.daily_halt_engaged", reason=reason, until=until)
        if self._redis is not None:
            try:
                await self._redis.set(HALT_UNTIL_KEY, str(until))
            except Exception:  # noqa: BLE001
                log.warning("risk.daily_halt_set_failed", exc_info=True)

    async def authorize(
        self, opportunity: Opportunity, decision: TradeDecision
    ) -> RiskDecision:
        if not decision.is_buy:
            return RiskDecision(False, 0.0, "decision is not a buy")

        # 1. Kill switch.
        if await self.is_killed():
            return RiskDecision(False, 0.0, "kill switch engaged")

        # 2. Daily loss circuit breaker.
        daily_limit = float(self._risk.get("daily_loss_limit_sol", 0.5))
        realized_today = await self._store.realized_pnl_today()
        if realized_today <= -abs(daily_limit):
            await self.halt_for_day(
                f"daily loss {realized_today:.3f} SOL <= -{daily_limit} SOL"
            )
            return RiskDecision(False, 0.0, f"daily loss limit hit ({realized_today:.3f} SOL)")

        # 3. Max open positions.
        max_open = int(self._risk.get("max_open_positions", 5))
        if await self._store.count_open() >= max_open:
            return RiskDecision(False, 0.0, f"max open positions ({max_open}) reached")

        # 4. No double-buy of an address already held.
        if opportunity.address and await self._store.has_open_for_address(opportunity.address):
            return RiskDecision(False, 0.0, "position already open for this token")

        # 5. Size clamp: min(requested, conviction ceiling, per-trade cap, hard ceiling).
        size = self._clamp_size(decision)
        if size <= 0:
            return RiskDecision(False, 0.0, "clamped size is zero")

        # 6. Total exposure cap.
        max_total = float(self._risk.get("max_total_exposure_sol", 1.0))
        current_total = await self._store.total_exposure_sol()
        remaining_total = max_total - current_total
        if remaining_total <= 0:
            return RiskDecision(False, 0.0, "total exposure cap reached")
        size = min(size, remaining_total)

        # 7. Per-source exposure cap.
        per_source = float(self._risk.get("max_exposure_per_source_sol", 0.4))
        current_source = await self._store.source_exposure_sol(opportunity.source)
        remaining_source = per_source - current_source
        if remaining_source <= 0:
            return RiskDecision(False, 0.0, f"per-source cap reached for {opportunity.source}")
        size = min(size, remaining_source)

        # Guard against a clamp that pushed size below a sane floor.
        min_floor = min(0.01, float(self._risk.get("default_trade_sol", 0.1)))
        if size < min_floor:
            return RiskDecision(False, 0.0, f"clamped size {size:.4f} below floor {min_floor}")

        # Slippage is a guardrail too: clamp the evaluator's request DOWN to the
        # config buy-slippage cap. The LLM (fed attacker-controlled message text)
        # can never widen tolerance beyond config.
        slippage = min(int(decision.max_slippage_bps), self._max_slippage_bps)
        slippage = max(10, slippage)

        log.info("risk.authorized", ticker=opportunity.ticker, size_sol=round(size, 4),
                 requested=decision.size_sol, slippage_bps=slippage)
        return RiskDecision(True, round(size, 6), "authorized", slippage_bps=slippage)

    def _clamp_size(self, decision: TradeDecision) -> float:
        per_trade_cap = self._cfg.max_trade_sol()  # already <= HARD ceiling
        default = float(self._risk.get("default_trade_sol", 0.1))

        requested = decision.size_sol if decision.size_sol > 0 else default

        # Conviction ceiling as a fraction of the per-trade cap.
        fraction = float(self._sizing.get(decision.conviction.value, 0.25)) \
            if decision.conviction is not Conviction.NONE else 0.0
        conviction_ceiling = per_trade_cap * fraction if fraction > 0 else per_trade_cap

        size = min(requested, conviction_ceiling, per_trade_cap, HARD_ABSOLUTE_MAX_TRADE_SOL)
        return max(0.0, size)
