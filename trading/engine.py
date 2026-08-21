"""TradingEngine — orchestrates a single opportunity end to end.

    opportunity
      -> SAFETY GATE   (hard pass/fail, runs FIRST, before spending an Opus call)
      -> EVALUATOR      (Opus: go/no-go + conviction + size)   [only if gate passes]
      -> RISK MANAGER   (code-enforced caps, clamps size)      [only if go]
      -> APPROVAL/AUTO  (human confirm by default, auto only in paper or gated live)
      -> EXECUTOR       (paper simulate | live buy)
      -> POSITION STORE (tracked for exit by PositionMonitor)

Safety is authoritative: a hard failure stops the pipeline and the evaluator is
never consulted. Sizing from the evaluator is only ever clamped down by risk.
"""

from __future__ import annotations

import structlog

from agents.trader.agent import OpportunityEvaluator
from skills.approval.gate import ApprovalGate
from skills.executor.base import Executor
from skills.safety.gate import SafetyGate
from storage.trade_store import TradeStore
from trading.config import TradingConfig
from trading.models import (
    Opportunity,
    Position,
    PositionStatus,
    TradeProposal,
)
from trading.risk import RiskManager

log = structlog.get_logger()

# Notification callback: async fn(text: str, notify_mode: str) -> None
from typing import Awaitable, Callable

Notifier = Callable[[str, str], Awaitable[None]]


class TradingEngine:
    def __init__(
        self,
        config: TradingConfig,
        safety_gate: SafetyGate,
        evaluator: OpportunityEvaluator,
        risk_manager: RiskManager,
        executor: Executor,
        trade_store: TradeStore,
        approval_gate: ApprovalGate | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self._cfg = config
        self._safety = safety_gate
        self._evaluator = evaluator
        self._risk = risk_manager
        self._executor = executor
        self._store = trade_store
        self._approval = approval_gate
        self._notify = notifier
        if approval_gate is not None:
            approval_gate.set_execute_callback(self._execute)

    async def handle_opportunity(self, opportunity: Opportunity) -> None:
        """Run one opportunity through the full decision + execution pipeline."""
        ticker = opportunity.ticker

        # 1. SAFETY GATE — authoritative, runs first.
        report = await self._safety.check(
            opportunity.address, opportunity.chain.value, opportunity.metadata
        )
        if not report.passed:
            log.info("engine.safety_blocked", ticker=ticker,
                     reasons=report.blocking_reasons())
            return

        # 2. EVALUATOR (Opus) — only on safety survivors.
        decision = await self._evaluator.evaluate(
            opportunity, report, self._cfg.max_trade_sol()
        )
        if not decision.is_buy:
            log.info("engine.evaluator_skip", ticker=ticker, reason=decision.reasoning)
            return

        # 3. RISK — code-enforced caps; clamps size.
        risk = await self._risk.authorize(opportunity, decision)
        if not risk.approved:
            log.info("engine.risk_blocked", ticker=ticker, reason=risk.reason)
            return

        proposal = TradeProposal(
            opportunity=opportunity, safety=report, decision=decision,
            approved_size_sol=risk.size_sol,
        )

        # 4. APPROVAL or AUTO.
        if self._cfg.is_auto:
            log.info("engine.auto_execute", ticker=ticker, paper=self._cfg.is_paper)
            await self._execute(proposal)
        elif self._approval is not None:
            await self._approval.request(proposal)
        else:
            # No approval channel and not auto → never buy silently.
            log.warning("engine.no_approval_channel", ticker=ticker)
            if self._notify:
                await self._notify(
                    f"⚠️ Buy proposal for {ticker} ({risk.size_sol:.3f} SOL) "
                    f"but no approval channel configured — skipped.", "normal")

    async def _execute(self, proposal: TradeProposal) -> None:
        opp = proposal.opportunity
        if proposal.is_expired():
            log.info("engine.proposal_expired", ticker=opp.ticker)
            return

        # Re-check the kill switch at the moment of execution.
        if await self._risk.is_killed():
            log.warning("engine.killed_at_execute", ticker=opp.ticker)
            if self._notify:
                await self._notify(f"🛑 Kill switch on — did not buy {opp.ticker}.", "normal")
            return

        result = await self._executor.buy(
            opp, proposal.approved_size_sol, proposal.decision.max_slippage_bps
        )
        await self._store.record_trade(result, opp.ticker, None)

        if not result.success:
            log.warning("engine.buy_failed", ticker=opp.ticker, error=result.error)
            if self._notify:
                await self._notify(f"❌ Buy failed for {opp.ticker}: {result.error}", "normal")
            return

        entry_price = result.price or 0.0
        position = Position(
            address=opp.address or "", ticker=opp.ticker, chain=opp.chain,
            venue=opp.venue, source=opp.source, entry_price=entry_price,
            amount_sol=result.sol_amount or proposal.approved_size_sol,
            token_amount=result.token_amount or 0.0,
            initial_token_amount=result.token_amount or 0.0,
            status=PositionStatus.OPEN, is_paper=result.is_paper,
            entry_tx=result.tx_signature, peak_price=entry_price,
        )
        await self._store.open_position(position)

        tag = "📝 PAPER" if result.is_paper else "💰 LIVE"
        if self._notify:
            await self._notify(
                f"{tag} BOUGHT <b>{opp.ticker}</b> — {position.amount_sol:.3f} SOL "
                f"@ {entry_price:.8f} SOL\n<code>{result.tx_signature}</code>", "normal")
        log.info("engine.bought", ticker=opp.ticker, paper=result.is_paper,
                 amount_sol=position.amount_sol, tx=result.tx_signature)
