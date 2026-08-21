"""OpportunityEvaluator — turns a safety-passed opportunity into a buy/skip
decision with conviction, size, and slippage. Opus-tier reasoning.

Fail-closed: any API/parse error yields a SKIP decision. The agent can never
*accidentally* buy.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import anthropic
import structlog

from agents.trader.prompts import EVALUATOR_SYSTEM_PROMPT
from trading.models import (
    Conviction,
    DecisionAction,
    Opportunity,
    SafetyReport,
    TradeDecision,
)

log = structlog.get_logger()

# "Opus 5" — the top reasoning tier, used ONLY for this low-volume, high-stakes
# judgment (gated behind aggregator buy-grade AND a passed safety gate, so a
# handful of calls/day). Classification stays on Haiku, scoring on Sonnet.
# Overridable via EVALUATOR_MODEL env for flexibility / model migration.
MODEL = os.getenv("EVALUATOR_MODEL", "claude-opus-5")
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0

_CONVICTION_MAP = {
    "high": Conviction.HIGH,
    "medium": Conviction.MEDIUM,
    "low": Conviction.LOW,
}


class OpportunityEvaluator:
    def __init__(self, api_key: str, model: str = MODEL) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def evaluate(
        self,
        opportunity: Opportunity,
        safety: SafetyReport,
        max_trade_sol: float,
    ) -> TradeDecision:
        """Return a buy/skip decision. SKIP on any error."""
        payload = self._build_payload(opportunity, safety, max_trade_sol)
        try:
            raw = await self._call_api(json.dumps(payload, indent=2))
            decision = self._parse(raw, max_trade_sol)
            log.info(
                "evaluator.decided",
                ticker=opportunity.ticker,
                action=decision.action.value,
                conviction=decision.conviction.value,
                size_sol=decision.size_sol,
            )
            return decision
        except Exception:
            log.warning("evaluator.error_skip", ticker=opportunity.ticker, exc_info=True)
            return self._skip("evaluator error — failing closed to SKIP")

    def _build_payload(
        self, opp: Opportunity, safety: SafetyReport, max_trade_sol: float
    ) -> dict[str, Any]:
        return {
            "ticker": opp.ticker,
            "address": opp.address,
            "chain": opp.chain.value,
            "venue": opp.venue.value,
            "discovery_source": opp.source,
            "source_reputation": opp.source_reputation,
            "aggregator_score": opp.aggregator_score,
            "alert_level": opp.alert_level,
            "social_metrics": {
                "unique_groups": opp.social_metrics.get("unique_groups"),
                "mention_count": opp.social_metrics.get("mention_count"),
                "convictions": opp.social_metrics.get("convictions"),
                "sources": opp.social_metrics.get("sources"),
            },
            "onchain": {
                "price_usd": opp.metadata.get("price_usd"),
                "market_cap": opp.metadata.get("market_cap"),
                "liquidity_usd": opp.metadata.get("liquidity_usd"),
                "volume_24h": opp.metadata.get("volume_24h"),
                "price_change_24h": opp.metadata.get("price_change_24h"),
                "age_days": opp.metadata.get("age_days"),
                "holder_count": opp.metadata.get("holder_count"),
                "top10_holder_pct": opp.metadata.get("top10_holder_pct"),
            },
            "safety_soft_warnings": [f"{c.name}: {c.detail}" for c in safety.soft_failures],
            "max_trade_sol": max_trade_sol,
        }

    async def _call_api(self, user_content: str) -> str:
        delay = INITIAL_RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=512,
                    system=[
                        {
                            "type": "text",
                            "text": EVALUATOR_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_content}],
                )
                return response.content[0].text
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 500, 529) and attempt < MAX_RETRIES:
                    log.warning("evaluator.api_retry", status=e.status_code,
                                attempt=attempt, delay=delay)
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise
        raise RuntimeError("Exhausted retries")

    def _parse(self, raw: str, max_trade_sol: float) -> TradeDecision:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        data = json.loads(text)  # raises → caught by evaluate() → SKIP

        decision_str = str(data.get("decision", "no_go")).lower()
        if decision_str != "go":
            return self._skip(data.get("reasoning", "evaluator returned no_go"))

        conviction = _CONVICTION_MAP.get(str(data.get("conviction", "")).lower(), Conviction.LOW)

        # Defensive numeric parsing; the RiskManager clamps hard afterwards, but
        # we sanitize here too so a garbage value never propagates.
        try:
            size = float(data.get("position_size_sol", 0.0))
        except (TypeError, ValueError):
            size = 0.0
        size = max(0.0, min(size, max_trade_sol))

        try:
            slippage = int(data.get("max_slippage_bps", 500))
        except (TypeError, ValueError):
            slippage = 500
        slippage = max(10, min(slippage, 5000))

        try:
            ttl = int(data.get("ttl_seconds", 120))
        except (TypeError, ValueError):
            ttl = 120
        ttl = max(30, min(ttl, 600))

        if size <= 0:
            return self._skip("evaluator returned non-positive size")

        return TradeDecision(
            action=DecisionAction.BUY,
            conviction=conviction,
            size_sol=size,
            max_slippage_bps=slippage,
            ttl_seconds=ttl,
            reasoning=str(data.get("reasoning", ""))[:500],
            model=self._model,
            raw=data,
        )

    @staticmethod
    def _skip(reason: str) -> TradeDecision:
        return TradeDecision(
            action=DecisionAction.SKIP,
            conviction=Conviction.NONE,
            size_sol=0.0,
            max_slippage_bps=0,
            ttl_seconds=0,
            reasoning=str(reason)[:500],
            model=MODEL,
        )
