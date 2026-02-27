from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import anthropic
import structlog
import yaml

from agents.scorer.prompts import SCORER_SYSTEM_PROMPT

log = structlog.get_logger()

MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0

# Load scoring weights once at import time
_weights_path = Path(__file__).parent.parent.parent / "config" / "scoring_weights.yml"
with open(_weights_path) as f:
    SCORING_WEIGHTS = yaml.safe_load(f)


class FundamentalsScorer:
    """Scores tokens using deterministic Python + LLM reasoning note.

    The numeric score uses pure Python + scoring_weights.yml thresholds.
    The LLM (Claude Sonnet) only generates a reasoning note + warning flags.
    """

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def score(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        """Score a token based on its on-chain metadata.

        Returns: total_score, score_breakdown, metadata_available, reasoning, flags.
        """
        if metadata is None:
            return {
                "total_score": 0,
                "score_breakdown": {},
                "metadata_available": False,
                "reasoning": "No on-chain data available",
                "flags": ["no_data"],
            }

        breakdown = self._compute_scores(metadata)
        total = sum(breakdown.values())

        # Get LLM reasoning
        reasoning, flags = await self._get_reasoning(metadata, breakdown, total)

        return {
            "total_score": total,
            "score_breakdown": breakdown,
            "metadata_available": True,
            "reasoning": reasoning,
            "flags": flags,
        }

    @staticmethod
    def _compute_scores(metadata: dict[str, Any]) -> dict[str, int]:
        """Deterministic scoring using scoring_weights.yml thresholds."""
        breakdown: dict[str, int] = {}

        # Token age scoring
        age_days = metadata.get("age_days")
        age_config = SCORING_WEIGHTS.get("token_age", {})
        if age_days is not None:
            if age_days <= age_config.get("critical_young", {}).get("max_days", 1):
                breakdown["token_age"] = age_config["critical_young"]["score"]
            elif age_days <= age_config.get("young", {}).get("max_days", 3):
                breakdown["token_age"] = age_config["young"]["score"]
            elif age_days <= age_config.get("moderate", {}).get("max_days", 14):
                breakdown["token_age"] = age_config["moderate"]["score"]
            else:
                breakdown["token_age"] = age_config.get("established", {}).get("score", 5)

        # Holder distribution scoring
        top10_pct = metadata.get("top10_holder_pct")
        holder_config = SCORING_WEIGHTS.get("holder_distribution", {})
        if top10_pct is not None:
            if top10_pct > holder_config.get("whale_dominated", {}).get("top10_pct_above", 60):
                breakdown["holder_distribution"] = holder_config["whale_dominated"]["score"]
            elif top10_pct > holder_config.get("concentrated", {}).get("top10_pct_above", 40):
                breakdown["holder_distribution"] = holder_config["concentrated"]["score"]
            elif top10_pct >= holder_config.get("healthy", {}).get("top10_pct_below", 40):
                # Between healthy threshold and concentrated — use healthy
                breakdown["holder_distribution"] = holder_config["healthy"]["score"]
            else:
                breakdown["holder_distribution"] = holder_config.get("distributed", {}).get("score", 20)

        # Liquidity scoring
        liquidity = metadata.get("liquidity_usd")
        liq_config = SCORING_WEIGHTS.get("liquidity", {})
        if liquidity is not None:
            if liquidity <= liq_config.get("danger", {}).get("max_usd", 50000):
                breakdown["liquidity"] = liq_config["danger"]["score"]
            elif liquidity <= liq_config.get("thin", {}).get("max_usd", 200000):
                breakdown["liquidity"] = liq_config["thin"]["score"]
            elif liquidity <= liq_config.get("adequate", {}).get("max_usd", 1000000):
                breakdown["liquidity"] = liq_config["adequate"]["score"]
            else:
                breakdown["liquidity"] = liq_config.get("deep", {}).get("score", 15)

        # Volume trend scoring
        price_change = metadata.get("price_change_24h")
        vol_config = SCORING_WEIGHTS.get("volume_trend", {})
        if price_change is not None:
            if price_change > vol_config.get("exploding", {}).get("change_24h_above", 100):
                breakdown["volume_trend"] = vol_config["exploding"]["score"]
            elif price_change > vol_config.get("growing", {}).get("change_24h_above", 30):
                breakdown["volume_trend"] = vol_config["growing"]["score"]
            elif price_change < vol_config.get("declining", {}).get("change_24h_below", -30):
                breakdown["volume_trend"] = vol_config["declining"]["score"]
            else:
                breakdown["volume_trend"] = vol_config.get("flat", {}).get("score", 0)

        return breakdown

    async def _get_reasoning(
        self,
        metadata: dict[str, Any],
        breakdown: dict[str, int],
        total_score: int,
    ) -> tuple[str, list[str]]:
        """Get LLM-generated reasoning note and warning flags."""
        user_content = json.dumps(
            {
                "price_usd": metadata.get("price_usd"),
                "market_cap": metadata.get("market_cap"),
                "liquidity_usd": metadata.get("liquidity_usd"),
                "volume_24h": metadata.get("volume_24h"),
                "price_change_24h": metadata.get("price_change_24h"),
                "age_days": metadata.get("age_days"),
                "holder_count": metadata.get("holder_count"),
                "top10_holder_pct": metadata.get("top10_holder_pct"),
                "score_breakdown": breakdown,
                "total_score": total_score,
            },
            indent=2,
        )

        try:
            raw = await self._call_api(user_content)
            return self._parse_reasoning(raw)
        except Exception:
            log.warning("scorer.reasoning_error", exc_info=True)
            return "Unable to generate reasoning", []

    async def _call_api(self, user_content: str) -> str:
        """Call Anthropic API with retry on transient errors."""
        delay = INITIAL_RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.messages.create(
                    model=MODEL,
                    max_tokens=256,
                    system=[
                        {
                            "type": "text",
                            "text": SCORER_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_content}],
                )
                return response.content[0].text
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 500, 529) and attempt < MAX_RETRIES:
                    log.warning(
                        "scorer.api_retry",
                        status=e.status_code,
                        attempt=attempt,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise
        raise RuntimeError("Exhausted retries")

    @staticmethod
    def _parse_reasoning(raw: str) -> tuple[str, list[str]]:
        """Parse LLM JSON response into reasoning + flags."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("scorer.parse_error", raw=raw[:300])
            return raw.strip()[:200], []

        reasoning = data.get("reasoning", "")
        flags = data.get("flags", [])
        if not isinstance(flags, list):
            flags = []
        return reasoning, flags
