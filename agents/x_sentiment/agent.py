from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import structlog

from agents.x_sentiment.prompts import X_SENTIMENT_SYSTEM_PROMPT

log = structlog.get_logger()

MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0


class XSentimentAnalyzer:
    """Analyzes X/Twitter sentiment for a token using Claude Haiku.

    Only called when a ticker has X mentions.
    Fail-safe: on any error, returns neutral default (no impact on scoring).
    """

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def analyze(
        self,
        ticker: str,
        x_mentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze X mentions for sentiment.

        Returns: bullish_pct, bearish_pct, neutral_pct, quality, key_narrative, reasoning.
        """
        if not x_mentions:
            return self._default_result(reason="no_x_mentions")

        user_content = json.dumps(
            {
                "ticker": ticker,
                "tweets": [
                    {
                        "account": m.get("group_name", ""),
                        "text": (m.get("raw_text") or "")[:200],
                        "engagement": m.get("engagement_data") or {},
                    }
                    for m in x_mentions[:10]
                ],
            },
            indent=2,
        )

        try:
            raw = await self._call_api(user_content)
            return self._parse_response(raw)
        except Exception:
            log.warning("x_sentiment.analysis_error", ticker=ticker, exc_info=True)
            return self._default_result(reason="llm_error")

    async def _call_api(self, user_content: str) -> str:
        """Call Anthropic API with retry on transient errors."""
        delay = INITIAL_RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.messages.create(
                    model=MODEL,
                    max_tokens=512,
                    system=[
                        {
                            "type": "text",
                            "text": X_SENTIMENT_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_content}],
                )
                return response.content[0].text
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 500, 529) and attempt < MAX_RETRIES:
                    log.warning(
                        "x_sentiment.api_retry",
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
    def _parse_response(raw: str) -> dict[str, Any]:
        """Parse LLM JSON response into sentiment result."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        data = json.loads(text)

        bullish_pct = float(data.get("bullish_pct", 0.33))
        bearish_pct = float(data.get("bearish_pct", 0.33))
        neutral_pct = float(data.get("neutral_pct", 0.34))
        quality = data.get("quality", "low")
        if quality not in ("high", "medium", "low"):
            quality = "low"

        return {
            "bullish_pct": bullish_pct,
            "bearish_pct": bearish_pct,
            "neutral_pct": neutral_pct,
            "quality": quality,
            "key_narrative": str(data.get("key_narrative", "")),
            "reasoning": str(data.get("reasoning", "")),
        }

    @staticmethod
    def _default_result(reason: str = "unknown") -> dict[str, Any]:
        """Fail-safe default: neutral sentiment, no impact on scoring."""
        return {
            "bullish_pct": 0.33,
            "bearish_pct": 0.33,
            "neutral_pct": 0.34,
            "quality": "low",
            "key_narrative": reason,
            "reasoning": reason,
        }
