from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import structlog

from agents.cross_ref.prompts import CROSS_REF_SYSTEM_PROMPT

log = structlog.get_logger()

MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0


class CrossReferenceAnalyst:
    """Assesses cross-platform corroboration using Claude Haiku.

    Only called when a ticker has mentions from both Telegram and X/Twitter.
    Fail-safe: on any error, returns NOT corroborated (act_now stays blocked).
    """

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def analyze(
        self,
        ticker: str,
        telegram_mentions: list[dict[str, Any]],
        twitter_mentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze cross-platform mentions for a ticker.

        Returns: corroborated, confidence, reasoning, suspicious, has_multi_source.
        """
        if not telegram_mentions or not twitter_mentions:
            return self._default_result(reason="single_source_only")

        user_content = json.dumps(
            {
                "ticker": ticker,
                "telegram_mentions": [
                    {
                        "group": m.get("group_name", ""),
                        "conviction": m.get("conviction", ""),
                        "context": m.get("context", ""),
                        "text": (m.get("raw_text") or "")[:200],
                    }
                    for m in telegram_mentions[:10]
                ],
                "twitter_mentions": [
                    {
                        "account": m.get("group_name", ""),
                        "text": (m.get("raw_text") or "")[:200],
                    }
                    for m in twitter_mentions[:10]
                ],
            },
            indent=2,
        )

        try:
            raw = await self._call_api(user_content)
            return self._parse_response(raw)
        except Exception:
            log.warning("cross_ref.analysis_error", ticker=ticker, exc_info=True)
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
                            "text": CROSS_REF_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_content}],
                )
                return response.content[0].text
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 500, 529) and attempt < MAX_RETRIES:
                    log.warning(
                        "cross_ref.api_retry",
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
        """Parse LLM JSON response into cross-ref result."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        data = json.loads(text)

        corroborated = bool(data.get("corroborated", False))
        confidence = data.get("confidence", "low")
        if confidence not in ("high", "medium", "low"):
            confidence = "low"

        return {
            "corroborated": corroborated,
            "confidence": confidence,
            "reasoning": str(data.get("reasoning", "")),
            "suspicious": bool(data.get("suspicious", False)),
            "has_multi_source": True,
        }

    @staticmethod
    def _default_result(reason: str = "unknown") -> dict[str, Any]:
        """Fail-safe default: NOT corroborated."""
        return {
            "corroborated": False,
            "confidence": "low",
            "reasoning": reason,
            "suspicious": False,
            "has_multi_source": False,
        }
