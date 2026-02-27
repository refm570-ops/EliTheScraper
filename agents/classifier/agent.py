from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import structlog

from agents.classifier.prompts import (
    FEW_SHOT_BLOCK,
    SYSTEM_PROMPT,
    VALID_CONVICTIONS,
    VALID_INTENTS,
)

log = structlog.get_logger()

MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0


class MessageClassifier:
    """Classifies batches of raw Telegram messages using Claude Haiku."""

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def classify_batch(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Classify a batch of raw messages.

        Returns list of classification dicts with id, ticker, intent,
        conviction, context.
        """
        if not messages:
            return []

        # Build input: only send id + text to the LLM
        input_items = [
            {"id": msg.get("message_id", i), "text": msg.get("text", "")}
            for i, msg in enumerate(messages)
        ]
        user_content = f"{FEW_SHOT_BLOCK}\n{json.dumps(input_items)}"

        raw_response = await self._call_api(user_content)
        parsed = self._parse_response(raw_response, len(messages))

        log.info(
            "classifier.batch_classified",
            input_count=len(messages),
            output_count=len(parsed),
        )
        return parsed

    async def _call_api(self, user_content: str) -> str:
        """Call Anthropic API with retry on 529/500 errors."""
        delay = INITIAL_RETRY_DELAY

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_content}],
                )
                return response.content[0].text
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 500, 529) and attempt < MAX_RETRIES:
                    log.warning(
                        "classifier.api_retry",
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
    def _parse_response(
        raw: str, expected_count: int
    ) -> list[dict[str, Any]]:
        """Parse and validate the JSON response from the LLM."""
        # Strip any accidental markdown fences
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            results = json.loads(text)
        except json.JSONDecodeError:
            log.error("classifier.json_parse_error", raw=raw[:500])
            return []

        if not isinstance(results, list):
            log.error("classifier.unexpected_type", type=type(results).__name__)
            return []

        # Validate each item
        validated: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            intent = item.get("intent")
            conviction = item.get("conviction")
            if intent not in VALID_INTENTS:
                intent = "NOISE"
            if conviction not in VALID_CONVICTIONS:
                conviction = None
            validated.append(
                {
                    "id": item.get("id"),
                    "ticker": item.get("ticker"),
                    "intent": intent,
                    "conviction": conviction,
                    "context": item.get("context"),
                }
            )

        if len(validated) != expected_count:
            log.warning(
                "classifier.count_mismatch",
                expected=expected_count,
                got=len(validated),
            )

        return validated
