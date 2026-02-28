from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import anthropic
import structlog
import yaml

from agents.aggregator.prompts import AGGREGATOR_SYSTEM_PROMPT

log = structlog.get_logger()

MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0

# Load alert rules once at import time
_rules_path = Path(__file__).parent.parent.parent / "config" / "alert_rules.yml"
with open(_rules_path) as f:
    ALERT_RULES = yaml.safe_load(f)


class SignalAggregator:
    """Applies alert_rules.yml tiers deterministically, LLM generates summary.

    Alert levels: act_now, interesting, watch, suppress.
    cross_platform requirement: act_now requires corroborated cross-platform signal.
    """

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def aggregate(
        self,
        ticker: str,
        social_metrics: dict[str, Any],
        score_result: dict[str, Any],
        cross_ref_result: dict[str, Any] | None = None,
        x_sentiment_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Determine alert level and generate summary.

        Returns: alert_level, notify_mode, summary, blocked_reason (if any).
        """
        alert_level, notify_mode, blocked_reason = self._determine_level(
            social_metrics, score_result, cross_ref_result, x_sentiment_result
        )

        if alert_level == "suppress":
            return {
                "alert_level": "suppress",
                "notify_mode": "silent",
                "summary": None,
                "blocked_reason": None,
            }

        # Get LLM summary
        summary = await self._get_summary(
            ticker, social_metrics, score_result, alert_level,
            cross_ref_result, x_sentiment_result,
        )

        return {
            "alert_level": alert_level,
            "notify_mode": notify_mode,
            "summary": summary,
            "blocked_reason": blocked_reason,
        }

    @staticmethod
    def _check_cross_platform(
        cross_ref_result: dict[str, Any] | None,
        requirement: str,
    ) -> bool:
        """Check whether cross-platform requirement is satisfied.

        Requirements:
          - "optional": always satisfied
          - "required": needs corroborated=True AND suspicious=False
          - "preferred": satisfied without cross-ref, enhanced with it
        """
        if requirement == "optional":
            return True
        if requirement == "preferred":
            return True
        if requirement == "required":
            if cross_ref_result is None:
                return False
            return (
                cross_ref_result.get("corroborated", False)
                and not cross_ref_result.get("suspicious", False)
            )
        return False

    @staticmethod
    def _determine_level(
        social_metrics: dict[str, Any],
        score_result: dict[str, Any],
        cross_ref_result: dict[str, Any] | None = None,
        x_sentiment_result: dict[str, Any] | None = None,
    ) -> tuple[str, str, str | None]:
        """Deterministic alert tier selection using alert_rules.yml."""
        levels = ALERT_RULES.get("alert_levels", {})
        unique_groups = social_metrics.get("unique_groups", 0)
        total_score = score_result.get("total_score", 0)

        # Apply X sentiment score boost/penalty
        if x_sentiment_result and x_sentiment_result.get("x_data_available"):
            total_score += x_sentiment_result.get("x_sentiment_score", 0)

        # Try act_now first
        act_now = levels.get("act_now", {})
        if (
            unique_groups >= act_now.get("min_groups", 3)
            and total_score >= act_now.get("min_onchain_score", 20)
        ):
            requirement = act_now.get("cross_platform", "required")
            if SignalAggregator._check_cross_platform(cross_ref_result, requirement):
                return ("act_now", act_now.get("notify", "sound_and_pin"), None)
            # Cross-platform not satisfied → downgrade to interesting
            return (
                "interesting",
                levels.get("interesting", {}).get("notify", "normal"),
                "cross_platform_required",
            )

        # Try interesting
        interesting = levels.get("interesting", {})
        if (
            unique_groups >= interesting.get("min_groups", 2)
            and total_score >= interesting.get("min_onchain_score", 0)
        ):
            return ("interesting", interesting.get("notify", "normal"), None)

        # Try watch
        watch = levels.get("watch", {})
        if (
            unique_groups >= watch.get("min_groups", 2)
            and total_score >= watch.get("min_onchain_score", -20)
        ):
            return ("watch", watch.get("notify", "silent"), None)

        # Doesn't meet any threshold → suppress
        return ("suppress", "silent", None)

    async def _get_summary(
        self,
        ticker: str,
        social_metrics: dict[str, Any],
        score_result: dict[str, Any],
        alert_level: str,
        cross_ref_result: dict[str, Any] | None = None,
        x_sentiment_result: dict[str, Any] | None = None,
    ) -> str:
        """Get LLM-generated summary sentence."""
        payload: dict[str, Any] = {
            "ticker": ticker,
            "alert_level": alert_level,
            "unique_groups": social_metrics.get("unique_groups"),
            "group_names": social_metrics.get("group_names"),
            "mention_count": social_metrics.get("mention_count"),
            "convictions": social_metrics.get("convictions"),
            "total_score": score_result.get("total_score"),
            "score_breakdown": score_result.get("score_breakdown"),
            "reasoning": score_result.get("reasoning"),
            "flags": score_result.get("flags"),
            "metadata_available": score_result.get("metadata_available"),
        }
        if cross_ref_result:
            payload["cross_platform"] = {
                "corroborated": cross_ref_result.get("corroborated"),
                "confidence": cross_ref_result.get("confidence"),
                "platforms": social_metrics.get("sources", []),
            }
        if x_sentiment_result and x_sentiment_result.get("x_data_available"):
            payload["x_sentiment"] = {
                "score": x_sentiment_result.get("x_sentiment_score"),
                "breakdown": x_sentiment_result.get("x_score_breakdown"),
                "narrative": x_sentiment_result.get("key_narrative"),
            }
        user_content = json.dumps(payload, indent=2)

        try:
            raw = await self._call_api(user_content)
            return self._parse_summary(raw)
        except Exception:
            log.warning("aggregator.summary_error", exc_info=True)
            return f"{ticker} triggered {alert_level} alert"

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
                            "text": AGGREGATOR_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_content}],
                )
                return response.content[0].text
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 500, 529) and attempt < MAX_RETRIES:
                    log.warning(
                        "aggregator.api_retry",
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
    def _parse_summary(raw: str) -> str:
        """Parse LLM JSON response into summary string."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
            return data.get("summary", text)
        except json.JSONDecodeError:
            # Fall back to raw text
            return text[:200]
