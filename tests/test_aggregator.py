from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agents.aggregator.agent import SignalAggregator


@pytest.fixture
def aggregator():
    return SignalAggregator(api_key="test-key")


class MockContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class MockResponse:
    def __init__(self, text: str) -> None:
        self.content = [MockContentBlock(text)]


# --- Deterministic level tests (no cross-ref) ---


def test_level_act_now_downgraded(aggregator: SignalAggregator) -> None:
    """Test that act_now is downgraded to interesting without cross-ref."""
    social = {"unique_groups": 4, "mention_count": 10}
    score = {"total_score": 25}

    level, notify, blocked = aggregator._determine_level(social, score)
    assert level == "interesting"
    assert blocked == "cross_platform_required"


def test_level_interesting(aggregator: SignalAggregator) -> None:
    """Test interesting level qualification."""
    social = {"unique_groups": 2, "mention_count": 5}
    score = {"total_score": 5}

    level, notify, blocked = aggregator._determine_level(social, score)
    assert level == "interesting"
    assert notify == "normal"
    assert blocked is None


def test_level_watch(aggregator: SignalAggregator) -> None:
    """Test watch level qualification."""
    social = {"unique_groups": 2, "mention_count": 3}
    score = {"total_score": -15}

    level, notify, blocked = aggregator._determine_level(social, score)
    assert level == "watch"
    assert notify == "silent"
    assert blocked is None


def test_level_suppress_low_groups(aggregator: SignalAggregator) -> None:
    """Test suppression when not enough groups."""
    social = {"unique_groups": 1, "mention_count": 5}
    score = {"total_score": 30}

    level, notify, blocked = aggregator._determine_level(social, score)
    assert level == "suppress"


def test_level_suppress_low_score(aggregator: SignalAggregator) -> None:
    """Test suppression when score too low even with enough groups."""
    social = {"unique_groups": 3, "mention_count": 5}
    score = {"total_score": -25}

    level, notify, blocked = aggregator._determine_level(social, score)
    assert level == "suppress"


# --- act_now unlocked with cross-ref (Phase 3) ---


def test_level_act_now_unlocked(aggregator: SignalAggregator) -> None:
    """Test that act_now is unlocked when cross-ref is corroborated and not suspicious."""
    social = {"unique_groups": 4, "mention_count": 10}
    score = {"total_score": 25}
    cross_ref = {
        "corroborated": True,
        "confidence": "high",
        "suspicious": False,
        "has_multi_source": True,
    }

    level, notify, blocked = aggregator._determine_level(social, score, cross_ref)
    assert level == "act_now"
    assert notify == "sound_and_pin"
    assert blocked is None


def test_level_act_now_still_blocked_not_corroborated(
    aggregator: SignalAggregator,
) -> None:
    """Test that act_now stays blocked when cross-ref is NOT corroborated."""
    social = {"unique_groups": 4, "mention_count": 10}
    score = {"total_score": 25}
    cross_ref = {
        "corroborated": False,
        "confidence": "low",
        "suspicious": False,
        "has_multi_source": True,
    }

    level, notify, blocked = aggregator._determine_level(social, score, cross_ref)
    assert level == "interesting"
    assert blocked == "cross_platform_required"


def test_level_act_now_blocked_by_suspicious(aggregator: SignalAggregator) -> None:
    """Test that act_now stays blocked when cross-ref detects suspicious activity."""
    social = {"unique_groups": 4, "mention_count": 10}
    score = {"total_score": 25}
    cross_ref = {
        "corroborated": True,
        "confidence": "high",
        "suspicious": True,
        "has_multi_source": True,
    }

    level, notify, blocked = aggregator._determine_level(social, score, cross_ref)
    assert level == "interesting"
    assert blocked == "cross_platform_required"


# --- _check_cross_platform tests ---


def test_check_cross_platform_optional(aggregator: SignalAggregator) -> None:
    """Test optional requirement is always satisfied."""
    assert SignalAggregator._check_cross_platform(None, "optional") is True
    assert SignalAggregator._check_cross_platform({}, "optional") is True


def test_check_cross_platform_preferred(aggregator: SignalAggregator) -> None:
    """Test preferred requirement is always satisfied."""
    assert SignalAggregator._check_cross_platform(None, "preferred") is True


def test_check_cross_platform_required_none(aggregator: SignalAggregator) -> None:
    """Test required requirement fails with no cross-ref result."""
    assert SignalAggregator._check_cross_platform(None, "required") is False


def test_check_cross_platform_required_corroborated(
    aggregator: SignalAggregator,
) -> None:
    """Test required requirement passes with corroborated + not suspicious."""
    result = {"corroborated": True, "suspicious": False}
    assert SignalAggregator._check_cross_platform(result, "required") is True


def test_check_cross_platform_required_suspicious(
    aggregator: SignalAggregator,
) -> None:
    """Test required requirement fails when suspicious."""
    result = {"corroborated": True, "suspicious": True}
    assert SignalAggregator._check_cross_platform(result, "required") is False


# --- Full aggregate tests (with mocked LLM) ---


@pytest.mark.asyncio
async def test_aggregate_suppress(aggregator: SignalAggregator) -> None:
    """Test that suppressed signals skip LLM call."""
    social = {"unique_groups": 1}
    score = {"total_score": 0}

    result = await aggregator.aggregate("$NOBODY", social, score)
    assert result["alert_level"] == "suppress"
    assert result["summary"] is None


@pytest.mark.asyncio
async def test_aggregate_interesting(aggregator: SignalAggregator) -> None:
    """Test interesting aggregation with LLM summary."""
    social = {
        "unique_groups": 2,
        "group_names": ["Alpha Calls", "Degen Chat"],
        "mention_count": 4,
        "convictions": {"STRONG": 2, "MODERATE": 2},
    }
    score = {
        "total_score": 10,
        "score_breakdown": {"token_age": 10, "liquidity": 10, "volume_trend": -10},
        "reasoning": "Moderate risk profile",
        "flags": [],
        "metadata_available": True,
    }

    llm_response = json.dumps({
        "summary": "$MONKE spotted in 2 groups with strong conviction and adequate liquidity."
    })

    with patch.object(
        aggregator._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=MockResponse(llm_response),
    ):
        result = await aggregator.aggregate("$MONKE", social, score)

    assert result["alert_level"] == "interesting"
    assert result["notify_mode"] == "normal"
    assert "$MONKE" in result["summary"]


@pytest.mark.asyncio
async def test_aggregate_act_now_with_cross_ref(aggregator: SignalAggregator) -> None:
    """Test act_now aggregation with corroborated cross-ref."""
    social = {
        "unique_groups": 4,
        "group_names": ["Alpha Calls", "Degen Chat", "Ape In", "Moon Shots"],
        "mention_count": 8,
        "convictions": {"STRONG": 5, "MODERATE": 3},
        "sources": ["telegram", "twitter"],
    }
    score = {
        "total_score": 30,
        "score_breakdown": {"token_age": 10, "liquidity": 15, "volume_trend": 15},
        "reasoning": "Strong metrics",
        "flags": [],
        "metadata_available": True,
    }
    cross_ref = {
        "corroborated": True,
        "confidence": "high",
        "suspicious": False,
        "has_multi_source": True,
    }

    llm_response = json.dumps({
        "summary": "$MONKE corroborated across platforms with strong conviction."
    })

    with patch.object(
        aggregator._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=MockResponse(llm_response),
    ):
        result = await aggregator.aggregate("$MONKE", social, score, cross_ref)

    assert result["alert_level"] == "act_now"
    assert result["notify_mode"] == "sound_and_pin"
    assert result["blocked_reason"] is None


# --- Parse summary tests ---


def test_parse_summary_valid(aggregator: SignalAggregator) -> None:
    """Test parsing valid summary JSON."""
    raw = '{"summary": "Token is bullish"}'
    assert aggregator._parse_summary(raw) == "Token is bullish"


def test_parse_summary_markdown(aggregator: SignalAggregator) -> None:
    """Test parsing markdown-wrapped summary."""
    raw = '```json\n{"summary": "Token is bullish"}\n```'
    assert aggregator._parse_summary(raw) == "Token is bullish"


def test_parse_summary_fallback(aggregator: SignalAggregator) -> None:
    """Test fallback on invalid JSON."""
    raw = "Just a plain text summary"
    assert aggregator._parse_summary(raw) == "Just a plain text summary"
