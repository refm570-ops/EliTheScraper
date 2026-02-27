from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agents.cross_ref.agent import CrossReferenceAnalyst


@pytest.fixture
def analyst():
    return CrossReferenceAnalyst(api_key="test-key")


class MockContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class MockResponse:
    def __init__(self, text: str) -> None:
        self.content = [MockContentBlock(text)]


# --- Single-source skip ---


@pytest.mark.asyncio
async def test_single_source_telegram_only(analyst: CrossReferenceAnalyst) -> None:
    """Test that analysis returns NOT corroborated when only Telegram mentions exist."""
    tg = [{"group_name": "Alpha Calls", "raw_text": "$MONKE looking good"}]
    result = await analyst.analyze("$MONKE", tg, [])

    assert result["corroborated"] is False
    assert result["has_multi_source"] is False
    assert result["reasoning"] == "single_source_only"


@pytest.mark.asyncio
async def test_single_source_twitter_only(analyst: CrossReferenceAnalyst) -> None:
    """Test that analysis returns NOT corroborated when only Twitter mentions exist."""
    tw = [{"group_name": "@cryptoalpha", "raw_text": "$MONKE bullish"}]
    result = await analyst.analyze("$MONKE", [], tw)

    assert result["corroborated"] is False
    assert result["has_multi_source"] is False


# --- Corroborated analysis ---


@pytest.mark.asyncio
async def test_corroborated_analysis(analyst: CrossReferenceAnalyst) -> None:
    """Test corroborated cross-platform analysis."""
    tg = [
        {"group_name": "Alpha Calls", "conviction": "STRONG", "context": "Aping in", "raw_text": "$MONKE aped"},
        {"group_name": "Degen Chat", "conviction": "MODERATE", "context": "Watching", "raw_text": "watching $MONKE"},
    ]
    tw = [
        {"group_name": "@cryptoalpha", "raw_text": "$MONKE chart looking great, TA confirms breakout"},
    ]

    llm_response = json.dumps({
        "corroborated": True,
        "confidence": "high",
        "reasoning": "$MONKE independently mentioned in TG alpha groups and on X with technical analysis — genuine convergence.",
        "suspicious": False,
    })

    with patch.object(
        analyst._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=MockResponse(llm_response),
    ):
        result = await analyst.analyze("$MONKE", tg, tw)

    assert result["corroborated"] is True
    assert result["confidence"] == "high"
    assert result["suspicious"] is False
    assert result["has_multi_source"] is True


# --- Suspicious detection ---


@pytest.mark.asyncio
async def test_suspicious_detection(analyst: CrossReferenceAnalyst) -> None:
    """Test that suspicious coordination is detected."""
    tg = [{"group_name": "Pump Group", "conviction": "STRONG", "context": "Buy now", "raw_text": "Buy $SCAM now!!!"}]
    tw = [{"group_name": "@pumper", "raw_text": "Buy $SCAM now!!!"}]

    llm_response = json.dumps({
        "corroborated": False,
        "confidence": "high",
        "reasoning": "Identical text across platforms posted simultaneously — coordinated pump.",
        "suspicious": True,
    })

    with patch.object(
        analyst._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=MockResponse(llm_response),
    ):
        result = await analyst.analyze("$SCAM", tg, tw)

    assert result["corroborated"] is False
    assert result["suspicious"] is True
    assert result["has_multi_source"] is True


# --- LLM failure fail-safe ---


@pytest.mark.asyncio
async def test_llm_failure_failsafe(analyst: CrossReferenceAnalyst) -> None:
    """Test that LLM failure returns NOT corroborated (conservative fail-safe)."""
    tg = [{"group_name": "Alpha Calls", "raw_text": "$MONKE aped"}]
    tw = [{"group_name": "@cryptoalpha", "raw_text": "$MONKE bullish"}]

    with patch.object(
        analyst._client.messages,
        "create",
        new_callable=AsyncMock,
        side_effect=Exception("API Error"),
    ):
        result = await analyst.analyze("$MONKE", tg, tw)

    assert result["corroborated"] is False
    assert result["has_multi_source"] is False
    assert result["reasoning"] == "llm_error"


# --- Parse response tests ---


def test_parse_valid_response(analyst: CrossReferenceAnalyst) -> None:
    """Test parsing valid JSON response."""
    raw = json.dumps({
        "corroborated": True,
        "confidence": "medium",
        "reasoning": "Independent signals.",
        "suspicious": False,
    })
    result = analyst._parse_response(raw)
    assert result["corroborated"] is True
    assert result["confidence"] == "medium"
    assert result["has_multi_source"] is True


def test_parse_markdown_wrapped(analyst: CrossReferenceAnalyst) -> None:
    """Test parsing markdown-wrapped JSON response."""
    raw = '```json\n{"corroborated": false, "confidence": "low", "reasoning": "test", "suspicious": false}\n```'
    result = analyst._parse_response(raw)
    assert result["corroborated"] is False


def test_parse_invalid_confidence(analyst: CrossReferenceAnalyst) -> None:
    """Test that invalid confidence defaults to low."""
    raw = json.dumps({
        "corroborated": True,
        "confidence": "very_high",
        "reasoning": "test",
        "suspicious": False,
    })
    result = analyst._parse_response(raw)
    assert result["confidence"] == "low"
