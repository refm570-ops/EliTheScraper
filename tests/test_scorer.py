from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agents.scorer.agent import FundamentalsScorer


@pytest.fixture
def scorer():
    return FundamentalsScorer(api_key="test-key")


class MockContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class MockResponse:
    def __init__(self, text: str) -> None:
        self.content = [MockContentBlock(text)]


# --- Deterministic scoring tests (no LLM needed) ---


def test_score_young_token(scorer: FundamentalsScorer) -> None:
    """Test scoring for a critically young token."""
    metadata = {"age_days": 0.5, "liquidity_usd": 30000, "price_change_24h": 50}
    breakdown = scorer._compute_scores(metadata)
    assert breakdown["token_age"] == -30  # critical_young
    assert breakdown["liquidity"] == -20  # danger
    assert breakdown["volume_trend"] == 15  # growing


def test_score_established_healthy_token(scorer: FundamentalsScorer) -> None:
    """Test scoring for an established, healthy token."""
    metadata = {
        "age_days": 30,
        "liquidity_usd": 2000000,
        "price_change_24h": 10,
        "top10_holder_pct": 15,
    }
    breakdown = scorer._compute_scores(metadata)
    assert breakdown["token_age"] == 5  # established
    assert breakdown["liquidity"] == 15  # deep
    assert breakdown["volume_trend"] == 0  # flat
    assert breakdown["holder_distribution"] == 20  # distributed


def test_score_whale_dominated(scorer: FundamentalsScorer) -> None:
    """Test scoring for whale-dominated token."""
    metadata = {"top10_holder_pct": 75}
    breakdown = scorer._compute_scores(metadata)
    assert breakdown["holder_distribution"] == -25


def test_score_declining_volume(scorer: FundamentalsScorer) -> None:
    """Test scoring for declining volume."""
    metadata = {"price_change_24h": -50}
    breakdown = scorer._compute_scores(metadata)
    assert breakdown["volume_trend"] == -15


def test_score_exploding_volume(scorer: FundamentalsScorer) -> None:
    """Test scoring for exploding volume."""
    metadata = {"price_change_24h": 150}
    breakdown = scorer._compute_scores(metadata)
    assert breakdown["volume_trend"] == 20


def test_score_missing_fields(scorer: FundamentalsScorer) -> None:
    """Test that missing metadata fields produce empty breakdown."""
    metadata = {}
    breakdown = scorer._compute_scores(metadata)
    assert breakdown == {}


def test_score_moderate_age_adequate_liquidity(scorer: FundamentalsScorer) -> None:
    """Test moderate age + adequate liquidity."""
    metadata = {"age_days": 10, "liquidity_usd": 500000}
    breakdown = scorer._compute_scores(metadata)
    assert breakdown["token_age"] == 10  # moderate
    assert breakdown["liquidity"] == 10  # adequate


# --- Full score method tests (with mocked LLM) ---


@pytest.mark.asyncio
async def test_score_none_metadata(scorer: FundamentalsScorer) -> None:
    """Test that None metadata returns score=0 with no LLM call."""
    result = await scorer.score(None)
    assert result["total_score"] == 0
    assert result["metadata_available"] is False
    assert "no_data" in result["flags"]


@pytest.mark.asyncio
async def test_score_with_metadata(scorer: FundamentalsScorer) -> None:
    """Test full score with metadata and mocked LLM."""
    metadata = {
        "price_usd": 0.005,
        "market_cap": 5000000,
        "liquidity_usd": 300000,
        "volume_24h": 500000,
        "price_change_24h": 60,
        "age_days": 5,
        "holder_count": 5000,
        "top10_holder_pct": 30,
    }

    llm_response = json.dumps({
        "reasoning": "Token is 5 days old with adequate liquidity and growing volume. Moderate risk profile.",
        "flags": [],
    })

    with patch.object(
        scorer._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=MockResponse(llm_response),
    ):
        result = await scorer.score(metadata)

    assert result["metadata_available"] is True
    assert result["total_score"] != 0
    assert "token_age" in result["score_breakdown"]
    assert "liquidity" in result["score_breakdown"]
    assert result["reasoning"] == "Token is 5 days old with adequate liquidity and growing volume. Moderate risk profile."


# --- Parse reasoning tests ---


def test_parse_reasoning_valid(scorer: FundamentalsScorer) -> None:
    """Test parsing valid JSON reasoning."""
    raw = '{"reasoning": "Good token", "flags": ["no_holder_data"]}'
    reasoning, flags = scorer._parse_reasoning(raw)
    assert reasoning == "Good token"
    assert flags == ["no_holder_data"]


def test_parse_reasoning_markdown_fences(scorer: FundamentalsScorer) -> None:
    """Test parsing reasoning wrapped in markdown."""
    raw = '```json\n{"reasoning": "Good", "flags": []}\n```'
    reasoning, flags = scorer._parse_reasoning(raw)
    assert reasoning == "Good"
    assert flags == []


def test_parse_reasoning_invalid_json(scorer: FundamentalsScorer) -> None:
    """Test fallback on invalid JSON."""
    raw = "This is just text, not JSON"
    reasoning, flags = scorer._parse_reasoning(raw)
    assert reasoning == "This is just text, not JSON"
    assert flags == []
