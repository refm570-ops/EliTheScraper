from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agents.x_sentiment.agent import XSentimentAnalyzer
from agents.x_sentiment.scorer import score_x_sentiment


# --- Fixtures ---


@pytest.fixture
def analyzer():
    return XSentimentAnalyzer(api_key="test-key")


class MockContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class MockResponse:
    def __init__(self, text: str) -> None:
        self.content = [MockContentBlock(text)]


# --- Deterministic scorer tests ---


def test_score_viral_bullish() -> None:
    """Viral engagement + bullish sentiment + high quality = high score."""
    engagement = {
        "total_likes": 400,
        "total_retweets": 100,
        "total_replies": 30,
        "total_quotes": 20,
        "max_author_followers": 50000,
        "avg_author_followers": 25000.0,
        "tweet_count": 5,
    }
    sentiment = {
        "bullish_pct": 0.8,
        "bearish_pct": 0.1,
        "neutral_pct": 0.1,
        "quality": "high",
    }
    result = score_x_sentiment(engagement, sentiment)
    assert result["x_data_available"] is True
    bd = result["x_score_breakdown"]
    # total_engagement = 400 + 100 + 20 = 520 -> +15
    assert bd["engagement"] == 15
    # max_followers = 50000 -> +5
    assert bd["reach"] == 5
    # bullish_pct = 0.8 > 0.7 -> +10
    assert bd["sentiment"] == 10
    # quality = high -> +5
    assert bd["quality"] == 5
    assert result["x_sentiment_score"] == 35


def test_score_low_engagement() -> None:
    """Low engagement below noise threshold = penalty."""
    engagement = {
        "total_likes": 2,
        "total_retweets": 0,
        "total_replies": 0,
        "total_quotes": 0,
        "max_author_followers": 500,
        "avg_author_followers": 500.0,
        "tweet_count": 1,
    }
    sentiment = {
        "bullish_pct": 0.5,
        "bearish_pct": 0.2,
        "neutral_pct": 0.3,
        "quality": "low",
    }
    result = score_x_sentiment(engagement, sentiment)
    bd = result["x_score_breakdown"]
    # total_engagement = 2 < 5 -> -10
    assert bd["engagement"] == -10
    # max_followers = 500 < 1000 -> -5
    assert bd["reach"] == -5
    # bullish_pct = 0.5 <= 0.5 -> 0 (not > 0.5)
    assert bd["sentiment"] == 0
    # quality = low -> 0
    assert bd["quality"] == 0
    assert result["x_sentiment_score"] == -15


def test_score_bearish_sentiment() -> None:
    """Strong bearish sentiment penalizes score."""
    engagement = {
        "total_likes": 50,
        "total_retweets": 10,
        "total_replies": 5,
        "total_quotes": 5,
        "max_author_followers": 10000,
        "avg_author_followers": 10000.0,
        "tweet_count": 3,
    }
    sentiment = {
        "bullish_pct": 0.1,
        "bearish_pct": 0.7,
        "neutral_pct": 0.2,
        "quality": "medium",
    }
    result = score_x_sentiment(engagement, sentiment)
    bd = result["x_score_breakdown"]
    # total_engagement = 50 + 10 + 5 = 65 -> +5
    assert bd["engagement"] == 5
    # max_followers = 10000 >= mid (10000), < influencer (100000) -> +5
    assert bd["reach"] == 5
    # bearish_pct = 0.7 > 0.6 -> -15
    assert bd["sentiment"] == -15
    # quality = medium -> +2
    assert bd["quality"] == 2
    assert result["x_sentiment_score"] == -3


def test_score_moderate_engagement_bullish() -> None:
    """Moderate engagement + lean bullish + medium quality."""
    engagement = {
        "total_likes": 80,
        "total_retweets": 10,
        "total_replies": 5,
        "total_quotes": 5,
        "max_author_followers": 120000,
        "avg_author_followers": 60000.0,
        "tweet_count": 4,
    }
    sentiment = {
        "bullish_pct": 0.55,
        "bearish_pct": 0.15,
        "neutral_pct": 0.30,
        "quality": "medium",
    }
    result = score_x_sentiment(engagement, sentiment)
    bd = result["x_score_breakdown"]
    # total_engagement = 80 + 10 + 5 = 95 -> +5 (< 100)
    assert bd["engagement"] == 5
    # max_followers = 120000 -> +10 (>= 100000)
    assert bd["reach"] == 10
    # bullish_pct = 0.55 > 0.5 -> +5
    assert bd["sentiment"] == 5
    # quality = medium -> +2
    assert bd["quality"] == 2
    assert result["x_sentiment_score"] == 22


def test_score_no_follower_data() -> None:
    """Missing follower data results in neutral reach score."""
    engagement = {
        "total_likes": 30,
        "total_retweets": 5,
        "total_replies": 2,
        "total_quotes": 0,
        "max_author_followers": None,
        "avg_author_followers": None,
        "tweet_count": 2,
    }
    sentiment = {
        "bullish_pct": 0.4,
        "bearish_pct": 0.3,
        "neutral_pct": 0.3,
        "quality": "low",
    }
    result = score_x_sentiment(engagement, sentiment)
    assert result["x_score_breakdown"]["reach"] == 0


# --- LLM analysis tests ---


@pytest.mark.asyncio
async def test_analyze_bullish_consensus(analyzer: XSentimentAnalyzer) -> None:
    """Test LLM analysis of bullish consensus."""
    mentions = [
        {
            "group_name": "@cryptoalpha",
            "raw_text": "$MONKE breaking out, volume surging",
            "engagement_data": {"likes": 100, "retweets": 30},
        },
        {
            "group_name": "@whalewatch",
            "raw_text": "$MONKE accumulation by whales confirmed",
            "engagement_data": {"likes": 80, "retweets": 20},
        },
    ]

    llm_response = json.dumps({
        "bullish_pct": 0.9,
        "bearish_pct": 0.0,
        "neutral_pct": 0.1,
        "quality": "high",
        "key_narrative": "Multiple accounts highlighting breakout pattern and whale accumulation",
        "reasoning": "Strong bullish consensus with substantive technical analysis.",
    })

    with patch.object(
        analyzer._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=MockResponse(llm_response),
    ):
        result = await analyzer.analyze("$MONKE", mentions)

    assert result["bullish_pct"] == 0.9
    assert result["bearish_pct"] == 0.0
    assert result["quality"] == "high"
    assert "breakout" in result["key_narrative"]


@pytest.mark.asyncio
async def test_analyze_bearish_majority(analyzer: XSentimentAnalyzer) -> None:
    """Test LLM analysis of bearish majority."""
    mentions = [
        {
            "group_name": "@bearish_trader",
            "raw_text": "$MONKE about to dump, insiders selling",
            "engagement_data": {"likes": 50, "retweets": 10},
        },
    ]

    llm_response = json.dumps({
        "bullish_pct": 0.1,
        "bearish_pct": 0.8,
        "neutral_pct": 0.1,
        "quality": "medium",
        "key_narrative": "Warning about insider selling and potential dump",
        "reasoning": "Predominantly bearish with concern about insider activity.",
    })

    with patch.object(
        analyzer._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=MockResponse(llm_response),
    ):
        result = await analyzer.analyze("$MONKE", mentions)

    assert result["bearish_pct"] == 0.8
    assert result["quality"] == "medium"


@pytest.mark.asyncio
async def test_analyze_mixed_signals(analyzer: XSentimentAnalyzer) -> None:
    """Test LLM analysis of mixed signals."""
    mentions = [
        {
            "group_name": "@bull",
            "raw_text": "$MONKE bullish chart setup",
            "engagement_data": {},
        },
        {
            "group_name": "@bear",
            "raw_text": "$MONKE looks toppy here",
            "engagement_data": {},
        },
    ]

    llm_response = json.dumps({
        "bullish_pct": 0.4,
        "bearish_pct": 0.4,
        "neutral_pct": 0.2,
        "quality": "low",
        "key_narrative": "Divided opinions on price direction",
        "reasoning": "Split sentiment between bulls and bears, low conviction either way.",
    })

    with patch.object(
        analyzer._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=MockResponse(llm_response),
    ):
        result = await analyzer.analyze("$MONKE", mentions)

    assert result["bullish_pct"] == 0.4
    assert result["bearish_pct"] == 0.4


# --- Fail-safe tests ---


@pytest.mark.asyncio
async def test_analyze_llm_error_returns_neutral(analyzer: XSentimentAnalyzer) -> None:
    """Test that LLM error returns neutral default."""
    mentions = [
        {"group_name": "@test", "raw_text": "test", "engagement_data": {}},
    ]

    with patch.object(
        analyzer._client.messages,
        "create",
        new_callable=AsyncMock,
        side_effect=Exception("API error"),
    ):
        result = await analyzer.analyze("$MONKE", mentions)

    assert result["bullish_pct"] == 0.33
    assert result["bearish_pct"] == 0.33
    assert result["quality"] == "low"
    assert result["reasoning"] == "llm_error"


@pytest.mark.asyncio
async def test_analyze_no_mentions_returns_neutral(analyzer: XSentimentAnalyzer) -> None:
    """Test that no mentions returns neutral default."""
    result = await analyzer.analyze("$MONKE", [])
    assert result["bullish_pct"] == 0.33
    assert result["quality"] == "low"
    assert result["reasoning"] == "no_x_mentions"


# --- Parse response tests ---


def test_parse_response_valid(analyzer: XSentimentAnalyzer) -> None:
    """Test parsing valid JSON response."""
    raw = json.dumps({
        "bullish_pct": 0.7,
        "bearish_pct": 0.2,
        "neutral_pct": 0.1,
        "quality": "high",
        "key_narrative": "Bullish narrative",
        "reasoning": "Strong bullish signals",
    })
    result = analyzer._parse_response(raw)
    assert result["bullish_pct"] == 0.7
    assert result["quality"] == "high"


def test_parse_response_markdown_wrapped(analyzer: XSentimentAnalyzer) -> None:
    """Test parsing markdown-wrapped response."""
    raw = '```json\n{"bullish_pct": 0.5, "bearish_pct": 0.3, "neutral_pct": 0.2, "quality": "medium", "key_narrative": "test", "reasoning": "test"}\n```'
    result = analyzer._parse_response(raw)
    assert result["bullish_pct"] == 0.5


def test_parse_response_invalid_quality(analyzer: XSentimentAnalyzer) -> None:
    """Test that invalid quality defaults to low."""
    raw = json.dumps({
        "bullish_pct": 0.5,
        "bearish_pct": 0.3,
        "neutral_pct": 0.2,
        "quality": "INVALID",
        "key_narrative": "test",
        "reasoning": "test",
    })
    result = analyzer._parse_response(raw)
    assert result["quality"] == "low"
