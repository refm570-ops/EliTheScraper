from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agents.classifier.agent import MessageClassifier
from agents.classifier.prompts import VALID_INTENTS


@pytest.fixture
def sample_messages() -> list[dict]:
    fixtures = Path(__file__).parent / "fixtures" / "tg_messages.json"
    with open(fixtures) as f:
        return json.load(f)


@pytest.fixture
def classifier() -> MessageClassifier:
    return MessageClassifier(api_key="test-key")


# A realistic mock response matching the fixture messages
MOCK_API_RESPONSE = json.dumps([
    {"id": 1001, "ticker": "$MONKE", "intent": "TICKER_CALL", "conviction": "STRONG", "context": "Bought in, bullish on chart and market cap"},
    {"id": 1002, "ticker": None, "intent": "NOISE", "conviction": None, "context": None},
    {"id": 1003, "ticker": None, "intent": "PRICE_ACTION", "conviction": None, "context": "Bearish BTC outlook, potential alt selloff"},
    {"id": 1004, "ticker": "$BONK", "intent": "TICKER_CALL", "conviction": "STRONG", "context": "Heavy position, high conviction buy"},
    {"id": 1005, "ticker": "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr", "intent": "TICKER_CALL", "conviction": "MODERATE", "context": "Watching with growing volume"},
    {"id": 1006, "ticker": None, "intent": "NOISE", "conviction": None, "context": None},
    {"id": 1007, "ticker": "$PEPE", "intent": "PRICE_ACTION", "conviction": None, "context": "Rug pull, liquidity removed"},
    {"id": 1008, "ticker": "$WIF", "intent": "TICKER_CALL", "conviction": "WEAK", "context": "Secondhand info, unverified pump claim"},
])


class MockContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class MockResponse:
    def __init__(self, text: str) -> None:
        self.content = [MockContentBlock(text)]


@pytest.mark.asyncio
async def test_classify_batch(classifier: MessageClassifier, sample_messages: list[dict]) -> None:
    """Test that classifier correctly parses a batch of messages."""
    with patch.object(
        classifier._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=MockResponse(MOCK_API_RESPONSE),
    ):
        results = await classifier.classify_batch(sample_messages)

    assert len(results) == 8

    # Check first message: strong ticker call
    assert results[0]["ticker"] == "$MONKE"
    assert results[0]["intent"] == "TICKER_CALL"
    assert results[0]["conviction"] == "STRONG"

    # Check noise messages
    assert results[1]["intent"] == "NOISE"
    assert results[5]["intent"] == "NOISE"

    # Check price action
    assert results[2]["intent"] == "PRICE_ACTION"

    # Check rug pull detection
    assert results[6]["ticker"] == "$PEPE"
    assert results[6]["intent"] == "PRICE_ACTION"

    # Check weak conviction (secondhand)
    assert results[7]["conviction"] == "WEAK"

    # Check contract address extraction
    assert results[4]["ticker"] == "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr"


@pytest.mark.asyncio
async def test_classify_empty_batch(classifier: MessageClassifier) -> None:
    """Test that empty batch returns empty list without API call."""
    results = await classifier.classify_batch([])
    assert results == []


@pytest.mark.asyncio
async def test_parse_response_with_markdown_fences(classifier: MessageClassifier) -> None:
    """Test that markdown-fenced JSON is handled."""
    fenced = f"```json\n{MOCK_API_RESPONSE}\n```"
    results = classifier._parse_response(fenced, 8)
    assert len(results) == 8
    assert results[0]["ticker"] == "$MONKE"


@pytest.mark.asyncio
async def test_parse_response_invalid_json(classifier: MessageClassifier) -> None:
    """Test that invalid JSON returns empty list."""
    results = classifier._parse_response("this is not json", 5)
    assert results == []


@pytest.mark.asyncio
async def test_parse_response_validates_intents(classifier: MessageClassifier) -> None:
    """Test that invalid intents default to NOISE."""
    bad_response = json.dumps([
        {"id": 1, "ticker": "$X", "intent": "INVALID_INTENT", "conviction": None, "context": None},
    ])
    results = classifier._parse_response(bad_response, 1)
    assert len(results) == 1
    assert results[0]["intent"] == "NOISE"


@pytest.mark.asyncio
async def test_all_intents_valid() -> None:
    """Verify all expected intents are in the valid set."""
    expected = {"TICKER_CALL", "PRICE_ACTION", "ANALYSIS", "NOISE"}
    assert VALID_INTENTS == expected
