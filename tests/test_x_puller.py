from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skills.x_puller.puller import XFeedPuller

FIXTURES = Path(__file__).parent / "fixtures"


class MockResponse:
    def __init__(self, data: dict, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.rpush = AsyncMock()
    return redis


@pytest.fixture
def timeline_data():
    with open(FIXTURES / "x_timeline_response.json") as f:
        return json.load(f)


@pytest.fixture
def puller(mock_redis):
    return XFeedPuller(
        bearer_token="test-token",
        redis=mock_redis,
    )


# --- Tweet to message mapping ---


def test_tweet_to_message() -> None:
    """Test that tweets are correctly converted to the shared message schema."""
    tweet = {
        "id": "123456",
        "text": "$MONKE looking bullish",
        "created_at": "2024-03-15T10:30:00.000Z",
    }
    msg = XFeedPuller._tweet_to_message(tweet, "cryptoalpha", "999")

    assert msg["source"] == "twitter"
    assert msg["group_id"] == "999"
    assert msg["group_name"] == "@cryptoalpha"
    assert msg["message_id"] == "123456"
    assert msg["sender_id"] == "999"
    assert msg["text"] == "$MONKE looking bullish"
    assert msg["timestamp"] == "2024-03-15T10:30:00.000Z"
    assert msg["has_media"] is False
    assert msg["reply_to"] is None


# --- Poll with mock httpx ---


@pytest.mark.asyncio
async def test_poll_pushes_tweets(
    puller: XFeedPuller, mock_redis: AsyncMock, timeline_data: dict
) -> None:
    """Test that polling fetches tweets and pushes them to Redis."""
    # Set up puller with a resolved account
    puller._accounts = {
        "cryptoalpha": {"user_id": "999", "category": "calls", "priority": "high"}
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=MockResponse(timeline_data))
    puller._client = mock_client

    count = await puller.poll()
    assert count == 3

    # Verify tweets were pushed to Redis
    assert mock_redis.rpush.call_count == 3

    # Verify since_id was updated to newest tweet
    mock_redis.set.assert_called_once_with("x:since_id:999", "1234567890123456789")


# --- since_id tracking ---


@pytest.mark.asyncio
async def test_poll_uses_since_id(
    puller: XFeedPuller, mock_redis: AsyncMock
) -> None:
    """Test that since_id is passed to the API when available."""
    puller._accounts = {
        "cryptoalpha": {"user_id": "999", "category": "calls", "priority": "high"}
    }
    mock_redis.get = AsyncMock(return_value="1234567890000000000")

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=MockResponse({"data": [], "meta": {"result_count": 0}})
    )
    puller._client = mock_client

    await puller.poll()

    # Verify since_id was included in request params
    call_args = mock_client.get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params", {})
    assert params.get("since_id") == "1234567890000000000"


# --- Rate limit (429) handling ---


@pytest.mark.asyncio
async def test_poll_handles_429(
    puller: XFeedPuller, mock_redis: AsyncMock
) -> None:
    """Test graceful handling of 429 rate limit responses."""
    puller._accounts = {
        "cryptoalpha": {"user_id": "999", "category": "calls", "priority": "high"}
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=MockResponse({}, status_code=429))
    puller._client = mock_client

    count = await puller.poll()
    assert count == 0
    mock_redis.rpush.assert_not_called()


# --- Empty account list ---


@pytest.mark.asyncio
async def test_poll_no_accounts(puller: XFeedPuller) -> None:
    """Test that polling with no accounts returns 0."""
    puller._client = AsyncMock()
    puller._accounts = {}
    count = await puller.poll()
    assert count == 0


# --- Empty response ---


@pytest.mark.asyncio
async def test_poll_empty_response(
    puller: XFeedPuller, mock_redis: AsyncMock
) -> None:
    """Test handling of empty tweet response."""
    puller._accounts = {
        "cryptoalpha": {"user_id": "999", "category": "calls", "priority": "high"}
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=MockResponse({"meta": {"result_count": 0}})
    )
    puller._client = mock_client

    count = await puller.poll()
    assert count == 0
    mock_redis.rpush.assert_not_called()
