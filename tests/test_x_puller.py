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
def timeline_data_with_metrics():
    with open(FIXTURES / "x_timeline_with_metrics.json") as f:
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
    # Engagement defaults to 0 when no public_metrics
    assert msg["engagement"] == {"likes": 0, "retweets": 0, "replies": 0, "quotes": 0}
    assert msg["author_followers"] is None


def test_tweet_to_message_with_engagement() -> None:
    """Test that engagement metrics are included in message schema."""
    tweet = {
        "id": "123456",
        "text": "$MONKE looking bullish",
        "created_at": "2024-03-15T10:30:00.000Z",
        "public_metrics": {
            "like_count": 150,
            "retweet_count": 45,
            "reply_count": 12,
            "quote_count": 8,
        },
    }
    msg = XFeedPuller._tweet_to_message(tweet, "cryptoalpha", "999", author_followers=50000)

    assert msg["engagement"]["likes"] == 150
    assert msg["engagement"]["retweets"] == 45
    assert msg["engagement"]["replies"] == 12
    assert msg["engagement"]["quotes"] == 8
    assert msg["author_followers"] == 50000


def test_parse_author_followers() -> None:
    """Test parsing author follower counts from API expansion response."""
    data = {
        "data": [],
        "includes": {
            "users": [
                {
                    "id": "999",
                    "username": "cryptoalpha",
                    "public_metrics": {"followers_count": 50000},
                }
            ]
        },
    }
    followers = XFeedPuller._parse_author_followers(data)
    assert followers == {"999": 50000}


def test_parse_author_followers_empty() -> None:
    """Test parsing when no expansion data is present."""
    data = {"data": []}
    followers = XFeedPuller._parse_author_followers(data)
    assert followers == {}


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


@pytest.mark.asyncio
async def test_poll_captures_engagement_metrics(
    puller: XFeedPuller, mock_redis: AsyncMock, timeline_data_with_metrics: dict
) -> None:
    """Test that polling captures engagement metrics and author followers."""
    puller._accounts = {
        "cryptoalpha": {"user_id": "999", "category": "calls", "priority": "high"}
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=MockResponse(timeline_data_with_metrics))
    puller._client = mock_client

    count = await puller.poll()
    assert count == 3

    # Verify the pushed messages include engagement data
    pushed_messages = [
        json.loads(call.args[1]) for call in mock_redis.rpush.call_args_list
    ]
    first_msg = pushed_messages[0]
    assert first_msg["engagement"]["likes"] == 150
    assert first_msg["engagement"]["retweets"] == 45
    assert first_msg["author_followers"] == 50000


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
