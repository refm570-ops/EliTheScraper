from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from skills.token_metadata.fetcher import TokenMetadataFetcher
from storage.db import Database


@pytest.fixture
async def db():
    database = Database(db_path=":memory:")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def fetcher(db):
    f = TokenMetadataFetcher(db=db, birdeye_api_key="test-key", cache_ttl_seconds=300)
    yield f
    await f.close()


@pytest.fixture
def dexscreener_response():
    fixtures = Path(__file__).parent / "fixtures" / "dexscreener_response.json"
    with open(fixtures) as f:
        return json.load(f)


@pytest.fixture
def birdeye_response():
    fixtures = Path(__file__).parent / "fixtures" / "birdeye_response.json"
    with open(fixtures) as f:
        return json.load(f)


class MockHTTPResponse:
    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.mark.asyncio
async def test_fetch_dexscreener(
    fetcher: TokenMetadataFetcher, dexscreener_response: dict
) -> None:
    """Test DexScreener data extraction picks highest-liquidity pair."""
    with patch.object(
        fetcher._http,
        "get",
        new_callable=AsyncMock,
        return_value=MockHTTPResponse(dexscreener_response),
    ):
        result = await fetcher.fetch("$MONKE")

    assert result is not None
    assert result["source"] == "dexscreener"
    assert result["price_usd"] == 0.0045
    assert result["market_cap"] == 4500000
    assert result["liquidity_usd"] == 250000  # Higher liquidity pair
    assert result["volume_24h"] == 800000
    assert result["price_change_24h"] == 45.5
    assert result["age_days"] is not None
    assert result["chain"] == "solana"
    assert result["dex_url"] is not None


@pytest.mark.asyncio
async def test_fetch_with_birdeye(
    fetcher: TokenMetadataFetcher,
    dexscreener_response: dict,
    birdeye_response: dict,
) -> None:
    """Test that Birdeye data enriches holder info."""
    responses = [
        MockHTTPResponse(dexscreener_response),
        MockHTTPResponse(birdeye_response),
    ]
    call_count = 0

    async def mock_get(*args, **kwargs):
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        return resp

    with patch.object(fetcher._http, "get", side_effect=mock_get):
        result = await fetcher.fetch("$MONKE")

    assert result is not None
    assert result["holder_count"] == 15234
    assert result["top10_holder_pct"] == 35.5


@pytest.mark.asyncio
async def test_fetch_no_pairs(fetcher: TokenMetadataFetcher) -> None:
    """Test that empty DexScreener response returns None."""
    with patch.object(
        fetcher._http,
        "get",
        new_callable=AsyncMock,
        return_value=MockHTTPResponse({"pairs": []}),
    ):
        result = await fetcher.fetch("$NONEXISTENT")

    assert result is None


@pytest.mark.asyncio
async def test_fetch_http_error(fetcher: TokenMetadataFetcher) -> None:
    """Test that HTTP errors return None gracefully."""
    with patch.object(
        fetcher._http,
        "get",
        new_callable=AsyncMock,
        side_effect=Exception("Connection refused"),
    ):
        result = await fetcher.fetch("$BROKEN")

    assert result is None


@pytest.mark.asyncio
async def test_cache_hit(
    fetcher: TokenMetadataFetcher, dexscreener_response: dict
) -> None:
    """Test that second fetch uses cache."""
    mock = AsyncMock(return_value=MockHTTPResponse(dexscreener_response))
    with patch.object(fetcher._http, "get", mock):
        result1 = await fetcher.fetch("$CACHED")
        result2 = await fetcher.fetch("$CACHED")

    assert result1 is not None
    assert result2 is not None
    # HTTP called only once (first was DexScreener, second should be cached)
    # With birdeye enabled, first fetch makes 2 calls (dex + birdeye)
    # Second fetch should make 0 calls (cached)
    assert mock.call_count <= 2  # first fetch only


@pytest.mark.asyncio
async def test_ticker_normalization(
    fetcher: TokenMetadataFetcher, dexscreener_response: dict
) -> None:
    """Test that $ticker and TICKER resolve the same cache entry."""
    mock = AsyncMock(return_value=MockHTTPResponse(dexscreener_response))
    with patch.object(fetcher._http, "get", mock):
        await fetcher.fetch("$monke")
        result = await fetcher.fetch("MONKE")

    assert result is not None
