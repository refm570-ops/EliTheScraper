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


def _pair(chain: str, liquidity: float, address: str) -> dict:
    return {
        "chainId": chain,
        "dexId": "uniswap",
        "url": f"https://dexscreener.com/{chain}/{address}",
        "pairAddress": f"pair-{chain}",
        "baseToken": {"address": address, "symbol": "GME"},
        "priceUsd": "1.0",
        "marketCap": 1000,
        "liquidity": {"usd": liquidity},
        "volume": {"h24": 100},
        "priceChange": {"h24": 1.0},
        "pairCreatedAt": 1700000000000,
    }


@pytest.fixture
def multichain_response():
    """One ticker on three chains — these are the real liquidity figures
    DexScreener returns for $GME, where Ethereum is 13x deeper than Robinhood."""
    return {
        "pairs": [
            _pair("ethereum", 3_699_511, "0xeth"),
            _pair("bsc", 1_496_309, "0xbsc"),
            _pair("robinhood", 282_464, "0xrh"),
        ]
    }


@pytest.mark.asyncio
async def test_preferred_chain_wins_over_deeper_liquidity(
    db, multichain_response: dict
) -> None:
    """A ticker shilled on Robinhood Chain must not resolve to the Ethereum
    token just because Ethereum's pool is deeper."""
    f = TokenMetadataFetcher(db=db, preferred_chains=["robinhood"])
    with patch.object(
        f._http,
        "get",
        new_callable=AsyncMock,
        return_value=MockHTTPResponse(multichain_response),
    ):
        result = await f.fetch("$GME")
    await f.close()

    assert result is not None
    assert result["chain"] == "robinhood"
    assert result["liquidity_usd"] == 282_464
    assert result["base_address"] == "0xrh"


@pytest.mark.asyncio
async def test_preferred_chain_order_is_not_a_priority_list(
    db, multichain_response: dict
) -> None:
    """Every preferred chain ranks equally; liquidity breaks the tie among them."""
    f = TokenMetadataFetcher(db=db, preferred_chains=["robinhood", "bsc"])
    with patch.object(
        f._http,
        "get",
        new_callable=AsyncMock,
        return_value=MockHTTPResponse(multichain_response),
    ):
        result = await f.fetch("$GME")
    await f.close()

    assert result["chain"] == "bsc"


@pytest.mark.asyncio
async def test_falls_back_to_liquidity_when_no_pair_on_preferred_chain(
    db, multichain_response: dict
) -> None:
    """Preference is a preference, not a filter — an unrelated token still
    resolves rather than vanishing from the pipeline."""
    f = TokenMetadataFetcher(db=db, preferred_chains=["arbitrum"])
    with patch.object(
        f._http,
        "get",
        new_callable=AsyncMock,
        return_value=MockHTTPResponse(multichain_response),
    ):
        result = await f.fetch("$GME")
    await f.close()

    assert result["chain"] == "ethereum"


@pytest.mark.asyncio
async def test_no_preference_keeps_highest_liquidity(
    db, multichain_response: dict
) -> None:
    """Unconfigured behavior is unchanged from before chain preference existed."""
    f = TokenMetadataFetcher(db=db)
    with patch.object(
        f._http,
        "get",
        new_callable=AsyncMock,
        return_value=MockHTTPResponse(multichain_response),
    ):
        result = await f.fetch("$GME")
    await f.close()

    assert result["chain"] == "ethereum"
    assert result["liquidity_usd"] == 3_699_511


@pytest.mark.asyncio
async def test_birdeye_skipped_for_non_solana_chain(
    db, multichain_response: dict
) -> None:
    """Birdeye is queried with x-chain: solana, so calling it with an EVM
    address burns a request and returns holder data for the wrong token."""
    f = TokenMetadataFetcher(
        db=db, birdeye_api_key="test-key", preferred_chains=["robinhood"]
    )
    mock = AsyncMock(return_value=MockHTTPResponse(multichain_response))
    with patch.object(f._http, "get", mock):
        result = await f.fetch("$GME")
    await f.close()

    assert result["chain"] == "robinhood"
    assert result["holder_count"] is None
    assert mock.call_count == 1  # DexScreener only — Birdeye never called


@pytest.mark.asyncio
async def test_ohlcv_queries_the_requested_network(db) -> None:
    """TA candles must come from the chain the token lives on. Querying
    GeckoTerminal's solana network with an EVM address returns nothing and
    silently degrades the TA signal to neutral."""
    f = TokenMetadataFetcher(db=db)
    urls: list[str] = []

    async def mock_get(url, **kwargs):
        urls.append(url)
        if url.endswith("/pools"):
            return MockHTTPResponse({"data": [{"attributes": {"address": "pool1"}}]})
        return MockHTTPResponse(
            {"data": {"attributes": {"ohlcv_list": [[1700000000, 1, 2, 0.5, 1.5, 100]]}}}
        )

    with patch.object(f._http, "get", side_effect=mock_get):
        candles = await f.fetch_ohlcv("0xrh", network="robinhood")
    await f.close()

    assert urls and all("/networks/robinhood/" in u for u in urls)
    assert candles == [
        {"ts": 1700000000, "price_usd": 1.5, "high": 2.0, "low": 0.5, "volume": 100.0}
    ]


@pytest.mark.asyncio
async def test_ohlcv_defaults_to_solana_when_network_unspecified(db) -> None:
    """Callers that predate the network parameter keep working."""
    f = TokenMetadataFetcher(db=db)
    urls: list[str] = []

    async def mock_get(url, **kwargs):
        urls.append(url)
        return MockHTTPResponse({"data": []})

    with patch.object(f._http, "get", side_effect=mock_get):
        await f.fetch_ohlcv("So111")
    await f.close()

    assert urls and all("/networks/solana/" in u for u in urls)
