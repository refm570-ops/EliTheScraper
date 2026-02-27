from __future__ import annotations

import pytest

from skills.social_metrics.counter import SocialMetricsCounter
from storage.db import Database
from storage.ticker_store import TickerStore


@pytest.fixture
async def db():
    database = Database(db_path=":memory:")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def ticker_store(db):
    return TickerStore(db=db)


@pytest.fixture
async def counter(ticker_store):
    return SocialMetricsCounter(ticker_store=ticker_store)


@pytest.mark.asyncio
async def test_get_metrics_empty(counter: SocialMetricsCounter) -> None:
    """Test metrics for a ticker with no mentions."""
    metrics = await counter.get_metrics("$NOBODY")
    assert metrics["mention_count"] == 0
    assert metrics["unique_groups"] == 0


@pytest.mark.asyncio
async def test_get_metrics_with_data(
    counter: SocialMetricsCounter, ticker_store: TickerStore
) -> None:
    """Test metrics after recording mentions."""
    for group in ["Alpha Calls", "Degen Chat"]:
        await ticker_store.record_mention(
            ticker="$MONKE",
            intent="TICKER_CALL",
            conviction="STRONG",
            context="test",
            group_id=None,
            group_name=group,
            sender_id=None,
            message_id=None,
            raw_text=None,
        )

    metrics = await counter.get_metrics("$MONKE")
    assert metrics["mention_count"] == 2
    assert metrics["unique_groups"] == 2


@pytest.mark.asyncio
async def test_meets_group_threshold(counter: SocialMetricsCounter) -> None:
    """Test group threshold check."""
    metrics = {"unique_groups": 3}
    assert counter.meets_group_threshold(metrics, min_groups=2)
    assert counter.meets_group_threshold(metrics, min_groups=3)
    assert not counter.meets_group_threshold(metrics, min_groups=4)


@pytest.mark.asyncio
async def test_meets_group_threshold_zero(counter: SocialMetricsCounter) -> None:
    """Test threshold with zero groups."""
    metrics = {"unique_groups": 0}
    assert not counter.meets_group_threshold(metrics, min_groups=1)
