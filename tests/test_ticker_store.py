from __future__ import annotations

import time

import pytest

from storage.db import Database
from storage.ticker_store import TickerStore
from storage.alert_log import AlertLog


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
async def alert_log(db):
    return AlertLog(db=db)


@pytest.mark.asyncio
async def test_record_and_query_mention(ticker_store: TickerStore) -> None:
    """Test recording a mention and querying social metrics."""
    await ticker_store.record_mention(
        ticker="$MONKE",
        intent="TICKER_CALL",
        conviction="STRONG",
        context="Aped in",
        group_id=-1001234567890,
        group_name="Alpha Calls",
        sender_id=123,
        message_id=1001,
        raw_text="just aped $MONKE",
    )

    metrics = await ticker_store.get_social_metrics("$MONKE", window_hours=1.0)
    assert metrics["mention_count"] == 1
    assert metrics["unique_groups"] == 1
    assert "Alpha Calls" in metrics["group_names"]
    assert metrics["convictions"]["STRONG"] == 1


@pytest.mark.asyncio
async def test_multiple_groups(ticker_store: TickerStore) -> None:
    """Test that metrics track multiple unique groups."""
    for i, group in enumerate(["Alpha Calls", "Degen Chat", "Alpha Calls"]):
        await ticker_store.record_mention(
            ticker="$BONK",
            intent="TICKER_CALL",
            conviction="MODERATE",
            context=f"mention {i}",
            group_id=-100123456789 + i,
            group_name=group,
            sender_id=100 + i,
            message_id=2000 + i,
            raw_text=f"test {i}",
        )

    metrics = await ticker_store.get_social_metrics("$BONK", window_hours=1.0)
    assert metrics["mention_count"] == 3
    assert metrics["unique_groups"] == 2  # Alpha Calls counted once


@pytest.mark.asyncio
async def test_ticker_normalization(ticker_store: TickerStore) -> None:
    """Test that ticker lookup is case-insensitive."""
    await ticker_store.record_mention(
        ticker="$monke",
        intent="TICKER_CALL",
        conviction="STRONG",
        context="test",
        group_id=None,
        group_name="Test",
        sender_id=None,
        message_id=None,
        raw_text=None,
    )

    metrics = await ticker_store.get_social_metrics("$MONKE", window_hours=1.0)
    assert metrics["mention_count"] == 1


@pytest.mark.asyncio
async def test_empty_metrics(ticker_store: TickerStore) -> None:
    """Test querying metrics for a ticker with no mentions."""
    metrics = await ticker_store.get_social_metrics("$NOBODY", window_hours=1.0)
    assert metrics["mention_count"] == 0
    assert metrics["unique_groups"] == 0
    assert metrics["group_names"] == []
    assert metrics["first_seen"] is None
    assert metrics["sources"] == []
    assert metrics["unique_sources"] == 0


@pytest.mark.asyncio
async def test_cleanup(ticker_store: TickerStore) -> None:
    """Test that cleanup removes old mentions."""
    await ticker_store.record_mention(
        ticker="$OLD",
        intent="TICKER_CALL",
        conviction="WEAK",
        context="old",
        group_id=None,
        group_name="Test",
        sender_id=None,
        message_id=None,
        raw_text=None,
    )

    # Force the created_at to be old
    await ticker_store._db.conn.execute(
        "UPDATE ticker_mentions SET created_at = ?",
        (time.time() - 200000,),
    )
    await ticker_store._db.conn.commit()

    deleted = await ticker_store.cleanup(max_age_hours=48.0)
    assert deleted == 1

    metrics = await ticker_store.get_social_metrics("$OLD", window_hours=999.0)
    assert metrics["mention_count"] == 0


# --- Source column tests (Phase 3) ---


@pytest.mark.asyncio
async def test_record_mention_with_source(ticker_store: TickerStore) -> None:
    """Test recording mentions with different sources."""
    await ticker_store.record_mention(
        ticker="$MONKE",
        intent="TICKER_CALL",
        conviction="STRONG",
        context="TG alpha",
        group_id=-100123,
        group_name="Alpha Calls",
        sender_id=123,
        message_id=1001,
        raw_text="aped $MONKE",
        source="telegram",
    )
    await ticker_store.record_mention(
        ticker="$MONKE",
        intent="TICKER_CALL",
        conviction="MODERATE",
        context="X mention",
        group_id=999,
        group_name="@cryptoalpha",
        sender_id=999,
        message_id=2001,
        raw_text="$MONKE looking bullish",
        source="twitter",
    )

    metrics = await ticker_store.get_social_metrics("$MONKE", window_hours=1.0)
    assert metrics["mention_count"] == 2
    assert metrics["unique_sources"] == 2
    assert "telegram" in metrics["sources"]
    assert "twitter" in metrics["sources"]


@pytest.mark.asyncio
async def test_default_source_is_telegram(ticker_store: TickerStore) -> None:
    """Test that the default source is 'telegram'."""
    await ticker_store.record_mention(
        ticker="$BONK",
        intent="TICKER_CALL",
        conviction="STRONG",
        context="test",
        group_id=None,
        group_name="Test",
        sender_id=None,
        message_id=None,
        raw_text=None,
    )

    metrics = await ticker_store.get_social_metrics("$BONK", window_hours=1.0)
    assert metrics["sources"] == ["telegram"]
    assert metrics["unique_sources"] == 1


@pytest.mark.asyncio
async def test_get_mentions_by_source(ticker_store: TickerStore) -> None:
    """Test querying mentions filtered by source."""
    await ticker_store.record_mention(
        ticker="$MONKE",
        intent="TICKER_CALL",
        conviction="STRONG",
        context="TG alpha",
        group_id=-100123,
        group_name="Alpha Calls",
        sender_id=123,
        message_id=1001,
        raw_text="aped $MONKE",
        source="telegram",
    )
    await ticker_store.record_mention(
        ticker="$MONKE",
        intent="TICKER_CALL",
        conviction="MODERATE",
        context="X mention",
        group_id=999,
        group_name="@cryptoalpha",
        sender_id=999,
        message_id=2001,
        raw_text="$MONKE chart breakout",
        source="twitter",
    )

    tg_mentions = await ticker_store.get_mentions_by_source("$MONKE", "telegram", window_hours=1.0)
    assert len(tg_mentions) == 1
    assert tg_mentions[0]["group_name"] == "Alpha Calls"

    tw_mentions = await ticker_store.get_mentions_by_source("$MONKE", "twitter", window_hours=1.0)
    assert len(tw_mentions) == 1
    assert tw_mentions[0]["group_name"] == "@cryptoalpha"

    empty = await ticker_store.get_mentions_by_source("$MONKE", "discord", window_hours=1.0)
    assert len(empty) == 0


# --- Alert log tests ---


@pytest.mark.asyncio
async def test_alert_log_record_and_check(alert_log: AlertLog) -> None:
    """Test alert logging and recently-alerted check."""
    assert not await alert_log.was_recently_alerted("$MONKE")

    await alert_log.record(
        ticker="$MONKE",
        alert_level="interesting",
        notify_mode="normal",
        score=15.0,
        summary="test summary",
    )

    assert await alert_log.was_recently_alerted("$MONKE")
    assert not await alert_log.was_recently_alerted("$OTHER")


@pytest.mark.asyncio
async def test_alert_log_cooldown_expired(alert_log: AlertLog) -> None:
    """Test that alerts outside the cooldown window are not flagged."""
    await alert_log.record(
        ticker="$EXPIRED",
        alert_level="watch",
        notify_mode="silent",
    )

    # Force the created_at to be old
    await alert_log._db.conn.execute(
        "UPDATE alert_log SET created_at = ?",
        (time.time() - 3600,),
    )
    await alert_log._db.conn.commit()

    assert not await alert_log.was_recently_alerted("$EXPIRED", cooldown_seconds=1800.0)
