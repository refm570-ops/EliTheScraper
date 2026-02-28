from __future__ import annotations

import json
import time
from typing import Any

import structlog

from storage.db import Database

log = structlog.get_logger()


class TickerStore:
    """Records ticker mentions and queries social metrics from SQLite."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record_mention(
        self,
        ticker: str,
        intent: str,
        conviction: str | None,
        context: str | None,
        group_id: int | None,
        group_name: str | None,
        sender_id: int | None,
        message_id: int | None,
        raw_text: str | None,
        source: str = "telegram",
        engagement_data: dict | None = None,
    ) -> None:
        """Insert a single ticker mention."""
        engagement_json = json.dumps(engagement_data) if engagement_data else None
        await self._db.conn.execute(
            """INSERT INTO ticker_mentions
               (ticker, intent, conviction, context, group_id, group_name,
                sender_id, message_id, raw_text, source, engagement_data, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker.upper(),
                intent,
                conviction,
                context,
                group_id,
                group_name,
                sender_id,
                message_id,
                raw_text,
                source,
                engagement_json,
                time.time(),
            ),
        )
        await self._db.conn.commit()

    async def get_social_metrics(
        self, ticker: str, window_hours: float = 6.0
    ) -> dict[str, Any]:
        """Query social metrics for a ticker within a time window.

        Returns: mention_count, unique_groups, group_names, conviction breakdown,
        first_seen, last_seen, sources, unique_sources.
        """
        cutoff = time.time() - (window_hours * 3600)
        normalized = ticker.upper()

        cursor = await self._db.conn.execute(
            """SELECT group_name, conviction, source, created_at
               FROM ticker_mentions
               WHERE ticker = ? AND created_at >= ?
               ORDER BY created_at ASC""",
            (normalized, cutoff),
        )
        rows = await cursor.fetchall()

        if not rows:
            return {
                "mention_count": 0,
                "unique_groups": 0,
                "group_names": [],
                "convictions": {},
                "first_seen": None,
                "last_seen": None,
                "sources": [],
                "unique_sources": 0,
            }

        groups: set[str] = set()
        sources: set[str] = set()
        convictions: dict[str, int] = {}
        first_seen = rows[0]["created_at"]
        last_seen = rows[-1]["created_at"]

        for row in rows:
            gn = row["group_name"]
            if gn:
                groups.add(gn)
            src = row["source"]
            if src:
                sources.add(src)
            conv = row["conviction"]
            if conv:
                convictions[conv] = convictions.get(conv, 0) + 1

        return {
            "mention_count": len(rows),
            "unique_groups": len(groups),
            "group_names": sorted(groups),
            "convictions": convictions,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "sources": sorted(sources),
            "unique_sources": len(sources),
        }

    async def get_mentions_by_source(
        self, ticker: str, source: str, window_hours: float = 6.0
    ) -> list[dict[str, Any]]:
        """Get mention details for a specific ticker+source within a time window."""
        cutoff = time.time() - (window_hours * 3600)
        normalized = ticker.upper()

        cursor = await self._db.conn.execute(
            """SELECT group_name, conviction, context, raw_text, engagement_data, created_at
               FROM ticker_mentions
               WHERE ticker = ? AND source = ? AND created_at >= ?
               ORDER BY created_at ASC
               LIMIT 10""",
            (normalized, source, cutoff),
        )
        rows = await cursor.fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            eng_raw = row["engagement_data"]
            engagement = json.loads(eng_raw) if eng_raw else None
            results.append(
                {
                    "group_name": row["group_name"],
                    "conviction": row["conviction"],
                    "context": row["context"],
                    "raw_text": row["raw_text"],
                    "engagement_data": engagement,
                    "created_at": row["created_at"],
                }
            )
        return results

    async def get_x_engagement_summary(
        self, ticker: str, window_hours: float = 6.0
    ) -> dict[str, Any]:
        """Aggregate engagement stats for X/Twitter mentions of a ticker."""
        cutoff = time.time() - (window_hours * 3600)
        normalized = ticker.upper()

        cursor = await self._db.conn.execute(
            """SELECT engagement_data
               FROM ticker_mentions
               WHERE ticker = ? AND source = 'twitter' AND created_at >= ?
               ORDER BY created_at ASC""",
            (normalized, cutoff),
        )
        rows = await cursor.fetchall()

        total_likes = 0
        total_retweets = 0
        total_replies = 0
        total_quotes = 0
        follower_counts: list[int] = []
        tweet_count = 0

        for row in rows:
            raw = row["engagement_data"]
            if not raw:
                tweet_count += 1
                continue
            data = json.loads(raw)
            total_likes += data.get("likes", 0)
            total_retweets += data.get("retweets", 0)
            total_replies += data.get("replies", 0)
            total_quotes += data.get("quotes", 0)
            af = data.get("author_followers")
            if af is not None:
                follower_counts.append(af)
            tweet_count += 1

        return {
            "total_likes": total_likes,
            "total_retweets": total_retweets,
            "total_replies": total_replies,
            "total_quotes": total_quotes,
            "max_author_followers": max(follower_counts) if follower_counts else None,
            "avg_author_followers": (
                sum(follower_counts) / len(follower_counts) if follower_counts else None
            ),
            "tweet_count": tweet_count,
        }

    async def cleanup(self, max_age_hours: float = 48.0) -> int:
        """Delete mentions older than max_age_hours. Returns deleted count."""
        cutoff = time.time() - (max_age_hours * 3600)
        cursor = await self._db.conn.execute(
            "DELETE FROM ticker_mentions WHERE created_at < ?", (cutoff,)
        )
        await self._db.conn.commit()
        deleted = cursor.rowcount
        if deleted:
            log.info("ticker_store.cleanup", deleted=deleted)
        return deleted
