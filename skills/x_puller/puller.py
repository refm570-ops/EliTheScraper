from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import redis.asyncio as aioredis
import structlog
import yaml

log = structlog.get_logger()

X_API_BASE = "https://api.x.com/2"
REDIS_KEY_RAW = "signals:messages:raw"


class XFeedPuller:
    """Polls X/Twitter timelines via API v2 and pushes tweets to the shared Redis queue."""

    def __init__(
        self,
        bearer_token: str,
        redis: aioredis.Redis,
        accounts_config_path: str = "config/x_accounts.yml",
    ) -> None:
        self._bearer_token = bearer_token
        self._redis = redis
        self._accounts_config_path = accounts_config_path
        self._client: httpx.AsyncClient | None = None
        # username -> {user_id, category, priority}
        self._accounts: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Load accounts config, create HTTP client, resolve usernames to IDs."""
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._bearer_token}"},
            timeout=30.0,
        )
        accounts = self._load_accounts(self._accounts_config_path)
        if not accounts:
            log.warning("x_puller.no_accounts_configured")
            return

        await self._resolve_user_ids(accounts)
        log.info("x_puller.initialized", accounts=len(self._accounts))

    @staticmethod
    def _load_accounts(path: str) -> list[dict[str, Any]]:
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
            return config.get("accounts", [])
        except FileNotFoundError:
            log.warning("x_puller.config_not_found", path=path)
            return []

    async def _resolve_user_ids(self, accounts: list[dict[str, Any]]) -> None:
        """Resolve usernames to X user IDs via GET /2/users/by."""
        usernames = [a["username"] for a in accounts if a.get("username")]
        if not usernames:
            return

        # API supports up to 100 usernames per request
        url = f"{X_API_BASE}/users/by"
        params = {"usernames": ",".join(usernames)}

        try:
            resp = await self._client.get(url, params=params)
            if resp.status_code == 429:
                log.warning("x_puller.rate_limited_on_init")
                return
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, Exception):
            log.error("x_puller.resolve_error", exc_info=True)
            return

        # Build lookup: username -> user_id
        id_map: dict[str, str] = {}
        for user in data.get("data", []):
            id_map[user["username"].lower()] = user["id"]

        for acct in accounts:
            username = acct["username"].lower()
            user_id = id_map.get(username)
            if user_id:
                self._accounts[username] = {
                    "user_id": user_id,
                    "category": acct.get("category", ""),
                    "priority": acct.get("priority", "medium"),
                }
            else:
                log.warning("x_puller.user_not_found", username=username)

    async def poll(self) -> int:
        """Poll all configured accounts for new tweets.

        Returns number of tweets pushed to the queue.
        """
        if not self._client or not self._accounts:
            return 0

        total = 0
        for username, info in self._accounts.items():
            try:
                count = await self._poll_account(username, info)
                total += count
            except Exception:
                log.error("x_puller.poll_error", username=username, exc_info=True)

        if total > 0:
            log.info("x_puller.poll_complete", tweets=total)
        return total

    async def _poll_account(self, username: str, info: dict[str, Any]) -> int:
        """Poll a single account for new tweets."""
        user_id = info["user_id"]
        since_key = f"x:since_id:{user_id}"

        # Get since_id from Redis
        since_id = await self._redis.get(since_key)

        url = f"{X_API_BASE}/users/{user_id}/tweets"
        params: dict[str, str] = {
            "exclude": "retweets,replies",
            "max_results": "10",
            "tweet.fields": "created_at,text,public_metrics",
            "expansions": "author_id",
            "user.fields": "public_metrics",
        }
        if since_id:
            params["since_id"] = since_id

        try:
            resp = await self._client.get(url, params=params)
        except httpx.HTTPError:
            log.error("x_puller.http_error", username=username, exc_info=True)
            return 0

        if resp.status_code == 429:
            log.warning("x_puller.rate_limited", username=username)
            return 0

        if resp.status_code != 200:
            log.warning(
                "x_puller.api_error",
                username=username,
                status=resp.status_code,
            )
            return 0

        data = resp.json()
        tweets = data.get("data", [])
        if not tweets:
            return 0

        # Parse author follower counts from expansions (may be absent on free tier)
        author_followers = self._parse_author_followers(data)

        # Update since_id to the newest tweet (first in the list)
        newest_id = tweets[0]["id"]
        await self._redis.set(since_key, newest_id)

        # Convert tweets to shared message schema and push to Redis
        count = 0
        for tweet in tweets:
            author_id = tweet.get("author_id", user_id)
            follower_count = author_followers.get(author_id)
            message = self._tweet_to_message(
                tweet, username, user_id, follower_count
            )
            await self._redis.rpush(REDIS_KEY_RAW, json.dumps(message))
            count += 1

        log.debug(
            "x_puller.account_polled",
            username=username,
            new_tweets=count,
        )
        return count

    @staticmethod
    def _parse_author_followers(data: dict[str, Any]) -> dict[str, int]:
        """Extract author_id → follower_count from API expansion includes."""
        followers: dict[str, int] = {}
        includes = data.get("includes", {})
        for user in includes.get("users", []):
            uid = user.get("id")
            pm = user.get("public_metrics", {})
            fc = pm.get("followers_count")
            if uid and fc is not None:
                followers[uid] = fc
        return followers

    @staticmethod
    def _tweet_to_message(
        tweet: dict[str, Any],
        username: str,
        user_id: str,
        author_followers: int | None = None,
    ) -> dict[str, Any]:
        """Convert an X API tweet object to the shared message schema."""
        pm = tweet.get("public_metrics", {})
        engagement = {
            "likes": pm.get("like_count", 0),
            "retweets": pm.get("retweet_count", 0),
            "replies": pm.get("reply_count", 0),
            "quotes": pm.get("quote_count", 0),
        }
        return {
            "source": "twitter",
            "group_id": user_id,
            "group_name": f"@{username}",
            "message_id": tweet["id"],
            "sender_id": user_id,
            "sender_name": username,
            "text": tweet.get("text", ""),
            "timestamp": tweet.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            "has_media": False,
            "reply_to": None,
            "engagement": engagement,
            "author_followers": author_followers,
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            log.info("x_puller.closed")
