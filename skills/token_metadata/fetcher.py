from __future__ import annotations

import json
import time
from typing import Any

import httpx
import structlog

from storage.db import Database

log = structlog.get_logger()

DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search"
BIRDEYE_TOKEN_OVERVIEW_URL = "https://public-api.birdeye.so/defi/token_overview"


class TokenMetadataFetcher:
    """Fetches on-chain token metadata from DexScreener and Birdeye.

    DexScreener (free, no key): price, mcap, liquidity, volume, age.
    Birdeye (API key): holder count, top-10 holder %.
    Results are cached in SQLite with configurable TTL.
    """

    def __init__(
        self,
        db: Database,
        birdeye_api_key: str | None = None,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        self._db = db
        self._birdeye_api_key = birdeye_api_key
        self._cache_ttl = cache_ttl_seconds
        self._http = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def fetch(self, ticker: str) -> dict[str, Any] | None:
        """Fetch metadata for a ticker. Returns None on failure."""
        normalized = ticker.upper().lstrip("$")

        # Check cache
        cached = await self._get_cached(normalized)
        if cached is not None:
            return cached

        # Fetch from DexScreener
        data = await self._fetch_dexscreener(normalized)
        if data is None:
            return None

        # Enrich with Birdeye holder data if we have a Solana address
        pair_address = data.get("pair_address")
        base_address = data.get("base_address")
        if self._birdeye_api_key and base_address:
            holder_data = await self._fetch_birdeye(base_address)
            if holder_data:
                data.update(holder_data)

        # Cache result
        await self._set_cached(normalized, data)
        return data

    async def _fetch_dexscreener(self, query: str) -> dict[str, Any] | None:
        """Search DexScreener for token data."""
        try:
            resp = await self._http.get(
                DEXSCREENER_SEARCH_URL, params={"q": query}
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception:
            log.warning("token_metadata.dexscreener_error", query=query, exc_info=True)
            return None

        pairs = body.get("pairs") or []
        if not pairs:
            log.debug("token_metadata.dexscreener_no_pairs", query=query)
            return None

        # Pick the highest-liquidity pair
        best = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0)

        price_usd = best.get("priceUsd")
        mcap = best.get("marketCap") or best.get("fdv")
        liquidity = (best.get("liquidity") or {}).get("usd")
        volume_24h = (best.get("volume") or {}).get("h24")
        price_change_24h = (best.get("priceChange") or {}).get("h24")
        pair_created = best.get("pairCreatedAt")  # ms epoch
        base_address = (best.get("baseToken") or {}).get("address")
        pair_address = best.get("pairAddress")
        chain = best.get("chainId")
        dex_url = best.get("url")

        age_days = None
        if pair_created:
            age_days = (time.time() - pair_created / 1000) / 86400

        return {
            "source": "dexscreener",
            "price_usd": float(price_usd) if price_usd else None,
            "market_cap": float(mcap) if mcap else None,
            "liquidity_usd": float(liquidity) if liquidity else None,
            "volume_24h": float(volume_24h) if volume_24h else None,
            "price_change_24h": float(price_change_24h) if price_change_24h else None,
            "age_days": round(age_days, 1) if age_days else None,
            "base_address": base_address,
            "pair_address": pair_address,
            "chain": chain,
            "dex_url": dex_url,
            "holder_count": None,
            "top10_holder_pct": None,
        }

    async def _fetch_birdeye(self, token_address: str) -> dict[str, Any] | None:
        """Fetch holder data from Birdeye."""
        try:
            resp = await self._http.get(
                BIRDEYE_TOKEN_OVERVIEW_URL,
                params={"address": token_address},
                headers={
                    "X-API-KEY": self._birdeye_api_key,
                    "x-chain": "solana",
                },
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception:
            log.warning(
                "token_metadata.birdeye_error",
                address=token_address,
                exc_info=True,
            )
            return None

        data = body.get("data") or {}
        holder_count = data.get("holder")
        # Birdeye doesn't always provide top10 %, but some endpoints do
        top10_pct = data.get("top10HolderPercent")

        if holder_count is None:
            return None

        return {
            "holder_count": holder_count,
            "top10_holder_pct": float(top10_pct) if top10_pct is not None else None,
        }

    async def _get_cached(self, ticker: str) -> dict[str, Any] | None:
        """Get cached metadata if not expired."""
        cursor = await self._db.conn.execute(
            "SELECT data, fetched_at FROM token_metadata_cache WHERE ticker = ?",
            (ticker,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        age = time.time() - row["fetched_at"]
        if age > self._cache_ttl:
            return None

        try:
            return json.loads(row["data"])
        except json.JSONDecodeError:
            return None

    async def _set_cached(self, ticker: str, data: dict[str, Any]) -> None:
        """Write metadata to cache."""
        await self._db.conn.execute(
            """INSERT OR REPLACE INTO token_metadata_cache (ticker, data, fetched_at)
               VALUES (?, ?, ?)""",
            (ticker, json.dumps(data), time.time()),
        )
        await self._db.conn.commit()
