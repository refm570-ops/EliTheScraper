"""Opportunity feeds for the paper-trading harness.

These replace the live Telegram/X source so trade QUALITY can be measured
without any Telegram credentials — the same downstream pipeline (safety →
evaluator → risk → paper execution → monitor) runs on each opportunity.

- DexScreenerFeed: live Solana tokens currently being promoted/boosted (a
  reasonable proxy for "tokens people are calling"). No API key required.
- MintListFeed: a fixed list of mints (for backtesting known winners/ruggers).
- DemoFeed: synthetic tokens for offline harness validation.
"""

from __future__ import annotations

import random
from typing import Any

import httpx
import structlog

from skills.token_metadata.fetcher import TokenMetadataFetcher
from trading.models import Chain, Opportunity, TokenVenue

log = structlog.get_logger()

DEX_BOOSTS_URL = "https://api.dexscreener.com/token-boosts/latest/v1"
DEX_PROFILES_URL = "https://api.dexscreener.com/token-profiles/latest/v1"


def detect_venue(metadata: dict[str, Any]) -> TokenVenue:
    dex_id = str(metadata.get("dex_id") or "").lower()
    labels = [str(x).lower() for x in (metadata.get("labels") or [])]
    if "pump" in dex_id or any("bonding" in x for x in labels):
        return TokenVenue.BONDING
    if dex_id:
        return TokenVenue.AMM
    return TokenVenue.UNKNOWN


class DexScreenerFeed:
    """Live Solana tokens from DexScreener's boosted/profiled lists."""

    def __init__(self, fetcher: TokenMetadataFetcher, min_liquidity_usd: float = 5000.0) -> None:
        self._fetcher = fetcher
        self._min_liq = min_liquidity_usd
        self._http = httpx.AsyncClient(timeout=15.0)
        self._seen: set[str] = set()

    async def close(self) -> None:
        await self._http.aclose()

    async def _candidate_mints(self) -> list[str]:
        mints: list[str] = []
        for url in (DEX_BOOSTS_URL, DEX_PROFILES_URL):
            try:
                resp = await self._http.get(url)
                resp.raise_for_status()
                for item in resp.json() or []:
                    if str(item.get("chainId")).lower() == "solana":
                        addr = item.get("tokenAddress")
                        if addr:
                            mints.append(addr)
            except Exception:
                log.warning("feed.dexscreener_list_error", url=url, exc_info=True)
        return mints

    async def poll(self) -> list[Opportunity]:
        opportunities: list[Opportunity] = []
        for mint in await self._candidate_mints():
            if mint in self._seen:
                continue
            self._seen.add(mint)
            meta = await self._fetcher.fetch_by_address(mint)
            if not meta:
                continue
            if (meta.get("liquidity_usd") or 0) < self._min_liq:
                continue
            opportunities.append(Opportunity(
                ticker=(meta.get("base_address") or mint)[:8],
                address=mint,
                chain=Chain.SOLANA,
                venue=detect_venue(meta),
                source="dexscreener_boost",
                metadata=meta,
            ))
        log.info("feed.dexscreener_poll", new_opportunities=len(opportunities))
        return opportunities


class MintListFeed:
    """Yields a fixed set of mints once (backtesting known outcomes)."""

    def __init__(self, fetcher: TokenMetadataFetcher, mints: list[str]) -> None:
        self._fetcher = fetcher
        self._mints = mints
        self._done = False

    async def close(self) -> None:
        return None

    async def poll(self) -> list[Opportunity]:
        if self._done:
            return []
        self._done = True
        out: list[Opportunity] = []
        for mint in self._mints:
            meta = await self._fetcher.fetch_by_address(mint)
            out.append(Opportunity(
                ticker=mint[:8], address=mint, chain=Chain.SOLANA,
                venue=detect_venue(meta or {}), source="backtest",
                metadata=meta or {},
            ))
        return out


class DemoFeed:
    """Synthetic tokens for offline harness validation (no network)."""

    def __init__(self, per_cycle: int = 3, max_total: int = 12, seed: int = 7) -> None:
        self._per_cycle = per_cycle
        self._max_total = max_total
        self._emitted = 0
        self._rng = random.Random(seed)

    async def close(self) -> None:
        return None

    async def poll(self) -> list[Opportunity]:
        out: list[Opportunity] = []
        for _ in range(self._per_cycle):
            if self._emitted >= self._max_total:
                break
            self._emitted += 1
            addr = f"DEMOmint{self._emitted:03d}"
            price = round(self._rng.uniform(0.0001, 0.01), 6)
            out.append(Opportunity(
                ticker=f"DEMO{self._emitted}", address=addr, chain=Chain.SOLANA,
                venue=TokenVenue.AMM, source="demo",
                metadata={"price_usd": price, "liquidity_usd": self._rng.uniform(8000, 120000),
                          "market_cap": self._rng.uniform(20000, 900000), "dex_id": "raydium"},
            ))
        return out
