from __future__ import annotations

import asyncio
import signal
import time
from typing import Any

import structlog

from agents.aggregator.agent import SignalAggregator
from agents.classifier.agent import MessageClassifier
from agents.cross_ref.agent import CrossReferenceAnalyst
from agents.scorer.agent import FundamentalsScorer
from agents.x_sentiment.agent import XSentimentAnalyzer
from agents.x_sentiment.scorer import score_x_sentiment
from pipeline.buffer import MessageBuffer
from skills.alert_dispatcher.bot import AlertDispatcher
from skills.alert_dispatcher.templates import format_phase2_alert
from skills.social_metrics.counter import SocialMetricsCounter
from skills.token_metadata.fetcher import TokenMetadataFetcher
from storage.alert_log import AlertLog
from storage.ticker_store import TickerStore
from trading.engine import TradingEngine
from trading.models import Chain, Opportunity, TokenVenue

log = structlog.get_logger()

# Alert levels that are eligible for an autonomous buy evaluation.
BUY_GRADE_LEVELS = ("act_now", "interesting")


class Orchestrator:
    """Main pipeline loop: buffer → classifier → enrich → score → aggregate → dispatch.

    Phase 2+3 flow per cycle:
    1. Pop batch + classify (existing)
    2. Filter NOISE (existing)
    3. Record all actionable mentions to TickerStore (with source)
    4. For each unique TICKER_CALL ticker:
       - Fetch metadata (cached)
       - Get social metrics (6h window)
       - Cross-ref analysis if multi-platform (Phase 3)
       - Score fundamentals
       - Aggregate → alert decision (with cross-ref result)
       - If not suppressed: rate limit check → format → enqueue → log
    """

    def __init__(
        self,
        buffer: MessageBuffer,
        classifier: MessageClassifier,
        dispatcher: AlertDispatcher,
        ticker_store: TickerStore,
        alert_log: AlertLog,
        metadata_fetcher: TokenMetadataFetcher,
        social_counter: SocialMetricsCounter,
        scorer: FundamentalsScorer,
        aggregator: SignalAggregator,
        cross_ref: CrossReferenceAnalyst | None = None,
        x_sentiment_analyzer: XSentimentAnalyzer | None = None,
        trading_engine: TradingEngine | None = None,
        ta_analyzer: Any = None,
        batch_size: int = 50,
        batch_interval: float = 300.0,  # 5 minutes
        heartbeat_interval: float = 60.0,
    ) -> None:
        self._buffer = buffer
        self._classifier = classifier
        self._dispatcher = dispatcher
        self._ticker_store = ticker_store
        self._alert_log = alert_log
        self._metadata_fetcher = metadata_fetcher
        self._social_counter = social_counter
        self._scorer = scorer
        self._aggregator = aggregator
        self._cross_ref = cross_ref
        self._x_sentiment_analyzer = x_sentiment_analyzer
        self._trading_engine = trading_engine
        self._ta_analyzer = ta_analyzer
        self._batch_size = batch_size
        self._batch_interval = batch_interval
        self._heartbeat_interval = heartbeat_interval
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._cycle_count = 0

    async def run(self) -> None:
        """Main loop — runs until shutdown signal."""
        self._running = True

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_shutdown)

        log.info(
            "orchestrator.started",
            batch_size=self._batch_size,
            interval=self._batch_interval,
        )

        # Start heartbeat
        heartbeat_task = asyncio.create_task(self._heartbeat())

        try:
            while self._running:
                await self._process_cycle()
        except asyncio.CancelledError:
            pass
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            # Drain remaining messages before exit
            await self._drain()
            log.info("orchestrator.stopped", total_cycles=self._cycle_count)

    def _request_shutdown(self) -> None:
        log.info("orchestrator.shutdown_requested")
        self._running = False
        self._shutdown_event.set()

    async def _process_cycle(self) -> None:
        """Single processing cycle: pop → classify → enrich → score → aggregate → dispatch."""
        self._cycle_count += 1
        cycle_start = time.monotonic()

        try:
            batch = await self._buffer.pop_batch(
                max_size=self._batch_size,
                max_wait_seconds=self._batch_interval,
            )

            if not batch:
                log.debug("orchestrator.empty_batch", cycle=self._cycle_count)
                return

            log.info(
                "orchestrator.processing_batch",
                cycle=self._cycle_count,
                batch_size=len(batch),
            )

            # Step 1: Classify
            classified = await self._classifier.classify_batch(batch)

            # Step 2: Filter NOISE
            actionable = [
                msg for msg in classified if msg.get("intent") != "NOISE"
            ]

            if not actionable:
                log.info(
                    "orchestrator.all_noise",
                    cycle=self._cycle_count,
                    batch_size=len(batch),
                )
                return

            # Build raw message lookup
            raw_by_id: dict[Any, dict[str, Any]] = {}
            for msg in batch:
                raw_by_id[msg.get("message_id")] = msg

            # Step 3: Record all actionable mentions to TickerStore (with source + engagement)
            for cls in actionable:
                ticker = cls.get("ticker")
                if not ticker:
                    continue
                raw_msg = raw_by_id.get(cls.get("id"), {})
                source = raw_msg.get("source", "telegram")
                engagement = raw_msg.get("engagement")
                author_followers = raw_msg.get("author_followers")
                engagement_data = None
                if engagement:
                    engagement_data = {**engagement, "author_followers": author_followers}
                await self._ticker_store.record_mention(
                    ticker=ticker,
                    intent=cls.get("intent", ""),
                    conviction=cls.get("conviction"),
                    context=cls.get("context"),
                    group_id=raw_msg.get("group_id"),
                    group_name=raw_msg.get("group_name"),
                    sender_id=raw_msg.get("sender_id"),
                    message_id=raw_msg.get("message_id"),
                    raw_text=raw_msg.get("text"),
                    source=source,
                    engagement_data=engagement_data,
                )

            # Step 4: Process unique TICKER_CALL tickers
            ticker_calls = [
                cls for cls in actionable
                if cls.get("intent") == "TICKER_CALL" and cls.get("ticker")
            ]
            seen_tickers: set[str] = set()
            alerts_sent = 0

            for cls in ticker_calls:
                ticker = cls["ticker"].upper()
                if ticker in seen_tickers:
                    continue
                seen_tickers.add(ticker)

                try:
                    alerts_sent += await self._process_ticker(ticker)
                except Exception:
                    log.error(
                        "orchestrator.ticker_error",
                        ticker=ticker,
                        exc_info=True,
                    )

            # Also dispatch non-TICKER_CALL actionable via Phase 1 path
            non_ticker_calls = [
                cls for cls in actionable
                if cls.get("intent") != "TICKER_CALL"
            ]
            if non_ticker_calls:
                await self._dispatcher.dispatch_batch(non_ticker_calls, batch)

            log.info(
                "orchestrator.batch_dispatched",
                cycle=self._cycle_count,
                total=len(batch),
                actionable=len(actionable),
                ticker_calls=len(seen_tickers),
                alerts_sent=alerts_sent,
                noise=len(batch) - len(actionable),
                duration=round(time.monotonic() - cycle_start, 2),
            )
        except Exception:
            log.error(
                "orchestrator.cycle_error",
                cycle=self._cycle_count,
                exc_info=True,
            )

    async def _process_ticker(self, ticker: str) -> int:
        """Process a single ticker through the Phase 2+3 enrichment pipeline.

        Returns 1 if an alert was sent, 0 otherwise.
        """
        # Rate limit check (survives restarts via SQLite)
        if await self._alert_log.was_recently_alerted(ticker):
            log.debug("orchestrator.ticker_recently_alerted", ticker=ticker)
            return 0

        # Fetch metadata (cached)
        metadata = await self._metadata_fetcher.fetch(ticker)

        # Get social metrics (6h window)
        social_metrics = await self._social_counter.get_metrics(ticker, window_hours=6.0)

        # Cross-reference analysis (Phase 3)
        cross_ref_result = None
        if self._cross_ref and social_metrics.get("unique_sources", 0) >= 2:
            tg_mentions = await self._ticker_store.get_mentions_by_source(
                ticker, "telegram", window_hours=6.0
            )
            x_mentions = await self._ticker_store.get_mentions_by_source(
                ticker, "twitter", window_hours=6.0
            )
            cross_ref_result = await self._cross_ref.analyze(
                ticker, tg_mentions, x_mentions
            )
            log.info(
                "orchestrator.cross_ref",
                ticker=ticker,
                corroborated=cross_ref_result.get("corroborated"),
                suspicious=cross_ref_result.get("suspicious"),
            )

        # Score fundamentals
        score_result = await self._scorer.score(metadata)

        # X Sentiment analysis (Phase 4) — only when X mentions exist
        x_sentiment_result = None
        sources = social_metrics.get("sources", [])
        if self._x_sentiment_analyzer and "twitter" in sources:
            try:
                x_mentions = await self._ticker_store.get_mentions_by_source(
                    ticker, "twitter", window_hours=6.0
                )
                engagement_summary = await self._ticker_store.get_x_engagement_summary(
                    ticker, window_hours=6.0
                )
                llm_sentiment = await self._x_sentiment_analyzer.analyze(
                    ticker, x_mentions
                )
                x_sentiment_result = score_x_sentiment(
                    engagement_summary, llm_sentiment
                )
                # Attach narrative and engagement summary for display
                x_sentiment_result["key_narrative"] = llm_sentiment.get("key_narrative", "")
                x_sentiment_result["engagement_summary"] = engagement_summary
                log.info(
                    "orchestrator.x_sentiment",
                    ticker=ticker,
                    score=x_sentiment_result["x_sentiment_score"],
                )
            except Exception:
                log.warning("orchestrator.x_sentiment_error", ticker=ticker, exc_info=True)

        # Aggregate → alert decision (with cross-ref + X sentiment result)
        agg_result = await self._aggregator.aggregate(
            ticker, social_metrics, score_result, cross_ref_result, x_sentiment_result
        )

        alert_level = agg_result["alert_level"]
        notify_mode = agg_result["notify_mode"]

        if alert_level == "suppress":
            log.debug("orchestrator.ticker_suppressed", ticker=ticker)
            return 0

        # Trading hook: for buy-grade opportunities, hand off to the trading
        # engine (safety → evaluate → risk → approve/execute). Wrapped so a
        # trading error can never break the alert path.
        if self._trading_engine is not None and alert_level in BUY_GRADE_LEVELS:
            try:
                await self._maybe_trade(
                    ticker, metadata, social_metrics, score_result, alert_level
                )
            except Exception:
                log.error("orchestrator.trade_hook_error", ticker=ticker, exc_info=True)

        # Format rich alert
        text = format_phase2_alert(
            ticker=ticker,
            alert_level=alert_level,
            score_result=score_result,
            social_metrics=social_metrics,
            aggregator_result=agg_result,
            metadata=metadata,
            cross_ref_result=cross_ref_result,
            x_sentiment_result=x_sentiment_result,
        )

        # Enqueue and log
        await self._dispatcher.enqueue(text, notify_mode)
        await self._alert_log.record(
            ticker=ticker,
            alert_level=alert_level,
            notify_mode=notify_mode,
            score=score_result.get("total_score"),
            summary=agg_result.get("summary"),
            metadata_available=score_result.get("metadata_available", False),
        )

        log.info(
            "orchestrator.alert_sent",
            ticker=ticker,
            alert_level=alert_level,
            score=score_result.get("total_score"),
            groups=social_metrics.get("unique_groups"),
        )
        return 1

    async def _maybe_trade(
        self,
        ticker: str,
        metadata: dict[str, Any] | None,
        social_metrics: dict[str, Any],
        score_result: dict[str, Any],
        alert_level: str,
    ) -> None:
        """Build an Opportunity from the enriched signal and hand it to the
        trading engine. The engine applies the safety gate, evaluator, and risk
        limits — this method only assembles the input.
        """
        metadata = metadata or {}
        address = metadata.get("base_address")
        chain_str = (metadata.get("chain") or "solana").lower()
        try:
            chain = Chain(chain_str)
        except ValueError:
            chain = Chain.UNKNOWN

        sources = social_metrics.get("sources") or []
        source = sources[0] if len(sources) == 1 else ("mixed" if sources else "telegram")

        # Venue routing: pump.fun bonding-curve tokens (dexId "pumpfun", pre-
        # graduation) must go to PumpPortal, not Jupiter. Graduated/AMM tokens
        # route to Jupiter. Unknown → executor defaults to Jupiter.
        dex_id = str(metadata.get("dex_id") or "").lower()
        labels = [str(x).lower() for x in (metadata.get("labels") or [])]
        if "pump" in dex_id or any("bonding" in x for x in labels):
            venue = TokenVenue.BONDING
        elif dex_id:
            venue = TokenVenue.AMM
        else:
            venue = TokenVenue.UNKNOWN

        # Technical-analysis signal (best-effort; chart structure for the
        # evaluator). Never let a TA/fetch failure block a trade evaluation.
        ta_signal = None
        if self._ta_analyzer is not None and address:
            try:
                candles = await self._metadata_fetcher.fetch_ohlcv(address)
                ta_signal = self._ta_analyzer.analyze(candles)
            except Exception:
                log.warning("orchestrator.ta_error", ticker=ticker, exc_info=True)

        opportunity = Opportunity(
            ticker=ticker,
            address=address,
            chain=chain,
            venue=venue,
            source=source,
            ta=ta_signal,
            aggregator_score=float(score_result.get("total_score", 0) or 0),
            alert_level=alert_level,
            social_metrics=social_metrics,
            metadata=metadata,
        )
        await self._trading_engine.handle_opportunity(opportunity)

    async def _drain(self) -> None:
        """Drain any remaining messages on shutdown."""
        log.info("orchestrator.draining")
        remaining = await self._buffer.length()
        if remaining > 0:
            batch = await self._buffer.pop_batch(
                max_size=remaining, max_wait_seconds=1.0
            )
            if batch:
                try:
                    classified = await self._classifier.classify_batch(batch)
                    actionable = [
                        m for m in classified if m.get("intent") != "NOISE"
                    ]
                    if actionable:
                        await self._dispatcher.dispatch_batch(actionable, batch)
                except Exception:
                    log.error("orchestrator.drain_error", exc_info=True)

    async def _heartbeat(self) -> None:
        """Periodic health log."""
        while True:
            buffer_len = await self._buffer.length()
            log.info(
                "orchestrator.heartbeat",
                cycles_completed=self._cycle_count,
                buffer_size=buffer_len,
                running=self._running,
            )
            await asyncio.sleep(self._heartbeat_interval)
