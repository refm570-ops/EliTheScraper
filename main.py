from __future__ import annotations

import asyncio
import os
import signal

import redis.asyncio as aioredis
import structlog
from dotenv import load_dotenv

from agents.aggregator.agent import SignalAggregator
from agents.classifier.agent import MessageClassifier
from agents.cross_ref.agent import CrossReferenceAnalyst
from agents.scorer.agent import FundamentalsScorer
from agents.x_sentiment.agent import XSentimentAnalyzer
from pipeline.buffer import MessageBuffer
from pipeline.orchestrator import Orchestrator
from pipeline.scheduler import PeriodicTask
from skills.alert_dispatcher.bot import AlertDispatcher
from skills.social_metrics.counter import SocialMetricsCounter
from skills.tg_listener.flood_guard import FloodGuard
from skills.tg_listener.listener import TelegramListener
from skills.tg_listener.session_manager import SessionManager
from skills.token_metadata.fetcher import TokenMetadataFetcher
from skills.x_puller.puller import XFeedPuller
from storage.alert_log import AlertLog
from storage.db import Database
from storage.ticker_store import TickerStore

# Configure structlog for JSON output
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
)

log = structlog.get_logger()


def load_config() -> dict[str, str]:
    """Load configuration from .env file."""
    load_dotenv()
    required = [
        "TG_API_ID",
        "TG_API_HASH",
        "TG_SESSION_NAME",
        "TG_ALERT_BOT_TOKEN",
        "TG_ALERT_CHAT_ID",
        "ANTHROPIC_API_KEY",
    ]
    config: dict[str, str] = {}
    missing: list[str] = []
    for key in required:
        val = os.getenv(key)
        if not val:
            missing.append(key)
        else:
            config[key] = val

    config["REDIS_URL"] = os.getenv("REDIS_URL", "redis://localhost:6379")
    config["DB_PATH"] = os.getenv("DB_PATH", "signals.db")
    config["CACHE_TTL_SECONDS"] = os.getenv("CACHE_TTL_SECONDS", "300")
    config["BIRDEYE_API_KEY"] = os.getenv("BIRDEYE_API_KEY", "")
    config["X_BEARER_TOKEN"] = os.getenv("X_BEARER_TOKEN", "")

    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    return config


async def main() -> None:
    config = load_config()
    log.info("main.config_loaded")

    # Redis
    redis = aioredis.from_url(config["REDIS_URL"], decode_responses=True)
    log.info("main.redis_connected", url=config["REDIS_URL"])

    # Database
    db = Database(db_path=config["DB_PATH"])
    await db.connect()

    # Storage
    ticker_store = TickerStore(db=db)
    alert_log = AlertLog(db=db)

    # TG Listener
    flood_guard = FloodGuard()
    session_mgr = SessionManager(
        session_name=config["TG_SESSION_NAME"],
        api_id=int(config["TG_API_ID"]),
        api_hash=config["TG_API_HASH"],
        flood_guard=flood_guard,
    )
    listener = TelegramListener(
        session_manager=session_mgr,
        redis=redis,
    )

    # Skills
    metadata_fetcher = TokenMetadataFetcher(
        db=db,
        birdeye_api_key=config["BIRDEYE_API_KEY"] or None,
        cache_ttl_seconds=float(config["CACHE_TTL_SECONDS"]),
    )
    social_counter = SocialMetricsCounter(ticker_store=ticker_store)

    # Agents
    api_key = config["ANTHROPIC_API_KEY"]
    classifier = MessageClassifier(api_key=api_key)
    scorer = FundamentalsScorer(api_key=api_key)
    aggregator = SignalAggregator(api_key=api_key)
    cross_ref = CrossReferenceAnalyst(api_key=api_key)
    x_sentiment = XSentimentAnalyzer(api_key=api_key)

    # X/Twitter integration (opt-in)
    x_puller: XFeedPuller | None = None
    x_poll_task: PeriodicTask | None = None
    x_bearer = config["X_BEARER_TOKEN"]
    if x_bearer:
        x_puller = XFeedPuller(bearer_token=x_bearer, redis=redis)
        await x_puller.initialize()
        x_poll_task = PeriodicTask(
            name="x_poll",
            func=x_puller.poll,
            interval_seconds=120,  # 2 minutes
        )
        log.info("main.x_integration_enabled")
    else:
        log.info("main.x_integration_disabled", reason="no X_BEARER_TOKEN")

    # Dispatcher
    dispatcher = AlertDispatcher(
        bot_token=config["TG_ALERT_BOT_TOKEN"],
        chat_id=config["TG_ALERT_CHAT_ID"],
    )

    # Orchestrator
    orchestrator = Orchestrator(
        buffer=MessageBuffer(redis=redis),
        classifier=classifier,
        dispatcher=dispatcher,
        ticker_store=ticker_store,
        alert_log=alert_log,
        metadata_fetcher=metadata_fetcher,
        social_counter=social_counter,
        scorer=scorer,
        aggregator=aggregator,
        cross_ref=cross_ref,
        x_sentiment_analyzer=x_sentiment,
    )

    # Periodic cleanup task (every 6 hours)
    cleanup_task = PeriodicTask(
        name="mention_cleanup",
        func=ticker_store.cleanup,
        interval_seconds=6 * 3600,
    )

    # Start components
    await dispatcher.start()
    cleanup_task.start()
    if x_poll_task:
        x_poll_task.start()

    # Run listener and orchestrator concurrently
    shutdown_event = asyncio.Event()

    def on_signal() -> None:
        log.info("main.shutdown_signal")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, on_signal)

    listener_task = asyncio.create_task(listener.start())
    orchestrator_task = asyncio.create_task(orchestrator.run())

    log.info("main.started")

    # Wait for shutdown
    await shutdown_event.wait()

    log.info("main.shutting_down")
    await listener.stop()
    orchestrator._request_shutdown()
    await orchestrator_task
    listener_task.cancel()
    try:
        await listener_task
    except asyncio.CancelledError:
        pass

    if x_poll_task:
        await x_poll_task.stop()
    if x_puller:
        await x_puller.close()
    await cleanup_task.stop()
    await dispatcher.stop()
    await metadata_fetcher.close()
    await db.close()
    await redis.aclose()
    log.info("main.stopped")


if __name__ == "__main__":
    asyncio.run(main())
