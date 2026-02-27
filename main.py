from __future__ import annotations

import asyncio
import os
import signal

import redis.asyncio as aioredis
import structlog
from dotenv import load_dotenv

from agents.classifier.agent import MessageClassifier
from pipeline.buffer import MessageBuffer
from pipeline.orchestrator import Orchestrator
from skills.alert_dispatcher.bot import AlertDispatcher
from skills.tg_listener.flood_guard import FloodGuard
from skills.tg_listener.listener import TelegramListener
from skills.tg_listener.session_manager import SessionManager

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

    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    return config


async def main() -> None:
    config = load_config()
    log.info("main.config_loaded")

    # Redis
    redis = aioredis.from_url(config["REDIS_URL"], decode_responses=True)
    log.info("main.redis_connected", url=config["REDIS_URL"])

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

    # Pipeline
    buffer = MessageBuffer(redis=redis)
    classifier = MessageClassifier(api_key=config["ANTHROPIC_API_KEY"])
    dispatcher = AlertDispatcher(
        bot_token=config["TG_ALERT_BOT_TOKEN"],
        chat_id=config["TG_ALERT_CHAT_ID"],
    )
    orchestrator = Orchestrator(
        buffer=buffer,
        classifier=classifier,
        dispatcher=dispatcher,
    )

    # Start components
    await dispatcher.start()

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

    await dispatcher.stop()
    await redis.aclose()
    log.info("main.stopped")


if __name__ == "__main__":
    asyncio.run(main())
