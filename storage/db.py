from __future__ import annotations

import aiosqlite
import structlog

log = structlog.get_logger()

SCHEMA = """\
CREATE TABLE IF NOT EXISTS ticker_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    intent TEXT NOT NULL,
    conviction TEXT,
    context TEXT,
    group_id INTEGER,
    group_name TEXT,
    sender_id INTEGER,
    message_id INTEGER,
    raw_text TEXT,
    source TEXT NOT NULL DEFAULT 'telegram',
    created_at REAL NOT NULL  -- time.time() epoch
);
CREATE INDEX IF NOT EXISTS idx_ticker_mentions_ticker_time
    ON ticker_mentions(ticker, created_at);
CREATE INDEX IF NOT EXISTS idx_ticker_mentions_source_ticker_time
    ON ticker_mentions(source, ticker, created_at);

CREATE TABLE IF NOT EXISTS token_metadata_cache (
    ticker TEXT PRIMARY KEY,
    data TEXT NOT NULL,  -- JSON blob
    fetched_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS alert_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    alert_level TEXT NOT NULL,
    notify_mode TEXT NOT NULL,
    score REAL,
    summary TEXT,
    metadata_available INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alert_log_ticker_time
    ON alert_log(ticker, created_at);

-- Trading subsystem -------------------------------------------------------
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    address TEXT NOT NULL,
    ticker TEXT NOT NULL,
    chain TEXT NOT NULL,
    venue TEXT NOT NULL,
    source TEXT NOT NULL,
    entry_price REAL NOT NULL,          -- SOL per token
    amount_sol REAL NOT NULL,           -- SOL deployed at entry
    token_amount REAL NOT NULL,         -- tokens currently held
    initial_token_amount REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    is_paper INTEGER NOT NULL DEFAULT 1,
    entry_tx TEXT,
    peak_price REAL,
    realized_pnl_sol REAL NOT NULL DEFAULT 0,
    take_profit_hits INTEGER NOT NULL DEFAULT 0,
    opened_at REAL NOT NULL,
    closed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_source ON positions(source, status);

-- Append-only audit of every buy/sell attempt (the financial equivalent of
-- alert_log). Never updated after insert.
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id TEXT,
    address TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,                 -- BUY | SELL
    sol_amount REAL,
    token_amount REAL,
    price REAL,
    tx_signature TEXT,
    status TEXT NOT NULL,               -- success | failed
    is_paper INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(created_at);
"""


class Database:
    """Async SQLite connection wrapper."""

    def __init__(self, db_path: str = "signals.db") -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        await self._migrate(self._conn)
        log.info("database.connected", path=self._db_path)

    @staticmethod
    async def _migrate(conn: aiosqlite.Connection) -> None:
        """Run lightweight migrations for schema changes."""
        cursor = await conn.execute("PRAGMA table_info(ticker_mentions)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "source" not in columns:
            await conn.execute(
                "ALTER TABLE ticker_mentions ADD COLUMN source TEXT NOT NULL DEFAULT 'telegram'"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ticker_mentions_source_ticker_time "
                "ON ticker_mentions(source, ticker, created_at)"
            )
            await conn.commit()
            log.info("database.migrated", added_column="source")
        if "engagement_data" not in columns:
            await conn.execute(
                "ALTER TABLE ticker_mentions ADD COLUMN engagement_data TEXT"
            )
            await conn.commit()
            log.info("database.migrated", added_column="engagement_data")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            log.info("database.closed")

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn
