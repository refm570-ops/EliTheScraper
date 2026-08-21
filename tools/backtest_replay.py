"""Historical replay backtest — measure trade quality over PAST data, fast.

Replays a dataset of per-token price candles through the REAL decision logic
(safety gate → evaluator → risk → paper entry → the same exit rules the live
monitor uses) in compressed time. Two days of candles replay in minutes.

This is the honest way to backtest: it reuses the production RiskManager,
PositionMonitor exit logic, and PaperExecutor accounting — only the price source
and clock are simulated.

Dataset format (JSON) — build it with tools/fetch_dataset.py where you have
network access:
  {
    "sol_usd": 150.0,
    "tokens": [
      {
        "ticker": "WIF", "address": "<mint>", "venue": "amm",
        "safety": {"pass": true},          # OPTIONAL recorded point-in-time gate
        "candles": [                        # chronological, oldest first
          {"ts": 1710000000, "price_usd": 0.0012, "liquidity_usd": 80000},
          ...
        ]
      }, ...
    ]
  }

Honest limitations (read these):
  - SAFETY IS TIME-DEPENDENT. RugCheck/authority state can't be queried "as of"
    a past block cheaply. If a token has no recorded "safety" block, use
    --skip-safety (assume pass, loud warning) or --live-safety (checks CURRENT
    state, which is only valid for still-existing tokens). Neither perfectly
    reconstructs the past — a real safety backtest needs recorded snapshots.
  - Time-based exits (max_hold_hours) are simulated from candle timestamps when
    present; if candles lack ts, only price-based exits (TP/stop/trailing) fire.
  - The evaluator sees only the FIRST candle's metadata (entry-time snapshot),
    which is correct — it must not see the future.

Usage:
  # with real Opus evaluator (needs ANTHROPIC_API_KEY) + recorded safety:
  python -m tools.backtest_replay --dataset data.json
  # mechanics/exit-tuning only, no key/network:
  python -m tools.backtest_replay --dataset data.json --stub-evaluator --skip-safety
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any

import structlog

from storage.db import Database
from storage.trade_store import TradeStore
from tools.paper_scan import Stats, DemoEvaluator, DemoSafety, build_report, _print_report, _write
from trading.config import TradingConfig, load_trading_config
from trading.models import (
    Chain,
    ExecutionResult,
    Opportunity,
    Position,
    PositionStatus,
    SafetyCheck,
    SafetyReport,
    Severity,
    TokenVenue,
    TradeSide,
)
from trading.monitor import PositionMonitor
from trading.risk import RiskManager

log = structlog.get_logger()


class ReplayExecutor:
    """Paper executor whose price is driven by the replay clock (SOL/token)."""

    is_paper = True

    def __init__(self) -> None:
        self._price: float | None = None

    def set_price(self, price_sol: float | None) -> None:
        self._price = price_sol

    async def close(self) -> None:
        return None

    async def buy(self, opportunity: Opportunity, sol_amount: float, max_slippage_bps: int) -> ExecutionResult:
        if not self._price or self._price <= 0:
            return ExecutionResult(False, TradeSide.BUY, address=opportunity.address,
                                   is_paper=True, error="replay: no price")
        tokens = sol_amount / self._price
        return ExecutionResult(True, TradeSide.BUY, address=opportunity.address,
                               tx_signature="REPLAY-BUY", sol_amount=sol_amount,
                               token_amount=tokens, price=self._price, is_paper=True)

    async def sell(self, position: Position, token_amount: float, max_slippage_bps: int) -> ExecutionResult:
        if not self._price or self._price <= 0:
            return ExecutionResult(False, TradeSide.SELL, address=position.address,
                                   is_paper=True, error="replay: no price")
        return ExecutionResult(True, TradeSide.SELL, address=position.address,
                               tx_signature="REPLAY-SELL", sol_amount=token_amount * self._price,
                               token_amount=token_amount, price=self._price, is_paper=True)

    async def current_price_sol(self, position: Position) -> float | None:
        return self._price


class RecordedSafety:
    """Uses per-token recorded safety verdicts from the dataset."""

    def __init__(self, skip: bool) -> None:
        self._skip = skip
        self._current: dict[str, Any] | None = None

    def set_token(self, safety_block: dict[str, Any] | None) -> None:
        self._current = safety_block

    async def close(self) -> None:
        return None

    async def check(self, address, chain="solana", metadata=None) -> SafetyReport:
        report = SafetyReport(address=address)
        if self._skip:
            report.add(SafetyCheck("skipped", True, Severity.HARD, "safety skipped (--skip-safety)"))
            return report
        blk = self._current or {}
        passed = bool(blk.get("pass", False))
        report.add(SafetyCheck("recorded", passed, Severity.HARD,
                               blk.get("reason", "recorded verdict")))
        return report


def _venue(v: str) -> TokenVenue:
    try:
        return TokenVenue(v)
    except ValueError:
        return TokenVenue.UNKNOWN


async def run(args) -> None:
    with open(args.dataset) as f:
        dataset = json.load(f)
    sol_usd = float(dataset.get("sol_usd", os.getenv("SOL_USD_PRICE", 150.0)))
    tokens = dataset.get("tokens", [])

    cfg: TradingConfig = load_trading_config()
    cfg.mode, cfg.autonomy = "paper", "auto"

    db = Database(args.db)
    await db.connect()
    store = TradeStore(db)
    risk = RiskManager(cfg, store, redis=None)
    executor = ReplayExecutor()

    # Evaluator: real Opus (needs key) or offline stub.
    if args.stub_evaluator:
        evaluator: Any = DemoEvaluator()
    else:
        from agents.trader.agent import OpportunityEvaluator
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY required (or pass --stub-evaluator).")
        evaluator = OpportunityEvaluator(api_key=key)

    if args.live_safety:
        from skills.safety.gate import SafetyGate
        rpc = None
        if os.getenv("SOLANA_RPC_URL"):
            from skills.safety.providers import SolanaRPC
            rpc = SolanaRPC(rpc_url=os.getenv("SOLANA_RPC_URL"))
        safety: Any = SafetyGate(safety_config=cfg.safety, rpc=rpc)
        recorded = None
    else:
        recorded = RecordedSafety(skip=args.skip_safety)
        safety = recorded

    # Disable the monitor's WALL-CLOCK time exit in replay (opened_at is a
    # historical timestamp, so real-now vs then would fire instantly). We
    # simulate time-based exit from candle timestamps below instead.
    exit_cfg = dict(cfg.exit)
    max_hold_hours = exit_cfg.pop("max_hold_hours", None)
    monitor = PositionMonitor(store, executor, exit_cfg, cfg.execution, fetcher=None,
                              risk_manager=None)  # no portfolio breaker in replay
    stats = Stats()
    log_fp = open(args.decisions, "w")

    try:
        for tok in tokens:
            candles = tok.get("candles") or []
            if not candles:
                continue
            first = candles[0]
            price0_usd = first.get("price_usd")
            if not price0_usd or price0_usd <= 0:
                continue
            meta = {
                "price_usd": price0_usd,
                "liquidity_usd": first.get("liquidity_usd"),
                "market_cap": first.get("market_cap"),
                "base_address": tok.get("address"),
                "dex_id": tok.get("dex_id"),
            }
            opp = Opportunity(
                ticker=tok.get("ticker") or (tok.get("address") or "?")[:8],
                address=tok.get("address"), chain=Chain.SOLANA,
                venue=_venue(tok.get("venue", "amm")), source=tok.get("source", "backtest"),
                metadata=meta,
            )
            if recorded is not None:
                recorded.set_token(tok.get("safety"))

            rec: dict[str, Any] = {"ticker": opp.ticker, "address": opp.address}
            stats.opportunities += 1

            report = await safety.check(opp.address, "solana", meta)
            rec["safety_passed"] = report.passed
            if not report.passed:
                rec["stage"] = "SKIP_SAFETY"; rec["safety_blocks"] = report.blocking_reasons()
                stats.safety_blocked += 1; stats.add_block_reason(report.blocking_reasons())
                _write(log_fp, rec); continue
            stats.safety_passed += 1

            decision = await evaluator.evaluate(opp, report, cfg.max_trade_sol())
            rec["decision"] = decision.action.value
            rec["conviction"] = decision.conviction.value
            rec["reasoning"] = decision.reasoning
            if not decision.is_buy:
                rec["stage"] = "SKIP_EVAL"; stats.eval_skip += 1; _write(log_fp, rec); continue
            stats.eval_buy += 1

            rd = await risk.authorize(opp, decision)
            rec["risk_approved"] = rd.approved; rec["risk_reason"] = rd.reason
            if not rd.approved:
                rec["stage"] = "SKIP_RISK"; stats.risk_blocked += 1; _write(log_fp, rec); continue

            # Entry at first candle.
            executor.set_price(price0_usd / sol_usd)
            result = await executor.buy(opp, rd.size_sol, rd.slippage_bps)
            if not result.success:
                rec["stage"] = "BUY_FAILED"; rec["error"] = result.error; _write(log_fp, rec); continue
            pos = Position(
                address=opp.address or "", ticker=opp.ticker, chain=opp.chain, venue=opp.venue,
                source=opp.source, entry_price=result.price or 0.0,
                amount_sol=result.sol_amount or rd.size_sol, token_amount=result.token_amount or 0.0,
                initial_token_amount=result.token_amount or 0.0, status=PositionStatus.OPEN,
                is_paper=True, entry_tx=result.tx_signature, peak_price=result.price or 0.0,
                opened_at=first.get("ts") or time.time(),
            )
            await store.open_position(pos)
            await store.record_trade(result, opp.ticker, pos.id)
            stats.buys += 1
            rec["stage"] = "BOUGHT"; rec["entry_price_usd"] = price0_usd

            # Replay subsequent candles through the real exit logic (TP ladder,
            # stop-loss, trailing) plus a simulated time-based exit.
            exit_reason = "still_open_at_end"
            first_ts = first.get("ts")
            for candle in candles[1:]:
                p = candle.get("price_usd")
                if not p or p <= 0:
                    continue
                price_sol = p / sol_usd
                executor.set_price(price_sol)
                await monitor._evaluate(pos)
                refreshed = await _get_position(store, pos.id)
                if refreshed:
                    pos = refreshed
                if pos.status == PositionStatus.CLOSED:
                    exit_reason = "closed"
                    break
                # Simulated time-based exit from candle timestamps.
                if (max_hold_hours and first_ts and candle.get("ts")
                        and candle["ts"] - first_ts >= float(max_hold_hours) * 3600):
                    await monitor._sell_all(pos, price_sol, reason=f"max hold {max_hold_hours}h (replay)")
                    pos = await _get_position(store, pos.id) or pos
                    exit_reason = "time_exit"
                    break
            rec["exit"] = exit_reason
            rec["realized_pnl_sol"] = round(pos.realized_pnl_sol, 5)
            _write(log_fp, rec)
    finally:
        log_fp.close()
        report = await build_report(store, stats)
        with open(args.report, "w") as rf:
            json.dump(report, rf, indent=2)
        _print_report(report, args.decisions)
        await db.close()


async def _get_position(store: TradeStore, pid: str) -> Position | None:
    cursor = await store._db.conn.execute("SELECT * FROM positions WHERE id = ?", (pid,))
    row = await cursor.fetchone()
    return store._row_to_position(row) if row else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Historical replay backtest")
    ap.add_argument("--dataset", required=True, help="JSON dataset (see module docstring / fetch_dataset.py)")
    ap.add_argument("--stub-evaluator", action="store_true", help="offline heuristic instead of Opus (no key)")
    ap.add_argument("--skip-safety", action="store_true", help="assume safety pass (no recorded verdicts)")
    ap.add_argument("--live-safety", action="store_true", help="check CURRENT on-chain safety (needs network)")
    ap.add_argument("--db", default="backtest.db")
    ap.add_argument("--decisions", default="backtest_decisions.jsonl")
    ap.add_argument("--report", default="backtest_report.json")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
