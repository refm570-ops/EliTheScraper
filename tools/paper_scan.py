"""Paper-trading / backtest harness — measure trade quality WITHOUT risking money.

Runs the real pipeline (safety gate → Opus evaluator → risk manager → PAPER
execution → position monitor with exits) over a live or fixed set of Solana
tokens, logging every decision and simulated fill, then prints a quality report.

Modes:
  live      DexScreener boosted/profiled Solana tokens as the opportunity feed
            (no Telegram creds needed). Run for hours/days: --hours N.
  backtest  A fixed list of mints (--mints file.txt), e.g. known ruggers/winners.
  demo      Fully offline synthetic run (no network / no API key) to validate
            the harness itself and show the report format.

Real modes need ANTHROPIC_API_KEY (evaluator) and network access to DexScreener
/ RugCheck / a Solana RPC (set SOLANA_RPC_URL to a paid endpoint for reliable
safety reads). This process NEVER trades real funds — execution is always paper.

Examples:
  python -m tools.paper_scan --demo
  python -m tools.paper_scan --mode live --hours 6 --db paper.db
  python -m tools.paper_scan --mode backtest --mints known_tokens.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Any

import structlog

from storage.db import Database
from storage.trade_store import TradeStore
from skills.executor.paper import PaperExecutor
from tools.feeds import DemoFeed, DexScreenerFeed, MintListFeed
from trading.config import TradingConfig, load_trading_config
from trading.models import (
    Conviction,
    DecisionAction,
    Opportunity,
    Position,
    PositionStatus,
    SafetyCheck,
    SafetyReport,
    Severity,
    TradeDecision,
)
from trading.monitor import PositionMonitor
from trading.risk import RiskManager

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self) -> None:
        self.opportunities = 0
        self.safety_passed = 0
        self.safety_blocked = 0
        self.eval_buy = 0
        self.eval_skip = 0
        self.risk_blocked = 0
        self.buys = 0
        self.block_reasons: dict[str, int] = {}

    def add_block_reason(self, reasons: list[str]) -> None:
        for r in reasons:
            key = r.split(":")[0]
            self.block_reasons[key] = self.block_reasons.get(key, 0) + 1


# ---------------------------------------------------------------------------
# Demo stand-ins (offline; used only with --demo)
# ---------------------------------------------------------------------------
class DemoFetcher:
    """Assigns each token a fate and random-walks its price so positions resolve."""

    def __init__(self, seed: int = 11) -> None:
        import random
        self._rng = random.Random(seed)
        self._state: dict[str, dict[str, Any]] = {}

    async def close(self) -> None:
        return None

    def _fate(self) -> str:
        r = self._rng.random()
        return "moon" if r < 0.4 else ("rug" if r < 0.8 else "wander")

    async def fetch_by_address(self, address: str) -> dict[str, Any]:
        st = self._state.get(address)
        if st is None:
            st = {"price": self._rng.uniform(0.0005, 0.01), "fate": self._fate()}
            self._state[address] = st
        else:
            fate = st["fate"]
            if fate == "moon":
                st["price"] *= self._rng.uniform(1.3, 1.9)
            elif fate == "rug":
                st["price"] *= self._rng.uniform(0.35, 0.7)
            else:
                st["price"] *= self._rng.uniform(0.9, 1.1)
        return {"price_usd": st["price"], "liquidity_usd": 40000, "dex_id": "raydium"}


class DemoSafety:
    """Passes most tokens, hard-blocks ~30% (to exercise the block path)."""

    def __init__(self, seed: int = 3) -> None:
        import random
        self._rng = random.Random(seed)

    async def close(self) -> None:
        return None

    async def check(self, address, chain="solana", metadata=None) -> SafetyReport:
        report = SafetyReport(address=address)
        if self._rng.random() < 0.3:
            report.add(SafetyCheck("mint_authority_revoked", False, Severity.HARD,
                                   "demo: mint authority active"))
        else:
            report.add(SafetyCheck("mint_authority_revoked", True, Severity.HARD, "ok"))
        return report


class DemoEvaluator:
    """Buys ~60% with random conviction; used only offline."""

    def __init__(self, seed: int = 5) -> None:
        import random
        self._rng = random.Random(seed)

    async def evaluate(self, opportunity: Opportunity, safety: SafetyReport,
                       max_trade_sol: float) -> TradeDecision:
        if self._rng.random() < 0.4:
            return TradeDecision(DecisionAction.SKIP, Conviction.NONE, 0.0, 0, "demo skip", model="demo")
        conv = self._rng.choice([Conviction.HIGH, Conviction.MEDIUM, Conviction.LOW])
        return TradeDecision(DecisionAction.BUY, conv, max_trade_sol, 500,
                             "demo buy", ttl_seconds=300, model="demo")


# ---------------------------------------------------------------------------
# Core evaluation of one opportunity (mirrors TradingEngine, instrumented)
# ---------------------------------------------------------------------------
async def evaluate_one(opp, safety, evaluator, risk, executor, store, stats, log_fp, cfg) -> None:
    rec: dict[str, Any] = {"ts": round(time.time(), 1), "ticker": opp.ticker,
                           "address": opp.address, "venue": opp.venue.value}
    stats.opportunities += 1

    report = await safety.check(opp.address, opp.chain.value, opp.metadata)
    rec["safety_passed"] = report.passed
    if not report.passed:
        rec["stage"] = "SKIP_SAFETY"
        rec["safety_blocks"] = report.blocking_reasons()
        stats.safety_blocked += 1
        stats.add_block_reason(report.blocking_reasons())
        _write(log_fp, rec)
        return
    stats.safety_passed += 1

    decision = await evaluator.evaluate(opp, report, cfg.max_trade_sol())
    rec["decision"] = decision.action.value
    rec["conviction"] = decision.conviction.value
    rec["reasoning"] = decision.reasoning
    if not decision.is_buy:
        rec["stage"] = "SKIP_EVAL"
        stats.eval_skip += 1
        _write(log_fp, rec)
        return
    stats.eval_buy += 1

    rd = await risk.authorize(opp, decision)
    rec["risk_approved"] = rd.approved
    rec["risk_reason"] = rd.reason
    if not rd.approved:
        rec["stage"] = "SKIP_RISK"
        stats.risk_blocked += 1
        _write(log_fp, rec)
        return
    rec["size_sol"] = rd.size_sol

    result = await executor.buy(opp, rd.size_sol, rd.slippage_bps)
    if not result.success:
        rec["stage"] = "BUY_FAILED"
        rec["error"] = result.error
        _write(log_fp, rec)
        return

    pos = Position(
        address=opp.address or "", ticker=opp.ticker, chain=opp.chain, venue=opp.venue,
        source=opp.source, entry_price=result.price or 0.0,
        amount_sol=result.sol_amount or rd.size_sol,
        token_amount=result.token_amount or 0.0,
        initial_token_amount=result.token_amount or 0.0,
        status=PositionStatus.OPEN, is_paper=True, entry_tx=result.tx_signature,
        peak_price=result.price or 0.0,
    )
    await store.open_position(pos)
    await store.record_trade(result, opp.ticker, pos.id)
    rec["stage"] = "BOUGHT"
    rec["entry_price"] = result.price
    stats.buys += 1
    _write(log_fp, rec)


def _write(fp, rec: dict[str, Any]) -> None:
    fp.write(json.dumps(rec) + "\n")
    fp.flush()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
async def build_report(store: TradeStore, stats: Stats) -> dict[str, Any]:
    cursor = await store._db.conn.execute("SELECT * FROM positions")
    rows = await cursor.fetchall()
    closed = [r for r in rows if r["status"] == "CLOSED"]
    open_ = [r for r in rows if r["status"] in ("OPEN", "CLOSING")]
    pnls = [r["realized_pnl_sol"] for r in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    report = {
        "funnel": {
            "opportunities": stats.opportunities,
            "safety_passed": stats.safety_passed,
            "safety_blocked": stats.safety_blocked,
            "evaluator_buy": stats.eval_buy,
            "evaluator_skip": stats.eval_skip,
            "risk_blocked": stats.risk_blocked,
            "positions_opened": stats.buys,
        },
        "safety_block_reasons": dict(sorted(stats.block_reasons.items(),
                                            key=lambda kv: -kv[1])),
        "positions": {
            "closed": len(closed),
            "still_open": len(open_),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(100 * len(wins) / len(closed), 1) if closed else None,
        },
        "pnl_sol": {
            "total_realized": round(sum(pnls), 4),
            "avg_per_closed": round(statistics.mean(pnls), 4) if pnls else None,
            "median_per_closed": round(statistics.median(pnls), 4) if pnls else None,
            "best": round(max(pnls), 4) if pnls else None,
            "worst": round(min(pnls), 4) if pnls else None,
        },
    }
    return report


def _print_report(report: dict[str, Any], decisions_path: str) -> None:
    f = report["funnel"]
    p = report["positions"]
    pnl = report["pnl_sol"]
    print("\n" + "=" * 60)
    print("  PAPER-TRADING QUALITY REPORT")
    print("=" * 60)
    print("  Funnel:")
    print(f"    opportunities seen ........ {f['opportunities']}")
    print(f"    passed safety gate ........ {f['safety_passed']}")
    print(f"    blocked by safety ......... {f['safety_blocked']}")
    print(f"    evaluator: BUY / SKIP ..... {f['evaluator_buy']} / {f['evaluator_skip']}")
    print(f"    blocked by risk limits .... {f['risk_blocked']}")
    print(f"    positions opened .......... {f['positions_opened']}")
    if report["safety_block_reasons"]:
        print("  Safety blocks by reason:")
        for k, v in report["safety_block_reasons"].items():
            print(f"    {k:.<30} {v}")
    print("  Outcomes:")
    print(f"    closed / still open ....... {p['closed']} / {p['still_open']}")
    print(f"    wins / losses ............. {p['wins']} / {p['losses']}")
    print(f"    win rate .................. {p['win_rate_pct']}%")
    print("  Simulated PnL (SOL):")
    print(f"    total realized ............ {pnl['total_realized']}")
    print(f"    avg / median per trade .... {pnl['avg_per_closed']} / {pnl['median_per_closed']}")
    print(f"    best / worst .............. {pnl['best']} / {pnl['worst']}")
    print("=" * 60)
    print(f"  Full decision log: {decisions_path}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _paper_config() -> TradingConfig:
    base = load_trading_config()
    # Force paper + auto so the harness executes simulated buys without prompts,
    # regardless of the committed config or env.
    base.mode = "paper"
    base.autonomy = "auto"
    base.trading_enabled = False
    base.allow_auto = False
    return base


async def run(args) -> None:
    cfg = _paper_config()
    db = Database(args.db)
    await db.connect()
    store = TradeStore(db)

    if args.mode == "demo":
        fetcher: Any = DemoFetcher()
        safety: Any = DemoSafety()
        evaluator: Any = DemoEvaluator()
        feed: Any = DemoFeed()
    else:
        from skills.token_metadata.fetcher import TokenMetadataFetcher
        from skills.safety.gate import SafetyGate
        from agents.trader.agent import OpportunityEvaluator

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY required for live/backtest modes (evaluator).")
        fetcher = TokenMetadataFetcher(db=db, birdeye_api_key=os.getenv("BIRDEYE_API_KEY") or None)
        rpc = None
        if os.getenv("SOLANA_RPC_URL"):
            from skills.safety.providers import SolanaRPC
            rpc = SolanaRPC(rpc_url=os.getenv("SOLANA_RPC_URL"))
        safety = SafetyGate(safety_config=cfg.safety, rpc=rpc)
        evaluator = OpportunityEvaluator(api_key=api_key)
        if args.mode == "backtest":
            with open(args.mints) as fh:
                mints = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
            feed = MintListFeed(fetcher, mints)
        else:
            feed = DexScreenerFeed(fetcher, min_liquidity_usd=float(cfg.safety.get("min_liquidity_usd", 5000)))

    risk = RiskManager(cfg, store, redis=None)
    sol_usd = float(os.getenv("SOL_USD_PRICE", cfg.execution.get("sol_usd_reference", 150.0)))
    executor = PaperExecutor(fetcher=fetcher, sol_usd_reference=sol_usd)
    monitor = PositionMonitor(
        store, executor, cfg.exit, cfg.execution, fetcher=fetcher,
        min_liquidity_usd=float(cfg.safety.get("min_liquidity_usd", 5000)),
        risk_manager=risk, daily_loss_limit_sol=float(cfg.risk.get("daily_loss_limit_sol", 0.5)),
    )

    stats = Stats()
    decisions_path = args.decisions
    log_fp = open(decisions_path, "w")

    deadline = time.time() + args.hours * 3600
    interval = args.interval
    max_ticks = args.max_ticks or (60 if args.mode == "demo" else 0)
    ticks = 0
    try:
        while True:
            opps = await feed.poll()
            for opp in opps:
                try:
                    await evaluate_one(opp, safety, evaluator, risk, executor, store, stats, log_fp, cfg)
                except Exception:
                    log.error("scan.evaluate_error", ticker=opp.ticker, exc_info=True)
            await monitor.tick()
            ticks += 1

            open_count = await store.count_open()
            time_up = time.time() >= deadline
            # Demo/backtest exit early once the feed is drained and positions resolve.
            if args.mode in ("demo", "backtest") and not opps and open_count == 0 and ticks > 1:
                break
            # Hard tick cap (safety against never-resolving positions in demo).
            if max_ticks and ticks >= max_ticks:
                break
            if time_up:
                break
            await asyncio.sleep(interval if args.mode != "demo" else 0.01)
    finally:
        log_fp.close()
        report = await build_report(store, stats)
        with open(args.report, "w") as rf:
            json.dump(report, rf, indent=2)
        _print_report(report, decisions_path)
        try:
            await feed.close()
        except Exception:
            pass
        await db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-trading / backtest harness")
    ap.add_argument("--mode", choices=["live", "backtest", "demo"], default="demo")
    ap.add_argument("--hours", type=float, default=6.0, help="max wall-clock run time (live mode)")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between feed polls (live)")
    ap.add_argument("--max-ticks", type=int, default=0, help="hard cap on cycles (0 = unlimited)")
    ap.add_argument("--mints", default="known_tokens.txt", help="mint list file (backtest mode)")
    ap.add_argument("--db", default="paper_scan.db")
    ap.add_argument("--decisions", default="paper_decisions.jsonl")
    ap.add_argument("--report", default="paper_report.json")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
