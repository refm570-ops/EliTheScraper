"""Trading journal — a human-readable list of every position and trade.

Reads the SQLite DB written by the live system (signals.db) or a paper/backtest
run (paper_scan.db / backtest.db) and prints a journal: each position with
entry, exit, status, and realized PnL, plus a summary and the append-only trade
audit. Optionally exports positions to CSV.

Usage:
  python -m tools.journal --db paper_scan.db
  python -m tools.journal --db backtest.db --csv journal.csv
  python -m tools.journal --db signals.db --trades      # also list raw trades
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import time
from typing import Any

from storage.db import Database


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts))


async def load(db_path: str) -> tuple[list[dict], list[dict]]:
    db = Database(db_path)
    await db.connect()
    pos_cur = await db.conn.execute("SELECT * FROM positions ORDER BY opened_at")
    positions = [dict(r) for r in await pos_cur.fetchall()]
    trd_cur = await db.conn.execute("SELECT * FROM trades ORDER BY created_at")
    trades = [dict(r) for r in await trd_cur.fetchall()]
    await db.close()
    return positions, trades


def print_journal(positions: list[dict], trades: list[dict], show_trades: bool) -> None:
    if not positions:
        print("No positions recorded in this database.")
        return

    print("\n" + "=" * 92)
    print("  TRADING JOURNAL")
    print("=" * 92)
    hdr = f"  {'opened':<16} {'ticker':<14} {'mode':<6} {'status':<8} {'size SOL':>9} {'PnL SOL':>10}  src"
    print(hdr)
    print("  " + "-" * 88)

    realized_total = 0.0
    wins = losses = 0
    for p in positions:
        mode = "paper" if p["is_paper"] else "LIVE"
        pnl = p["realized_pnl_sol"] or 0.0
        realized_total += pnl
        if p["status"] == "CLOSED":
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
        print(f"  {_fmt_ts(p['opened_at']):<16} {p['ticker'][:14]:<14} {mode:<6} "
              f"{p['status']:<8} {p['amount_sol']:>9.4f} {pnl:>10.4f}  {p['source']}")

    closed = [p for p in positions if p["status"] == "CLOSED"]
    open_ = [p for p in positions if p["status"] in ("OPEN", "CLOSING")]
    print("  " + "-" * 88)
    print(f"  positions: {len(positions)}  (closed {len(closed)}, open {len(open_)})   "
          f"wins {wins} / losses {losses}   "
          f"win rate {round(100*wins/len(closed),1) if closed else 'n/a'}%")
    print(f"  realized PnL: {realized_total:+.4f} SOL   |   trades logged: {len(trades)}")
    print("=" * 92)

    if show_trades and trades:
        print("\n  TRADE AUDIT (append-only)")
        print(f"  {'time':<16} {'ticker':<12} {'side':<5} {'SOL':>10} {'tokens':>14} {'status':<8} tx")
        print("  " + "-" * 88)
        for t in trades:
            print(f"  {_fmt_ts(t['created_at']):<16} {t['ticker'][:12]:<12} {t['side']:<5} "
                  f"{(t['sol_amount'] or 0):>10.4f} {(t['token_amount'] or 0):>14.2f} "
                  f"{t['status']:<8} {t['tx_signature'] or ''}")
        print()


def export_csv(positions: list[dict], path: str) -> None:
    if not positions:
        return
    cols = ["opened_at", "closed_at", "ticker", "address", "source", "venue", "is_paper",
            "status", "entry_price", "amount_sol", "token_amount", "realized_pnl_sol",
            "take_profit_hits", "entry_tx"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in positions:
            w.writerow(p)
    print(f"  exported {len(positions)} positions -> {path}")


async def run(args) -> None:
    positions, trades = await load(args.db)
    print_journal(positions, trades, show_trades=args.trades)
    if args.csv:
        export_csv(positions, args.csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Trading journal / position list")
    ap.add_argument("--db", default="signals.db", help="SQLite DB (signals.db / paper_scan.db / backtest.db)")
    ap.add_argument("--csv", help="also export positions to this CSV path")
    ap.add_argument("--trades", action="store_true", help="also print the raw trade audit log")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
