"""Build a backtest dataset (price candles) from GeckoTerminal OHLCV.

Run this WHERE YOU HAVE NETWORK ACCESS (your machine). It reads a file of
Solana mint addresses (one per line), pulls hourly OHLCV for each token's top
pool, and writes the JSON dataset that tools/backtest_replay.py consumes.

GeckoTerminal public API (no key, rate-limited ~30 req/min):
  pools:  GET /api/v2/networks/solana/tokens/{mint}/pools
  ohlcv:  GET /api/v2/networks/solana/pools/{pool}/ohlcv/hour?aggregate=1&limit=48

Note: this reconstructs PRICE history only. It cannot reconstruct point-in-time
safety (mint/freeze authority, LP lock) as-of the past — so backtest_replay runs
with --skip-safety unless you add recorded "safety" verdicts yourself. Judge the
evaluator's price/EV calls here; judge the safety gate separately with the live
gate on current tokens.

Usage:
  python -m tools.fetch_dataset --mints mints.txt --hours 48 --out data.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx

GT = "https://api.geckoterminal.com/api/v2"


async def _top_pool(client: httpx.AsyncClient, mint: str) -> str | None:
    r = await client.get(f"{GT}/networks/solana/tokens/{mint}/pools")
    r.raise_for_status()
    data = r.json().get("data") or []
    return data[0]["attributes"]["address"] if data else None


async def _ohlcv(client: httpx.AsyncClient, pool: str, hours: int) -> list[dict]:
    r = await client.get(
        f"{GT}/networks/solana/pools/{pool}/ohlcv/hour",
        params={"aggregate": 1, "limit": hours},
    )
    r.raise_for_status()
    rows = (((r.json().get("data") or {}).get("attributes") or {}).get("ohlcv_list")) or []
    # GeckoTerminal returns newest-first [ts, o, h, l, c, v]; sort oldest-first.
    rows = sorted(rows, key=lambda x: x[0])
    return [{"ts": int(x[0]), "price_usd": float(x[4]), "volume": float(x[5])} for x in rows]


async def run(args) -> None:
    with open(args.mints) as f:
        mints = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    tokens = []
    async with httpx.AsyncClient(timeout=20.0) as client:
        for mint in mints:
            try:
                pool = await _top_pool(client, mint)
                if not pool:
                    print(f"  no pool for {mint}, skipping")
                    continue
                candles = await _ohlcv(client, pool, args.hours)
                if not candles:
                    print(f"  no candles for {mint}, skipping")
                    continue
                tokens.append({"ticker": mint[:8], "address": mint, "venue": "amm",
                               "candles": candles})
                print(f"  {mint[:10]}…  {len(candles)} candles")
                await asyncio.sleep(2.2)  # rate-limit courtesy
            except Exception as e:  # noqa: BLE001
                print(f"  error for {mint}: {e}")

    out = {"sol_usd": args.sol_usd, "fetched_at": time.time(), "tokens": tokens}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nWrote {len(tokens)} tokens -> {args.out}")
    print(f"Now: python -m tools.backtest_replay --dataset {args.out} --skip-safety "
          f"[--stub-evaluator | (set ANTHROPIC_API_KEY)]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch backtest dataset from GeckoTerminal")
    ap.add_argument("--mints", required=True, help="file of Solana mint addresses, one per line")
    ap.add_argument("--hours", type=int, default=48, help="hours of hourly candles (48 = 2 days)")
    ap.add_argument("--sol-usd", type=float, default=150.0)
    ap.add_argument("--out", default="dataset.json")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
