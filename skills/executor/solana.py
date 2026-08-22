"""SolanaExecutor — real buys/sells on Solana (LIVE mode only).

Venue routing:
  - BONDING (pump.fun pre-graduation) -> PumpPortal trade-local: returns an
    UNSIGNED serialized tx we sign locally (non-custodial) and send ourselves.
  - AMM / graduated / unknown        -> Jupiter aggregator quote+swap.

Signing is local (Wallet). Slippage is hard-capped by the caller (RiskManager /
config) regardless of what the evaluator proposed. Every send is confirmed with
bounded retries; a failed trade returns ExecutionResult(success=False), never
raises, so the book stays consistent.

Requires the `trade` extra (solders, base58). Instantiated only when live.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx
import structlog

from skills.executor.base import Executor
from skills.executor.wallet import Wallet
from trading.models import (
    ExecutionResult,
    Opportunity,
    Position,
    TokenVenue,
    TradeSide,
)

log = structlog.get_logger()

PUMPPORTAL_LOCAL_URL = "https://pumpportal.fun/api/trade-local"
JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
WSOL_MINT = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000


class SolanaExecutor(Executor):
    is_paper = False

    def __init__(
        self,
        wallet: Wallet,
        rpc_url: str,
        priority_fee_microlamports: int = 200_000,
        use_jito: bool = False,
        jito_tip_sol: float = 0.0005,
        confirm_timeout_seconds: float = 60.0,
        confirm_retries: int = 3,
        min_gas_reserve_sol: float = 0.05,
    ) -> None:
        self._wallet = wallet
        self._rpc_url = rpc_url
        self._priority_fee = priority_fee_microlamports
        self._use_jito = use_jito
        self._jito_tip_sol = jito_tip_sol
        self._confirm_timeout = confirm_timeout_seconds
        self._confirm_retries = confirm_retries
        self._min_gas_reserve = min_gas_reserve_sol
        self._decimals_cache: dict[str, int] = {}
        self._http = httpx.AsyncClient(timeout=20.0)

    async def close(self) -> None:
        await self._http.aclose()

    # ---- public API --------------------------------------------------------
    async def buy(
        self, opportunity: Opportunity, sol_amount: float, max_slippage_bps: int
    ) -> ExecutionResult:
        try:
            # Gas reserve: never spend below the reserve needed to fund the
            # eventual exit transaction.
            balance = await self._get_balance_sol()
            if balance - sol_amount < self._min_gas_reserve:
                raise RuntimeError(
                    f"insufficient balance {balance:.4f} SOL for {sol_amount:.4f} "
                    f"+ {self._min_gas_reserve:.4f} gas reserve"
                )

            if opportunity.venue is TokenVenue.BONDING:
                sig = await self._pumpportal_trade(
                    "buy", opportunity.address, sol_amount, max_slippage_bps,
                    denominated_in_sol=True,
                )
                price = await self.current_price_sol_for(opportunity.address)
                token_amount = (sol_amount / price) if price else None
            else:
                sig, out_base_units = await self._jupiter_swap(
                    input_mint=WSOL_MINT, output_mint=opportunity.address,
                    amount=int(sol_amount * LAMPORTS_PER_SOL),
                    max_slippage_bps=max_slippage_bps,
                )
                decimals = await self._get_decimals(opportunity.address)
                token_amount = out_base_units / (10 ** decimals)  # whole tokens
                price = (sol_amount / token_amount) if token_amount else None
            return ExecutionResult(
                success=True, side=TradeSide.BUY, address=opportunity.address,
                tx_signature=sig, sol_amount=sol_amount, token_amount=token_amount,
                price=price, is_paper=False,
            )
        except Exception as e:  # noqa: BLE001
            log.error("solana.buy_failed", ticker=opportunity.ticker, error=str(e))
            return ExecutionResult(False, TradeSide.BUY, address=opportunity.address,
                                   is_paper=False, error=str(e))

    async def sell(
        self, position: Position, token_amount: float, max_slippage_bps: int
    ) -> ExecutionResult:
        try:
            if position.venue is TokenVenue.BONDING:
                # PumpPortal sells are denominated in whole tokens directly.
                sig = await self._pumpportal_trade(
                    "sell", position.address, token_amount, max_slippage_bps,
                    denominated_in_sol=False,
                )
            else:
                decimals = await self._get_decimals(position.address)
                raw_amount = int(token_amount * (10 ** decimals))
                sig, _ = await self._jupiter_swap(
                    input_mint=position.address, output_mint=WSOL_MINT,
                    amount=raw_amount, max_slippage_bps=max_slippage_bps,
                )
            price = await self.current_price_sol_for(position.address)
            sol_received = (token_amount * price) if price else None
            return ExecutionResult(
                success=True, side=TradeSide.SELL, address=position.address,
                tx_signature=sig, sol_amount=sol_received, token_amount=token_amount,
                price=price, is_paper=False,
            )
        except Exception as e:  # noqa: BLE001
            log.error("solana.sell_failed", ticker=position.ticker, error=str(e))
            return ExecutionResult(False, TradeSide.SELL, address=position.address,
                                   is_paper=False, error=str(e))

    async def current_price_sol(self, position: Position) -> float | None:
        return await self.current_price_sol_for(position.address)

    async def current_price_sol_for(self, address: str | None) -> float | None:
        """Price in SOL per WHOLE token via a Jupiter quote (1 token -> SOL)."""
        if not address:
            return None
        try:
            decimals = await self._get_decimals(address)
            probe = 10 ** decimals  # exactly one whole token in base units
            quote = await self._jupiter_quote(address, WSOL_MINT, probe, 500)
            out_lamports = int(quote["outAmount"])
            return out_lamports / LAMPORTS_PER_SOL  # SOL per whole token
        except Exception:
            return None

    async def _get_decimals(self, mint: str) -> int:
        if mint in self._decimals_cache:
            return self._decimals_cache[mint]
        result = await self._rpc(
            "getAccountInfo", [mint, {"encoding": "jsonParsed", "commitment": "confirmed"}]
        )
        info = ((((result or {}).get("value") or {}).get("data") or {}).get("parsed") or {}).get("info") or {}
        decimals = int(info.get("decimals", 6))
        self._decimals_cache[mint] = decimals
        return decimals

    async def _get_balance_sol(self) -> float:
        result = await self._rpc("getBalance", [self._wallet.pubkey])
        lamports = result.get("value", 0) if isinstance(result, dict) else (result or 0)
        return lamports / LAMPORTS_PER_SOL

    # ---- PumpPortal (bonding curve) ---------------------------------------
    async def _pumpportal_trade(
        self, action: str, mint: str, amount: float, max_slippage_bps: int,
        denominated_in_sol: bool,
    ) -> str:
        payload = {
            "publicKey": self._wallet.pubkey,
            "action": action,
            "mint": mint,
            "amount": amount,
            "denominatedInSol": "true" if denominated_in_sol else "false",
            "slippage": max_slippage_bps / 100.0,  # PumpPortal wants percent
            "priorityFee": self._jito_tip_sol if self._use_jito else 0.00005,
            "pool": "pump",
        }
        resp = await self._http.post(PUMPPORTAL_LOCAL_URL, json=payload)
        resp.raise_for_status()
        tx_bytes = resp.content  # serialized unsigned VersionedTransaction
        signed = self._wallet.sign_versioned(tx_bytes)
        return await self._send_and_confirm(bytes(signed))

    # ---- Jupiter (AMM) -----------------------------------------------------
    async def _jupiter_quote(
        self, input_mint: str, output_mint: str, amount: int, max_slippage_bps: int
    ) -> dict[str, Any]:
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(max_slippage_bps),
        }
        resp = await self._http.get(JUPITER_QUOTE_URL, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _jupiter_swap(
        self, input_mint: str, output_mint: str, amount: int, max_slippage_bps: int
    ) -> tuple[str, int]:
        """Execute a Jupiter swap. Returns (tx_signature, out_amount_base_units)."""
        quote = await self._jupiter_quote(input_mint, output_mint, amount, max_slippage_bps)
        out_amount = int(quote.get("outAmount", 0))
        body = {
            "quoteResponse": quote,
            "userPublicKey": self._wallet.pubkey,
            "wrapAndUnwrapSol": True,
            "prioritizationFeeLamports": self._priority_fee,
        }
        resp = await self._http.post(JUPITER_SWAP_URL, json=body)
        resp.raise_for_status()
        swap_tx_b64 = resp.json()["swapTransaction"]
        tx_bytes = base64.b64decode(swap_tx_b64)
        signed = self._wallet.sign_versioned(tx_bytes)
        sig = await self._send_and_confirm(bytes(signed))
        return sig, out_amount

    # ---- send + confirm ----------------------------------------------------
    async def _send_and_confirm(self, signed_tx: bytes) -> str:
        tx_b64 = base64.b64encode(signed_tx).decode()
        delay = 2.0
        last_err = "unknown"
        for attempt in range(1, self._confirm_retries + 1):
            try:
                sig = await self._rpc(
                    "sendTransaction",
                    [tx_b64, {"encoding": "base64", "skipPreflight": True, "maxRetries": 0}],
                )
                if await self._confirm(sig):
                    log.info("solana.tx_confirmed", signature=sig, attempt=attempt)
                    return sig
                last_err = f"not confirmed within {self._confirm_timeout}s"
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                log.warning("solana.send_retry", attempt=attempt, error=last_err)
            await asyncio.sleep(delay)
            delay *= 2
        raise RuntimeError(f"send failed after {self._confirm_retries} attempts: {last_err}")

    async def _confirm(self, signature: str) -> bool:
        deadline = self._confirm_timeout
        waited = 0.0
        step = 2.0
        while waited < deadline:
            result = await self._rpc("getSignatureStatuses", [[signature], {"searchTransactionHistory": True}])
            statuses = (result or {}).get("value") or [None]
            status = statuses[0]
            if status:
                if status.get("err"):
                    raise RuntimeError(f"transaction failed on-chain: {status['err']}")
                conf = status.get("confirmationStatus")
                if conf in ("confirmed", "finalized"):
                    return True
            await asyncio.sleep(step)
            waited += step
        return False

    async def _rpc(self, method: str, params: list[Any]) -> Any:
        resp = await self._http.post(
            self._rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"RPC {method} error: {body['error']}")
        return body.get("result")
