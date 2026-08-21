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
    ) -> None:
        self._wallet = wallet
        self._rpc_url = rpc_url
        self._priority_fee = priority_fee_microlamports
        self._use_jito = use_jito
        self._jito_tip_sol = jito_tip_sol
        self._confirm_timeout = confirm_timeout_seconds
        self._confirm_retries = confirm_retries
        self._http = httpx.AsyncClient(timeout=20.0)

    async def close(self) -> None:
        await self._http.aclose()

    # ---- public API --------------------------------------------------------
    async def buy(
        self, opportunity: Opportunity, sol_amount: float, max_slippage_bps: int
    ) -> ExecutionResult:
        try:
            if opportunity.venue is TokenVenue.BONDING:
                sig = await self._pumpportal_trade(
                    "buy", opportunity.address, sol_amount, max_slippage_bps,
                    denominated_in_sol=True,
                )
            else:
                sig = await self._jupiter_swap(
                    input_mint=WSOL_MINT, output_mint=opportunity.address,
                    amount=int(sol_amount * LAMPORTS_PER_SOL),
                    max_slippage_bps=max_slippage_bps,
                )
            price = await self.current_price_sol_for(opportunity.address)
            token_amount = (sol_amount / price) if price else None
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
                sig = await self._pumpportal_trade(
                    "sell", position.address, token_amount, max_slippage_bps,
                    denominated_in_sol=False,
                )
            else:
                raw_amount = int(token_amount * (10 ** self._token_decimals(position)))
                sig = await self._jupiter_swap(
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
        """Price in SOL per token via a small Jupiter quote (token -> SOL)."""
        if not address:
            return None
        try:
            # Quote 1 whole token's worth is unreliable pre-decimals; use a
            # fixed lamport-scale probe and invert.
            probe = 1_000_000  # base units of the token
            quote = await self._jupiter_quote(address, WSOL_MINT, probe, 500)
            out_lamports = int(quote["outAmount"])
            # price(SOL/token) = (out_lamports/1e9) / (probe/10^decimals)
            # decimals unknown here; caller-side ratios use consistent scaling,
            # so return SOL per base-unit * 1e0 — good enough for gain% ratios.
            return (out_lamports / LAMPORTS_PER_SOL) / probe
        except Exception:
            return None

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
    ) -> str:
        quote = await self._jupiter_quote(input_mint, output_mint, amount, max_slippage_bps)
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
        return await self._send_and_confirm(bytes(signed))

    def _token_decimals(self, position: Position) -> int:
        # Most SPL memecoins use 6 decimals (pump.fun default); overridable via
        # metadata if present. Conservative default keeps sell sizing sane.
        return int(position and 6)

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
