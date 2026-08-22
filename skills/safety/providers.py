"""Data providers for the safety gate.

Two tiers, by trust:
  - SolanaRPC: direct on-chain reads. AUTHORITATIVE — we read the chain itself.
    Used for the checks that decide whether we can even sell (mint authority,
    freeze authority, Token-2022 transfer-fee extension). Never outsourced.
  - RugCheck / GoPlus: third-party aggregators for convenience signals (LP
    lock, honeypot flags, holder concentration). Cross-checked against RPC
    where possible; treated as fail-closed when unreachable.

Every method either returns parsed data or raises; the gate decides how a
failure maps to a check result (fail-closed by config).
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

log = structlog.get_logger()

DEFAULT_SOLANA_RPC = "https://api.mainnet-beta.solana.com"
RUGCHECK_REPORT_URL = "https://api.rugcheck.xyz/v1/tokens/{mint}/report"
GOPLUS_SOLANA_URL = "https://api.gopluslabs.io/api/v1/solana/token_security"

# System program address that authorities are set to when "revoked" is
# represented as the null address rather than a missing field.
_NULL_ADDRESSES = {
    "11111111111111111111111111111111",
    None,
    "",
}


class SolanaRPC:
    """Minimal JSON-RPC client for the authoritative on-chain reads."""

    def __init__(self, rpc_url: str | None = None, timeout: float = 10.0) -> None:
        self._url = rpc_url or DEFAULT_SOLANA_RPC
        self._http = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def _rpc(self, method: str, params: list[Any]) -> Any:
        resp = await self._http.post(
            self._url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"RPC error for {method}: {body['error']}")
        return body.get("result")

    async def get_mint_info(self, mint: str) -> dict[str, Any]:
        """Read the SPL mint account. Authoritative source of authorities.

        Returns a dict with keys:
          mint_authority, freeze_authority (str | None),
          is_token_2022 (bool), transfer_fee_bps (int | None),
          decimals, supply.
        Raises if the account can't be read or parsed.
        """
        result = await self._rpc(
            "getAccountInfo",
            [mint, {"encoding": "jsonParsed", "commitment": "confirmed"}],
        )
        value = (result or {}).get("value")
        if not value:
            raise RuntimeError(f"mint account not found: {mint}")

        owner_program = value.get("owner")
        is_token_2022 = owner_program == "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

        parsed = (((value.get("data") or {}).get("parsed")) or {})
        info = parsed.get("info") or {}

        mint_authority = info.get("mintAuthority")
        freeze_authority = info.get("freezeAuthority")

        # Token-2022 transfer-fee extension → sell/buy tax.
        transfer_fee_bps: int | None = None
        for ext in info.get("extensions", []) or []:
            if ext.get("extension") == "transferFeeConfig":
                state = ext.get("state") or {}
                newer = state.get("newerTransferFee") or {}
                bps = newer.get("transferFeeBasisPoints")
                if bps is not None:
                    transfer_fee_bps = int(bps)

        return {
            "mint_authority": mint_authority if mint_authority not in _NULL_ADDRESSES else None,
            "freeze_authority": freeze_authority if freeze_authority not in _NULL_ADDRESSES else None,
            "is_token_2022": is_token_2022,
            "transfer_fee_bps": transfer_fee_bps,
            "decimals": info.get("decimals"),
            "supply": info.get("supply"),
        }


class RugCheckClient:
    """RugCheck aggregated report — LP lock, top holders, risk flags."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._http = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def report(self, mint: str) -> dict[str, Any]:
        """Fetch and normalize a RugCheck report. Raises on failure."""
        resp = await self._http.get(RUGCHECK_REPORT_URL.format(mint=mint))
        resp.raise_for_status()
        body = resp.json()

        # Normalize the fields the gate cares about; RugCheck's schema is broad.
        risks = body.get("risks") or []
        risk_names = {(r.get("name") or "").lower() for r in risks}

        # LP locked/burned percentage across markets.
        lp_locked_pct = None
        markets = body.get("markets") or []
        for m in markets:
            lp = m.get("lp") or {}
            pct = lp.get("lpLockedPct")
            if pct is not None:
                lp_locked_pct = max(lp_locked_pct or 0.0, float(pct))

        top_holders = body.get("topHolders") or []
        # Exclude LP/pool accounts flagged as insiders where possible.
        top_holder_pct = None
        top10_pct = None
        if top_holders:
            pcts = sorted(
                (float(h.get("pct", 0.0)) for h in top_holders), reverse=True
            )
            if pcts:
                top_holder_pct = pcts[0]
                top10_pct = sum(pcts[:10])

        return {
            "rugged": bool(body.get("rugged", False)),
            "score": body.get("score"),
            "risk_names": risk_names,
            "lp_locked_pct": lp_locked_pct,
            "top_holder_pct": top_holder_pct,
            "top10_holder_pct": top10_pct,
            "honeypot": "honeypot" in risk_names,
            "raw_risk_count": len(risks),
        }


class GoPlusClient:
    """GoPlus token-security — tax and honeypot flags (Solana coverage newer)."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._http = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def token_security(self, mint: str) -> dict[str, Any]:
        """Fetch GoPlus Solana token security. Raises on failure."""
        resp = await self._http.get(GOPLUS_SOLANA_URL, params={"contract_addresses": mint})
        resp.raise_for_status()
        body = resp.json()
        result = (body.get("result") or {})
        # Result is keyed by mint (case can vary).
        entry: dict[str, Any] = {}
        for key, val in result.items():
            if key.lower() == mint.lower():
                entry = val
                break
        if not entry and result:
            entry = next(iter(result.values()))

        def _pct(v: Any) -> float | None:
            try:
                return float(v) * 100.0
            except (TypeError, ValueError):
                return None

        return {
            "buy_tax_pct": _pct(entry.get("buy_tax")),
            "sell_tax_pct": _pct(entry.get("sell_tax")),
            "is_honeypot": entry.get("is_honeypot") in ("1", 1, True),
            "transferable": entry.get("transfer_pausable") not in ("1", 1, True),
            "raw": entry,
        }
