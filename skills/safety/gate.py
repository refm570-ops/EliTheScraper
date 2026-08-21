"""SafetyGate — the authoritative pre-trade rug/honeypot filter.

Runs BEFORE the evaluator LLM: cheap, deterministic, and the pipeline stops if
it does not pass. Philosophy: FAIL CLOSED. If a check cannot be verified (a
provider errors or returns nothing) and config.safety.fail_closed is true, the
check is recorded as a HARD failure. A rug that merely breaks the checker must
never slip through.
"""

from __future__ import annotations

from typing import Any

import structlog

from skills.safety.providers import GoPlusClient, RugCheckClient, SolanaRPC
from trading.models import SafetyCheck, SafetyReport, Severity

log = structlog.get_logger()


class SafetyGate:
    def __init__(
        self,
        safety_config: dict[str, Any],
        rpc: SolanaRPC | None = None,
        rugcheck: RugCheckClient | None = None,
        goplus: GoPlusClient | None = None,
    ) -> None:
        self._cfg = safety_config or {}
        self._rpc = rpc or SolanaRPC()
        self._rugcheck = rugcheck or RugCheckClient()
        self._goplus = goplus or GoPlusClient()

    async def close(self) -> None:
        await self._rpc.close()
        await self._rugcheck.close()
        await self._goplus.close()

    @property
    def _fail_closed(self) -> bool:
        return bool(self._cfg.get("fail_closed", True))

    async def check(
        self,
        address: str | None,
        chain: str = "solana",
        metadata: dict[str, Any] | None = None,
    ) -> SafetyReport:
        report = SafetyReport(address=address)
        metadata = metadata or {}

        # Without a contract address we cannot verify anything on-chain.
        if not address:
            report.add(SafetyCheck(
                "address_present", False, Severity.HARD,
                "no contract address available to verify",
            ))
            return report

        if chain != "solana":
            # On-chain authority reads here are Solana-specific.
            report.add(SafetyCheck(
                "supported_chain", not self._fail_closed, Severity.HARD,
                f"chain {chain} not supported by safety gate",
            ))
            if self._fail_closed:
                return report

        # ---- Gather provider data (fail-closed on error) -------------------
        mint_info = await self._safe(report, "rpc_mint_read", self._rpc.get_mint_info, address)
        rug = await self._safe(report, "rugcheck", self._rugcheck.report, address)
        goplus = await self._safe(report, "goplus", self._goplus.token_security, address)

        # ---- Hard checks ---------------------------------------------------
        self._check_authorities(report, mint_info)
        self._check_rugged(report, rug)
        self._check_liquidity(report, metadata)
        self._check_lp_lock(report, rug)
        self._check_holders(report, rug, metadata)
        self._check_sellable(report, rug, goplus)
        self._check_tax(report, mint_info, goplus)

        # ---- Soft checks (advisory, surfaced to evaluator) -----------------
        self._check_bundled(report, rug)

        log.info(
            "safety.checked",
            address=address,
            passed=report.passed,
            hard_failures=[c.name for c in report.hard_failures],
            provider_errors=report.provider_errors,
        )
        return report

    async def _safe(self, report: SafetyReport, name: str, fn, *args) -> dict[str, Any] | None:
        """Call a provider; on error record it and return None (gate fails closed)."""
        try:
            return await fn(*args)
        except Exception as e:  # noqa: BLE001 - provider errors must not crash the gate
            report.provider_errors.append(f"{name}: {type(e).__name__}: {e}")
            log.warning("safety.provider_error", provider=name, error=str(e))
            return None

    # ---- individual checks -------------------------------------------------
    def _check_authorities(self, report: SafetyReport, mint_info: dict | None) -> None:
        want_mint = self._cfg.get("require_mint_authority_revoked", True)
        want_freeze = self._cfg.get("require_freeze_authority_revoked", True)

        if mint_info is None:
            if self._fail_closed:
                if want_mint:
                    report.add(SafetyCheck("mint_authority_revoked", False, Severity.HARD,
                                           "could not read mint account (fail-closed)"))
                if want_freeze:
                    report.add(SafetyCheck("freeze_authority_revoked", False, Severity.HARD,
                                           "could not read mint account (fail-closed)"))
            return

        if want_mint:
            revoked = mint_info.get("mint_authority") is None
            report.add(SafetyCheck(
                "mint_authority_revoked", revoked, Severity.HARD,
                "mint authority active — supply can be inflated" if not revoked else "revoked",
                value=mint_info.get("mint_authority"),
            ))
        if want_freeze:
            revoked = mint_info.get("freeze_authority") is None
            report.add(SafetyCheck(
                "freeze_authority_revoked", revoked, Severity.HARD,
                "freeze authority active — your tokens can be frozen (unsellable)"
                if not revoked else "revoked",
                value=mint_info.get("freeze_authority"),
            ))

    def _check_rugged(self, report: SafetyReport, rug: dict | None) -> None:
        if rug is None:
            return  # covered by other fail-closed checks
        if rug.get("rugged"):
            report.add(SafetyCheck("not_rugged", False, Severity.HARD,
                                   "RugCheck flags token as already rugged"))

    def _check_liquidity(self, report: SafetyReport, metadata: dict) -> None:
        min_liq = self._cfg.get("min_liquidity_usd", 5000)
        liq = metadata.get("liquidity_usd")
        if liq is None:
            if self._fail_closed:
                report.add(SafetyCheck("min_liquidity", False, Severity.HARD,
                                       "liquidity unknown (fail-closed)"))
            return
        ok = liq >= min_liq
        report.add(SafetyCheck("min_liquidity", ok, Severity.HARD,
                               f"liquidity ${liq:,.0f} < ${min_liq:,.0f} floor" if not ok
                               else f"${liq:,.0f}", value=liq))

    def _check_lp_lock(self, report: SafetyReport, rug: dict | None) -> None:
        if not self._cfg.get("require_lp_locked_or_burned", True):
            return
        min_pct = self._cfg.get("min_lp_locked_pct", 80)
        if rug is None:
            if self._fail_closed:
                report.add(SafetyCheck("lp_locked_or_burned", False, Severity.HARD,
                                       "LP lock status unknown (fail-closed)"))
            return
        pct = rug.get("lp_locked_pct")
        if pct is None:
            if self._fail_closed:
                report.add(SafetyCheck("lp_locked_or_burned", False, Severity.HARD,
                                       "LP lock percentage unavailable (fail-closed)"))
            return
        ok = pct >= min_pct
        report.add(SafetyCheck("lp_locked_or_burned", ok, Severity.HARD,
                               f"only {pct:.0f}% of LP locked/burned (need {min_pct}%)"
                               if not ok else f"{pct:.0f}% locked/burned", value=pct))

    def _check_holders(self, report: SafetyReport, rug: dict | None, metadata: dict) -> None:
        max_top = self._cfg.get("max_top_holder_pct", 30)
        max_top10 = self._cfg.get("max_top10_holder_pct", 70)

        top_holder = (rug or {}).get("top_holder_pct")
        top10 = (rug or {}).get("top10_holder_pct")
        if top10 is None:
            top10 = metadata.get("top10_holder_pct")  # Birdeye fallback

        if top_holder is not None:
            ok = top_holder <= max_top
            report.add(SafetyCheck("top_holder_concentration", ok, Severity.HARD,
                                   f"top holder {top_holder:.0f}% > {max_top}% cap"
                                   if not ok else f"top {top_holder:.0f}%", value=top_holder))
        if top10 is not None:
            ok = top10 <= max_top10
            report.add(SafetyCheck("top10_concentration", ok, Severity.HARD,
                                   f"top-10 {top10:.0f}% > {max_top10}% cap"
                                   if not ok else f"top-10 {top10:.0f}%", value=top10))
        elif self._fail_closed:
            report.add(SafetyCheck("top10_concentration", False, Severity.HARD,
                                   "holder distribution unknown (fail-closed)"))

    def _check_sellable(self, report: SafetyReport, rug: dict | None, goplus: dict | None) -> None:
        if not self._cfg.get("require_sellable", True):
            return
        signals: list[bool] = []
        if rug is not None:
            signals.append(not rug.get("honeypot", False))
        if goplus is not None:
            signals.append(not goplus.get("is_honeypot", False))
        if not signals:
            if self._fail_closed:
                report.add(SafetyCheck("sellable", False, Severity.HARD,
                                       "no honeypot signal available (fail-closed)"))
            return
        ok = all(signals)
        report.add(SafetyCheck("sellable", ok, Severity.HARD,
                               "honeypot / non-sellable flagged" if not ok else "sell simulated ok"))

    def _check_tax(self, report: SafetyReport, mint_info: dict | None, goplus: dict | None) -> None:
        max_tax = self._cfg.get("max_tax_pct", 10)
        taxes: list[float] = []
        if mint_info and mint_info.get("transfer_fee_bps") is not None:
            taxes.append(mint_info["transfer_fee_bps"] / 100.0)
        if goplus:
            for k in ("buy_tax_pct", "sell_tax_pct"):
                if goplus.get(k) is not None:
                    taxes.append(goplus[k])
        if not taxes:
            return  # no tax info; not fail-closed (most SPL tokens have none)
        worst = max(taxes)
        ok = worst <= max_tax
        report.add(SafetyCheck("transfer_tax", ok, Severity.HARD,
                               f"transfer tax {worst:.1f}% > {max_tax}% cap" if not ok
                               else f"{worst:.1f}%", value=worst))

    def _check_bundled(self, report: SafetyReport, rug: dict | None) -> None:
        # Best-effort / low reliability → SOFT (advisory for the evaluator).
        if rug is None:
            return
        risk_names = rug.get("risk_names") or set()
        bundled = any("bundl" in r or "sniper" in r for r in risk_names)
        report.add(SafetyCheck("bundle_sniper", not bundled, Severity.SOFT,
                               "bundle/sniper risk flagged by RugCheck" if bundled else "ok"))
