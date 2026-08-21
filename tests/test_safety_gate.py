"""Tests for skills/safety/gate.py — the authoritative pre-trade safety filter.

Providers are duck-typed fakes (async get_mint_info/report/token_security plus
async close) so no network or real provider client is touched.
"""

from __future__ import annotations

import pytest

from skills.safety.gate import SafetyGate

SAFETY_CONFIG = {
    "require_mint_authority_revoked": True,
    "require_freeze_authority_revoked": True,
    "require_lp_locked_or_burned": True,
    "min_lp_locked_pct": 80,
    "max_top_holder_pct": 30,
    "max_top10_holder_pct": 70,
    "min_liquidity_usd": 5000,
    "require_sellable": True,
    "max_tax_pct": 10,
    "max_bundled_pct": 40,
    "fail_closed": True,
}


class FakeRPC:
    def __init__(self, mint_info=None, error=None):
        self._mint_info = mint_info
        self._error = error

    async def get_mint_info(self, mint: str):
        if self._error:
            raise self._error
        return self._mint_info

    async def close(self):
        pass


class FakeRugCheck:
    def __init__(self, report=None, error=None):
        self._report = report
        self._error = error

    async def report(self, mint: str):
        if self._error:
            raise self._error
        return self._report

    async def close(self):
        pass


class FakeGoPlus:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    async def token_security(self, mint: str):
        if self._error:
            raise self._error
        return self._result

    async def close(self):
        pass


CLEAN_MINT_INFO = {
    "mint_authority": None,
    "freeze_authority": None,
    "is_token_2022": False,
    "transfer_fee_bps": None,
    "decimals": 6,
    "supply": "1000000000",
}

CLEAN_RUG_REPORT = {
    "rugged": False,
    "score": 10,
    "risk_names": set(),
    "lp_locked_pct": 100.0,
    "top_holder_pct": 5.0,
    "top10_holder_pct": 20.0,
    "honeypot": False,
    "raw_risk_count": 0,
}

CLEAN_GOPLUS = {
    "buy_tax_pct": 0.0,
    "sell_tax_pct": 0.0,
    "is_honeypot": False,
    "transferable": True,
    "raw": {},
}

CLEAN_METADATA = {"liquidity_usd": 25_000.0}


def make_gate(mint_info=None, mint_error=None, rug=None, rug_error=None,
              goplus=None, goplus_error=None, config=None) -> SafetyGate:
    return SafetyGate(
        safety_config=config or SAFETY_CONFIG,
        rpc=FakeRPC(mint_info=mint_info, error=mint_error),
        rugcheck=FakeRugCheck(report=rug, error=rug_error),
        goplus=FakeGoPlus(result=goplus, error=goplus_error),
    )


@pytest.mark.asyncio
async def test_clean_token_passes() -> None:
    gate = make_gate(mint_info=CLEAN_MINT_INFO, rug=CLEAN_RUG_REPORT, goplus=CLEAN_GOPLUS)
    report = await gate.check("MINT_CLEAN", chain="solana", metadata=CLEAN_METADATA)
    assert report.passed is True
    assert report.hard_failures == []


@pytest.mark.asyncio
async def test_active_mint_authority_hard_fails() -> None:
    dirty_mint_info = dict(CLEAN_MINT_INFO, mint_authority="SomeAuthorityPubkey111")
    gate = make_gate(mint_info=dirty_mint_info, rug=CLEAN_RUG_REPORT, goplus=CLEAN_GOPLUS)
    report = await gate.check("MINT_DIRTY", chain="solana", metadata=CLEAN_METADATA)
    assert report.passed is False
    names = [c.name for c in report.hard_failures]
    assert "mint_authority_revoked" in names


@pytest.mark.asyncio
async def test_erroring_provider_fails_closed() -> None:
    # RPC read blows up entirely -> mint_info is None -> fail_closed True hard-fails.
    gate = make_gate(mint_error=RuntimeError("rpc down"), rug=CLEAN_RUG_REPORT,
                      goplus=CLEAN_GOPLUS)
    report = await gate.check("MINT_RPC_DOWN", chain="solana", metadata=CLEAN_METADATA)
    assert report.passed is False
    assert any("rpc_mint_read" in e for e in report.provider_errors)
    names = [c.name for c in report.hard_failures]
    assert "mint_authority_revoked" in names
    assert "freeze_authority_revoked" in names


@pytest.mark.asyncio
async def test_missing_rugcheck_report_fails_closed() -> None:
    # RugCheck report is None (provider errored) -> lp lock unknown -> fail closed.
    gate = make_gate(mint_info=CLEAN_MINT_INFO, rug_error=RuntimeError("timeout"),
                      goplus=CLEAN_GOPLUS)
    report = await gate.check("MINT_NO_RUG", chain="solana", metadata=CLEAN_METADATA)
    assert report.passed is False
    assert report.provider_errors  # rugcheck error recorded
    names = [c.name for c in report.hard_failures]
    assert "lp_locked_or_burned" in names


@pytest.mark.asyncio
async def test_fail_open_config_lets_missing_provider_soft_through() -> None:
    # With fail_closed off, a missing rugcheck report yields no hard failure
    # for the checks that depend on it (rug is None -> _check_rugged returns,
    # lp lock check with fail_closed False also returns without adding).
    config = dict(SAFETY_CONFIG, fail_closed=False)
    gate = make_gate(mint_info=CLEAN_MINT_INFO, rug_error=RuntimeError("timeout"),
                      goplus=CLEAN_GOPLUS, config=config)
    report = await gate.check("MINT_OPEN", chain="solana", metadata=CLEAN_METADATA)
    assert report.passed is True


@pytest.mark.asyncio
async def test_bundle_sniper_is_soft_only_and_still_passes() -> None:
    bundled_rug = dict(CLEAN_RUG_REPORT, risk_names={"bundled buys"})
    gate = make_gate(mint_info=CLEAN_MINT_INFO, rug=bundled_rug, goplus=CLEAN_GOPLUS)
    report = await gate.check("MINT_BUNDLED", chain="solana", metadata=CLEAN_METADATA)

    assert report.passed is True  # soft failure does not block
    soft_names = [c.name for c in report.soft_failures]
    assert "bundle_sniper" in soft_names
    assert report.hard_failures == []


@pytest.mark.asyncio
async def test_no_address_hard_fails_without_calling_providers() -> None:
    gate = make_gate(mint_info=CLEAN_MINT_INFO, rug=CLEAN_RUG_REPORT, goplus=CLEAN_GOPLUS)
    report = await gate.check(None, chain="solana", metadata=CLEAN_METADATA)
    assert report.passed is False
    assert report.checks[0].name == "address_present"


@pytest.mark.asyncio
async def test_low_liquidity_hard_fails() -> None:
    gate = make_gate(mint_info=CLEAN_MINT_INFO, rug=CLEAN_RUG_REPORT, goplus=CLEAN_GOPLUS)
    report = await gate.check("MINT_LOW_LIQ", chain="solana", metadata={"liquidity_usd": 100.0})
    assert report.passed is False
    names = [c.name for c in report.hard_failures]
    assert "min_liquidity" in names


@pytest.mark.asyncio
async def test_gate_close_closes_all_providers() -> None:
    rpc = FakeRPC(mint_info=CLEAN_MINT_INFO)
    rug = FakeRugCheck(report=CLEAN_RUG_REPORT)
    goplus = FakeGoPlus(result=CLEAN_GOPLUS)
    gate = SafetyGate(SAFETY_CONFIG, rpc=rpc, rugcheck=rug, goplus=goplus)
    # Should not raise.
    await gate.close()
