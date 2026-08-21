"""Tests for trading/models.py domain dataclasses."""

from __future__ import annotations

import time

from trading.models import (
    Chain,
    Conviction,
    DecisionAction,
    Opportunity,
    SafetyCheck,
    SafetyReport,
    Severity,
    TokenVenue,
    TradeDecision,
    TradeProposal,
)


def make_decision(ttl_seconds: int = 120) -> TradeDecision:
    return TradeDecision(
        action=DecisionAction.BUY,
        conviction=Conviction.HIGH,
        size_sol=0.1,
        max_slippage_bps=1000,
        reasoning="test",
        ttl_seconds=ttl_seconds,
    )


def make_opportunity() -> Opportunity:
    return Opportunity(ticker="$FOO", address="MINT1", chain=Chain.SOLANA, venue=TokenVenue.AMM)


# ---------------------------------------------------------------------------
# SafetyReport.passed
# ---------------------------------------------------------------------------

def test_safety_report_passes_with_no_checks() -> None:
    report = SafetyReport(address="MINT1")
    assert report.passed is True


def test_safety_report_passes_with_only_soft_failure() -> None:
    report = SafetyReport(address="MINT1")
    report.add(SafetyCheck("bundle_sniper", False, Severity.SOFT, "bundled"))
    assert report.passed is True
    assert report.soft_failures[0].name == "bundle_sniper"
    assert report.hard_failures == []


def test_safety_report_fails_with_hard_failure() -> None:
    report = SafetyReport(address="MINT1")
    report.add(SafetyCheck("bundle_sniper", False, Severity.SOFT, "bundled"))
    report.add(SafetyCheck("mint_authority_revoked", False, Severity.HARD, "active authority"))
    assert report.passed is False
    assert [c.name for c in report.hard_failures] == ["mint_authority_revoked"]
    assert report.blocking_reasons() == ["mint_authority_revoked: active authority"]


def test_safety_report_passing_hard_check_does_not_block() -> None:
    report = SafetyReport(address="MINT1")
    report.add(SafetyCheck("mint_authority_revoked", True, Severity.HARD, "revoked"))
    assert report.passed is True


def test_safety_report_to_dict_shape() -> None:
    report = SafetyReport(address="MINT1")
    report.add(SafetyCheck("mint_authority_revoked", True, Severity.HARD, "revoked"))
    d = report.to_dict()
    assert d["address"] == "MINT1"
    assert d["passed"] is True
    assert d["hard_failures"] == []
    assert len(d["checks"]) == 1
    assert d["checks"][0]["name"] == "mint_authority_revoked"


# ---------------------------------------------------------------------------
# TradeProposal.is_expired
# ---------------------------------------------------------------------------

def test_trade_proposal_not_expired_immediately() -> None:
    proposal = TradeProposal(
        opportunity=make_opportunity(),
        safety=SafetyReport(address="MINT1"),
        decision=make_decision(ttl_seconds=120),
        approved_size_sol=0.1,
    )
    assert proposal.is_expired() is False


def test_trade_proposal_expired_after_ttl() -> None:
    created = time.time() - 1000
    proposal = TradeProposal(
        opportunity=make_opportunity(),
        safety=SafetyReport(address="MINT1"),
        decision=make_decision(ttl_seconds=120),
        approved_size_sol=0.1,
        created_at=created,
    )
    assert proposal.is_expired() is True
    assert proposal.expires_at == created + 120


def test_trade_proposal_is_expired_with_explicit_now() -> None:
    proposal = TradeProposal(
        opportunity=make_opportunity(),
        safety=SafetyReport(address="MINT1"),
        decision=make_decision(ttl_seconds=60),
        approved_size_sol=0.1,
        created_at=1000.0,
    )
    assert proposal.is_expired(now=1059.0) is False
    assert proposal.is_expired(now=1060.0) is True
    assert proposal.is_expired(now=1200.0) is True
