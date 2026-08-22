"""Tests for trading/config.py — the master safety-switch config object."""

from __future__ import annotations

import pytest

from trading.config import HARD_ABSOLUTE_MAX_TRADE_SOL, TradingConfig, load_trading_config


def test_default_config_is_safe() -> None:
    """A bare TradingConfig() must default to the safest possible posture."""
    cfg = TradingConfig()
    assert cfg.mode == "paper"
    assert cfg.autonomy == "approval"
    assert cfg.is_paper is True
    assert cfg.is_live is False
    assert cfg.is_auto is False


def test_max_trade_sol_never_exceeds_hard_ceiling() -> None:
    cfg = TradingConfig(risk={"max_trade_sol": 999.0})
    assert cfg.max_trade_sol() == HARD_ABSOLUTE_MAX_TRADE_SOL


def test_max_trade_sol_uses_configured_value_when_under_ceiling() -> None:
    cfg = TradingConfig(risk={"max_trade_sol": 0.1})
    assert cfg.max_trade_sol() == pytest.approx(0.1)


def test_max_trade_sol_default_when_unconfigured() -> None:
    cfg = TradingConfig()
    assert cfg.max_trade_sol() == pytest.approx(0.25)


def test_is_auto_false_when_autonomy_is_approval() -> None:
    # Even paper + trading_enabled + allow_auto, approval autonomy always wins.
    cfg = TradingConfig(
        mode="paper",
        autonomy="approval",
        trading_enabled=True,
        allow_auto=True,
    )
    assert cfg.is_auto is False


def test_is_live_requires_mode_live_and_trading_enabled() -> None:
    # mode=live alone is not enough.
    cfg = TradingConfig(mode="live", trading_enabled=False)
    assert cfg.is_live is False

    # trading_enabled alone (still paper) is not enough.
    cfg = TradingConfig(mode="paper", trading_enabled=True)
    assert cfg.is_live is False

    # Both together -> live.
    cfg = TradingConfig(mode="live", trading_enabled=True)
    assert cfg.is_live is True


def test_is_auto_in_live_requires_allow_auto() -> None:
    # live + auto but allow_auto False -> not auto.
    cfg = TradingConfig(
        mode="live", autonomy="auto", trading_enabled=True, allow_auto=False
    )
    assert cfg.is_auto is False

    # live + auto + trading_enabled + allow_auto -> auto.
    cfg = TradingConfig(
        mode="live", autonomy="auto", trading_enabled=True, allow_auto=True
    )
    assert cfg.is_auto is True

    # live + auto + allow_auto but trading NOT enabled -> is_live False, so the
    # config is treated as paper (is_paper True) and auto is allowed freely,
    # same as any other non-live config.
    cfg = TradingConfig(
        mode="live", autonomy="auto", trading_enabled=False, allow_auto=True
    )
    assert cfg.is_live is False
    assert cfg.is_paper is True
    assert cfg.is_auto is True


def test_paper_plus_auto_is_auto_true() -> None:
    """Paper mode may auto-execute freely regardless of allow_auto."""
    cfg = TradingConfig(mode="paper", autonomy="auto", allow_auto=False)
    assert cfg.is_paper is True
    assert cfg.is_auto is True


def test_load_trading_config_missing_file_falls_back_to_safe_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.delenv("TRADING_AUTONOMY", raising=False)
    monkeypatch.delenv("TRADING_ENABLED", raising=False)
    monkeypatch.delenv("TRADING_ALLOW_AUTO", raising=False)

    missing = tmp_path / "does-not-exist.yml"
    cfg = load_trading_config(missing)
    assert cfg.mode == "paper"
    assert cfg.autonomy == "approval"
    assert cfg.is_live is False
    assert cfg.is_auto is False


def test_load_trading_config_env_overrides_mode_and_autonomy(tmp_path, monkeypatch) -> None:
    yml = tmp_path / "trading.yml"
    yml.write_text("mode: paper\nautonomy: approval\n")

    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("TRADING_AUTONOMY", "auto")
    monkeypatch.setenv("TRADING_ENABLED", "true")
    monkeypatch.setenv("TRADING_ALLOW_AUTO", "true")

    cfg = load_trading_config(yml)
    assert cfg.mode == "live"
    assert cfg.autonomy == "auto"
    assert cfg.is_live is True
    assert cfg.is_auto is True
