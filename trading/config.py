"""Central loader for trading configuration.

Merges config/trading.yml with environment-variable overrides for the master
switches (mode / autonomy / kill switches), so the safe defaults committed to
the repo can never be silently turned into a live-trading config by editing YAML
alone — going live requires explicit env vars on the host.

An absolute spend ceiling is hard-coded here, ABOVE anything config or an LLM
can request. It is the last backstop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger()

# Hard-coded absolute ceiling on a single trade, in SOL. Nothing — not config,
# not the evaluator LLM, not autonomy mode — may exceed this. Change requires a
# code edit and review.
HARD_ABSOLUTE_MAX_TRADE_SOL = 1.0

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "trading.yml"


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class TradingConfig:
    mode: str = "paper"            # paper | live
    autonomy: str = "approval"    # approval | auto
    # Kill switches / gating (env-driven, default OFF)
    trading_enabled: bool = False     # must be true for live mode to place buys
    allow_auto: bool = False          # must be true for autonomy=auto to be honored
    risk: dict[str, Any] = field(default_factory=dict)
    sizing: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    exit: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        """Live trading is active ONLY when mode=live AND the env kill switch is on."""
        return self.mode == "live" and self.trading_enabled

    @property
    def is_paper(self) -> bool:
        return not self.is_live

    @property
    def is_auto(self) -> bool:
        """Whether to execute without human approval.

        Paper mode may auto-execute freely (it risks nothing) so the loop can be
        validated hands-off. Live auto-execution is honored ONLY when explicitly
        allowed via env (TRADING_ALLOW_AUTO) on top of live mode.
        """
        if self.autonomy != "auto":
            return False
        if self.is_paper:
            return True
        return self.allow_auto and self.is_live

    def max_trade_sol(self) -> float:
        configured = float(self.risk.get("max_trade_sol", 0.25))
        return min(configured, HARD_ABSOLUTE_MAX_TRADE_SOL)


def load_trading_config(path: Path | None = None) -> TradingConfig:
    cfg_path = path or _CONFIG_PATH
    try:
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning("trading_config.missing", path=str(cfg_path))
        raw = {}

    mode = os.getenv("TRADING_MODE", raw.get("mode", "paper")).strip().lower()
    autonomy = os.getenv("TRADING_AUTONOMY", raw.get("autonomy", "approval")).strip().lower()
    if mode not in ("paper", "live"):
        log.warning("trading_config.bad_mode", mode=mode)
        mode = "paper"
    if autonomy not in ("approval", "auto"):
        autonomy = "approval"

    config = TradingConfig(
        mode=mode,
        autonomy=autonomy,
        trading_enabled=_env_bool("TRADING_ENABLED", False),
        allow_auto=_env_bool("TRADING_ALLOW_AUTO", False),
        risk=raw.get("risk", {}),
        sizing=raw.get("sizing", {}),
        execution=raw.get("execution", {}),
        exit=raw.get("exit", {}),
        safety=raw.get("safety", {}),
    )

    log.info(
        "trading_config.loaded",
        mode=config.mode,
        autonomy=config.autonomy,
        is_live=config.is_live,
        is_auto=config.is_auto,
        max_trade_sol=config.max_trade_sol(),
    )
    return config
