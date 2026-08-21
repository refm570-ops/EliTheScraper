"""Domain models for the trading subsystem.

Dataclasses (rather than the loose dicts used elsewhere) are used here on
purpose: this code moves money, so explicit, typed, self-documenting shapes
reduce the chance of a silent field-name mistake causing a bad trade.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class Chain(str, Enum):
    SOLANA = "solana"
    ETHEREUM = "ethereum"
    BASE = "base"
    UNKNOWN = "unknown"


class TokenVenue(str, Enum):
    """Where a token trades — decides the execution route."""

    BONDING = "bonding"  # pump.fun-style bonding curve (pre-graduation)
    AMM = "amm"          # graduated / standard AMM pool (Jupiter, Raydium…)
    UNKNOWN = "unknown"


class Conviction(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class DecisionAction(str, Enum):
    BUY = "BUY"
    SKIP = "SKIP"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


# ---------------------------------------------------------------------------
# Opportunity — the input to the buy decision.
# ---------------------------------------------------------------------------
@dataclass
class Opportunity:
    """A candidate token surfaced by the signal pipeline, worth evaluating."""

    ticker: str
    address: str | None = None
    chain: Chain = Chain.SOLANA
    venue: TokenVenue = TokenVenue.UNKNOWN
    source: str = "unknown"            # discovery source (telegram/twitter/launchpad)
    aggregator_score: float = 0.0      # SignalAggregator on-chain score
    alert_level: str | None = None     # act_now / interesting / watch
    social_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)   # TokenMetadataFetcher output
    source_reputation: float | None = None                   # 0..1, from discovery layer
    ta: dict[str, Any] | None = None                         # TAAnalyzer signal (chart structure)
    created_at: float = field(default_factory=time.time)

    @property
    def liquidity_usd(self) -> float | None:
        return self.metadata.get("liquidity_usd")


# ---------------------------------------------------------------------------
# Safety — authoritative pass/fail gate.
# ---------------------------------------------------------------------------
class Severity(str, Enum):
    HARD = "HARD"      # must pass; a failure blocks the buy unconditionally
    SOFT = "SOFT"      # advisory; surfaced to the evaluator but not blocking


@dataclass
class SafetyCheck:
    name: str
    passed: bool
    severity: Severity
    detail: str = ""
    value: Any = None


@dataclass
class SafetyReport:
    address: str | None
    checks: list[SafetyCheck] = field(default_factory=list)
    provider_errors: list[str] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)

    def add(self, check: SafetyCheck) -> None:
        self.checks.append(check)

    @property
    def hard_failures(self) -> list[SafetyCheck]:
        return [c for c in self.checks if c.severity is Severity.HARD and not c.passed]

    @property
    def soft_failures(self) -> list[SafetyCheck]:
        return [c for c in self.checks if c.severity is Severity.SOFT and not c.passed]

    @property
    def passed(self) -> bool:
        """A report passes only if NO hard check failed."""
        return not self.hard_failures

    def blocking_reasons(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.hard_failures]

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "passed": self.passed,
            "hard_failures": self.blocking_reasons(),
            "soft_failures": [f"{c.name}: {c.detail}" for c in self.soft_failures],
            "provider_errors": self.provider_errors,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity.value,
                    "detail": c.detail,
                    "value": c.value,
                }
                for c in self.checks
            ],
            "checked_at": self.checked_at,
        }


# ---------------------------------------------------------------------------
# Decision — the evaluator agent's structured output.
# ---------------------------------------------------------------------------
@dataclass
class TradeDecision:
    action: DecisionAction
    conviction: Conviction
    size_sol: float                 # requested size (pre risk-clamp)
    max_slippage_bps: int
    reasoning: str
    ttl_seconds: int = 120          # memecoin edges decay; proposal void after this
    model: str = ""                 # which model produced the decision
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.action is DecisionAction.BUY


# ---------------------------------------------------------------------------
# Proposal — bundles everything needed to approve/execute a buy.
# ---------------------------------------------------------------------------
@dataclass
class TradeProposal:
    opportunity: Opportunity
    safety: SafetyReport
    decision: TradeDecision
    approved_size_sol: float        # post risk-clamp size actually to be sent
    approved_slippage_bps: int = 500  # post risk-clamp slippage (<= config cap)
    id: str = field(default_factory=_new_id)
    created_at: float = field(default_factory=time.time)

    @property
    def expires_at(self) -> float:
        return self.created_at + self.decision.ttl_seconds

    def is_expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) >= self.expires_at


# ---------------------------------------------------------------------------
# Execution result — what the executor returns.
# ---------------------------------------------------------------------------
@dataclass
class ExecutionResult:
    success: bool
    side: TradeSide
    address: str | None = None
    tx_signature: str | None = None
    sol_amount: float | None = None       # SOL spent (buy) or received (sell)
    token_amount: float | None = None     # tokens received (buy) or sold (sell)
    price: float | None = None            # SOL per token at fill
    is_paper: bool = False
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Position — an open holding tracked for exit.
# ---------------------------------------------------------------------------
@dataclass
class Position:
    address: str
    ticker: str
    chain: Chain
    venue: TokenVenue
    source: str
    entry_price: float                    # SOL per token
    amount_sol: float                     # SOL deployed at entry
    token_amount: float                   # tokens currently held
    initial_token_amount: float           # tokens at entry (for ladder portions)
    status: PositionStatus = PositionStatus.OPEN
    is_paper: bool = False
    entry_tx: str | None = None
    peak_price: float | None = None       # for trailing stop
    realized_pnl_sol: float = 0.0
    take_profit_hits: int = 0             # ladder rungs already taken
    id: str = field(default_factory=_new_id)
    opened_at: float = field(default_factory=time.time)
    closed_at: float | None = None

    def unrealized_pnl_sol(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.token_amount

    def gain_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price * 100.0
