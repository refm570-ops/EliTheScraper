"""Executor interface.

Prices are always expressed as SOL per token so Position math stays consistent
across paper and live. buy() spends SOL to acquire tokens; sell() disposes of
tokens for SOL. Neither raises on a failed trade — both return an
ExecutionResult with success=False and an error string, so the caller can
record the attempt without a crash mid-book.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trading.models import ExecutionResult, Opportunity, Position


class Executor(ABC):
    is_paper: bool = True

    @abstractmethod
    async def buy(
        self,
        opportunity: Opportunity,
        sol_amount: float,
        max_slippage_bps: int,
    ) -> ExecutionResult:
        ...

    @abstractmethod
    async def sell(
        self,
        position: Position,
        token_amount: float,
        max_slippage_bps: int,
    ) -> ExecutionResult:
        ...

    @abstractmethod
    async def current_price_sol(self, position: Position) -> float | None:
        """Current price in SOL per token, for exit evaluation. None if unknown."""
        ...

    async def close(self) -> None:  # pragma: no cover - default no-op
        return None
