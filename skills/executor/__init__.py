"""Execution engine — places buys/sells on Solana, or simulates them (paper)."""

from skills.executor.base import Executor
from skills.executor.paper import PaperExecutor

__all__ = ["Executor", "PaperExecutor"]
