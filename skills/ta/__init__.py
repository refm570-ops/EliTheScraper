"""Technical-analysis signals for the buy agent.

Vendors the self-contained TA detectors from the sibling `tradingbot` project
(support/resistance, bull/bear-flag, volume) and exposes a single TAAnalyzer
that turns a token's OHLCV candles into a compact TASignal the evaluator and
exit monitor can reason over. Chart structure was the missing dimension: the
agent traded on social + on-chain data but never looked at price action.
"""

from skills.ta.analyzer import TAAnalyzer

__all__ = ["TAAnalyzer"]
