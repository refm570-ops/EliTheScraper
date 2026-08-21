"""Pre-trade safety gate.

Authoritative, fail-closed rug/honeypot checks that HARD-BLOCK a buy. The
opportunity-evaluator agent cannot override a hard failure here — the gate runs
first and the pipeline stops if it does not pass.
"""

from skills.safety.gate import SafetyGate

__all__ = ["SafetyGate"]
