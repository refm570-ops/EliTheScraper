"""Autonomous buy-agent subsystem.

Turns opportunity signals from the discovery/scoring pipeline into buy
decisions, gates every buy behind a hard safety check and coded risk limits,
executes on Solana (paper or live), and manages exits.

Design principles:
  - SAFE BY DEFAULT: paper mode + human approval unless explicitly opted out.
  - Hard limits are enforced in code, never left to an LLM prompt.
  - The safety gate is authoritative: the evaluator agent cannot override it.
"""
