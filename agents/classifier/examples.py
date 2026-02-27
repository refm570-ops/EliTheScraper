from __future__ import annotations

# Edge case examples for testing the classifier
EDGE_CASES: list[dict[str, object]] = [
    {
        "input": {"id": 100, "text": "just aped $MONKE, chart looks clean, 2M mcap"},
        "expected": {
            "id": 100,
            "ticker": "$MONKE",
            "intent": "TICKER_CALL",
            "conviction": "STRONG",
        },
    },
    {
        "input": {"id": 101, "text": "gm everyone, how we feeling today?"},
        "expected": {
            "id": 101,
            "ticker": None,
            "intent": "NOISE",
            "conviction": None,
        },
    },
    {
        "input": {
            "id": 102,
            "text": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D looks interesting, just launched",
        },
        "expected": {
            "id": 102,
            "ticker": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "intent": "TICKER_CALL",
            "conviction": "MODERATE",
        },
    },
    {
        "input": {"id": 103, "text": "BTC breaking 70k means alts will pump"},
        "expected": {
            "id": 103,
            "ticker": None,
            "intent": "PRICE_ACTION",
            "conviction": None,
        },
    },
    {
        "input": {"id": 104, "text": "$PEPE got rugged, dev pulled liquidity"},
        "expected": {
            "id": 104,
            "ticker": "$PEPE",
            "intent": "PRICE_ACTION",
            "conviction": None,
        },
    },
    {
        "input": {
            "id": 105,
            "text": "watching 7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr closely, volume picking up",
        },
        "expected": {
            "id": 105,
            "ticker": "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",
            "intent": "TICKER_CALL",
            "conviction": "MODERATE",
        },
    },
    {
        "input": {
            "id": 106,
            "text": "someone in another group mentioned $WIF might pump, idk tho",
        },
        "expected": {
            "id": 106,
            "ticker": "$WIF",
            "intent": "TICKER_CALL",
            "conviction": "WEAK",
        },
    },
    {
        "input": {"id": 107, "text": "full send on $BONK, loaded a bag at 0.00001"},
        "expected": {
            "id": 107,
            "ticker": "$BONK",
            "intent": "TICKER_CALL",
            "conviction": "STRONG",
        },
    },
    {
        "input": {
            "id": 108,
            "text": "the market structure for SOL looks solid, weekly close above 100",
        },
        "expected": {
            "id": 108,
            "ticker": None,
            "intent": "ANALYSIS",
            "conviction": None,
        },
    },
    {
        "input": {"id": 109, "text": "ngmi if you're not in memes rn"},
        "expected": {
            "id": 109,
            "ticker": None,
            "intent": "NOISE",
            "conviction": None,
        },
    },
]
