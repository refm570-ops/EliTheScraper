from __future__ import annotations

SYSTEM_PROMPT = """\
You are a crypto signal classifier. You receive batches of messages from Telegram crypto groups.

For EACH message, output a JSON object with:
- id: The message id from the input
- ticker: The token symbol (e.g., "$MONKE") or contract address if mentioned. null if no ticker.
- intent: One of "TICKER_CALL", "PRICE_ACTION", "ANALYSIS", "NOISE"
- conviction: One of "STRONG", "MODERATE", "WEAK", null (only for TICKER_CALL)
- context: 1-sentence summary of why this matters. null for NOISE.

Crypto slang reference:
- "ape/aped/aping" = bought aggressively → STRONG conviction
- "loaded/full send/max bid" = heavy position → STRONG
- "watching/eyeing/interesting" = considering → MODERATE
- "someone said/heard about" = secondhand → WEAK
- "rug/rugged" = scam, token collapsed → classify as PRICE_ACTION
- "gm/gn/wagmi/ngmi" = social noise → NOISE
- Contract addresses: Solana (base58, 32-44 chars), EVM (0x + 40 hex chars)

CRITICAL: Output ONLY a JSON array. No markdown. No backticks. No explanation.
Each element corresponds to the same-indexed input message.\
"""

# Few-shot examples included in the user prompt
FEW_SHOT_BLOCK = """\
Example input:
[
  {"id": 1, "text": "just aped $MONKE, chart looks clean, 2M mcap"},
  {"id": 2, "text": "gm everyone how we doing"},
  {"id": 3, "text": "BTC looking weak, alts might dump"}
]

Example output:
[
  {"id": 1, "ticker": "$MONKE", "intent": "TICKER_CALL", "conviction": "STRONG", "context": "Bought in, bullish on chart and market cap"},
  {"id": 2, "ticker": null, "intent": "NOISE", "conviction": null, "context": null},
  {"id": 3, "ticker": null, "intent": "PRICE_ACTION", "conviction": null, "context": "Bearish BTC outlook, potential alt selloff"}
]

Now classify the following messages:\
"""

VALID_INTENTS = {"TICKER_CALL", "PRICE_ACTION", "ANALYSIS", "NOISE"}
VALID_CONVICTIONS = {"STRONG", "MODERATE", "WEAK", None}
