---
name: crypto-alpha-signals
description: Multi-agent system for extracting, classifying, scoring, and alerting on crypto token signals from Telegram groups and Twitter/X feeds. Combines real-time message ingestion, LLM-powered classification, on-chain data enrichment, cross-platform correlation, and automated alerting via Telegram bot.
---

# Crypto Alpha Signal System

## Overview

A pipeline that reads 30+ Telegram crypto groups and Twitter/X feeds, filters noise, extracts ticker mentions, scores them using on-chain data and social signals, and delivers actionable alerts to a private Telegram bot.

## Architecture

```
DATA SKILLS (deterministic) → AGENTS (LLM-powered) → OUTPUT SKILLS (deterministic)
```

### System Components

| Component | Type | Model/Tech | Purpose |
|-----------|------|------------|---------|
| TG Listener | Skill | Telethon (Python) | Passive real-time message capture |
| X Feed Puller | Skill | X API v2 or scraping | Pull tweets from followed accounts |
| Token Metadata Fetcher | Skill | DexScreener/Birdeye API | On-chain data for any ticker |
| Social Metrics Fetcher | Skill | Internal DB queries | Count mentions across sources |
| Message Classifier | Agent | Claude Haiku | Extract tickers, kill noise |
| Cross-Reference Analyst | Agent | Claude Haiku | Correlate across platforms |
| Fundamentals Scorer | Agent | Claude Sonnet | Score token quality |
| Signal Aggregator | Agent | Claude Sonnet | Final confidence + alert decision |
| Alert Dispatcher | Skill | Telegram Bot API | Send formatted alerts |

## Directory Structure

```
crypto-alpha-system/
├── SKILL.md
├── .env.example
├── docker-compose.yml
├── pyproject.toml
│
├── skills/
│   ├── tg_listener/
│   │   ├── listener.py
│   │   ├── session_manager.py
│   │   └── flood_guard.py
│   │
│   ├── x_puller/          (Phase 3)
│   ├── token_metadata/    (Phase 2)
│   ├── social_metrics/    (Phase 2)
│   │
│   └── alert_dispatcher/
│       ├── bot.py
│       ├── templates.py
│       └── rate_limiter.py
│
├── agents/
│   ├── classifier/
│   │   ├── agent.py
│   │   ├── prompts.py
│   │   └── examples.py
│   │
│   ├── cross_ref/         (Phase 2)
│   ├── scorer/            (Phase 2)
│   └── aggregator/        (Phase 2)
│
├── pipeline/
│   ├── orchestrator.py
│   ├── buffer.py
│   └── scheduler.py
│
├── storage/
│   ├── models.py
│   ├── ticker_store.py
│   └── alert_log.py
│
├── config/
│   ├── groups.example.yml      # copy to groups.yml (gitignored) with real IDs
│   ├── x_accounts.example.yml  # copy to x_accounts.yml (gitignored)
│   ├── scoring_weights.yml
│   └── alert_rules.yml
│
└── tests/
    ├── test_classifier.py
    └── fixtures/
        └── tg_messages.json
```

## TG Listener Safety Rules

- Use personal account session (NOT Bot API)
- NEVER pull message history — only listen to new messages via `events.NewMessage`
- NEVER call `client.get_messages()`, `client.get_participants()`, or any history/member API
- NEVER bulk-download media or member lists
- Run from home IP or Israel-based VPS (same geo as normal usage)
- Start with 5 groups, scale to 30 over 1 week
- Add random jitter (1-3s) to any proactive API call
- Reconnect gracefully on disconnect with exponential backoff
- Implement exponential backoff on FloodWaitError (start 60s, max 15min)

## Message Schema

```json
{
  "source": "telegram",
  "group_id": -1001234567890,
  "group_name": "Alpha Calls",
  "message_id": 45678,
  "sender_id": 123456,
  "sender_name": "whale_watcher",
  "text": "just aped $MONKE, chart looks clean, 2M mcap",
  "timestamp": "2025-02-27T14:30:00Z",
  "has_media": false,
  "reply_to": null
}
```

## Agent 1: Message Classifier

**Model:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)

Classification outputs:
- `ticker`: Token symbol or contract address (null if none)
- `intent`: TICKER_CALL | PRICE_ACTION | ANALYSIS | NOISE
- `conviction`: STRONG | MODERATE | WEAK | null

Crypto slang reference:
- "ape/aped/aping" = bought aggressively → STRONG conviction
- "loaded/full send/max bid" = heavy position → STRONG
- "watching/eyeing/interesting" = considering → MODERATE
- "someone said/heard about" = secondhand → WEAK
- "rug/rugged" = scam, token collapsed → PRICE_ACTION
- "gm/gn/wagmi/ngmi" = social noise → NOISE

## Development Phases

### Phase 1: Foundation
- TG Listener skill — real-time, 5 groups
- Redis buffer
- Agent 1 classifier with Haiku
- Simple alert to TG bot
- Test with live data for 48 hours

### Phase 2: Intelligence
- Token Metadata fetcher (DexScreener)
- Agent 3 scorer
- Social Metrics counter
- Agent 4 aggregator
- Full alert cards
- Scale to 30 groups

### Phase 3: X Integration
- X Feed Puller
- Agent 2 cross-reference with X data

### Phase 4: Tuning
- Analyze alert accuracy
- Tune scoring weights
- Dashboard for historical signals
