from __future__ import annotations

from html import escape
from typing import Any

# Alert level headers
_LEVEL_HEADERS = {
    "act_now": "\U0001f6a8 ACT NOW",
    "interesting": "\U0001f50d INTERESTING",
    "watch": "\U0001f440 WATCH",
}


def format_phase1_alert(
    classification: dict[str, Any],
    raw_message: dict[str, Any] | None = None,
) -> str:
    """Simple Phase 1 alert format: ticker + source + conviction + raw text."""
    ticker = classification.get("ticker") or "Unknown"
    intent = classification.get("intent", "")
    conviction = classification.get("conviction") or "N/A"
    context = classification.get("context") or ""

    group_name = ""
    raw_text = ""
    if raw_message:
        group_name = raw_message.get("group_name", "Unknown")
        raw_text = raw_message.get("text", "")

    # Conviction emoji
    conv_emoji = {"STRONG": "\U0001f525", "MODERATE": "\U0001f7e1", "WEAK": "\u26aa"}.get(
        conviction, ""
    )

    lines = [
        f"<b>{conv_emoji} {ticker}</b>",
        f"<b>Intent:</b> {intent}",
        f"<b>Conviction:</b> {conviction}",
        f"<b>Source:</b> {group_name}",
    ]

    if context:
        lines.append(f"<b>Context:</b> {context}")

    if raw_text:
        # Truncate long messages
        preview = raw_text[:300]
        if len(raw_text) > 300:
            preview += "..."
        lines.append(f"\n<i>{preview}</i>")

    return "\n".join(lines)


def format_phase2_alert(
    ticker: str,
    alert_level: str,
    score_result: dict[str, Any],
    social_metrics: dict[str, Any],
    aggregator_result: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    cross_ref_result: dict[str, Any] | None = None,
    x_sentiment_result: dict[str, Any] | None = None,
) -> str:
    """Rich Phase 2+3 alert card with on-chain data, social metrics, scoring, and cross-platform."""
    header = _LEVEL_HEADERS.get(alert_level, alert_level.upper())
    lines: list[str] = []

    # Header
    lines.append(f"<b>{header} \u2014 {escape(ticker)}</b>")
    lines.append("")

    # Aggregator summary
    summary = aggregator_result.get("summary")
    if summary:
        lines.append(f"<i>{escape(summary)}</i>")
        lines.append("")

    # On-chain score breakdown
    total_score = score_result.get("total_score", 0)
    breakdown = score_result.get("score_breakdown", {})
    metadata_available = score_result.get("metadata_available", False)

    lines.append(f"<b>\U0001f4ca On-Chain Score:</b> {total_score}")
    if metadata_available and breakdown:
        parts = [f"{k}: {v:+d}" for k, v in breakdown.items()]
        lines.append(f"  {' | '.join(parts)}")
    elif not metadata_available:
        lines.append("  <i>On-chain data unavailable</i>")

    # Scorer reasoning
    reasoning = score_result.get("reasoning")
    if reasoning and reasoning != "No on-chain data available":
        lines.append(f"  {escape(reasoning)}")

    lines.append("")

    # Social metrics
    mention_count = social_metrics.get("mention_count", 0)
    unique_groups = social_metrics.get("unique_groups", 0)
    group_names = social_metrics.get("group_names", [])
    convictions = social_metrics.get("convictions", {})
    sources = social_metrics.get("sources", [])

    lines.append(f"<b>\U0001f4e2 Social:</b> {mention_count} mentions in {unique_groups} groups")
    if group_names:
        lines.append(f"  Groups: {', '.join(escape(g) for g in group_names)}")
    if convictions:
        conv_parts = [f"{k}: {v}" for k, v in convictions.items()]
        lines.append(f"  Conviction: {' | '.join(conv_parts)}")
    if sources:
        lines.append(f"  Platforms: {', '.join(sources)}")

    # Cross-platform status
    if cross_ref_result and cross_ref_result.get("has_multi_source"):
        corroborated = cross_ref_result.get("corroborated", False)
        confidence = cross_ref_result.get("confidence", "low")
        suspicious = cross_ref_result.get("suspicious", False)
        if corroborated:
            lines.append(f"  Cross-platform: \u2705 Corroborated ({confidence})")
        else:
            lines.append("  Cross-platform: \u274c Not corroborated")
        if suspicious:
            lines.append("  \u26a0\ufe0f Suspicious coordination detected")

    # X Sentiment section (Phase 4)
    if x_sentiment_result and x_sentiment_result.get("x_data_available"):
        lines.append("")
        x_score = x_sentiment_result.get("x_sentiment_score", 0)
        x_bd = x_sentiment_result.get("x_score_breakdown", {})
        direction = "bullish" if x_score > 0 else ("bearish" if x_score < 0 else "neutral")
        lines.append(
            f"<b>\U0001f4ca X Sentiment:</b> {x_score:+d} ({direction})"
        )
        parts = [f"{k}: {v:+d}" for k, v in x_bd.items()]
        if parts:
            lines.append(f"  {' | '.join(parts)}")
        narrative = x_sentiment_result.get("key_narrative")
        if narrative:
            lines.append(f"  <i>{escape(narrative)}</i>")
        eng_summary = x_sentiment_result.get("engagement_summary", {})
        tweet_count = eng_summary.get("tweet_count", 0)
        total_eng = (
            eng_summary.get("total_likes", 0)
            + eng_summary.get("total_retweets", 0)
            + eng_summary.get("total_quotes", 0)
        )
        if tweet_count:
            lines.append(f"  Buzz: {total_eng} engagements from {tweet_count} tweets")

    # Warning flags
    flags = score_result.get("flags", [])
    blocked_reason = aggregator_result.get("blocked_reason")
    if blocked_reason:
        flags = [*flags, blocked_reason]
    # Add X bearish warning if applicable
    if (
        x_sentiment_result
        and x_sentiment_result.get("x_data_available")
        and x_sentiment_result.get("x_sentiment_score", 0) < -10
    ):
        flags = [*flags, "x_bearish_sentiment"]
    if flags:
        lines.append("")
        lines.append(f"\u26a0\ufe0f <b>Flags:</b> {', '.join(escape(f) for f in flags)}")

    # On-chain details (if metadata available)
    if metadata:
        lines.append("")
        detail_parts: list[str] = []
        if metadata.get("market_cap"):
            detail_parts.append(f"MCap: ${_fmt_number(metadata['market_cap'])}")
        if metadata.get("liquidity_usd"):
            detail_parts.append(f"Liq: ${_fmt_number(metadata['liquidity_usd'])}")
        if metadata.get("volume_24h"):
            detail_parts.append(f"Vol24h: ${_fmt_number(metadata['volume_24h'])}")
        if metadata.get("age_days") is not None:
            detail_parts.append(f"Age: {metadata['age_days']}d")
        if metadata.get("holder_count"):
            detail_parts.append(f"Holders: {metadata['holder_count']:,}")
        if detail_parts:
            lines.append(f"<b>\U0001f4b0 Market:</b> {' | '.join(detail_parts)}")

        # DexScreener link
        dex_url = metadata.get("dex_url")
        if dex_url:
            lines.append(f'\U0001f517 <a href="{escape(dex_url)}">DexScreener</a>')

    return "\n".join(lines)


def _fmt_number(n: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.2f}"
