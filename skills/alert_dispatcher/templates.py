from __future__ import annotations

from typing import Any


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
