from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Load scoring weights once at import time
_weights_path = Path(__file__).parent.parent.parent / "config" / "scoring_weights.yml"
with open(_weights_path) as f:
    _ALL_WEIGHTS = yaml.safe_load(f)

X_WEIGHTS = _ALL_WEIGHTS.get("x_sentiment", {})


def score_x_sentiment(
    engagement_summary: dict[str, Any],
    sentiment_result: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic scoring of X sentiment + engagement data.

    Returns: x_sentiment_score, x_score_breakdown, x_data_available.
    """
    eng = X_WEIGHTS.get("engagement", {})
    reach_cfg = X_WEIGHTS.get("reach", {})
    sent_cfg = X_WEIGHTS.get("sentiment", {})

    # --- Engagement score (-10 to +15) ---
    total_engagement = (
        engagement_summary.get("total_likes", 0)
        + engagement_summary.get("total_retweets", 0)
        + engagement_summary.get("total_quotes", 0)
    )
    noise_thresh = eng.get("noise", 5)
    low_thresh = eng.get("low", 25)
    moderate_thresh = eng.get("moderate", 100)
    significant_thresh = eng.get("significant", 500)

    if total_engagement < noise_thresh:
        engagement_score = -10
    elif total_engagement < low_thresh:
        engagement_score = 0
    elif total_engagement < moderate_thresh:
        engagement_score = 5
    elif total_engagement < significant_thresh:
        engagement_score = 10
    else:
        engagement_score = 15

    # --- Reach score (-5 to +10) ---
    max_followers = engagement_summary.get("max_author_followers")
    small_thresh = reach_cfg.get("small", 1000)
    mid_thresh = reach_cfg.get("mid", 10000)
    influencer_thresh = reach_cfg.get("influencer", 100000)

    if max_followers is None:
        reach_score = 0
    elif max_followers < small_thresh:
        reach_score = -5
    elif max_followers < mid_thresh:
        reach_score = 0
    elif max_followers < influencer_thresh:
        reach_score = 5
    else:
        reach_score = 10

    # --- Sentiment score (-15 to +10) ---
    bullish_pct = sentiment_result.get("bullish_pct", 0.33)
    bearish_pct = sentiment_result.get("bearish_pct", 0.33)
    bearish_strong = sent_cfg.get("bearish_strong", 0.6)
    bearish_lean = sent_cfg.get("bearish_lean", 0.4)
    bullish_strong = sent_cfg.get("bullish_strong", 0.7)
    bullish_lean = sent_cfg.get("bullish_lean", 0.5)

    if bearish_pct > bearish_strong:
        sentiment_score = -15
    elif bearish_pct > bearish_lean:
        sentiment_score = -5
    elif bullish_pct > bullish_strong:
        sentiment_score = 10
    elif bullish_pct > bullish_lean:
        sentiment_score = 5
    else:
        sentiment_score = 0

    # --- Quality bonus (0 to +5) ---
    quality = sentiment_result.get("quality", "low")
    if quality == "high":
        quality_score = 5
    elif quality == "medium":
        quality_score = 2
    else:
        quality_score = 0

    total = engagement_score + reach_score + sentiment_score + quality_score

    return {
        "x_sentiment_score": total,
        "x_score_breakdown": {
            "engagement": engagement_score,
            "reach": reach_score,
            "sentiment": sentiment_score,
            "quality": quality_score,
        },
        "x_data_available": True,
    }
