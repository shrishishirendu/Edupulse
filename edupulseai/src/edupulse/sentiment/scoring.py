"""Lightweight lexicon-based sentiment scoring."""

from __future__ import annotations

import re
from typing import Iterable

# Curated lexicons (small, deterministic; easy to extend/replace)
POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "positive",
    "happy",
    "love",
    "like",
    "enjoy",
    "fantastic",
    "wonderful",
    "satisfied",
    "pleased",
    "delighted",
    "helpful",
    "supportive",
    "improved",
    "improving",
    "progress",
    "commend",
    "recommend",
    "brilliant",
    "awesome",
    "outstanding",
    "terrific",
    "super",
    "superb",
    "encourage",
    "strong",
    "clear",
    "confident",
    "creative",
    "flexible",
    "friendly",
    "approachable",
    "engaging",
    "motivated",
    "motivating",
    "responsive",
    "reliable",
    "effective",
    "efficient",
    "organized",
    "valuable",
    "useful",
    "helped",
    "grateful",
    "appreciate",
    "appreciated",
    "caring",
    "kind",
    "greatly",
    "solid",
    "timely",
    "quick",
    "impressive",
    "excited",
    "excellent",
    "fantastic",
    "phenomenal",
    "remarkable",
    "encouraging",
    "positive",
    "meaningful",
    "interesting",
    "insightful",
    "support",
    "supported",
    "smooth",
    "easy",
    "intuitive",
    "fun",
    "uplifting",
    "accessible",
    "achieved",
    "accomplished",
    "celebrate",
    "celebrated",
    "engaged",
    "helpful",
    "helping",
    "mentored",
    "mentoring",
    "clarified",
    "clarity",
    "prepared",
    "well-prepared",
    "organized",
    "effective",
    "efficient",
    "responsive",
    "proactive",
    "encouraged",
}

NEGATIVE_WORDS = {
    "bad",
    "poor",
    "terrible",
    "awful",
    "negative",
    "sad",
    "hate",
    "dislike",
    "angry",
    "frustrated",
    "frustrating",
    "confusing",
    "confused",
    "unclear",
    "difficult",
    "hard",
    "late",
    "slow",
    "unresponsive",
    "rude",
    "unhelpful",
    "broken",
    "bug",
    "issue",
    "problem",
    "problems",
    "error",
    "errors",
    "fail",
    "failed",
    "failing",
    "worse",
    "worst",
    "boring",
    "useless",
    "waste",
    "disappointed",
    "disappointing",
    "concerned",
    "concern",
    "concerns",
    "anxious",
    "stress",
    "stressed",
    "overwhelmed",
    "struggle",
    "struggling",
    "struggled",
    "drop",
    "dropout",
    "quit",
    "quitting",
    "absent",
    "absentee",
    "missed",
    "missing",
    "late",
    "delay",
    "delayed",
    "ignored",
    "ignored",
    "forgotten",
    "forget",
    "forgot",
    "messy",
    "chaotic",
    "lack",
    "lacking",
    "incomplete",
    "insufficient",
    "poorly",
    "unfair",
    "unjust",
    "uncomfortable",
    "unpleasant",
    "annoyed",
    "annoying",
    "inconsistent",
    "unprepared",
    "unprofessional",
    "disorganized",
    "noise",
    "noisy",
    "distracting",
    "inefficient",
    "ineffective",
    "unreliable",
    "uncaring",
    "harsh",
    "punish",
    "punished",
    "penalty",
    "penalized",
    "stressful",
    "tired",
    "exhausted",
    "burnout",
    "burned",
    "overload",
    "overloaded",
}

NEGATIONS = {"not", "no", "never", "n't"}
INTENSIFIERS = {"very", "extremely", "super", "really"}

WORD_RE = re.compile(r"[a-zA-Z']+")


def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, and limit repeated punctuation."""
    text = (text or "").lower().strip()
    text = re.sub(r"[!?]{2,}", "!!", text)
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _tokens(text: str) -> Iterable[str]:
    normalized = normalize_text(text)
    return WORD_RE.findall(normalized)


def _word_score(token: str) -> int:
    if token in POSITIVE_WORDS:
        return 1
    if token in NEGATIVE_WORDS:
        return -1
    return 0


def sentiment_score(text: str) -> float:
    """Compute a simple lexicon-based sentiment score in [-1, 1]."""
    tokens = list(_tokens(text))
    if not tokens:
        return 0.0

    score = 0.0
    negate_next = False

    for idx, tok in enumerate(tokens):
        if tok in NEGATIONS:
            negate_next = True
            continue

        base = _word_score(tok)
        if base == 0:
            continue

        # Check for intensifier immediately before the sentiment word
        if idx > 0 and tokens[idx - 1] in INTENSIFIERS:
            base *= 1.5

        if negate_next:
            base *= -1
            negate_next = False

        score += base

    # Normalize to [-1, 1]
    score = max(min(score, 10.0), -10.0)  # clamp raw score
    return float(score / 10.0)


def sentiment_label(score: float, cfg: dict | None = None) -> str:
    """Return label based on configured thresholds."""
    thresholds = (cfg or {}).get("sentiment", {}).get("thresholds", {})
    neg_thresh = thresholds.get("negative", -0.2)
    pos_thresh = thresholds.get("positive", 0.2)

    if score >= pos_thresh:
        return "positive"
    if score <= neg_thresh:
        return "negative"
    return "neutral"
