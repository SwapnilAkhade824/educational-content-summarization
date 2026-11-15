"""Student feedback sentiment analysis using VADER."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentError(Exception):
    """Raised when sentiment analysis fails."""


# Standard VADER thresholds (documented).
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


@dataclass
class SentimentResult:
    """Sentiment result for a single feedback entry."""

    feedback_text: str
    compound_score: float
    positive_score: float
    negative_score: float
    neutral_score: float
    sentiment_label: str


@dataclass
class BatchSentimentResult:
    """Aggregated sentiment results for multiple feedback entries."""

    total: int
    positive: int
    negative: int
    neutral: int
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    results: List[SentimentResult] = field(default_factory=list)


def _ensure_vader() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        try:
            nltk.data.find("sentiment/vader_lexicon")
        except LookupError as exc:
            raise SentimentError(
                "Required NLTK resource 'vader_lexicon' is not available. "
                "Download it with: python -c \"import nltk; nltk.download('vader_lexicon')\"."
            ) from exc


def _normalize_feedback(text: str) -> str:
    if not isinstance(text, str):
        raise SentimentError("Feedback must be a string.")
    # Collapse whitespace; keep punctuation/case for VADER.
    return re.sub(r"\s+", " ", text).strip()


def _label_from_compound(compound: float) -> str:
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyze_feedback(text: str) -> SentimentResult:
    """Analyze sentiment of a single student feedback entry using VADER."""
    _ensure_vader()
    normalized = _normalize_feedback(text)

    if not normalized:
        return SentimentResult(
            feedback_text="",
            compound_score=0.0,
            positive_score=0.0,
            negative_score=0.0,
            neutral_score=1.0,
            sentiment_label="Neutral",
        )

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(normalized)
    compound = float(scores["compound"])

    return SentimentResult(
        feedback_text=normalized,
        compound_score=compound,
        positive_score=float(scores["pos"]),
        negative_score=float(scores["neg"]),
        neutral_score=float(scores["neu"]),
        sentiment_label=_label_from_compound(compound),
    )


def analyze_batch(feedbacks: Sequence[str]) -> BatchSentimentResult:
    """Analyze sentiment for multiple feedback entries."""
    if not isinstance(feedbacks, (list, tuple)):
        raise SentimentError("Feedbacks must be provided as a list or tuple.")

    results = [analyze_feedback(str(fb)) for fb in feedbacks]
    total = len(results)
    positive = sum(1 for r in results if r.sentiment_label == "Positive")
    negative = sum(1 for r in results if r.sentiment_label == "Negative")
    neutral = sum(1 for r in results if r.sentiment_label == "Neutral")

    if total == 0:
        pos_pct = neg_pct = neu_pct = 0.0
    else:
        pos_pct = positive / total * 100.0
        neg_pct = negative / total * 100.0
        neu_pct = neutral / total * 100.0

    return BatchSentimentResult(
        total=total,
        positive=positive,
        negative=negative,
        neutral=neutral,
        positive_pct=pos_pct,
        negative_pct=neg_pct,
        neutral_pct=neu_pct,
        results=results,
    )
