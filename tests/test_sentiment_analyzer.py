"""Tests for sentiment analysis."""

from __future__ import annotations

import pytest

from sentiment.sentiment_analyzer import (
    SentimentError,
    analyze_batch,
    analyze_feedback,
)

POSITIVE = "This course is amazing and very helpful. I loved the explanations!"
NEGATIVE = "This material is terrible and confusing. I hated it."
NEUTRAL = "The chapter covers several topics."
MIXED = "The content is great but the examples are confusing."


def test_positive_feedback() -> None:
    result = analyze_feedback(POSITIVE)
    assert result.sentiment_label == "Positive"
    assert result.compound_score >= 0.05
    assert result.positive_score > result.negative_score


def test_negative_feedback() -> None:
    result = analyze_feedback(NEGATIVE)
    assert result.sentiment_label == "Negative"
    assert result.compound_score <= -0.05
    assert result.negative_score > result.positive_score


def test_neutral_feedback() -> None:
    result = analyze_feedback(NEUTRAL)
    assert result.sentiment_label == "Neutral"
    assert -0.05 < result.compound_score < 0.05


def test_mixed_feedback() -> None:
    result = analyze_feedback(MIXED)
    assert result.sentiment_label in {"Positive", "Negative", "Neutral"}
    assert result.positive_score >= 0
    assert result.negative_score >= 0
    assert result.neutral_score >= 0


def test_empty_feedback() -> None:
    result = analyze_feedback("")
    assert result.sentiment_label == "Neutral"
    assert result.compound_score == 0.0
    assert result.feedback_text == ""


def test_whitespace_feedback() -> None:
    result = analyze_feedback("   ")
    assert result.sentiment_label == "Neutral"


def test_invalid_input_raises() -> None:
    with pytest.raises(SentimentError):
        analyze_feedback(123)  # type: ignore[arg-type]


def test_batch_analysis() -> None:
    batch = analyze_batch([POSITIVE, NEGATIVE, NEUTRAL, ""])
    assert batch.total == 4
    assert batch.positive >= 1
    assert batch.negative >= 1
    assert batch.neutral >= 1
    assert abs((batch.positive_pct + batch.negative_pct + batch.neutral_pct) - 100.0) < 1e-9
    assert len(batch.results) == 4


def test_batch_empty() -> None:
    batch = analyze_batch([])
    assert batch.total == 0
    assert batch.positive == 0
    assert batch.positive_pct == 0.0


def test_batch_invalid_input() -> None:
    with pytest.raises(SentimentError):
        analyze_batch("not a list")  # type: ignore[arg-type]


def test_scores_in_range() -> None:
    result = analyze_feedback(POSITIVE)
    for score in (result.compound_score, result.positive_score, result.negative_score, result.neutral_score):
        assert 0.0 <= abs(score) <= 1.0
    assert result.positive_score + result.negative_score + result.neutral_score > 0
