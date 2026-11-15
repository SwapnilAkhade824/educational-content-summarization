"""Student feedback sentiment analysis (VADER)."""

from .sentiment_analyzer import (
    BatchSentimentResult,
    SentimentError,
    SentimentResult,
    analyze_batch,
    analyze_feedback,
)

__all__ = [
    "BatchSentimentResult",
    "SentimentError",
    "SentimentResult",
    "analyze_batch",
    "analyze_feedback",
]
