"""Tests for TextRank summarization."""

from __future__ import annotations

import pytest

from summarization.textrank import TextRankError, summarize_text

SHORT_TEXT = (
    "Machine learning is a subset of artificial intelligence. "
    "It enables computers to learn from data without explicit programming. "
    "Data preprocessing is an important step in machine learning. "
    "Students should understand these concepts clearly."
)

LONG_TEXT = (
    "Natural language processing is a field of artificial intelligence. "
    "It focuses on the interaction between computers and human language. "
    "Text summarization is an important application of NLP. "
    "Extractive summarization selects important sentences from a document. "
    "TextRank is a graph-based ranking algorithm. "
    "It uses sentence similarity to build a graph. "
    "PageRank is then applied to rank sentences. "
    "The top-ranked sentences form the summary. "
    "This approach does not require training data. "
    "It is fully unsupervised and reproducible."
)

REPEATED_TEXT = (
    "Machine learning is important. "
    "Machine learning is important. "
    "Deep learning is a subset of machine learning. "
    "Students should study both."
)


def test_short_text_summary() -> None:
    result = summarize_text(SHORT_TEXT, num_sentences=2)
    assert result.summary_text
    assert len(result.selected_sentences) == 2
    assert result.original_word_count > result.summary_word_count
    assert 0 < result.compression_ratio < 1


def test_long_text_summary() -> None:
    result = summarize_text(LONG_TEXT, num_sentences=3)
    assert len(result.selected_sentences) == 3
    # Original order preserved
    indices = [LONG_TEXT.find(s) for s in result.selected_sentences]
    assert indices == sorted(indices)


def test_empty_input() -> None:
    result = summarize_text("")
    assert result.summary_text == ""
    assert result.selected_sentences == []
    assert result.sentence_scores == {}
    assert result.summary_word_count == 0


def test_single_sentence() -> None:
    text = "Only one sentence here."
    result = summarize_text(text)
    assert result.summary_text == text
    assert result.selected_sentences == [text]


def test_repeated_sentences_deduplicated() -> None:
    result = summarize_text(REPEATED_TEXT, num_sentences=2)
    # Duplicate sentence should not appear twice
    assert len(result.selected_sentences) == len(set(s.lower() for s in result.selected_sentences))
    assert len(result.selected_sentences) <= 2


def test_different_summary_lengths() -> None:
    r1 = summarize_text(SHORT_TEXT, num_sentences=1)
    r2 = summarize_text(SHORT_TEXT, num_sentences=3)
    assert len(r1.selected_sentences) == 1
    assert len(r2.selected_sentences) == 3


def test_ratio_parameter() -> None:
    result = summarize_text(LONG_TEXT, ratio=0.5)
    # 10 sentences -> ~5
    assert 1 <= len(result.selected_sentences) <= 10


def test_zero_sentences_returns_empty() -> None:
    result = summarize_text(SHORT_TEXT, num_sentences=0)
    assert result.selected_sentences == []
    assert result.summary_text == ""


def test_sentence_scores_present() -> None:
    result = summarize_text(SHORT_TEXT, num_sentences=2)
    assert len(result.sentence_scores) > 0
    for sentence, score in result.sentence_scores.items():
        assert isinstance(score, float)


def test_reproducibility() -> None:
    r1 = summarize_text(SHORT_TEXT, num_sentences=2)
    r2 = summarize_text(SHORT_TEXT, num_sentences=2)
    assert r1.summary_text == r2.summary_text
    assert r1.selected_sentences == r2.selected_sentences


def test_invalid_input_raises() -> None:
    with pytest.raises(TextRankError):
        summarize_text(123)  # type: ignore[arg-type]


def test_paragraphs_similar_sentences() -> None:
    text = (
        "Machine learning is a subset of AI. "
        "Machine learning is a subset of artificial intelligence. "
        "Deep learning uses neural networks. "
        "Neural networks are used in deep learning."
    )
    result = summarize_text(text, num_sentences=2)
    assert len(result.selected_sentences) == 2


def test_compression_ratio() -> None:
    result = summarize_text(SHORT_TEXT, num_sentences=1)
    expected = result.summary_word_count / result.original_word_count
    assert abs(result.compression_ratio - expected) < 1e-9
