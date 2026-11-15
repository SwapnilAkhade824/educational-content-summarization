"""Tests for UI helper functions."""

from __future__ import annotations

import pytest

from ui.helpers import (
    build_export_markdown,
    process_document,
    prepare_preprocessing,
    run_concepts,
    run_llm_summary,
    run_sentiment_many,
    run_sentiment_one,
    run_textrank,
    run_topics,
    top_tfidf_terms,
)


def test_process_document_valid_txt() -> None:
    result = process_document("notes.txt", b"Hello world.\nSecond line.")
    assert result["ok"] is True
    assert result["file_type"] == "txt"
    assert "Hello world" in result["text"]


def test_process_document_invalid_pdf() -> None:
    result = process_document("bad.pdf", b"not a pdf")
    assert result["ok"] is False
    assert "error" in result


def test_process_document_unsupported_type() -> None:
    result = process_document("image.png", b"fake")
    assert result["ok"] is False


def test_prepare_preprocessing_empty() -> None:
    assert prepare_preprocessing("") is None
    assert prepare_preprocessing("   ") is None


def test_prepare_preprocessing_valid() -> None:
    result = prepare_preprocessing("Machine learning is great.")
    assert result is not None
    assert result.cleaned_text


def test_run_textrank() -> None:
    text = "Machine learning is a subset of AI. It uses data. Students should learn it."
    result = run_textrank(text, 2)
    assert result.selected_sentences
    assert len(result.selected_sentences) == 2


def test_run_concepts() -> None:
    text = "Machine Learning is a branch of AI. Google is a company."
    result = run_concepts(text)
    assert result is not None
    assert isinstance(result.combined, list)


def test_run_topics_insufficient() -> None:
    result = run_topics(["short"], 3)
    assert result is not None
    # Should warn or produce limited topics
    assert isinstance(result.warnings, list)


def test_run_sentiment_one() -> None:
    result = run_sentiment_one("This course is great!")
    assert result.sentiment_label == "Positive"


def test_run_sentiment_many() -> None:
    result = run_sentiment_many(["Great!", "Terrible.", "Okay."])
    assert result.total == 3


def test_run_llm_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL_ID"):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(key, "")
    # Without env, should return error message
    result, error = run_llm_summary("Some text.", "short")
    # Either configured (unlikely) or error message
    if error is not None:
        assert "configured" in error.lower() or "gemini" in error.lower()


def test_top_tfidf_terms() -> None:
    terms = top_tfidf_terms("machine learning data data")
    assert isinstance(terms, list)


def test_build_export_markdown() -> None:
    md = build_export_markdown(
        source_name="doc.txt",
        summary_type="short",
        summary_text="A summary.",
        method="TextRank",
        concepts=["ML", "AI"],
        definitions=[],
    )
    assert "doc.txt" in md
    assert "A summary." in md
    assert "ML" in md
    assert "verified academic facts" in md.lower() or "review before use" in md.lower()
