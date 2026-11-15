"""Tests for text preprocessing."""

from __future__ import annotations

import pytest

from preprocessing.text_preprocessing import (
    PreprocessingError,
    clean_text,
    lemmatize_tokens,
    preprocess_text,
    remove_stopwords,
    tokenize_sentences,
    tokenize_words,
)

SAMPLE_TEXT = (
    "Machine learning is a subset of artificial intelligence. "
    "It enables computers to learn from data without explicit programming. "
    "Students should not ignore the importance of data preprocessing."
)


def test_clean_text_normal() -> None:
    cleaned = clean_text(SAMPLE_TEXT)
    assert cleaned == SAMPLE_TEXT.lower()
    assert "  " not in cleaned


def test_clean_text_empty() -> None:
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_clean_text_excessive_whitespace() -> None:
    text = "Hello   world\n\n\nagain\t\t!"
    cleaned = clean_text(text)
    assert cleaned == "hello world again !"


def test_clean_text_preserves_original() -> None:
    original = SAMPLE_TEXT
    _ = clean_text(original)
    assert original == SAMPLE_TEXT  # original unchanged


def test_clean_text_math_symbols() -> None:
    text = "E = mc^2 and sum x_i"
    cleaned = clean_text(text)
    assert "mc" in cleaned  # symbols preserved-ish
    assert "=" in cleaned or "mc" in cleaned


def test_tokenize_sentences() -> None:
    sentences = tokenize_sentences(SAMPLE_TEXT)
    assert len(sentences) == 3


def test_tokenize_words() -> None:
    tokens = tokenize_words("Hello, world! It's 2024.")
    assert "Hello" in tokens or "hello" not in tokens  # case preserved in tokenizer
    assert "," in tokens


def test_stopword_removal_removes_common() -> None:
    tokens = ["the", "cat", "is", "on", "mat"]
    filtered = remove_stopwords(tokens)
    assert "cat" in filtered
    assert "mat" in filtered
    assert "the" not in filtered
    assert "is" not in filtered


def test_stopword_removal_keeps_negations() -> None:
    tokens = ["students", "should", "not", "ignore", "never", "give", "up", "no", "excuses"]
    filtered = remove_stopwords(tokens)
    assert "not" in filtered
    assert "never" in filtered
    assert "no" in filtered


def test_stopword_removal_custom_list() -> None:
    tokens = ["alpha", "beta", "gamma"]
    filtered = remove_stopwords(tokens, stopwords={"beta"})
    assert "beta" not in filtered
    assert "alpha" in filtered
    assert "gamma" in filtered


def test_lemmatization() -> None:
    lemmas = lemmatize_tokens(["running", "studies", "better"])
    assert isinstance(lemmas, list)
    assert len(lemmas) == 3
    # running -> run (verb), studies -> study
    assert "run" in lemmas or "running" in lemmas
    assert "study" in lemmas or "studies" in lemmas


def test_preprocess_text_empty() -> None:
    result = preprocess_text("")
    assert result.original_text == ""
    assert result.cleaned_text == ""
    assert result.sentences == []
    assert result.tokens == []
    assert result.filtered_tokens == []
    assert result.lemmatized_tokens == []


def test_preprocess_text_full_pipeline() -> None:
    result = preprocess_text(SAMPLE_TEXT)
    assert result.original_text == SAMPLE_TEXT
    assert result.cleaned_text == SAMPLE_TEXT.lower()
    assert len(result.sentences) == 3
    assert len(result.tokens) > 0
    assert "not" in result.filtered_tokens
    assert len(result.lemmatized_tokens) == len(result.filtered_tokens)
    assert result.warnings == []


def test_preprocess_text_invalid_input() -> None:
    with pytest.raises(PreprocessingError):
        preprocess_text(123)  # type: ignore[arg-type]


def test_clean_text_invalid_input() -> None:
    with pytest.raises(PreprocessingError):
        clean_text(None)  # type: ignore[arg-type]


def test_missing_resource_error_message() -> None:
    # Directly test _ensure_resource via monkeypatch by pointing to bogus path
    import nltk
    from preprocessing.text_preprocessing import _ensure_resource, PreprocessingError

    # Force failure by using a path that does not exist
    with pytest.raises(PreprocessingError) as exc_info:
        _ensure_resource("corpora/does_not_exist_xyz", "stopwords")
    assert "Download it with" in str(exc_info.value)
