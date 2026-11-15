"""Tests for TF-IDF vectorization."""

from __future__ import annotations

import pytest

from preprocessing.vectorizer import (
    VectorizationError,
    fit_tfidf,
    transform_tfidf,
)

DOCS = [
    "machine learning enables computers to learn from data",
    "data preprocessing is important for machine learning",
    "artificial intelligence includes machine learning",
]


def test_fit_transform_normal() -> None:
    vectorizer, result = fit_tfidf(DOCS)
    assert result.matrix_shape[0] == 3
    assert result.matrix_shape[1] > 0
    assert len(result.feature_names) == result.matrix_shape[1]
    assert len(result.top_terms) <= 10
    assert all(isinstance(t, tuple) and len(t) == 2 for t in result.top_terms)


def test_fit_transform_single_doc() -> None:
    vectorizer, result = fit_tfidf(["hello world"])
    assert result.matrix_shape == (1, 2)


def test_empty_corpus_raises() -> None:
    with pytest.raises(VectorizationError):
        fit_tfidf([])


def test_all_empty_docs_raises() -> None:
    with pytest.raises(VectorizationError):
        fit_tfidf(["", "   "])


def test_unseen_transform() -> None:
    vectorizer, _ = fit_tfidf(DOCS)
    result = transform_tfidf(vectorizer, ["new unseen document about data"])
    assert result.matrix_shape[0] == 1
    assert result.matrix_shape[1] == len(vectorizer.get_feature_names_out())


def test_transform_requires_fitted_vectorizer() -> None:
    with pytest.raises(VectorizationError):
        transform_tfidf(None, ["hello"])  # type: ignore[arg-type]


def test_top_terms_sorted_descending() -> None:
    _, result = fit_tfidf(DOCS)
    weights = [w for _, w in result.top_terms]
    assert weights == sorted(weights, reverse=True)


def test_feature_names_contain_expected() -> None:
    _, result = fit_tfidf(DOCS)
    assert "machine" in result.feature_names
    assert "learning" in result.feature_names
    assert "data" in result.feature_names


# ---------------------------------------------------------------------------
# Specific verification tests for stopword exclusion, lemmatization, and matrix
# ---------------------------------------------------------------------------


def test_common_stopwords_excluded() -> None:
    """Verify common English stopwords are absent from vocabulary and matrix."""
    text = (
        "This is an educational text with a problem and the solution to that issue "
        "or of any other concept."
    )
    vectorizer, result = fit_tfidf([text])
    excluded_stopwords = ["a", "an", "the", "and", "is", "to", "of", "or", "that", "this", "with"]
    vocab = set(result.feature_names)
    for sw in excluded_stopwords:
        assert sw not in vocab, f"Stopword '{sw}' leaked into TF-IDF vocabulary!"
        assert sw not in vectorizer.vocabulary_, f"Stopword '{sw}' present in TfidfVectorizer.vocabulary_!"


def test_meaningful_educational_terms_remain() -> None:
    """Verify essential educational terminology is retained."""
    text = "Students study gradient descent algorithms and neural network architectures."
    vectorizer, result = fit_tfidf([text])
    vocab = set(result.feature_names)
    assert "gradient" in vocab
    assert "descent" in vocab
    assert "algorithm" in vocab or "algorithms" in vocab
    assert "neural" in vocab
    assert "network" in vocab


def test_protected_negations_preserved() -> None:
    """Verify protected negations 'no', 'not', 'never' remain in vocabulary."""
    text = (
        "Students should not skip preprocessing. Never ignore outliers. There are no shortcuts."
    )
    vectorizer, result = fit_tfidf([text])
    vocab = set(result.feature_names)
    assert "not" in vocab, "Protected negation 'not' was removed!"
    assert "never" in vocab, "Protected negation 'never' was removed!"
    assert "no" in vocab, "Protected negation 'no' was removed!"


def test_lemmatized_tokens_consumed_directly() -> None:
    """Verify vectorizer consumes pre-lemmatized token sequences directly."""
    lemmatized = ["student", "study", "neural", "network", "not", "fail"]
    vectorizer, result = fit_tfidf([lemmatized])
    assert result.matrix_shape == (1, 6)
    assert set(result.feature_names) == {"fail", "network", "neural", "not", "student", "study"}
    # Ensure no secondary tokenization corrupted tokens
    for token in lemmatized:
        assert token in result.feature_names


def test_empty_inputs_after_stopword_removal() -> None:
    """Verify inputs consisting entirely of stopwords are handled safely."""
    with pytest.raises(VectorizationError):
        fit_tfidf(["is the and of a to"])


def test_multiple_documents_token_lists() -> None:
    """Verify multiple preprocessed document token lists can be vectorized."""
    doc1 = ["supervised", "learning", "classification"]
    doc2 = ["unsupervised", "learning", "clustering"]
    vectorizer, result = fit_tfidf([doc1, doc2])
    assert result.matrix_shape == (2, 5)
    assert set(result.feature_names) == {
        "classification",
        "clustering",
        "learning",
        "supervised",
        "unsupervised",
    }


def test_vocabulary_matches_actual_feature_matrix() -> None:
    """Verify vocabulary length matches matrix column dimension exactly."""
    text1 = "Deep learning uses multilayer artificial neural networks."
    text2 = "Machine learning builds predictive mathematical models from data."
    vectorizer, result = fit_tfidf([text1, text2])
    assert len(result.feature_names) == result.matrix.shape[1]
    assert len(vectorizer.get_feature_names_out()) == result.matrix.shape[1]
    assert len(vectorizer.vocabulary_) == result.matrix.shape[1]
    assert result.matrix_shape == (2, len(result.feature_names))
