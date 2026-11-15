"""Tests for LDA topic modelling."""

from __future__ import annotations

import pytest

from topic_modeling.lda_model import TopicModellingError, run_lda

DOCS = [
    "Machine learning is a subset of artificial intelligence. It uses data and algorithms to learn patterns.",
    "Natural language processing deals with text and human language. It includes tokenization and parsing.",
    "Deep learning uses neural networks with many layers. It is applied to image and speech recognition.",
    "Data preprocessing cleans and transforms raw data. It is essential for machine learning pipelines.",
]

SUBJECT_DOCS = [
    "Photosynthesis converts sunlight into energy in plants. Chlorophyll captures light.",
    "Newton's laws describe motion and forces. Gravity pulls objects toward Earth.",
    "The French Revolution changed European politics. Bastille was stormed in 1789.",
]


def test_multiple_documents() -> None:
    result = run_lda(DOCS, n_topics=2, random_state=42)
    assert len(result.topics) == 2
    assert len(result.document_topics) >= len(DOCS)
    for topic in result.topics:
        assert len(topic.keywords) > 0
        assert len(topic.keywords) == len(topic.weights)
    assert result.vocabulary_size > 0


def test_multiple_subjects() -> None:
    result = run_lda(SUBJECT_DOCS, n_topics=3, random_state=42)
    assert len(result.topics) == 3
    assert len(result.document_topics) == len(SUBJECT_DOCS)


def test_empty_input() -> None:
    result = run_lda([])
    assert result.topics == []
    assert result.document_topics == []
    assert any("No non-empty" in w for w in result.warnings)


def test_all_empty_docs() -> None:
    result = run_lda(["", "   "])
    assert result.topics == []
    assert any("No non-empty" in w for w in result.warnings)


def test_one_short_document() -> None:
    result = run_lda(["Machine learning is great."], n_topics=2, random_state=42)
    # Should handle gracefully with warnings, not crash
    assert isinstance(result.warnings, list)
    # Either topics produced or warning about insufficient data
    assert (len(result.topics) >= 1) or (len(result.warnings) > 0)


def test_repeated_content() -> None:
    docs = [DOCS[0]] * 5
    result = run_lda(docs, n_topics=2, random_state=42)
    assert len(result.topics) == 2
    assert len(result.document_topics) == 5


def test_too_few_unique_terms() -> None:
    result = run_lda(["a a a a", "b b b"], n_topics=2, random_state=42)
    # Small vocab warning or reduced topics
    assert (len(result.warnings) > 0) or (len(result.topics) <= 2)


def test_different_topic_counts() -> None:
    r1 = run_lda(DOCS, n_topics=2, random_state=42)
    r2 = run_lda(DOCS, n_topics=4, random_state=42)
    assert len(r1.topics) == 2
    assert len(r2.topics) == 4


def test_reproducibility() -> None:
    r1 = run_lda(DOCS, n_topics=3, random_state=7)
    r2 = run_lda(DOCS, n_topics=3, random_state=7)
    assert [t.keywords for t in r1.topics] == [t.keywords for t in r2.topics]
    assert [d.dominant_topic for d in r1.document_topics] == [d.dominant_topic for d in r2.document_topics]


def test_document_topic_distribution_sums_to_one() -> None:
    result = run_lda(DOCS, n_topics=3, random_state=42)
    for dt in result.document_topics:
        assert abs(sum(dt.topic_distribution) - 1.0) < 1e-6


def test_dominant_topic_valid() -> None:
    result = run_lda(DOCS, n_topics=3, random_state=42)
    for dt in result.document_topics:
        assert 0 <= dt.dominant_topic < len(result.topics)


def test_invalid_n_topics() -> None:
    with pytest.raises(TopicModellingError):
        run_lda(DOCS, n_topics=0)


def test_invalid_n_top_words() -> None:
    with pytest.raises(TopicModellingError):
        run_lda(DOCS, n_top_words=0)


def test_invalid_documents_type() -> None:
    with pytest.raises(TopicModellingError):
        run_lda("not a list")  # type: ignore[arg-type]


def test_perplexity_reported() -> None:
    result = run_lda(DOCS, n_topics=2, random_state=42)
    # Perplexity may or may not be available; if present, it's a number
    if result.perplexity is not None:
        assert isinstance(result.perplexity, float)


def test_model_config_present() -> None:
    result = run_lda(DOCS, n_topics=2, n_top_words=5, max_iter=10, random_state=1, chunk_size=3)
    assert result.model_config["n_topics_requested"] == 2
    assert result.model_config["n_top_words"] == 5
    assert result.model_config["max_iter"] == 10
    assert result.model_config["random_state"] == 1
    assert result.model_config["chunk_size"] == 3
    assert result.model_config["vectorizer"] == "CountVectorizer (BoW)"


def test_long_document_chunking() -> None:
    long_doc = " ".join(DOCS)
    result = run_lda([long_doc], n_topics=2, chunk_size=2, random_state=42)
    assert len(result.document_topics) > 1
