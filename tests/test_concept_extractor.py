"""Tests for information extraction."""

from __future__ import annotations

import pytest

from information_extraction.concept_extractor import (
    ConceptExtractionError,
    extract_concepts,
    extract_definitions,
    extract_entities,
    extract_keyphrases,
)

TECH_TEXT = (
    "Machine Learning is a branch of artificial intelligence that enables computers to learn from data. "
    "Natural Language Processing refers to the ability of computers to understand human language. "
    "Deep learning means a subset of machine learning using neural networks. "
    "Google and Microsoft are leading organizations in AI research. "
    "The study was published on January 1, 2024 in New York."
)

NO_DEF_TEXT = (
    "Computers process data quickly. "
    "Algorithms improve performance. "
    "No definitions are provided here."
)

REPEATED_TEXT = (
    "Machine Learning is a field of AI. "
    "Machine learning is a field of AI. "
    "Deep learning is a subset of machine learning."
)


def test_entities_extracted() -> None:
    entities = extract_entities(TECH_TEXT)
    texts = {e.text for e in entities}
    labels = {e.label for e in entities}
    assert any("Google" in t for t in texts) or "Google" in texts
    assert any("Microsoft" in t for t in texts) or "Microsoft" in texts
    assert len(entities) > 0
    for e in entities:
        assert e.start >= 0
        assert e.end > e.start


def test_keyphrases() -> None:
    kps = extract_keyphrases(TECH_TEXT, top_n=5)
    assert len(kps) > 0
    phrases = {k.text.lower() for k in kps}
    # At least one meaningful technical phrase
    assert any("machine" in p or "learning" in p or "processing" in p for p in phrases)


def test_multiple_definitions() -> None:
    defs = extract_definitions(TECH_TEXT)
    concepts = {d.concept.lower() for d in defs}
    assert len(defs) >= 2
    # Machine Learning definition
    ml = next((d for d in defs if "machine" in d.concept.lower()), None)
    assert ml is not None
    assert "artificial intelligence" in ml.definition.lower() or "branch" in ml.definition.lower()


def test_no_definitions() -> None:
    defs = extract_definitions(NO_DEF_TEXT)
    assert defs == []


def test_empty_document() -> None:
    result = extract_concepts("")
    assert result.entities == []
    assert result.keyphrases == []
    assert result.definitions == []
    assert result.combined == []


def test_repeated_concepts_deduplicated() -> None:
    defs = extract_definitions(REPEATED_TEXT)
    concepts = [d.concept.lower() for d in defs]
    assert len(concepts) == len(set(concepts))


def test_incomplete_sentences() -> None:
    text = "Machine learning is. Deep learning means. Incomplete"
    defs = extract_definitions(text)
    # Should not crash; may return few or none
    assert isinstance(defs, list)


def test_extract_concepts_combined() -> None:
    result = extract_concepts(TECH_TEXT)
    assert len(result.combined) > 0
    types = {c.type for c in result.combined}
    assert "entity" in types or "concept_definition" in types or "keyphrase" in types


def test_definition_pattern_is_a() -> None:
    text = "Python is a programming language used for data science."
    defs = extract_definitions(text)
    assert len(defs) == 1
    assert defs[0].concept.lower() == "python"
    assert "programming language" in defs[0].definition.lower()


def test_invalid_input_raises() -> None:
    with pytest.raises(ConceptExtractionError):
        extract_entities(123)  # type: ignore[arg-type]
    with pytest.raises(ConceptExtractionError):
        extract_keyphrases(None)  # type: ignore[arg-type]
    with pytest.raises(ConceptExtractionError):
        extract_definitions(42)  # type: ignore[arg-type]


def test_entity_positions() -> None:
    text = "Google was founded in California."
    entities = extract_entities(text)
    google = next((e for e in entities if "Google" in e.text), None)
    assert google is not None
    assert text[google.start:google.end] == google.text
