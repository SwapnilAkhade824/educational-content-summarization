"""Educational concept, entity, keyphrase, and definition extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, List, Optional

import spacy


class ConceptExtractionError(Exception):
    """Raised when concept extraction fails."""


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    method: str = "spacy_ner"


@dataclass
class Keyphrase:
    text: str
    score: float = 0.0
    method: str = "noun_chunk_frequency"


@dataclass
class Definition:
    concept: str
    definition: str
    pattern: str
    start: int = 0
    method: str = "rule_based_definition"


@dataclass
class ExtractedItem:
    text: str
    type: str
    description: Optional[str] = None
    method: str = ""
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class ConceptExtractionResult:
    text: str
    entities: List[Entity] = field(default_factory=list)
    keyphrases: List[Keyphrase] = field(default_factory=list)
    definitions: List[Definition] = field(default_factory=list)
    combined: List[ExtractedItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# Common generic words to avoid as standalone keyphrases or concepts.
_GENERIC_WORDS = {
    "this", "that", "these", "those", "it", "they", "we", "you", "i",
    "he", "she", "the", "a", "an", "and", "or", "but", "if", "then",
    "thing", "things", "something", "anything", "everything", "nothing",
    "there", "here", "which", "what", "who", "one", "each", "both",
}

_DEFINITION_PATTERNS = [
    (r"^(.+?)\s+(?:is|are|can\s+be)\s+defined\s+as\s+(.+)$", "is defined as", "defined as"),
    (r"^(.+?)\s+refers\s+to\s+(.+)$", "refers to", "refers to"),
    (r"^(.+?)\s+means\s+(.+)$", "means", "means"),
    (r"^(.+?)\s+(?:is|are)\s+known\s+as\s+(.+)$", "is known as", "known as"),
    (r"^(.+?)\s+is\s+(?:a|an|the)\s+(.+)$", "is a/an/the", " is "),
]


def _load_spacy_model(model_name: str):
    try:
        import spacy as _spacy
    except ImportError as exc:  # pragma: no cover - spacy is a declared dependency
        raise ConceptExtractionError(
            "spaCy is not installed. Install it with: pip install spacy"
        ) from exc
    try:
        return _spacy.load(model_name)
    except OSError as exc:
        raise ConceptExtractionError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc


@lru_cache(maxsize=4)
def _cached_model(model_name: str):
    return _load_spacy_model(model_name)


def _entities_from_doc(doc: Any) -> List[Entity]:
    entities: List[Entity] = []
    seen: set[tuple[str, str]] = set()
    for ent in doc.ents:
        key = (ent.text.strip().lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            Entity(
                text=ent.text.strip(),
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            )
        )
    return entities


def extract_entities(
    text: str, model_name: str = "en_core_web_sm"
) -> List[Entity]:
    """Extract named entities using spaCy NER."""
    if not isinstance(text, str):
        raise ConceptExtractionError("Input text must be a string.")
    if not text.strip():
        return []

    nlp = _cached_model(model_name)
    doc = nlp(text)
    return _entities_from_doc(doc)


def _keyphrases_from_doc(doc: Any, top_n: int = 10) -> List[Keyphrase]:
    counts: dict[str, int] = {}
    for chunk in doc.noun_chunks:
        phrase = " ".join(chunk.text.split())
        if not phrase:
            continue
        lower = phrase.lower()
        if lower in _GENERIC_WORDS:
            continue
        # Skip chunks whose head is not a noun/proper noun.
        if chunk.root.pos_ not in {"NOUN", "PROPN"}:
            continue
        # Skip very short or stopword-only chunks.
        words = [w for w in phrase.split() if w.lower() not in _GENERIC_WORDS]
        if not words:
            continue
        counts[phrase] = counts.get(phrase, 0) + 1

    # Score by frequency, with a small boost for multi-word phrases.
    scored = []
    for phrase, count in counts.items():
        score = float(count) + 0.1 * (len(phrase.split()) - 1)
        scored.append((phrase, score))
    scored.sort(key=lambda x: (-x[1], x[0]))

    result: List[Keyphrase] = []
    for phrase, score in scored[:top_n]:
        result.append(Keyphrase(text=phrase, score=score))
    return result


def extract_keyphrases(
    text: str, model_name: str = "en_core_web_sm", top_n: int = 10
) -> List[Keyphrase]:
    """Extract keyphrases from noun chunks, ranked by frequency."""
    if not isinstance(text, str):
        raise ConceptExtractionError("Input text must be a string.")
    if not text.strip():
        return []

    nlp = _cached_model(model_name)
    doc = nlp(text)
    return _keyphrases_from_doc(doc, top_n=top_n)


def _clean_concept(concept: str) -> str:
    concept = concept.strip(" ,;:-")
    # Strip common introductory transition words/clauses (e.g. "However, X", "In AI, X").
    concept = re.sub(
        r"^(?:however|therefore|furthermore|moreover|specifically|generally|formally|in\s+[\w\s]{1,25}),\s+",
        "",
        concept,
        flags=re.IGNORECASE,
    )
    # Remove leading articles.
    concept = re.sub(r"^(the|a|an)\s+", "", concept, flags=re.IGNORECASE)
    return concept.strip()


def _clean_definition(definition: str) -> str:
    return definition.strip().rstrip(".!?").strip() + ""


def extract_definitions(text: str) -> List[Definition]:
    """Extract rule-based definitions from educational text."""
    if not isinstance(text, str):
        raise ConceptExtractionError("Input text must be a string.")
    if not text.strip():
        return []

    definitions: List[Definition] = []
    seen: set[str] = set()

    # Split into paragraphs and separate standalone heading lines (e.g. "Page 1: ...")
    # so PDF headings without trailing periods do not merge into the first sentence.
    preprocessed_blocks = re.sub(
        r"(?m)^((?:page|chapter|section|stage|part)\s+\d+\s*[:.\-][^\n]*)\n+",
        r"\1.\n\n",
        text,
        flags=re.IGNORECASE,
    )
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", preprocessed_blocks) if p.strip()]
    sentences: List[str] = []
    try:
        import nltk

        nltk.data.find("tokenizers/punkt")
        from nltk.tokenize import sent_tokenize

        for para in paragraphs:
            for s in sent_tokenize(para):
                if s.strip():
                    sentences.append(s.strip())
    except Exception:
        for para in paragraphs:
            for s in para.split("."):
                if s.strip():
                    sentences.append(s.strip())

    offset = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        start = text.find(sentence[:20], offset)
        if start == -1:
            start = offset
        offset = start + len(sentence)

        # Collapse PDF intra-sentence line wraps so regex patterns match across wrapped lines.
        normalized_sentence = re.sub(r"\s+", " ", sentence).strip().rstrip(".!? ")
        if not normalized_sentence or len(normalized_sentence) > 500:
            continue
        if re.match(r"^(?:page|chapter|section|stage|part)\s+\d+\s*[:.\-]", normalized_sentence, flags=re.IGNORECASE):
            continue

        lower_sent = f" {normalized_sentence.lower()} "
        for pattern, label, trigger in _DEFINITION_PATTERNS:
            if trigger not in lower_sent:
                continue
            match = re.match(pattern, normalized_sentence, flags=re.IGNORECASE)
            if not match:
                continue
            concept = _clean_concept(match.group(1))
            definition = _clean_definition(match.group(2))
            if not concept or not definition:
                continue
            max_words = 5 if label == "is a/an/the" else 7
            if len(concept.split()) > max_words:
                continue
            if any(v in concept.lower().split() for v in ("assumes", "states", "shows", "proves", "where", "when", "because")):
                continue
            key = concept.lower()
            if key in _GENERIC_WORDS or key in seen:
                continue
            seen.add(key)
            definitions.append(
                Definition(
                    concept=concept,
                    definition=definition,
                    pattern=label,
                    start=start,
                )
            )
            break  # one definition per sentence

    return definitions


def extract_concepts(
    text: str,
    model_name: str = "en_core_web_sm",
    top_n_keyphrases: int = 10,
    step_callback: Optional[Callable[[int, int, str], None]] = None,
) -> ConceptExtractionResult:
    """Run all information-extraction steps and combine results."""
    if not isinstance(text, str):
        raise ConceptExtractionError("Input text must be a string.")

    def _notify(step: int, msg: str) -> None:
        if step_callback is not None:
            try:
                step_callback(step, 3, msg)
            except Exception:
                pass

    result = ConceptExtractionResult(text=text)
    warnings: List[str] = []

    # Definitions are rule-based and do not require spaCy.
    _notify(1, "Phase 1/3: Extracting rule-based syntactic concept definitions...")
    try:
        result.definitions = extract_definitions(text)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"Definition extraction failed: {exc}")

    # spaCy-dependent steps: run nlp(text) once and reuse the parsed Doc for
    # both Named Entity Recognition and noun-chunk keyphrase extraction.
    try:
        _notify(2, f"Phase 2/3: Running spaCy neural pipeline ({model_name}) & extracting named entities...")
        if not text.strip():
            result.entities = []
            result.keyphrases = []
        else:
            nlp = _cached_model(model_name)
            doc = nlp(text)
            result.entities = _entities_from_doc(doc)
            _notify(3, "Phase 3/3: Extracting and ranking noun-chunk keyphrases from parsed syntax tree...")
            result.keyphrases = _keyphrases_from_doc(doc, top_n=top_n_keyphrases)
    except ConceptExtractionError as exc:
        warnings.append(str(exc))

    # Combined, displayable list.
    combined: List[ExtractedItem] = []
    for entity in result.entities:
        combined.append(
            ExtractedItem(
                text=entity.text,
                type="entity",
                description=entity.label,
                method=entity.method,
                start=entity.start,
                end=entity.end,
            )
        )
    for keyphrase in result.keyphrases:
        combined.append(
            ExtractedItem(
                text=keyphrase.text,
                type="keyphrase",
                description=None,
                method=keyphrase.method,
                start=None,
                end=None,
            )
        )
    for definition in result.definitions:
        combined.append(
            ExtractedItem(
                text=definition.concept,
                type="concept_definition",
                description=definition.definition,
                method=definition.method,
                start=definition.start,
                end=None,
            )
        )

    result.combined = combined
    result.warnings = warnings
    return result
