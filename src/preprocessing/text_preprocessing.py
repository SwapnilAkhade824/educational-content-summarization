"""Text cleaning, tokenization, stopword removal, and lemmatization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

class PreprocessingError(Exception):
    """Raised when text preprocessing fails."""

# Negations that must be retained to preserve polarity/meaning.
PROTECTED_NEGATIONS: Set[str] = {"no", "not", "never"}


@dataclass
class PreprocessingResult:
    """Structured output of the preprocessing pipeline."""

    original_text: str
    cleaned_text: str
    sentences: List[str]
    tokens: List[str]
    filtered_tokens: List[str]
    lemmatized_tokens: List[str]
    warnings: List[str] = field(default_factory=list)


def _ensure_resource(resource_path: str, download_hint: str) -> None:
    candidates = [resource_path]
    if not resource_path.endswith(".zip"):
        candidates.append(resource_path + ".zip")
    for candidate in candidates:
        try:
            nltk.data.find(candidate)
            return
        except LookupError:
            continue
    raise PreprocessingError(
        f"Required NLTK resource '{resource_path}' is not available. "
        f"Download it with: python -c \"import nltk; nltk.download('{download_hint}')\"."
    )


def clean_text(
    text: str,
    lowercase: bool = True,
    normalize_unicode: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Clean text for analysis without mutating the caller's original string.

    The original text is never modified; this returns a new cleaned string.
    """
    if not isinstance(text, str):
        raise PreprocessingError("Input text must be a string.")

    cleaned = text
    if normalize_unicode:
        # NFC keeps canonical equivalence without aggressive compatibility changes.
        cleaned = unicodedata.normalize("NFC", cleaned)

    # Remove control characters / formatting artifacts.
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    if lowercase:
        cleaned = cleaned.lower()

    if collapse_whitespace:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    _ensure_resource("tokenizers/punkt", "punkt")
    if not text or not text.strip():
        return []
    try:
        return sent_tokenize(text)
    except Exception as exc:  # pragma: no cover - defensive
        raise PreprocessingError(f"Sentence tokenization failed: {exc}") from exc


def tokenize_words(text: str) -> List[str]:
    """Split text into word tokens using NLTK."""
    _ensure_resource("tokenizers/punkt", "punkt")
    if not text or not text.strip():
        return []
    try:
        return word_tokenize(text)
    except Exception as exc:  # pragma: no cover - defensive
        raise PreprocessingError(f"Word tokenization failed: {exc}") from exc


def _default_stopwords() -> Set[str]:
    _ensure_resource("corpora/stopwords", "stopwords")
    try:
        words = set(nltk_stopwords.words("english"))
    except Exception as exc:
        raise PreprocessingError(f"Failed to load NLTK stopwords: {exc}") from exc
    # Always retain negations.
    for neg in PROTECTED_NEGATIONS:
        words.discard(neg)
    return words


def remove_stopwords(
    tokens: Iterable[str],
    stopwords: Optional[Iterable[str]] = None,
    keep_negations: bool = True,
) -> List[str]:
    """Remove stopwords while retaining important negations.

    NLTK English stopwords are common function words (articles, prepositions,
    pronouns, auxiliary verbs, etc.) that carry little semantic content on
    their own and add noise to TF-IDF and topic models. Negations such as
    'no', 'not', and 'never' are retained because they invert meaning.
    """
    if stopwords is None:
        stop_set = _default_stopwords()
    else:
        stop_set = {str(w).lower() for w in stopwords}

    if keep_negations:
        for neg in PROTECTED_NEGATIONS:
            stop_set.discard(neg)

    result: List[str] = []
    for token in tokens:
        lower = token.lower()
        if lower in stop_set:
            continue
        # Drop pure-punctuation tokens.
        if not any(ch.isalnum() for ch in token):
            continue
        result.append(token)
    return result


def lemmatize_tokens(tokens: Iterable[str]) -> List[str]:
    """Lemmatize tokens using WordNet, with POS-aware mapping when available.

    Tokens are normalized to lowercase before morphological lookup so that
    sentence-initial capitalized words (e.g., 'Models' -> 'model') are
    lemmatized consistently for downstream TF-IDF vectorization.
    """
    _ensure_resource("corpora/wordnet", "wordnet")
    try:
        _ensure_resource("corpora/omw-1.4", "omw-1.4")
    except PreprocessingError:
        # omw is optional for basic lemmatization; continue without it.
        pass

    lemmatizer = WordNetLemmatizer()

    token_list = [str(t).lower() for t in tokens]
    if not token_list:
        return []

    # Try POS tagging for better lemmatization; fall back gracefully if tagger
    # resources (averaged_perceptron_tagger / averaged_perceptron_tagger_eng)
    # are not installed.
    pos_tags = None
    try:
        _ensure_resource("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")
        pos_tags = nltk.pos_tag(token_list)
    except (PreprocessingError, LookupError):
        pos_tags = None

    def _to_wordnet_pos(tag: str) -> str:
        if tag.startswith("J"):
            return "a"
        if tag.startswith("V"):
            return "v"
        if tag.startswith("R"):
            return "r"
        return "n"

    # Common gerunds/deverbal nouns in educational NLP/ML text that should
    # remain nouns when POS tags are unavailable.
    _PROTECTED_GERUND_NOUNS = {
        "learning", "processing", "modeling", "modelling", "clustering",
        "training", "testing", "programming", "encoding", "decoding",
        "embedding", "pooling", "padding", "meaning", "understanding",
        "reasoning", "setting", "building", "warning", "finding",
        "ranking", "scoring", "weighting", "mapping", "splitting",
        "chunking", "parsing", "stemming", "tagging", "computing",
        "engineering", "mining", "overfitting", "underfitting",
    }

    result: List[str] = []
    if pos_tags is not None:
        for token, tag in pos_tags:
            result.append(lemmatizer.lemmatize(token, pos=_to_wordnet_pos(tag)))
    else:
        for token in token_list:
            lemma = lemmatizer.lemmatize(token, pos="n")
            if lemma == token and token not in _PROTECTED_GERUND_NOUNS:
                if token.endswith(("ing", "ed")):
                    lemma = lemmatizer.lemmatize(token, pos="v")
            result.append(lemma)
    return result


def preprocess_text(
    original_text: str,
    lowercase: bool = True,
    normalize_unicode: bool = True,
    collapse_whitespace: bool = True,
    stopwords: Optional[Iterable[str]] = None,
    keep_negations: bool = True,
) -> PreprocessingResult:
    """Run the full preprocessing pipeline on a single document's text.

    The original text is retained unchanged in the result.
    """
    if not isinstance(original_text, str):
        raise PreprocessingError("Input text must be a string.")

    warnings: List[str] = []
    cleaned = clean_text(
        original_text,
        lowercase=lowercase,
        normalize_unicode=normalize_unicode,
        collapse_whitespace=collapse_whitespace,
    )

    sentences = tokenize_sentences(cleaned)
    tokens = tokenize_words(cleaned)
    filtered = remove_stopwords(tokens, stopwords=stopwords, keep_negations=keep_negations)

    try:
        lemmatized = lemmatize_tokens(filtered)
    except PreprocessingError as exc:
        lemmatized = list(filtered)
        warnings.append(str(exc))

    return PreprocessingResult(
        original_text=original_text,
        cleaned_text=cleaned,
        sentences=sentences,
        tokens=tokens,
        filtered_tokens=filtered,
        lemmatized_tokens=lemmatized,
        warnings=warnings,
    )
