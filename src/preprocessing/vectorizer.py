"""TF-IDF vectorization for educational text documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .text_preprocessing import preprocess_text
except ImportError:  # pragma: no cover
    from preprocessing.text_preprocessing import preprocess_text


class VectorizationError(Exception):
    """Raised when TF-IDF vectorization fails."""


@dataclass
class VectorizationResult:
    """Structured output of a TF-IDF vectorization."""

    matrix: Any
    feature_names: List[str]
    matrix_shape: Tuple[int, int]
    top_terms: List[Tuple[str, float]] = field(default_factory=list)


def _identity(x: Any) -> Any:
    return x


def _prepare_document_tokens(
    documents: Sequence[Union[str, Sequence[str]]]
) -> List[List[str]]:
    """Validate and normalize documents into preprocessed token lists.

    If a document is already a list/tuple of tokens (e.g. lemmatized tokens),
    it is consumed directly. If it is a string, it is processed through the
    full preprocessing pipeline:
    Text Cleaning -> Word Tokenization -> Stopword Removal -> Lemmatization.
    """
    if not isinstance(documents, (list, tuple)):
        raise VectorizationError("Documents must be provided as a list or tuple.")
    if len(documents) == 0:
        raise VectorizationError("Cannot vectorize an empty document list.")

    processed_docs: List[List[str]] = []
    has_non_empty = False

    for doc in documents:
        if isinstance(doc, str):
            if not doc.strip():
                processed_docs.append([])
            else:
                tokens = preprocess_text(doc).lemmatized_tokens
                processed_docs.append(tokens)
                if tokens:
                    has_non_empty = True
        elif isinstance(doc, (list, tuple)):
            tokens = [str(t) for t in doc if str(t).strip()]
            processed_docs.append(tokens)
            if tokens:
                has_non_empty = True
        else:
            raise VectorizationError(
                f"Unsupported document type: {type(doc)}. Expected str or Sequence[str]."
            )

    if not has_non_empty:
        raise VectorizationError(
            "Cannot vectorize a corpus with no non-empty documents or terms."
        )

    return processed_docs


def top_weighted_terms(
    matrix: Any,
    feature_names: Sequence[str],
    top_n: int = 10,
    doc_index: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Return the top-weighted terms from a TF-IDF matrix.

    If doc_index is None, terms are ranked by their mean weight across all
    documents in the matrix.
    """
    if matrix is None or getattr(matrix, "shape", (0, 0))[1] == 0:
        return []
    import numpy as np

    if doc_index is None:
        weights = matrix.mean(axis=0)
        if hasattr(weights, "A1"):
            weights = weights.A1
        else:
            weights = np.asarray(weights).ravel()
    else:
        row = matrix[doc_index]
        if hasattr(row, "toarray"):
            weights = row.toarray().ravel()
        else:
            weights = np.asarray(row).ravel()

    names = list(feature_names)
    if len(names) != len(weights):
        return []
    ranked = sorted(zip(names, (float(w) for w in weights)), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def fit_tfidf(
    documents: Sequence[Union[str, Sequence[str]]],
    top_n: int = 10,
) -> tuple[TfidfVectorizer, VectorizationResult]:
    """Fit a TfidfVectorizer on preprocessed documents and transform them.

    Accepts preprocessed token sequences (e.g. lemmatized tokens) or text strings.
    Text strings are processed through the full NLP pipeline (cleaning, tokenization,
    stopword removal with negation preservation, and lemmatization) before vectorization.
    TfidfVectorizer is configured to consume preprocessed tokens directly without applying
    a second incompatible preprocessing or tokenization step.

    Returns:
        tuple of (fitted TfidfVectorizer, VectorizationResult)

    Raises:
        VectorizationError: On empty input or vocabulary failure.
    """
    processed_docs = _prepare_document_tokens(documents)

    vectorizer = TfidfVectorizer(
        preprocessor=_identity,
        tokenizer=_identity,
        token_pattern=None,
        lowercase=False,
    )
    try:
        matrix = vectorizer.fit_transform(processed_docs)
    except ValueError as exc:
        raise VectorizationError(f"TF-IDF vectorization failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise VectorizationError(f"TF-IDF vectorization failed: {exc}") from exc

    feature_names = list(vectorizer.get_feature_names_out())
    if not feature_names or matrix.shape[1] == 0:
        raise VectorizationError("TF-IDF vocabulary is empty after preprocessing.")

    result = VectorizationResult(
        matrix=matrix,
        feature_names=feature_names,
        matrix_shape=tuple(matrix.shape),
        top_terms=top_weighted_terms(matrix, feature_names, top_n=top_n),
    )
    return vectorizer, result


def transform_tfidf(
    vectorizer: TfidfVectorizer,
    documents: Sequence[Union[str, Sequence[str]]],
    top_n: int = 10,
) -> VectorizationResult:
    """Transform unseen documents using an already-fitted vectorizer."""
    if vectorizer is None:
        raise VectorizationError("A fitted vectorizer is required.")
    processed_docs = _prepare_document_tokens(documents)
    try:
        matrix = vectorizer.transform(processed_docs)
    except Exception as exc:
        raise VectorizationError(f"TF-IDF transformation failed: {exc}") from exc

    feature_names = list(vectorizer.get_feature_names_out())
    result = VectorizationResult(
        matrix=matrix,
        feature_names=feature_names,
        matrix_shape=tuple(matrix.shape),
        top_terms=top_weighted_terms(matrix, feature_names, top_n=top_n),
    )
    return result
