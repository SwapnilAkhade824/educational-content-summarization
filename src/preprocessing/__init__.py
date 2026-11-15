"""Text preprocessing and vectorization for the Educational Content Summarization System."""

from .text_preprocessing import (
    PreprocessingError,
    PreprocessingResult,
    PROTECTED_NEGATIONS,
    clean_text,
    lemmatize_tokens,
    preprocess_text,
    remove_stopwords,
    tokenize_sentences,
    tokenize_words,
)
from .vectorizer import (
    VectorizationError,
    VectorizationResult,
    fit_tfidf,
    top_weighted_terms,
    transform_tfidf,
)

__all__ = [
    "PreprocessingError",
    "PreprocessingResult",
    "PROTECTED_NEGATIONS",
    "VectorizationError",
    "VectorizationResult",
    "clean_text",
    "fit_tfidf",
    "lemmatize_tokens",
    "preprocess_text",
    "remove_stopwords",
    "tokenize_sentences",
    "tokenize_words",
    "top_weighted_terms",
    "transform_tfidf",
]
