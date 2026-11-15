"""Topic modelling using Latent Dirichlet Allocation (scikit-learn)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

class TopicModellingError(Exception):
    """Raised when topic modelling fails."""


@dataclass
class Topic:
    """A single discovered topic."""

    topic_id: int
    keywords: List[str]
    weights: List[float]


@dataclass
class DocumentTopic:
    """Topic distribution for a document chunk."""

    chunk_id: int
    source_index: int
    text_snippet: str
    topic_distribution: List[float]
    dominant_topic: int


@dataclass
class LDAResult:
    """Structured output of LDA topic modelling."""

    topics: List[Topic] = field(default_factory=list)
    document_topics: List[DocumentTopic] = field(default_factory=list)
    model_config: dict = field(default_factory=dict)
    vocabulary_size: int = 0
    perplexity: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    chunks: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    bow_matrix_shape: Tuple[int, int] = (0, 0)
    bow_matrix: Any = None
    top_bow_terms: List[Tuple[str, int]] = field(default_factory=list)


def _ensure_list(documents) -> List[str]:
    if not isinstance(documents, (list, tuple)):
        raise TopicModellingError("Documents must be provided as a list or tuple.")
    return [str(d) for d in documents]


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    """Split a long document into sentence-based chunks."""
    text = text.strip()
    if not text:
        return []
    sentences: List[str]
    try:
        import nltk

        nltk.data.find("tokenizers/punkt")
        from nltk.tokenize import sent_tokenize

        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return [text]

    chunks: List[str] = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [text]


def run_lda(
    documents: Sequence[str],
    n_topics: int = 3,
    n_top_words: int = 10,
    max_iter: int = 20,
    random_state: int = 42,
    chunk_size: int = 5,
    step_callback: Optional[Callable[[int, int, str], None]] = None,
) -> LDAResult:
    """Run LDA topic modelling on a collection of documents."""
    if not isinstance(n_topics, int) or n_topics < 1:
        raise TopicModellingError("n_topics must be a positive integer.")
    if not isinstance(n_top_words, int) or n_top_words < 1:
        raise TopicModellingError("n_top_words must be a positive integer.")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise TopicModellingError("max_iter must be a positive integer.")

    def _notify(step: int, msg: str) -> None:
        if step_callback is not None:
            try:
                step_callback(step, 4, msg)
            except Exception:
                pass

    docs = _ensure_list(documents)
    warnings: List[str] = []

    _notify(1, "Phase 1/4: Segmenting input documents into sentence chunks...")
    # Build chunks.
    chunks: List[Tuple[int, str]] = []
    for idx, doc in enumerate(docs):
        for chunk in _chunk_text(doc, chunk_size):
            chunks.append((idx, chunk))

    non_empty_chunks = [(si, c) for si, c in chunks if c.strip()]
    if not non_empty_chunks:
        warnings.append(
            "No non-empty documents provided. LDA requires meaningful text. "
            "Provide multiple educational documents or longer passages."
        )
        return LDAResult(warnings=warnings)

    source_indices = [si for si, _ in non_empty_chunks]
    chunk_texts = [c for _, c in non_empty_chunks]

    _notify(2, f"Phase 2/4: Building CountVectorizer Bag-of-Words matrix across {len(chunk_texts)} chunk(s)...")
    # Vectorize with CountVectorizer (BoW) for LDA.
    # Prefer multi-character alphabetic tokens (excluding 1-letter symbols like 'k' and digits)
    # while falling back to general word tokens for minimal synthetic test inputs.
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    extended_stopwords = list(
        set(ENGLISH_STOP_WORDS)
        | {"using", "used", "use", "example", "examples", "contain", "contains", "based", "given", "also"}
    )
    try:
        vectorizer = CountVectorizer(
            stop_words=extended_stopwords,
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
            min_df=1,
        )
        X = vectorizer.fit_transform(chunk_texts)
    except ValueError:
        try:
            vectorizer = CountVectorizer(
                stop_words="english",
                lowercase=True,
                token_pattern=r"(?u)\b\w+\b",
                min_df=1,
            )
            X = vectorizer.fit_transform(chunk_texts)
        except ValueError as exc:
            warnings.append(f"Vectorization failed: {exc}")
            return LDAResult(warnings=warnings, chunks=chunk_texts)

    feature_names = list(vectorizer.get_feature_names_out())
    vocab_size = len(feature_names)
    bow_shape = (int(X.shape[0]), int(X.shape[1]))

    term_counts = np.asarray(X.sum(axis=0)).ravel()
    top_bow = sorted(
        zip(feature_names, (int(c) for c in term_counts)),
        key=lambda item: (-item[1], item[0]),
    )[:15]

    if vocab_size == 0:
        warnings.append(
            "Vocabulary is empty after preprocessing. LDA cannot run. "
            "Provide documents with more varied and meaningful terms."
        )
        return LDAResult(warnings=warnings, chunks=chunk_texts)

    if vocab_size < 5:
        warnings.append(
            f"Vocabulary is very small ({vocab_size} terms). "
            "Topics may not be meaningful; provide more diverse text."
        )

    effective_topics = min(n_topics, vocab_size)
    if effective_topics < n_topics:
        warnings.append(
            f"Requested {n_topics} topics but vocabulary only supports "
            f"{effective_topics}. Reducing topic count."
        )
    if effective_topics < 1:
        effective_topics = 1

    if len(non_empty_chunks) < effective_topics:
        warnings.append(
            f"Number of document chunks ({len(non_empty_chunks)}) is less than "
            f"number of topics ({effective_topics}); topic assignments may be unstable."
        )

    _notify(3, f"Phase 3/4: Fitting LatentDirichletAllocation ({effective_topics} topics, max_iter={max_iter})...")
    # Train LDA.
    try:
        lda = LatentDirichletAllocation(
            n_components=effective_topics,
            max_iter=max_iter,
            random_state=random_state,
            learning_method="batch",
        )
        lda.fit(X)
    except Exception as exc:
        raise TopicModellingError(f"LDA training failed: {exc}") from exc

    _notify(4, "Phase 4/4: Computing topic-keyword weights and chunk topic distributions...")
    # Extract topics.
    topics: List[Topic] = []
    for topic_idx, component in enumerate(lda.components_):
        # Normalize to probabilities.
        total = float(component.sum())
        if total <= 0:
            weights = component
        else:
            weights = component / total
        top_indices = np.argsort(weights)[::-1][:n_top_words]
        keywords = [feature_names[i] for i in top_indices]
        top_weights = [float(weights[i]) for i in top_indices]
        topics.append(
            Topic(topic_id=topic_idx, keywords=keywords, weights=top_weights)
        )

    # Document-topic distributions.
    doc_topic_dist = lda.transform(X)
    document_topics: List[DocumentTopic] = []
    for i, (si, chunk) in enumerate(non_empty_chunks):
        dist = [float(v) for v in doc_topic_dist[i]]
        dominant = int(np.argmax(dist)) if dist else 0
        snippet = chunk[:120] + ("..." if len(chunk) > 120 else "")
        document_topics.append(
            DocumentTopic(
                chunk_id=i,
                source_index=si,
                text_snippet=snippet,
                topic_distribution=dist,
                dominant_topic=dominant,
            )
        )

    # Perplexity (not coherence).
    perplexity: Optional[float] = None
    try:
        perplexity = float(lda.perplexity(X))
    except Exception:
        perplexity = None

    model_config = {
        "n_topics_requested": n_topics,
        "n_topics_effective": effective_topics,
        "n_top_words": n_top_words,
        "max_iter": max_iter,
        "random_state": random_state,
        "chunk_size": chunk_size,
        "learning_method": "batch",
        "vectorizer": "CountVectorizer (BoW)",
    }

    return LDAResult(
        topics=topics,
        document_topics=document_topics,
        model_config=model_config,
        vocabulary_size=vocab_size,
        perplexity=perplexity,
        warnings=warnings,
        chunks=chunk_texts,
        feature_names=feature_names,
        bow_matrix_shape=bow_shape,
        bow_matrix=X,
        top_bow_terms=top_bow,
    )
