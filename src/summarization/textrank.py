"""Extractive summarization using the TextRank algorithm."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextRankError(Exception):
    """Raised when TextRank summarization fails."""


@dataclass
class TextRankResult:
    """Structured output of TextRank summarization."""

    original_text: str
    summary_text: str
    selected_sentences: List[str]
    sentence_scores: Dict[str, float]
    original_word_count: int
    summary_word_count: int
    compression_ratio: float
    warnings: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    similarity_matrix: Any = None
    sentence_vectors_shape: Tuple[int, int] = (0, 0)
    vocabulary: List[str] = field(default_factory=list)
    graph_nodes: int = 0
    graph_edges: int = 0


def _ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.data.find("tokenizers/punkt.zip")
        except LookupError as exc:
            raise TextRankError(
                "Required NLTK resource 'punkt' is not available. "
                "Download it with: python -c \"import nltk; nltk.download('punkt')\"."
            ) from exc


def _word_count(text: str) -> int:
    return len(text.split())


def _segment_sentences(text: str) -> List[str]:
    _ensure_punkt()
    if not text or not text.strip():
        return []
    try:
        preprocessed_blocks = re.sub(
            r"(?m)^((?:page|chapter|section|stage|part)\s+\d+\s*[:.\-][^\n]*)\n+",
            r"\n\n",
            text,
            flags=re.IGNORECASE,
        )
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", preprocessed_blocks) if p.strip()]
        sentences: List[str] = []
        for para in paragraphs:
            for s in sent_tokenize(para):
                cleaned = re.sub(r"\s+", " ", s).strip()
                if cleaned:
                    sentences.append(cleaned)
    except Exception as exc:  # pragma: no cover - defensive
        raise TextRankError(f"Sentence segmentation failed: {exc}") from exc
    return sentences


def _deduplicate(sentences: List[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for sentence in sentences:
        key = re.sub(r"\s+", " ", sentence.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(sentence)
    return unique


def summarize_text(
    text: str,
    num_sentences: Optional[int] = None,
    ratio: Optional[float] = None,
    damping: float = 0.85,
    step_callback: Optional[Callable[[int, int, str], None]] = None,
) -> TextRankResult:
    """Summarize text using TextRank.

    Args:
        text: Input educational text.
        num_sentences: Maximum number of sentences in the summary.
        ratio: Fraction of sentences to include (used if num_sentences is None).
        damping: PageRank damping factor.
        step_callback: Optional callback receiving (step_index, total_steps, description).

    Returns:
        TextRankResult with selected sentences in original order and intermediate artifacts.
    """
    if not isinstance(text, str):
        raise TextRankError("Input text must be a string.")

    def _notify(step: int, msg: str) -> None:
        if step_callback is not None:
            try:
                step_callback(step, 5, msg)
            except Exception:
                pass

    warnings: List[str] = []
    original_word_count = _word_count(text)

    _notify(1, "Phase 1/5: Segmenting and deduplicating candidate sentences...")
    sentences = _segment_sentences(text)
    unique = _deduplicate(sentences)

    if not unique:
        return TextRankResult(
            original_text=text,
            summary_text="",
            selected_sentences=[],
            sentence_scores={},
            original_word_count=original_word_count,
            summary_word_count=0,
            compression_ratio=0.0,
            warnings=warnings,
            sentences=[],
            similarity_matrix=None,
            sentence_vectors_shape=(0, 0),
            vocabulary=[],
            graph_nodes=0,
            graph_edges=0,
        )

    # Determine how many sentences to select.
    n = len(unique)
    if num_sentences is not None:
        if num_sentences <= 0:
            count = 0
        else:
            count = min(num_sentences, n)
    elif ratio is not None:
        if ratio <= 0:
            count = 0
        elif ratio >= 1:
            count = n
        else:
            count = min(n, max(1, round(ratio * n)))
    else:
        count = min(3, n)

    # Compute sentence scores.
    scores: Dict[str, float] = {}
    similarity_matrix: Any = None
    vec_shape: Tuple[int, int] = (0, 0)
    vocab_list: List[str] = []
    graph_nodes = n
    graph_edges = 0

    if n == 1:
        _notify(2, "Phase 2/5: Single sentence detected; assigning unit centrality score...")
        scores[unique[0]] = 1.0
    else:
        _notify(2, f"Phase 2/5: Vectorizing {n} candidate sentences with TF-IDF...")
        try:
            vectorizer = TfidfVectorizer(sublinear_tf=True)
            matrix = vectorizer.fit_transform(unique)
            vec_shape = (int(matrix.shape[0]), int(matrix.shape[1]))
            vocab_list = list(vectorizer.get_feature_names_out())
        except ValueError as exc:
            warnings.append(f"TF-IDF vectorization failed: {exc}")
            # Fallback to uniform scores.
            for sentence in unique:
                scores[sentence] = 1.0 / n
            matrix = None

        if matrix is not None:
            _notify(3, f"Phase 3/5: Computing {n}×{n} pairwise cosine similarity matrix...")
            similarity = cosine_similarity(matrix)
            # Zero out self-similarity.
            for i in range(n):
                similarity[i, i] = 0.0
            similarity_matrix = similarity

            _notify(4, "Phase 4/5: Constructing similarity graph and running PageRank...")
            graph = nx.Graph()
            for i in range(n):
                graph.add_node(i)
            for i in range(n):
                for j in range(i + 1, n):
                    weight = float(similarity[i, j])
                    if weight > 0.0:
                        graph.add_edge(i, j, weight=weight)

            graph_edges = graph.number_of_edges()
            if graph_edges == 0:
                warnings.append(
                    "No sentence similarity edges found; using uniform scores."
                )
                for sentence in unique:
                    scores[sentence] = 1.0 / n
            else:
                try:
                    pagerank = nx.pagerank(graph, alpha=damping, weight="weight")
                except Exception as exc:  # pragma: no cover - defensive
                    warnings.append(f"PageRank failed: {exc}")
                    for sentence in unique:
                        scores[sentence] = 1.0 / n
                    pagerank = {}
                for i, sentence in enumerate(unique):
                    scores[sentence] = float(pagerank.get(i, 0.0))

    _notify(5, f"Phase 5/5: Selecting top {count} sentence(s) and restoring document order...")
    if count == 0:
        selected: List[str] = []
    else:
        # Rank by score descending (ties broken by original order).
        ranked_indices = sorted(
            range(n), key=lambda i: scores[unique[i]], reverse=True
        )
        chosen = set(ranked_indices[:count])
        selected = [unique[i] for i in range(n) if i in chosen]

    summary_text = " ".join(selected)
    summary_word_count = _word_count(summary_text)
    compression_ratio = (
        summary_word_count / original_word_count if original_word_count else 0.0
    )

    return TextRankResult(
        original_text=text,
        summary_text=summary_text,
        selected_sentences=selected,
        sentence_scores=scores,
        original_word_count=original_word_count,
        summary_word_count=summary_word_count,
        compression_ratio=compression_ratio,
        warnings=warnings,
        sentences=unique,
        similarity_matrix=similarity_matrix,
        sentence_vectors_shape=vec_shape,
        vocabulary=vocab_list,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
    )
