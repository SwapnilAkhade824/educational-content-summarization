"""Streamlit entry point for the Educational Content Summarization System.

Single-page sequential pipeline layout displaying every NLP processing stage
with actual algorithm outputs, live per-stage loading indicators, and granular
step-by-step phase progress feedback. Student feedback sentiment analysis is
kept in a separate section because it is not part of the document pipeline.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st

try:
    from extraction.text_extractor import extract_document, ExtractionError
    from preprocessing.text_preprocessing import (
        PROTECTED_NEGATIONS,
        clean_text,
        lemmatize_tokens,
        preprocess_text,
        remove_stopwords,
        tokenize_sentences,
        tokenize_words,
    )
    from preprocessing.vectorizer import fit_tfidf
    from summarization.textrank import summarize_text
    from summarization.llm_summarizer import (
        LLMError,
        generate_summary,
        interpret_topics,
    )
    from information_extraction.concept_extractor import extract_concepts
    from topic_modeling.lda_model import run_lda
    from sentiment.sentiment_analyzer import analyze_feedback, analyze_batch
    from utils.config import load_config
    from summarization.llm_client import count_tokens, test_connection
    from ui.helpers import build_export_markdown
except ImportError:  # pragma: no cover
    from src.extraction.text_extractor import extract_document, ExtractionError
    from src.preprocessing.text_preprocessing import (
        PROTECTED_NEGATIONS,
        clean_text,
        lemmatize_tokens,
        preprocess_text,
        remove_stopwords,
        tokenize_sentences,
        tokenize_words,
    )
    from src.preprocessing.vectorizer import fit_tfidf
    from src.summarization.textrank import summarize_text
    from src.summarization.llm_summarizer import (
        LLMError,
        generate_summary,
        interpret_topics,
    )
    from src.information_extraction.concept_extractor import extract_concepts
    from src.topic_modeling.lda_model import run_lda
    from src.sentiment.sentiment_analyzer import analyze_feedback, analyze_batch
    from src.utils.config import load_config
    from src.summarization.llm_client import count_tokens, test_connection
    from src.ui.helpers import build_export_markdown


# ---------------------------------------------------------------------------
# One-time NLTK resource bootstrap (needed on fresh cloud hosts)
# ---------------------------------------------------------------------------

_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "vader_lexicon": "sentiment/vader_lexicon.zip",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


@st.cache_resource(show_spinner="Preparing NLTK language resources (first run only)...")
def _ensure_nltk_resources() -> List[str]:
    """Download any missing NLTK corpora/models once per server process."""
    import nltk

    downloaded: List[str] = []
    for package, path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                if nltk.download(package, quiet=True):
                    downloaded.append(package)
            except Exception:  # pragma: no cover - offline host
                pass
    return downloaded


# ---------------------------------------------------------------------------
# Status badge helpers
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "Not Started": "⬜",
    "Running": "🔄",
    "Completed": "✅",
    "Warning": "⚠️",
    "Failed": "❌",
}


def _badge(status: str) -> str:
    icon = _STATUS_ICONS.get(status, "⬜")
    return f"{icon} **{status}**"


# ---------------------------------------------------------------------------
# Session state management & document-change invalidation
# ---------------------------------------------------------------------------

_STAGE_KEYS = [
    "stage_extraction",
    "stage_sentence_tok",
    "stage_word_tok",
    "stage_cleaning",
    "stage_stopword",
    "stage_lemma",
    "stage_tfidf",
    "stage_textrank",
    "stage_info_extraction",
    "stage_lda",
    "stage_gemini",
]

_ALL_KEYS = (
    _STAGE_KEYS
    + [
        "raw_text",
        "extracted_data",
        "sentences",
        "word_tokens",
        "cleaned_text",
        "original_text",
        "filtered_tokens",
        "removed_stopwords",
        "lemma_pairs",
        "lemmatized_tokens",
        "tfidf_result",
        "tfidf_vectorizer",
        "textrank_result",
        "similarity_matrix",
        "concepts_result",
        "lda_result",
        "lda_interpretations",
        "gemini_results",
        "gemini_status_info",
        "gemini_error",
        "gemini_chunk_cache",
        "gemini_token_count",
        "sentiment_single",
        "sentiment_batch",
        "pipeline_ran",
        "stage_logs",
        "current_doc_fingerprint",
    ]
)

_GEMINI_SUMMARY_PHASES = [
    ("short", "Short Summary"),
    ("detailed", "Detailed Summary"),
    ("bullets", "Bullet-Point Study Notes"),
    ("definitions", "Important Definitions"),
    ("revision", "Exam Revision Notes"),
]


def _init_state() -> None:
    for key in _ALL_KEYS:
        if key not in st.session_state:
            if key in _STAGE_KEYS:
                st.session_state[key] = "Not Started"
            elif key == "pipeline_ran":
                st.session_state[key] = False
            elif key == "stage_logs":
                st.session_state[key] = {}
            else:
                st.session_state[key] = None


def _reset_state() -> None:
    for key in _ALL_KEYS:
        if key in st.session_state:
            del st.session_state[key]
    _init_state()


def _reset_document_pipeline_state(new_fingerprint: Optional[str] = None) -> None:
    """Invalidate all document-dependent pipeline results when the source document changes."""
    preserved_single = st.session_state.get("sentiment_single")
    preserved_batch = st.session_state.get("sentiment_batch")
    _reset_state()
    st.session_state["sentiment_single"] = preserved_single
    st.session_state["sentiment_batch"] = preserved_batch
    st.session_state["current_doc_fingerprint"] = new_fingerprint


def _compute_doc_fingerprint(filename: str, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    return f"{filename}::{len(content)}::{digest}"


def _ensure_fresh_document(filename: str, content: bytes) -> bool:
    """Return True if the uploaded document changed and stale results were invalidated."""
    fp = _compute_doc_fingerprint(filename, content)
    current = st.session_state.get("current_doc_fingerprint")
    if current is not None and current != fp:
        _reset_document_pipeline_state(fp)
        return True
    st.session_state["current_doc_fingerprint"] = fp
    return False


def _set_stage(key: str, status: str) -> None:
    st.session_state[key] = status


_STAGE_RESULT_KEYS: Dict[str, str] = {
    "stage_extraction": "extracted_data",
    "stage_sentence_tok": "sentences",
    "stage_word_tok": "word_tokens",
    "stage_cleaning": "cleaned_text",
    "stage_stopword": "filtered_tokens",
    "stage_lemma": "lemma_pairs",
    "stage_tfidf": "tfidf_result",
    "stage_textrank": "textrank_result",
    "stage_info_extraction": "concepts_result",
    "stage_lda": "lda_result",
    "stage_gemini": "gemini_results",
}


def _reconcile_interrupted_stages() -> None:
    """Prevent stages from staying stuck on 'Running' if a previous Streamlit script rerun was interrupted."""
    for stage_key, result_key in _STAGE_RESULT_KEYS.items():
        if st.session_state.get(stage_key) == "Running":
            if st.session_state.get(result_key):
                st.session_state[stage_key] = "Completed"
            else:
                st.session_state[stage_key] = "Not Started"


def _log_stage_step(stage_key: str, message: str, reset: bool = False) -> None:
    logs = st.session_state.get("stage_logs")
    if not isinstance(logs, dict):
        logs = {}
        st.session_state["stage_logs"] = logs
    if reset or stage_key not in logs:
        logs[stage_key] = []
    logs[stage_key].append(message)


def _render_stage_log(stage_key: str) -> None:
    logs = st.session_state.get("stage_logs") or {}
    steps = logs.get(stage_key) or []
    if steps:
        with st.expander(f"Execution phase log ({len(steps)} steps)", expanded=False):
            for idx, item in enumerate(steps, 1):
                st.caption(f"{idx}. {item}")


# ---------------------------------------------------------------------------
# Individual stage runners with live step-by-step phase feedback
# ---------------------------------------------------------------------------


def _run_extraction(
    file_content: bytes,
    filename: str,
    step_hook: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """Stage 1: Text Extraction."""
    _set_stage("stage_extraction", "Running")
    _log_stage_step("stage_extraction", f"Phase 1/3: Validating file '{filename}' ({len(file_content):,} bytes)...", reset=True)
    if step_hook:
        step_hook(1, 3, f"Phase 1/3: Validating file '{filename}' ({len(file_content):,} bytes)...")
    try:
        _log_stage_step("stage_extraction", "Phase 2/3: Extracting document text and page structure...")
        if step_hook:
            step_hook(2, 3, "Phase 2/3: Extracting document text and page structure...")
        result = extract_document(filename, file_content)
        st.session_state["extracted_data"] = result
        st.session_state["raw_text"] = result.extracted_text
        st.session_state["original_text"] = result.extracted_text

        char_count = len(result.extracted_text or "")
        word_count = len((result.extracted_text or "").split())
        _log_stage_step(
            "stage_extraction",
            f"Phase 3/3: Extracted {result.page_count} page(s), {char_count:,} characters, {word_count:,} words.",
        )
        if step_hook:
            step_hook(3, 3, f"Phase 3/3: Extracted {result.page_count} page(s), {word_count:,} words.")

        if not result.extracted_text.strip():
            _set_stage("stage_extraction", "Warning")
            return False
        if result.extraction_warnings:
            _set_stage("stage_extraction", "Warning")
        else:
            _set_stage("stage_extraction", "Completed")
        return True
    except ExtractionError as exc:
        _set_stage("stage_extraction", "Failed")
        _log_stage_step("stage_extraction", f"Extraction error: {exc}")
        st.error(f"Extraction failed: {exc}")
        return False
    except Exception as exc:  # pragma: no cover
        _set_stage("stage_extraction", "Failed")
        _log_stage_step("stage_extraction", f"Unexpected error: {exc}")
        st.error(f"Unexpected extraction error: {exc}")
        return False


def _run_sentence_tokenization(
    step_hook: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """Stage 2: Sentence Tokenization."""
    text = st.session_state.get("raw_text", "")
    if not text or not text.strip():
        _set_stage("stage_sentence_tok", "Warning")
        return False
    _set_stage("stage_sentence_tok", "Running")
    _log_stage_step("stage_sentence_tok", "Phase 1/2: Loading NLTK Punkt sentence boundary detector...", reset=True)
    if step_hook:
        step_hook(1, 2, "Phase 1/2: Segmenting raw text into sentences with NLTK Punkt...")
    try:
        sentences = tokenize_sentences(text)
        st.session_state["sentences"] = sentences
        _log_stage_step("stage_sentence_tok", f"Phase 2/2: Segmented document into {len(sentences)} sentences.")
        if step_hook:
            step_hook(2, 2, f"Phase 2/2: Segmented {len(sentences)} sentences.")
        _set_stage("stage_sentence_tok", "Completed")
        return True
    except Exception as exc:
        _set_stage("stage_sentence_tok", "Failed")
        st.error(f"Sentence tokenization failed: {exc}")
        return False


def _run_word_tokenization(
    step_hook: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """Stage 3: Word Tokenization."""
    text = st.session_state.get("raw_text", "")
    if not text or not text.strip():
        _set_stage("stage_word_tok", "Warning")
        return False
    _set_stage("stage_word_tok", "Running")
    _log_stage_step("stage_word_tok", "Phase 1/2: Splitting text into word and punctuation tokens...", reset=True)
    if step_hook:
        step_hook(1, 2, "Phase 1/2: Tokenizing words using NLTK word_tokenize...")
    try:
        tokens = tokenize_words(text)
        st.session_state["word_tokens"] = tokens
        _log_stage_step("stage_word_tok", f"Phase 2/2: Extracted {len(tokens):,} raw word tokens.")
        if step_hook:
            step_hook(2, 2, f"Phase 2/2: Extracted {len(tokens):,} word tokens.")
        _set_stage("stage_word_tok", "Completed")
        return True
    except Exception as exc:
        _set_stage("stage_word_tok", "Failed")
        st.error(f"Word tokenization failed: {exc}")
        return False


def _run_text_cleaning(
    step_hook: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """Stage 4: Text Cleaning."""
    text = st.session_state.get("raw_text", "")
    if not text or not text.strip():
        _set_stage("stage_cleaning", "Warning")
        return False
    _set_stage("stage_cleaning", "Running")
    _log_stage_step(
        "stage_cleaning",
        "Phase 1/2: Applying Unicode NFC normalization, control-character removal, lowercasing, and whitespace collapse...",
        reset=True,
    )
    if step_hook:
        step_hook(1, 2, "Phase 1/2: Normalizing Unicode (NFC), lowercasing, and collapsing whitespace...")
    try:
        cleaned = clean_text(text)
        st.session_state["cleaned_text"] = cleaned
        _log_stage_step(
            "stage_cleaning",
            f"Phase 2/2: Cleaned text ready ({len(text):,} -> {len(cleaned):,} characters).",
        )
        if step_hook:
            step_hook(2, 2, f"Phase 2/2: Normalized {len(text):,} -> {len(cleaned):,} characters.")
        _set_stage("stage_cleaning", "Completed")
        return True
    except Exception as exc:
        _set_stage("stage_cleaning", "Failed")
        st.error(f"Text cleaning failed: {exc}")
        return False


def _run_stopword_removal(
    step_hook: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """Stage 5: Stopword Removal."""
    tokens = st.session_state.get("word_tokens")
    if not tokens:
        _set_stage("stage_stopword", "Warning")
        return False
    _set_stage("stage_stopword", "Running")
    _log_stage_step(
        "stage_stopword",
        "Phase 1/2: Filtering NLTK English stopwords and punctuation while protecting negations (no, not, never)...",
        reset=True,
    )
    if step_hook:
        step_hook(1, 2, "Phase 1/2: Filtering stopwords while preserving negations (no, not, never)...")
    try:
        filtered = remove_stopwords(tokens)
        filtered_lower_set = {t.lower() for t in filtered}
        removed = [
            t.lower()
            for t in tokens
            if any(c.isalnum() for c in t) and t.lower() not in filtered_lower_set
        ]
        st.session_state["filtered_tokens"] = filtered
        st.session_state["removed_stopwords"] = removed
        _log_stage_step(
            "stage_stopword",
            f"Phase 2/2: Retained {len(filtered):,} content tokens; removed {len(removed):,} stopword occurrences ({len(set(removed))} unique).",
        )
        if step_hook:
            step_hook(2, 2, f"Phase 2/2: Retained {len(filtered):,} tokens ({len(removed):,} stopwords removed).")
        _set_stage("stage_stopword", "Completed")
        return True
    except Exception as exc:
        _set_stage("stage_stopword", "Failed")
        st.error(f"Stopword removal failed: {exc}")
        return False


def _run_lemmatization(
    step_hook: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """Stage 6: Lemmatization."""
    filtered = st.session_state.get("filtered_tokens")
    if not filtered:
        _set_stage("stage_lemma", "Warning")
        return False
    _set_stage("stage_lemma", "Running")
    _log_stage_step(
        "stage_lemma",
        f"Phase 1/2: Running WordNet morphological lemmatization on {len(filtered):,} filtered tokens...",
        reset=True,
    )
    if step_hook:
        step_hook(1, 2, f"Phase 1/2: Lemmatizing {len(filtered):,} filtered tokens via WordNet...")
    try:
        lemmatized = lemmatize_tokens(filtered)
        pairs = list(zip(filtered, lemmatized))
        st.session_state["lemmatized_tokens"] = lemmatized
        st.session_state["lemma_pairs"] = pairs
        changed_count = sum(1 for orig, lem in pairs if orig.lower() != lem.lower())
        _log_stage_step(
            "stage_lemma",
            f"Phase 2/2: Produced {len(lemmatized):,} lemmas ({changed_count:,} tokens morphologically reduced to base form).",
        )
        if step_hook:
            step_hook(2, 2, f"Phase 2/2: Lemmatized {len(lemmatized):,} tokens ({changed_count:,} reduced).")
        _set_stage("stage_lemma", "Completed")
        return True
    except Exception as exc:
        _set_stage("stage_lemma", "Failed")
        st.error(f"Lemmatization failed: {exc}")
        return False


def _run_tfidf(
    step_hook: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """Stage 7: TF-IDF Vectorization."""
    lemmatized = st.session_state.get("lemmatized_tokens")
    if not lemmatized:
        cleaned = st.session_state.get("cleaned_text")
        if cleaned and cleaned.strip():
            res = preprocess_text(cleaned)
            lemmatized = res.lemmatized_tokens
            st.session_state["lemmatized_tokens"] = lemmatized
    if not lemmatized:
        _set_stage("stage_tfidf", "Warning")
        return False
    _set_stage("stage_tfidf", "Running")
    _log_stage_step(
        "stage_tfidf",
        f"Phase 1/2: Passing {len(lemmatized):,} stopword-filtered, lemmatized tokens into TfidfVectorizer (identity analyzer)...",
        reset=True,
    )
    if step_hook:
        step_hook(1, 2, "Phase 1/2: Computing TF-IDF matrix from lemmatized tokens...")
    try:
        vectorizer, result = fit_tfidf([lemmatized], top_n=15)
        st.session_state["tfidf_result"] = result
        st.session_state["tfidf_vectorizer"] = vectorizer
        _log_stage_step(
            "stage_tfidf",
            f"Phase 2/2: Constructed {result.matrix_shape[0]}×{result.matrix_shape[1]} TF-IDF matrix ({len(result.feature_names):,} unique vocabulary terms).",
        )
        if step_hook:
            step_hook(2, 2, f"Phase 2/2: Built {result.matrix_shape[0]}×{result.matrix_shape[1]} TF-IDF feature matrix.")
        _set_stage("stage_tfidf", "Completed")
        return True
    except Exception as exc:
        _set_stage("stage_tfidf", "Failed")
        st.error(f"TF-IDF vectorization failed: {exc}")
        return False


def _run_textrank_stage(
    num_sentences: int = 3,
    step_hook: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """Stage 8: TextRank Summarization."""
    text = st.session_state.get("raw_text", "")
    if not text or not text.strip():
        _set_stage("stage_textrank", "Warning")
        return False
    _set_stage("stage_textrank", "Running")
    st.session_state.setdefault("stage_logs", {})["stage_textrank"] = []

    def _on_step(step: int, total: int, msg: str) -> None:
        _log_stage_step("stage_textrank", msg)
        if step_hook:
            step_hook(step, total, msg)

    try:
        result = summarize_text(
            text,
            num_sentences=num_sentences,
            step_callback=_on_step,
        )
        st.session_state["textrank_result"] = result
        st.session_state["similarity_matrix"] = result.similarity_matrix

        if result.warnings:
            _set_stage("stage_textrank", "Warning")
        else:
            _set_stage("stage_textrank", "Completed")
        return True
    except Exception as exc:
        _set_stage("stage_textrank", "Failed")
        st.error(f"TextRank summarization failed: {exc}")
        return False


def _run_info_extraction(
    step_hook: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """Stage 9: Information Extraction."""
    text = st.session_state.get("raw_text", "")
    if not text or not text.strip():
        _set_stage("stage_info_extraction", "Warning")
        return False
    _set_stage("stage_info_extraction", "Running")
    st.session_state.setdefault("stage_logs", {})["stage_info_extraction"] = []

    def _on_step(step: int, total: int, msg: str) -> None:
        _log_stage_step("stage_info_extraction", msg)
        if step_hook:
            step_hook(step, total, msg)

    try:
        result = extract_concepts(text, top_n_keyphrases=15, step_callback=_on_step)
        st.session_state["concepts_result"] = result
        _log_stage_step(
            "stage_info_extraction",
            f"Completed: {len(result.entities)} entities, {len(result.keyphrases)} keyphrases, {len(result.definitions)} definitions.",
        )
        if result.warnings:
            _set_stage("stage_info_extraction", "Warning")
        else:
            _set_stage("stage_info_extraction", "Completed")
        return True
    except Exception as exc:
        _set_stage("stage_info_extraction", "Failed")
        st.error(f"Information extraction failed: {exc}")
        return False


def _run_lda_stage(
    n_topics: int = 3,
    step_hook: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """Stage 10: LDA Topic Modelling."""
    text = st.session_state.get("raw_text", "")
    if not text or not text.strip():
        _set_stage("stage_lda", "Warning")
        return False
    _set_stage("stage_lda", "Running")
    st.session_state.setdefault("stage_logs", {})["stage_lda"] = []

    def _on_step(step: int, total: int, msg: str) -> None:
        _log_stage_step("stage_lda", msg)
        if step_hook:
            step_hook(step, total, msg)

    try:
        result = run_lda([text], n_topics=n_topics, random_state=42, step_callback=_on_step)
        st.session_state["lda_result"] = result
        if result.warnings:
            _set_stage("stage_lda", "Warning")
        else:
            _set_stage("stage_lda", "Completed")
        return True
    except Exception as exc:
        _set_stage("stage_lda", "Failed")
        st.error(f"LDA topic modelling failed: {exc}")
        return False


def _run_gemini_stage(
    step_hook: Optional[Callable[[int, int, str, Dict[str, Any]], None]] = None,
) -> bool:
    """Stage 11: Gemini Summarization with live phase & chunk progress feedback."""
    text = st.session_state.get("raw_text", "")
    cfg = load_config()
    if not cfg.is_configured:
        _set_stage("stage_gemini", "Warning")
        return False
    if not text or not text.strip():
        _set_stage("stage_gemini", "Warning")
        return False
    _set_stage("stage_gemini", "Running")
    st.session_state["gemini_error"] = None
    st.session_state.setdefault("stage_logs", {})["stage_gemini"] = []

    if st.session_state.get("gemini_chunk_cache") is None:
        st.session_state["gemini_chunk_cache"] = {}

    total_phases = len(_GEMINI_SUMMARY_PHASES)
    _log_stage_step(
        "stage_gemini",
        f"Pre-check: Model '{cfg.model_id}' configured (thinking_level={cfg.thinking_level}, max_single_request_tokens={cfg.max_single_request_tokens}).",
        reset=True,
    )

    try:
        tok_count = count_tokens(cfg, text)
        st.session_state["gemini_token_count"] = tok_count
        if tok_count is not None:
            mode = (
                "Single-Request Optimization"
                if tok_count <= cfg.max_single_request_tokens
                else "Hierarchical Resumable Chunking"
            )
            _log_stage_step("stage_gemini", f"Token count via Gemini SDK: {tok_count:,} tokens -> {mode}.")
    except Exception:
        st.session_state["gemini_token_count"] = None

    try:
        results = st.session_state.get("gemini_results") or {}
        for phase_idx, (stype, phase_label) in enumerate(_GEMINI_SUMMARY_PHASES, start=1):
            if stype in results and results[stype]:
                _log_stage_step(
                    "stage_gemini",
                    f"Phase {phase_idx}/{total_phases} ({phase_label}): Loaded completed summary from session cache.",
                )
                if step_hook:
                    step_hook(
                        phase_idx,
                        total_phases,
                        f"Phase {phase_idx}/{total_phases}: {phase_label} (cached)",
                        st.session_state.get("gemini_status_info") or {},
                    )
                continue

            def _progress_cb(
                status_dict: Dict[str, Any],
                _p_idx: int = phase_idx,
                _p_label: str = phase_label,
                _stype: str = stype,
            ) -> None:
                enriched = dict(status_dict)
                enriched["phase_index"] = _p_idx
                enriched["total_phases"] = total_phases
                enriched["phase_label"] = _p_label
                enriched["summary_type"] = _stype
                st.session_state["gemini_status_info"] = enriched
                chunk_msg = (
                    f"Chunk {enriched.get('current_chunk', 1)}/{enriched.get('total_chunks', 1)} "
                    f"— {enriched.get('retry_status', 'Running')}"
                )
                if step_hook:
                    step_hook(
                        _p_idx,
                        total_phases,
                        f"Phase {_p_idx}/{total_phases} ({_p_label}): {chunk_msg}",
                        enriched,
                    )

            _log_stage_step("stage_gemini", f"Phase {phase_idx}/{total_phases}: Generating {phase_label}...")
            result = generate_summary(
                text,
                summary_type=stype,
                config=cfg,
                progress_callback=_progress_cb,
                chunk_cache=st.session_state["gemini_chunk_cache"],
            )
            results[stype] = result.summary_text
            st.session_state["gemini_results"] = results
            _log_stage_step(
                "stage_gemini",
                f"Phase {phase_idx}/{total_phases} ({phase_label}): Completed ({result.chunk_count} request chunk(s)).",
            )

        _set_stage("stage_gemini", "Completed")
        st.session_state["gemini_error"] = None
        return True
    except LLMError as exc:
        _set_stage("stage_gemini", "Failed")
        st.session_state["gemini_error"] = str(exc)
        _log_stage_step("stage_gemini", f"Failed with LLMError: {exc}")
        return False
    except Exception as exc:  # pragma: no cover
        _set_stage("stage_gemini", "Failed")
        st.session_state["gemini_error"] = f"Unexpected Gemini error: {exc}"
        _log_stage_step("stage_gemini", f"Unexpected error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Display functions for each stage
# ---------------------------------------------------------------------------


_STAGE_TITLES: Dict[int, str] = {
    1: "Stage 1: Text Extraction",
    2: "Stage 2: Sentence Tokenization",
    3: "Stage 3: Word Tokenization",
    4: "Stage 4: Text Cleaning",
    5: "Stage 5: Stopword Removal",
    6: "Stage 6: Lemmatization",
    7: "Stage 7: TF-IDF Vectorization",
    8: "Stage 8: TextRank Summarization",
    9: "Stage 9: Information Extraction",
    10: "Stage 10: LDA Topic Modelling",
    11: "Stage 11: Gemini Summarization",
}


def _display_stage_1(show_header: bool = True) -> None:
    """Display Stage 1: Text Extraction."""
    if show_header:
        st.markdown(f"### Stage 1: Text Extraction {_badge(st.session_state['stage_extraction'])}")
    _render_stage_log("stage_extraction")
    data = st.session_state.get("extracted_data")
    if data is None:
        return
    char_count = len(data.extracted_text) if data.extracted_text else 0
    word_count = len(data.extracted_text.split()) if data.extracted_text else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original filename", data.filename)
    c2.metric("Page count", data.page_count)
    c3.metric("Character count", f"{char_count:,}")
    c4.metric("Word count", f"{word_count:,}")

    if data.extraction_warnings:
        for w in data.extraction_warnings:
            st.warning(w)
    with st.expander("Extracted text", expanded=False):
        st.text(data.extracted_text[:5000] if data.extracted_text else "(empty)")
        if char_count > 5000:
            st.caption(f"Showing first 5,000 of {char_count:,} total characters.")


def _display_stage_2(show_header: bool = True) -> None:
    """Display Stage 2: Sentence Tokenization."""
    if show_header:
        st.markdown(f"### Stage 2: Sentence Tokenization {_badge(st.session_state['stage_sentence_tok'])}")
    _render_stage_log("stage_sentence_tok")
    sentences = st.session_state.get("sentences")
    if sentences is None:
        return
    st.write(f"**Sentence count:** {len(sentences)}")
    with st.expander("Numbered sentence list", expanded=False):
        df = pd.DataFrame(
            [{"Sentence #": i, "Segmented Sentence": s, "Word Count": len(s.split())} for i, s in enumerate(sentences, 1)]
        )
        st.dataframe(df, hide_index=True, alt="Segmented sentences table")


def _display_stage_3(show_header: bool = True) -> None:
    """Display Stage 3: Word Tokenization."""
    if show_header:
        st.markdown(f"### Stage 3: Word Tokenization {_badge(st.session_state['stage_word_tok'])}")
    _render_stage_log("stage_word_tok")
    tokens = st.session_state.get("word_tokens")
    if tokens is None:
        return
    st.write(f"**Token count:** {len(tokens):,}")
    with st.expander("Word token list (first 200 tokens)", expanded=False):
        st.write(tokens[:200])
        if len(tokens) > 200:
            st.caption(f"Showing first 200 of {len(tokens):,} total tokens.")


def _display_stage_4(show_header: bool = True) -> None:
    """Display Stage 4: Text Cleaning."""
    if show_header:
        st.markdown(f"### Stage 4: Text Cleaning {_badge(st.session_state['stage_cleaning'])}")
    _render_stage_log("stage_cleaning")
    original = st.session_state.get("original_text")
    cleaned = st.session_state.get("cleaned_text")
    if cleaned is None:
        return
    orig_len = len(original) if original else 0
    clean_len = len(cleaned)
    st.write(
        f"**Normalization results:** Unicode NFC normalized, control characters stripped, "
        f"lowercased, and whitespace collapsed ({orig_len:,} → {clean_len:,} characters)."
    )
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Original text (first 2,000 chars)", expanded=False):
            st.text(original[:2000] if original else "")
    with col2:
        with st.expander("Cleaned text (first 2,000 chars)", expanded=False):
            st.text(cleaned[:2000])


def _display_stage_5(show_header: bool = True) -> None:
    """Display Stage 5: Stopword Removal."""
    if show_header:
        st.markdown(f"### Stage 5: Stopword Removal {_badge(st.session_state['stage_stopword'])}")
    _render_stage_log("stage_stopword")
    raw_tokens = st.session_state.get("word_tokens") or []
    filtered = st.session_state.get("filtered_tokens")
    removed = st.session_state.get("removed_stopwords")
    if filtered is None:
        return
    removed_list = removed or []
    retained_negations = sorted({t.lower() for t in filtered if t.lower() in PROTECTED_NEGATIONS})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tokens before removal", f"{len(raw_tokens):,}")
    c2.metric("Removed stopword occurrences", f"{len(removed_list):,}")
    c3.metric("Unique stopwords removed", f"{len(set(removed_list)):,}")
    c4.metric("Remaining content tokens", f"{len(filtered):,}")

    st.caption(
        f"Protected negations configured: {', '.join(sorted(PROTECTED_NEGATIONS))}. "
        f"Present and retained in document: {', '.join(retained_negations) if retained_negations else 'none present'}."
    )

    with st.expander("Tokens before removal (first 200)", expanded=False):
        st.write(raw_tokens[:200])
    with st.expander("Identified & removed stopwords", expanded=False):
        if removed_list:
            counts: Dict[str, int] = {}
            for w in removed_list:
                counts[w] = counts.get(w, 0) + 1
            sw_df = pd.DataFrame(
                [{"Stopword": k, "Occurrences Removed": v} for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
            )
            st.dataframe(sw_df, hide_index=True, alt="Removed stopwords table")
        else:
            st.write("No stopwords removed.")
    with st.expander("Remaining tokens (first 200)", expanded=False):
        st.write(filtered[:200])


def _display_stage_6(show_header: bool = True) -> None:
    """Display Stage 6: Lemmatization."""
    if show_header:
        st.markdown(f"### Stage 6: Lemmatization {_badge(st.session_state['stage_lemma'])}")
    _render_stage_log("stage_lemma")
    pairs = st.session_state.get("lemma_pairs")
    if pairs is None:
        return
    changed = [(orig, lem) for orig, lem in pairs if orig.lower() != lem.lower()]
    unique_lemmas = len({lem for _, lem in pairs})

    c1, c2, c3 = st.columns(3)
    c1.metric("Input filtered tokens", f"{len(pairs):,}")
    c2.metric("Morphologically changed tokens", f"{len(changed):,}")
    c3.metric("Unique lemmatized forms", f"{unique_lemmas:,}")

    with st.expander(f"Original Token | Lemmatized Token table ({len(pairs):,} total tokens)", expanded=False):
        all_df = pd.DataFrame(
            [
                {
                    "Original Token": orig,
                    "Lemmatized Token": lem,
                    "Changed": "Yes" if orig.lower() != lem.lower() else "No",
                }
                for orig, lem in pairs[:300]
            ]
        )
        st.dataframe(all_df, hide_index=True, alt="Original to lemmatized token table")
        if len(pairs) > 300:
            st.caption(f"Showing first 300 of {len(pairs):,} token pairs.")

    with st.expander(f"Changed tokens only ({len(changed):,} changed pairs)", expanded=False):
        if changed:
            df = pd.DataFrame(changed[:200], columns=["Original Token", "Lemmatized Token"])
            st.dataframe(df, hide_index=True, alt="Changed lemmatized tokens table")
        else:
            st.write("No tokens changed during lemmatization.")


def _display_stage_7(show_header: bool = True) -> None:
    """Display Stage 7: TF-IDF Vectorization."""
    if show_header:
        st.markdown(f"### Stage 7: TF-IDF Vectorization {_badge(st.session_state['stage_tfidf'])}")
    _render_stage_log("stage_tfidf")
    result = st.session_state.get("tfidf_result")
    if result is None:
        return
    c1, c2 = st.columns(2)
    c1.metric("Vocabulary size", f"{len(result.feature_names):,}")
    c2.metric("Matrix dimensions", f"{result.matrix_shape[0]} × {result.matrix_shape[1]}")

    with st.expander("Vocabulary terms (first 150)", expanded=False):
        st.write(result.feature_names[:150])
    with st.expander("Top weighted TF-IDF terms", expanded=False):
        if result.top_terms:
            df = pd.DataFrame(
                [{"Term": term, "TF-IDF Weight": round(weight, 6)} for term, weight in result.top_terms]
            )
            st.dataframe(df, hide_index=True, alt="Top weighted TF-IDF terms")
        else:
            st.write("No top terms available.")
    with st.expander("Numerical feature matrix (first 25 terms)", expanded=False):
        if result.matrix is not None:
            arr = result.matrix.toarray() if hasattr(result.matrix, "toarray") else np.asarray(result.matrix)
            display_arr = arr[:20, :25]
            st.dataframe(
                pd.DataFrame(
                    np.round(display_arr, 4),
                    columns=result.feature_names[: display_arr.shape[1]],
                ),
                alt="TF-IDF numerical feature matrix",
            )


def _display_stage_8(show_header: bool = True) -> None:
    """Display Stage 8: TextRank Summarization."""
    if show_header:
        st.markdown(f"### Stage 8: TextRank Summarization {_badge(st.session_state['stage_textrank'])}")
    _render_stage_log("stage_textrank")
    result = st.session_state.get("textrank_result")
    if result is None:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidate sentences", len(result.sentences) if result.sentences else len(result.sentence_scores))
    c2.metric("Original word count", f"{result.original_word_count:,}")
    c3.metric("Summary word count", f"{result.summary_word_count:,}")
    c4.metric("Compression ratio", f"{result.compression_ratio:.2%}")

    with st.expander("1. Input candidate sentences & TF-IDF sentence vectorization", expanded=False):
        vec_rows, vec_cols = result.sentence_vectors_shape
        st.write(
            f"**Sentence TF-IDF matrix shape:** `{vec_rows} sentences × {vec_cols} terms` "
            f"(sublinear TF scaling)."
        )
        if result.vocabulary:
            st.caption(f"Sample sentence TF-IDF terms: {', '.join(result.vocabulary[:40])}")
        input_sents = result.sentences or list(result.sentence_scores.keys())
        sent_df = pd.DataFrame(
            [{"Node ID": f"S{i+1}", "Sentence": s} for i, s in enumerate(input_sents)]
        )
        st.dataframe(sent_df, hide_index=True, alt="TextRank input sentences")

    sim_matrix = result.similarity_matrix
    if sim_matrix is None:
        sim_matrix = st.session_state.get("similarity_matrix")
    with st.expander("2. Pairwise sentence cosine similarity matrix", expanded=False):
        if sim_matrix is not None:
            labels = [f"S{i+1}" for i in range(len(sim_matrix))]
            df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
            st.dataframe(df.round(4), alt="Sentence cosine similarity matrix")
        else:
            st.write("Not available (single sentence or fewer).")

    with st.expander("3. Graph construction explanation & topology", expanded=False):
        st.write(
            "An undirected weighted graph $G = (V, E)$ is constructed where each vertex $v_i \\in V$ "
            "represents a unique candidate sentence. An edge $e_{ij} \\in E$ connects sentences $S_i$ and $S_j$ "
            "whenever their TF-IDF cosine similarity is strictly positive ($w_{ij} > 0$), with self-loops "
            "zeroed out ($w_{ii} = 0$). PageRank ($\\alpha = 0.85$) iteratively computes the stationary "
            "centrality score of each sentence node."
        )
        st.write(f"- **Graph vertices (sentences):** {result.graph_nodes}")
        st.write(f"- **Graph edges (non-zero similarity pairs):** {result.graph_edges}")

    with st.expander("4. PageRank importance scores & ranked sentences", expanded=False):
        if result.sentence_scores:
            input_sents = result.sentences or list(result.sentence_scores.keys())
            idx_map = {s: idx + 1 for idx, s in enumerate(input_sents)}
            ranked = sorted(result.sentence_scores.items(), key=lambda x: x[1], reverse=True)
            scores_data = [
                {
                    "Rank": rank,
                    "Node": f"S{idx_map.get(s, rank)}",
                    "PageRank Score": round(score, 6),
                    "Selected": "Yes ✅" if s in result.selected_sentences else "No",
                    "Sentence": s,
                }
                for rank, (s, score) in enumerate(ranked, 1)
            ]
            st.dataframe(pd.DataFrame(scores_data), hide_index=True, alt="PageRank sentence scores")

    with st.expander("5. Selected sentences (restored to original document order)", expanded=False):
        for i, s in enumerate(result.selected_sentences, 1):
            st.write(f"{i}. {s}")

    st.write("**Final Extractive Summary (TextRank):**")
    st.info(result.summary_text if result.summary_text else "_No summary generated._")


def _display_stage_9(show_header: bool = True) -> None:
    """Display Stage 9: Information Extraction."""
    if show_header:
        st.markdown(f"### Stage 9: Information Extraction {_badge(st.session_state['stage_info_extraction'])}")
    _render_stage_log("stage_info_extraction")
    result = st.session_state.get("concepts_result")
    if result is None:
        return
    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    c1, c2, c3 = st.columns(3)
    c1.metric("Named entities (spaCy NER)", len(result.entities))
    c2.metric("Noun-chunk keyphrases", len(result.keyphrases))
    c3.metric("Extracted definitions", len(result.definitions))

    with st.expander(f"Named entities ({len(result.entities)})", expanded=False):
        if result.entities:
            df = pd.DataFrame(
                [
                    {
                        "Entity Text": e.text,
                        "Entity Label": e.label,
                        "Start Char": e.start,
                        "End Char": e.end,
                    }
                    for e in result.entities
                ]
            )
            st.dataframe(df, hide_index=True, alt="Named entities table")
        else:
            st.write("No named entities found.")

    with st.expander(f"Keyphrases & important concepts ({len(result.keyphrases)})", expanded=False):
        if result.keyphrases:
            df = pd.DataFrame(
                [{"Keyphrase / Concept": k.text, "Frequency-Length Score": round(k.score, 3)} for k in result.keyphrases]
            )
            st.dataframe(df, hide_index=True, alt="Keyphrases table")
        else:
            st.write("No keyphrases found.")

    with st.expander(f"Rule-based extracted definitions ({len(result.definitions)})", expanded=False):
        if result.definitions:
            df = pd.DataFrame(
                [
                    {
                        "Concept": d.concept,
                        "Definition": d.definition,
                        "Syntactic Pattern": d.pattern,
                        "Start Char": d.start,
                    }
                    for d in result.definitions
                ]
            )
            st.dataframe(df, hide_index=True, alt="Extracted definitions table")
        else:
            st.write("No definitions found.")


def _display_stage_10(show_header: bool = True) -> None:
    """Display Stage 10: LDA Topic Modelling."""
    if show_header:
        st.markdown(f"### Stage 10: LDA Topic Modelling {_badge(st.session_state['stage_lda'])}")
    _render_stage_log("stage_lda")
    result = st.session_state.get("lda_result")
    if result is None:
        return
    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    n_topics_eff = result.model_config.get("n_topics_effective", len(result.topics))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Number of topics", n_topics_eff)
    c2.metric("Input chunks", len(result.chunks) if result.chunks else len(result.document_topics))
    c3.metric("CountVectorizer vocab", result.vocabulary_size)
    c4.metric("Model perplexity", f"{result.perplexity:.2f}" if result.perplexity is not None else "N/A")

    with st.expander("1. Input document chunks", expanded=False):
        if result.chunks:
            chunk_df = pd.DataFrame(
                [
                    {"Chunk ID": idx, "Word Count": len(c.split()), "Chunk Text": c}
                    for idx, c in enumerate(result.chunks)
                ]
            )
            st.dataframe(chunk_df, hide_index=True, alt="LDA input document chunks")
        elif result.document_topics:
            chunk_df = pd.DataFrame(
                [{"Chunk ID": d.chunk_id, "Snippet": d.text_snippet} for d in result.document_topics]
            )
            st.dataframe(chunk_df, hide_index=True, alt="LDA document snippets")

    with st.expander("2. CountVectorizer (Bag-of-Words) representation", expanded=False):
        st.write(
            f"**Matrix dimensions:** `{result.bow_matrix_shape[0]} chunks × {result.bow_matrix_shape[1]} terms` "
            f"(English stopwords removed)."
        )
        if result.top_bow_terms:
            bow_df = pd.DataFrame(result.top_bow_terms, columns=["Term", "Corpus Term Count"])
            st.dataframe(bow_df, hide_index=True, alt="Top CountVectorizer terms")
        if result.bow_matrix is not None and result.feature_names:
            arr = (
                result.bow_matrix.toarray()
                if hasattr(result.bow_matrix, "toarray")
                else np.asarray(result.bow_matrix)
            )
            sub = arr[:15, :20]
            st.dataframe(
                pd.DataFrame(sub, columns=result.feature_names[: sub.shape[1]]),
                alt="CountVectorizer matrix preview",
            )

    with st.expander("3. Topic keywords & normalized word weights", expanded=False):
        if result.topics:
            for topic in result.topics:
                st.write(f"**Topic {topic.topic_id}:** {', '.join(topic.keywords)}")
                df = pd.DataFrame(
                    [{"Keyword": k, "Normalized Weight": round(w, 4)} for k, w in zip(topic.keywords, topic.weights)]
                )
                st.dataframe(df, hide_index=True, alt=f"Topic {topic.topic_id} keyword weights")
        else:
            st.write("No topics generated.")

    with st.expander("4. Document-topic probability distributions & dominant topic assignments", expanded=False):
        if result.document_topics:
            rows = []
            for d in result.document_topics:
                row: Dict[str, Any] = {
                    "Chunk ID": d.chunk_id,
                    "Dominant Topic": f"Topic {d.dominant_topic}",
                    "Max Probability": round(max(d.topic_distribution), 4) if d.topic_distribution else 0.0,
                }
                for i, v in enumerate(d.topic_distribution):
                    row[f"Topic {i} Prob"] = round(v, 4)
                row["Text Snippet"] = d.text_snippet
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Document topic distributions")
        else:
            st.write("No document-topic data.")

    # Optional CO3 Cloud-AI Topic Interpretation (Google Gemini)
    cfg = load_config()
    if cfg.is_configured and result.topics:
        if st.button("Interpret LDA topics with Gemini (Cloud AI — CO3)", key="btn_interpret_topics"):
            with st.status("Interpreting LDA topics with Google Gemini (1 batched request)...", expanded=True) as status_box:
                try:
                    st.write("Sending local LDA topic keywords and weights to Gemini...")
                    interps = interpret_topics(result.topics, config=cfg)
                    st.session_state["lda_interpretations"] = interps
                    status_box.update(label="Cloud-AI topic interpretation complete", state="complete", expanded=False)
                except Exception as exc:
                    status_box.update(label="Cloud-AI topic interpretation failed", state="error")
                    st.error(f"Topic interpretation failed: {exc}")

    interps = st.session_state.get("lda_interpretations")
    if interps:
        with st.expander("5. Cloud-AI Topic Interpretations (Gemini — LLM-assisted)", expanded=True):
            for item in interps:
                st.markdown(f"**Topic {item.topic_id}: {item.title}**")
                st.write(item.explanation)
                st.caption(f"Source LDA keywords: {', '.join(item.keywords)}")


def _display_stage_11(show_header: bool = True) -> None:
    """Display Stage 11: Gemini Summarization."""
    if show_header:
        st.markdown(f"### Stage 11: Gemini Summarization {_badge(st.session_state['stage_gemini'])}")
    _render_stage_log("stage_gemini")
    cfg = load_config()
    c_mod, c_stat, c_tok = st.columns(3)
    c_mod.metric("Configured model", cfg.model_id or "Not set")
    c_stat.metric("Generation status", st.session_state.get("stage_gemini", "Not Started"))
    tok_count = st.session_state.get("gemini_token_count")
    c_tok.metric("Document token count (SDK)", f"{tok_count:,}" if tok_count is not None else "N/A")

    if not cfg.is_configured:
        st.info(
            "Gemini is not configured. Set your API key in .env. "
            "All traditional NLP stages (Stages 1–10) and VADER sentiment analysis work independently."
        )
        return

    status_info = st.session_state.get("gemini_status_info")
    if status_info:
        c1, c2, c3, c4 = st.columns(4)
        p_idx = status_info.get("phase_index", 0)
        p_tot = status_info.get("total_phases", 5)
        c1.metric("Summarization phase", f"{p_idx} / {p_tot}" if p_idx else "Completed")
        c2.metric(
            "Chunk progress",
            f"{status_info.get('current_chunk', 0)} / {status_info.get('total_chunks', 0)}",
        )
        c3.metric("Completed chunks", status_info.get("completed_chunks", 0))
        c4.metric("Current step status", status_info.get("retry_status", "N/A"))

    error_msg = st.session_state.get("gemini_error")
    if error_msg:
        st.error(f"Gemini error: {error_msg}")
        if st.button("🔁 Retry Gemini Summarization (resume uncompleted phases/chunks)", key="retry_gemini_btn"):
            _execute_stage_with_live_feedback(11)
            st.rerun()

    results = st.session_state.get("gemini_results")
    if not results:
        return

    for key, label in _GEMINI_SUMMARY_PHASES:
        with st.expander(label, expanded=(key == "short")):
            st.markdown(results.get(key, "_Not generated yet._"))


# ---------------------------------------------------------------------------
# Live interactive stage execution wrapper
# ---------------------------------------------------------------------------


def _execute_stage_with_live_feedback(
    stage_num: int,
    file_content: Optional[bytes] = None,
    filename: Optional[str] = None,
    num_sentences: int = 3,
    n_topics: int = 3,
    slot: Optional[Any] = None,
    title_placeholder: Optional[Any] = None,
) -> bool:
    """Execute a single stage inside its UI container with live phase & progress feedback."""
    target = slot if slot is not None else st.container()
    title = _STAGE_TITLES.get(stage_num, f"Stage {stage_num}")
    stage_key = _STAGE_KEYS[stage_num - 1]

    if title_placeholder is not None:
        title_placeholder.markdown(f"### {title} {_badge('Running')}")

    with target:
        status_box = st.status(f"Running {title}...", state="running", expanded=True)
        with status_box:
            progress_bar = st.progress(0, text=f"Initializing {title}...")
            step_log_area = st.container()

            def _update_step(step: int, total: int, msg: str) -> None:
                pct = min(100, max(0, int(round((step / max(1, total)) * 100))))
                progress_bar.progress(pct, text=msg)
                step_log_area.write(f"- {msg}")

            ok = False
            if stage_num == 1:
                if file_content is not None and filename is not None:
                    ok = _run_extraction(file_content, filename, step_hook=_update_step)
            elif stage_num == 2:
                ok = _run_sentence_tokenization(step_hook=_update_step)
            elif stage_num == 3:
                ok = _run_word_tokenization(step_hook=_update_step)
            elif stage_num == 4:
                ok = _run_text_cleaning(step_hook=_update_step)
            elif stage_num == 5:
                ok = _run_stopword_removal(step_hook=_update_step)
            elif stage_num == 6:
                ok = _run_lemmatization(step_hook=_update_step)
            elif stage_num == 7:
                ok = _run_tfidf(step_hook=_update_step)
            elif stage_num == 8:
                ok = _run_textrank_stage(num_sentences=num_sentences, step_hook=_update_step)
            elif stage_num == 9:
                ok = _run_info_extraction(step_hook=_update_step)
            elif stage_num == 10:
                ok = _run_lda_stage(n_topics=n_topics, step_hook=_update_step)
            elif stage_num == 11:
                def _gemini_hook(phase_idx: int, total_phases: int, msg: str, info: Dict[str, Any]) -> None:
                    cur_c = info.get("current_chunk", 1)
                    tot_c = max(1, info.get("total_chunks", 1))
                    frac = ((phase_idx - 1) + (cur_c / tot_c)) / max(1, total_phases)
                    pct = min(100, max(5, int(round(frac * 100))))
                    progress_bar.progress(pct, text=msg)
                    step_log_area.write(f"- {msg}")

                ok = _run_gemini_stage(step_hook=_gemini_hook)

            if ok:
                progress_bar.progress(100, text=f"{title} completed.")
                status_box.update(label=f"{title} — Completed", state="complete", expanded=False)
            else:
                st_state = st.session_state.get(stage_key, "Failed")
                final_state = "error" if st_state == "Failed" else "complete"
                status_box.update(label=f"{title} — {st_state}", state=final_state, expanded=(final_state == "error"))

    if title_placeholder is not None:
        title_placeholder.markdown(f"### {title} {_badge(st.session_state.get(stage_key, 'Not Started'))}")

    return ok


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Educational Content Summarizer", layout="wide")
    _ensure_nltk_resources()
    _init_state()
    _reconcile_interrupted_stages()

    # ---- SIDEBAR ----
    st.sidebar.title("Educational Content Summarizer")
    uploaded = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    num_sentences = st.sidebar.slider("TextRank sentences", 1, 10, 3)
    n_topics = st.sidebar.slider("LDA topics", 2, 8, 3)

    run_pipeline = st.sidebar.button("▶ Run Complete Pipeline", type="primary", width="stretch")
    reset = st.sidebar.button("🔄 Reset", width="stretch")

    if reset:
        _reset_state()
        st.rerun()

    # Detect source document change & automatically invalidate stale results
    if uploaded is not None:
        doc_changed = _ensure_fresh_document(uploaded.name, uploaded.getvalue())
        if doc_changed:
            st.info(f"New source document detected ({uploaded.name}). Previous stage outputs were cleared.")

    # Document info in sidebar
    data = st.session_state.get("extracted_data")
    if data is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Document Information**")
        st.sidebar.write(f"File: {data.filename}")
        st.sidebar.write(f"Type: {data.file_type}")
        st.sidebar.write(f"Pages: {data.page_count}")
        word_count = len(data.extracted_text.split()) if data.extracted_text else 0
        st.sidebar.write(f"Words: {word_count:,}")

    # Gemini status in sidebar
    cfg = load_config()
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Gemini Status**")
    st.sidebar.write(f"Configured: {'Yes' if cfg.is_configured else 'No'}")
    st.sidebar.write(f"Model: {cfg.model_id or 'not set'}")

    # ---- MAIN PAGE ----
    st.title("Educational Content Summarization System")
    st.caption(
        "Academic NLP demonstration pipeline exposing every intermediate algorithm stage, "
        "step-by-step phase progress, and verifiable mathematical outputs."
    )

    display_fns = {
        1: _display_stage_1,
        2: _display_stage_2,
        3: _display_stage_3,
        4: _display_stage_4,
        5: _display_stage_5,
        6: _display_stage_6,
        7: _display_stage_7,
        8: _display_stage_8,
        9: _display_stage_9,
        10: _display_stage_10,
        11: _display_stage_11,
    }

    # Build ordered visual containers for all 11 stages, each with its own top-right Run button
    selected_individual_stage: Optional[int] = None
    title_placeholders: Dict[int, Any] = {}
    live_slots: Dict[int, Any] = {}
    body_slots: Dict[int, Any] = {}

    for stage_idx in range(1, 12):
        stage_key = _STAGE_KEYS[stage_idx - 1]
        stage_title = _STAGE_TITLES[stage_idx]
        with st.container(border=True):
            col_title, col_btn = st.columns([6, 1], vertical_alignment="center")
            title_placeholders[stage_idx] = col_title.empty()
            title_placeholders[stage_idx].markdown(
                f"### {stage_title} {_badge(st.session_state[stage_key])}"
            )
            if col_btn.button(
                "▶ Run",
                key=f"btn_run_stage_{stage_idx}",
                width="stretch",
                disabled=(uploaded is None),
            ):
                selected_individual_stage = stage_idx
            live_slots[stage_idx] = st.empty()
            body_slots[stage_idx] = st.container()

    def _run_and_render_stage(
        s_idx: int,
        file_content: Optional[bytes] = None,
        filename: Optional[str] = None,
    ) -> bool:
        ok = _execute_stage_with_live_feedback(
            s_idx,
            file_content=file_content,
            filename=filename,
            num_sentences=num_sentences,
            n_topics=n_topics,
            slot=live_slots[s_idx],
            title_placeholder=title_placeholders[s_idx],
        )
        live_slots[s_idx].empty()
        with body_slots[s_idx]:
            display_fns[s_idx](show_header=False)
        return ok

    # Handle Complete Pipeline Run progressively across the 11 stage containers
    if run_pipeline:
        if uploaded is None:
            st.warning("Please upload a PDF or TXT document first.")
        else:
            file_content = uploaded.getvalue()
            filename = uploaded.name
            _reset_document_pipeline_state(_compute_doc_fingerprint(filename, file_content))
            for idx in range(1, 12):
                s_key = _STAGE_KEYS[idx - 1]
                title_placeholders[idx].markdown(
                    f"### {_STAGE_TITLES[idx]} {_badge(st.session_state[s_key])}"
                )

            extracted_ok = _run_and_render_stage(1, file_content=file_content, filename=filename)
            if extracted_ok:
                for s_idx in range(2, 12):
                    _run_and_render_stage(s_idx)

            st.session_state["pipeline_ran"] = True
            st.success("Pipeline execution finished. All completed stage outputs are displayed above.")

    # Handle Individual Stage Run with prerequisite auto-check
    elif selected_individual_stage is not None and uploaded is not None:
        file_content = uploaded.getvalue()
        filename = uploaded.name

        prereq_ran: set[int] = set()
        if selected_individual_stage > 1 and not st.session_state.get("raw_text"):
            _run_and_render_stage(1, file_content=file_content, filename=filename)
            prereq_ran.add(1)
        if selected_individual_stage in (5, 6, 7) and not st.session_state.get("word_tokens"):
            _run_and_render_stage(3)
            prereq_ran.add(3)
        if selected_individual_stage in (6, 7) and not st.session_state.get("filtered_tokens"):
            _run_and_render_stage(5)
            prereq_ran.add(5)
        if selected_individual_stage == 7 and not st.session_state.get("lemmatized_tokens"):
            _run_and_render_stage(6)
            prereq_ran.add(6)
        if selected_individual_stage == 11:
            if st.session_state.get("stage_gemini") == "Completed":
                st.session_state["gemini_results"] = {}

        for s_idx in range(1, 12):
            if s_idx == selected_individual_stage:
                _run_and_render_stage(s_idx, file_content=file_content, filename=filename)
            elif s_idx not in prereq_ran:
                with body_slots[s_idx]:
                    display_fns[s_idx](show_header=False)
    else:
        # Normal display pass
        for s_idx in range(1, 12):
            with body_slots[s_idx]:
                display_fns[s_idx](show_header=False)


    # ---- Sentiment Analysis (separate section) ----
    st.markdown("---")
    st.header("Student Feedback Sentiment Analysis (VADER)")
    st.caption(
        "This section is independent of the document processing pipeline. "
        "VADER sentiment analysis is evaluated strictly on student course feedback."
    )

    with st.container(border=True):
        single = st.text_area("Single feedback entry", height=80, key="feedback_single_input")
        if st.button("Analyze Single Feedback", key="btn_single_feedback"):
            if single.strip():
                with st.status("Running VADER sentiment lexicon analysis...", expanded=False) as s_box:
                    st.session_state["sentiment_single"] = analyze_feedback(single)
                    s_box.update(label="VADER analysis complete", state="complete")
            else:
                st.warning("Enter feedback text.")

        result = st.session_state.get("sentiment_single")
        if result is not None:
            st.write(f"**Feedback text:** {result.feedback_text}")
            col0, col1, col2, col3, col4 = st.columns(5)
            col0.metric("Sentiment label", result.sentiment_label)
            col1.metric("Compound score", f"{result.compound_score:.4f}")
            col2.metric("Positive score", f"{result.positive_score:.4f}")
            col3.metric("Negative score", f"{result.negative_score:.4f}")
            col4.metric("Neutral score", f"{result.neutral_score:.4f}")

    with st.container(border=True):
        st.markdown("#### Batch Feedback Distribution")
        batch_text = st.text_area(
            "Multiple feedback entries (one per line)", height=100, key="feedback_batch_input"
        )
        if st.button("Analyze Batch Feedback", key="btn_batch_feedback"):
            lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
            if lines:
                with st.status(f"Analyzing {len(lines)} feedback entries with VADER...", expanded=False) as b_box:
                    st.session_state["sentiment_batch"] = analyze_batch(lines)
                    b_box.update(label=f"Analyzed {len(lines)} entries", state="complete")
            else:
                st.warning("Enter at least one feedback entry.")

        batch = st.session_state.get("sentiment_batch")
        if batch is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total entries", batch.total)
            col2.metric("Positive", f"{batch.positive} ({batch.positive_pct:.1f}%)")
            col3.metric("Negative", f"{batch.negative} ({batch.negative_pct:.1f}%)")
            col4.metric("Neutral", f"{batch.neutral} ({batch.neutral_pct:.1f}%)")
            with st.expander("Individual batch feedback scores", expanded=True):
                df = pd.DataFrame(
                    [
                        {
                            "Feedback": r.feedback_text,
                            "Sentiment Label": r.sentiment_label,
                            "Compound": round(r.compound_score, 4),
                            "Positive": round(r.positive_score, 4),
                            "Negative": round(r.negative_score, 4),
                            "Neutral": round(r.neutral_score, 4),
                        }
                        for r in batch.results
                    ]
                )
                st.dataframe(df, hide_index=True, alt="Batch feedback sentiment results")

    # ---- Export ----
    st.markdown("---")
    st.header("Export Study Notes")
    tr = st.session_state.get("textrank_result")
    gemini = st.session_state.get("gemini_results")
    concepts_r = st.session_state.get("concepts_result")

    summary_text = ""
    method = "TextRank"
    if tr is not None:
        summary_text = tr.summary_text
    if gemini is not None and gemini.get("short"):
        summary_text = gemini["short"]
        method = "Gemini"

    concepts = []
    definitions = []
    if concepts_r is not None:
        concepts = [k.text for k in concepts_r.keyphrases] or [e.text for e in concepts_r.entities]
        definitions = concepts_r.definitions

    source_name = "uploaded_document"
    ext_data = st.session_state.get("extracted_data")
    if ext_data is not None:
        source_name = ext_data.filename

    if summary_text or concepts:
        markdown = build_export_markdown(
            source_name=source_name,
            summary_type="short",
            summary_text=summary_text,
            method=method,
            concepts=concepts,
            definitions=definitions,
        )
        with st.expander("Export preview", expanded=False):
            st.text_area("Markdown preview", markdown, height=200, key="export_preview")
        st.download_button(
            label="Download Study Notes (Markdown)",
            data=markdown,
            file_name="study_notes.md",
            mime="text/markdown",
        )
    else:
        st.info("Run the pipeline first to generate exportable study notes.")


if __name__ == "__main__":
    main()
