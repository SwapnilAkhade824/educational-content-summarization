"""Testable helper functions for the Streamlit application."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from extraction.text_extractor import ExtractionError, extract_document
    from information_extraction.concept_extractor import extract_concepts
    from preprocessing.text_preprocessing import preprocess_text
    from preprocessing.vectorizer import fit_tfidf
    from sentiment.sentiment_analyzer import analyze_batch, analyze_feedback
    from summarization.llm_summarizer import LLMError, generate_summary
    from summarization.textrank import summarize_text
    from topic_modeling.lda_model import run_lda
    from utils.config import load_config
except ImportError:  # pragma: no cover - src-layout fallback
    from ..extraction.text_extractor import ExtractionError, extract_document
    from ..information_extraction.concept_extractor import extract_concepts
    from ..preprocessing.text_preprocessing import preprocess_text
    from ..preprocessing.vectorizer import fit_tfidf
    from ..sentiment.sentiment_analyzer import analyze_batch, analyze_feedback
    from ..summarization.llm_summarizer import LLMError, generate_summary
    from ..summarization.textrank import summarize_text
    from ..topic_modeling.lda_model import run_lda
    from ..utils.config import load_config


def process_document(filename: str, content: bytes) -> Dict[str, Any]:
    """Extract text from an uploaded document."""
    try:
        result = extract_document(filename, content)
    except ExtractionError as exc:
        return {
            "ok": False,
            "filename": filename,
            "size": len(content),
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "ok": False,
            "filename": filename,
            "size": len(content),
            "error": f"Unexpected error: {exc}",
        }

    return {
        "ok": True,
        "filename": result.filename,
        "file_type": result.file_type,
        "size": len(content),
        "page_count": result.page_count,
        "warnings": list(result.extraction_warnings),
        "text": result.extracted_text,
        "pages": [(p.page_number, p.page_text) for p in result.pages],
    }


def prepare_preprocessing(text: str) -> Optional[Any]:
    """Run preprocessing on text, returning None for empty input."""
    if not text or not text.strip():
        return None
    return preprocess_text(text)


def run_textrank(text: str, num_sentences: int) -> Any:
    """Run TextRank summarization."""
    return summarize_text(text, num_sentences=num_sentences)


def run_llm_summary(text: str, summary_type: str, config: Any = None) -> Any:
    """Run LLM summarization, returning (result, error_message)."""
    cfg = config if config is not None else load_config()
    if not getattr(cfg, "is_configured", False) and config is None:
        return None, "Gemini is not configured. Set GEMINI_API_KEY in .env."
    if not getattr(cfg, "api_key", None) or not getattr(cfg, "model_id", None):
        return None, "Gemini is not configured. Set GEMINI_API_KEY in .env."
    try:
        result = generate_summary(text, summary_type=summary_type, config=cfg)
        return result, None
    except LLMError as exc:
        return None, str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Unexpected error: {exc}"


def run_topic_interpretation(topics: Any, config: Any = None) -> Any:
    """Run optional LLM topic interpretation, returning (result, error_message)."""
    cfg = config if config is not None else load_config()
    if not getattr(cfg, "api_key", None) or not getattr(cfg, "model_id", None):
        return None, "Gemini is not configured. Set GEMINI_API_KEY in .env."
    try:
        from summarization.llm_summarizer import interpret_topics

        return interpret_topics(topics, config=cfg), None
    except LLMError as exc:
        return None, str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Unexpected error: {exc}"


def run_concepts(text: str) -> Any:
    """Run information extraction."""
    return extract_concepts(text)


def run_topics(texts: List[str], n_topics: int) -> Any:
    """Run LDA topic modelling."""
    return run_lda(texts, n_topics=n_topics, random_state=42)


def run_sentiment_one(text: str) -> Any:
    """Analyze a single feedback entry."""
    return analyze_feedback(text)


def run_sentiment_many(texts: List[str]) -> Any:
    """Analyze multiple feedback entries."""
    return analyze_batch(texts)


def top_tfidf_terms(text: str, top_n: int = 10) -> List[Any]:
    """Compute top TF-IDF terms for a single document's cleaned text."""
    if not text or not text.strip():
        return []
    try:
        _, result = fit_tfidf([text])
        return result.top_terms[:top_n]
    except Exception:
        return []


def build_export_markdown(
    source_name: str,
    summary_type: str,
    summary_text: str,
    method: str,
    concepts: Optional[List[str]] = None,
    definitions: Optional[List[Any]] = None,
) -> str:
    """Build a Markdown export of generated study material."""
    concepts = concepts or []
    definitions = definitions or []
    lines: List[str] = []
    lines.append("# Educational Study Notes")
    lines.append("")
    lines.append(f"**Source document:** {source_name}")
    lines.append(f"**Summary type:** {summary_type}")
    lines.append(f"**Generation method:** {method}")
    lines.append("")
    lines.append("> Generated by an automated NLP pipeline. Review before use; not verified academic facts.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(summary_text or "_No summary generated._")
    lines.append("")
    lines.append("## Key Concepts")
    lines.append("")
    if concepts:
        for c in concepts:
            lines.append(f"- {c}")
    else:
        lines.append("_No concepts extracted._")
    lines.append("")
    lines.append("## Important Definitions")
    lines.append("")
    if definitions:
        for d in definitions:
            concept = getattr(d, "concept", None) or (d.get("concept") if isinstance(d, dict) else str(d))
            definition = getattr(d, "definition", None) or (d.get("definition") if isinstance(d, dict) else "")
            lines.append(f"- **{concept}**: {definition}")
    else:
        lines.append("_No definitions extracted._")
    lines.append("")
    return "\n".join(lines)
