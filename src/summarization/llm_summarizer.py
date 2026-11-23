"""Google Gemini generative summarization (official google-genai SDK).

The module keeps the existing chunking architecture and prompt templates.
Gemini is optional: configuration comes only from environment variables, and
when GEMINI_API_KEY is not set the rest of the application continues to work
while summarization raises a clear configuration error instead of fabricating
output.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

try:
    from ..utils.config import LLMConfig, load_config
except ImportError:  # pragma: no cover - supports src-layout imports
    from utils.config import LLMConfig, load_config

try:
    from ..summarization.llm_client import (
        LLMClientError,
        chat_completion,
        chat_completion_with_metadata,
        check_ollama_health,
        count_tokens,
        create_client,
    )
except ImportError:  # pragma: no cover - supports src-layout imports
    from summarization.llm_client import (
        LLMClientError,
        chat_completion,
        chat_completion_with_metadata,
        check_ollama_health,
        count_tokens,
        create_client,
    )

from .llm_client import _complete_once as _llm_complete_once

LLMError = LLMClientError
LLMConfigError = LLMClientError

from .prompts import SYSTEM_PROMPT, build_chunk_prompt, build_prompt


class LLMConfigError(LLMError):
    """Raised when the LLM is not configured."""


# Backward-compatible aliases.
LLMError = LLMError
LLMConfigError = LLMConfigError
LocalLLMError = LLMError
LocalLLMConfigError = LLMConfigError
OpenAICompatibleError = LLMError
GroqLLMError = LLMError
GroqConfigError = LLMConfigError
GroqLLM = LLMError


@dataclass
class LLMSummaryResult:
    """Result of a LLM-based educational summary."""

    summary_type: str
    summary_text: str
    generated_text: str = ""
    chunk_count: int = 1
    chunks_completed: int = 1
    chunks_failed: int = 0
    warnings: List[str] = field(default_factory=list)
    model_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.generated_text:
            self.generated_text = self.summary_text


# In-memory resumable chunk cache: cache_key -> {chunk_index: chunk_summary}
_CHUNK_CACHE: Dict[str, Dict[int, str]] = {}


def get_chunk_cache() -> Dict[str, Dict[int, str]]:
    """Return the global in-memory chunk cache."""
    return _CHUNK_CACHE


def clear_chunk_cache() -> None:
    """Clear all cached chunk summaries."""
    _CHUNK_CACHE.clear()


@dataclass
class TopicInterpretation:
    """LLM-assisted interpretation of a single LDA topic."""

    topic_id: Optional[int]
    title: str
    explanation: str
    keywords: List[str] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    llm_assisted: bool = True
    model_id: Optional[str] = None


def _logger() -> logging.Logger:
    try:
        try:
            from ..utils.logging import configure_logging
        except ImportError:  # pragma: no cover
            from utils.logging import configure_logging

        return configure_logging()
    except Exception:  # pragma: no cover - defensive
        return logging.getLogger("educational_summarizer")


def _validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise LLMError("Input text must be a string.")
    normalized = " ".join(text.split())
    if not normalized:
        raise LLMError("Input text is empty.")
    return text


def _resolve_config(config: Optional[LLMConfig]) -> LLMConfig:
    cfg = config if config is not None else load_config()
    if not cfg.is_configured:
        if getattr(cfg, "ollama_enabled", False):
            reachable, installed = check_ollama_health(
                getattr(cfg, "ollama_base_url", "http://localhost:11434"),
                timeout=1.5,
            )
            if reachable and installed:
                return cfg
        raise LLMConfigError(
            "Gemini is not configured. Set GEMINI_API_KEY in .env "
            "(and optionally GEMINI_MODEL_ID)."
        )
    return cfg


def _create_client(config: LLMConfig, client: Any = None) -> Any:
    if client is not None:
        return client
    if not config.is_configured:
        # Local Ollama-only path when Gemini key is absent.
        return None
    try:
        return create_client(config)
    except LLMError as exc:
        raise LLMError(f"Failed to create LLM client: {exc}") from exc


def _complete_once(client: Any, config: LLMConfig, messages: List[dict]) -> str:
    # Delegate to the shared provider client, which supports Gemini and
    # OpenAI-compatible test doubles.
    from .llm_client import _complete_once as _shared_complete_once

    return _shared_complete_once(client, config, messages)


def _complete(client: Any, config: LLMConfig, messages: List[dict]) -> str:
    return chat_completion(config, messages, client=client)


def _complete_with_meta(client: Any, config: LLMConfig, messages: List[dict]) -> tuple[str, str]:
    return chat_completion_with_metadata(config, messages, client=client)


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 400) -> List[str]:
    """Split text into chunks respecting sentence boundaries with overlap.

    ``chunk_size`` and ``overlap`` are character budgets. They are intentionally
    conservative relative to the model context window; callers may tune them.
    """
    if not isinstance(text, str):
        raise LLMError("Input text must be a string.")
    text = text.strip()
    if not text:
        return []

    # Sentence segmentation with fallback.
    try:
        import nltk

        nltk.data.find("tokenizers/punkt")
        from nltk.tokenize import sent_tokenize

        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        sentences = [s.strip() for s in text.split(". ") if s.strip()]

    if not sentences:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        if current and current_len + len(sentence) > chunk_size:
            chunks.append(" ".join(current))
            # Build overlap from trailing sentences.
            overlap_sents: List[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current = overlap_sents + [sentence]
            current_len = sum(len(s) for s in current)
        else:
            current.append(sentence)
            current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def _normalize_topic_items(topics: Any) -> List[tuple]:
    """Normalize supported topic inputs into (topic_id, keywords, weights)."""
    if hasattr(topics, "topics"):
        topics = topics.topics
    items: List[tuple] = []
    if isinstance(topics, dict):
        topics = [topics]
    for index, topic in enumerate(topics or []):
        topic_id = getattr(topic, "topic_id", None)
        keywords = getattr(topic, "keywords", None)
        weights = getattr(topic, "weights", None)
        if keywords is None and isinstance(topic, dict):
            topic_id = topic.get("topic_id", topic_id)
            keywords = topic.get("keywords", [])
            weights = topic.get("weights", [])
        if keywords is None:
            # Allow passing keyword lists directly.
            if isinstance(topic, (list, tuple)):
                if topic and isinstance(topic[0], str):
                    keywords = list(topic)
                    weights = [1.0] * len(keywords)
                else:
                    keywords = []
                    weights = []
            else:
                keywords = []
                weights = []
        keywords = [str(k) for k in (keywords or [])]
        weights = [float(w) for w in (weights or [])]
        if len(weights) < len(keywords):
            weights = weights + [0.0] * (len(keywords) - len(weights))
        if topic_id is None:
            topic_id = index
        items.append((topic_id, keywords, weights[: len(keywords)]))
    return items


def _clean_markdown_prefix(value: str) -> str:
    """Strip leading/trailing Markdown bolding, headings, and quotes."""
    import re

    cleaned = re.sub(r"^[#*\-\s]+", "", value.strip())
    cleaned = re.sub(r"[*`]+$", "", cleaned).strip()
    return cleaned.strip("\"' ")


def _parse_topic_interpretation(
    text: str,
    topic_id: Optional[int],
    keywords: List[str],
    weights: List[float],
) -> TopicInterpretation:
    import re

    default_title = f"Topic {topic_id}" if topic_id is not None else "Topic"
    title = default_title
    explanation_lines: List[str] = []
    in_explanation = False

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        # Normalize markdown bold/headers like "**Title**: ..." or "### Title: ..."
        normalized = re.sub(r"^[#>*\-\s]+", "", stripped)
        normalized = re.sub(r"\*\*([^*]+)\*\*", r"\1", normalized).strip()
        lower = normalized.lower()

        # Skip generic header lines like "LLM-Assisted Topic Interpretation"
        if lower in {
            "llm-assisted topic interpretation",
            "llm-assisted topic interpretations",
            f"topic {topic_id}",
            f"topic id: {topic_id}",
        } or lower.startswith("label:"):
            continue

        title_match = re.match(
            r"^(?:topic\s*\d*\s*)?title\s*[:\-]\s*(.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if title_match:
            candidate = _clean_markdown_prefix(title_match.group(1))
            if candidate:
                title = candidate
            in_explanation = False
            continue

        expl_match = re.match(
            r"^explanation\s*[:\-]\s*(.*)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if expl_match:
            first_part = _clean_markdown_prefix(expl_match.group(1))
            if first_part:
                explanation_lines.append(first_part)
            in_explanation = True
            continue

        if in_explanation:
            explanation_lines.append(stripped)

    if explanation_lines:
        explanation = " ".join(explanation_lines).strip()
    else:
        # Fallback: remove title and boilerplate lines from raw text
        fallback_lines = []
        for raw_line in text.splitlines():
            s = raw_line.strip()
            norm = re.sub(r"^[#>*\-\s]+|\*\*$", "", s).lower()
            if not s or "llm-assisted topic interpretation" in norm or norm.startswith("title:"):
                continue
            fallback_lines.append(s)
        explanation = " ".join(fallback_lines).strip() or text.strip()

    if title == default_title and "\n" in text:
        for raw_line in text.splitlines():
            candidate = _clean_markdown_prefix(raw_line)
            if candidate and "llm-assisted" not in candidate.lower() and not candidate.lower().startswith("explanation"):
                title = candidate[:80]
                break

    return TopicInterpretation(
        topic_id=topic_id,
        title=title,
        explanation=explanation,
        keywords=list(keywords),
        weights=list(weights),
    )


def _split_batched_topic_blocks(text: str, topic_ids: List[Optional[int]]) -> Dict[Optional[int], str]:
    """Split a single batched LLM response into per-topic text blocks."""
    import re

    blocks: Dict[Optional[int], List[str]] = {}
    current_id: Optional[int] = topic_ids[0] if topic_ids else 0

    for line in text.splitlines():
        stripped = line.strip()
        normalized = re.sub(r"^[#>*\-\s]+", "", stripped)
        normalized = re.sub(r"\*\*([^*]+)\*\*", r"\1", normalized).strip()
        m = re.match(r"^topic(?:\s+id)?\s*(\d+)\b\s*[:\-]?(.*)$", normalized, flags=re.IGNORECASE)
        if m:
            matched_id = int(m.group(1))
            if matched_id in topic_ids:
                current_id = matched_id
                remainder = m.group(2).strip()
                blocks.setdefault(current_id, [])
                if remainder:
                    blocks[current_id].append(remainder)
                continue
        blocks.setdefault(current_id, []).append(line)

    return {tid: "\n".join(lines).strip() for tid, lines in blocks.items()}


def interpret_topics(
    topics: Any,
    config: Optional[LLMConfig] = None,
    client: Any = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> List[TopicInterpretation]:
    """Generate an optional LLM-assisted title and explanation for LDA topics.

    All topics are sent in a single batched Gemini request to minimize API
    round-trips and avoid rate-limit / deadline errors. The original LDA
    keywords and weights are preserved in each result.
    """
    cfg = _resolve_config(config)
    llm_client = _create_client(cfg, client)
    normalized_items = _normalize_topic_items(topics)
    if not normalized_items:
        return []

    if len(normalized_items) == 1:
        topic_id, keywords, weights = normalized_items[0]
        keyword_lines = "\n".join(
            f"- {keyword}: {weight:.4f}" for keyword, weight in zip(keywords, weights)
        )
        user_prompt = (
            "You are interpreting the output of an LDA topic model. "
            "Use only the supplied keywords and weights. Do not invent topics, "
            "facts, or definitions. If the keywords are unclear, say so.\n\n"
            f"Topic ID: {topic_id}\n"
            "Keywords and weights:\n"
            f"{keyword_lines or '- (none)'}\n\n"
            "Return plain text in this exact format (no markdown headings):\n"
            "Title: <short academic topic title>\n"
            "Explanation: <2-3 sentences explaining the topic>"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        generated, used_model = _complete_with_meta(llm_client, cfg, messages)
        parsed = _parse_topic_interpretation(generated, topic_id, keywords, weights)
        parsed.model_id = used_model
        return [parsed]

    # Batched single-request path for multiple topics (1 API call instead of N calls)
    topic_sections: List[str] = []
    topic_ids: List[Optional[int]] = []
    for topic_id, keywords, weights in normalized_items:
        topic_ids.append(topic_id)
        kw_str = ", ".join(f"{k} ({w:.4f})" for k, w in zip(keywords, weights))
        topic_sections.append(f"Topic {topic_id}:\nKeywords: {kw_str or '(none)'}")

    batched_prompt = (
        "You are interpreting the output of an LDA topic model. "
        "Use only the supplied keywords and weights for each topic. "
        "Do not invent external facts or definitions.\n\n"
        + "\n\n".join(topic_sections)
        + "\n\nFor every topic above, return plain text in this exact format (do not add extra headings):\n"
        "Topic <ID>:\n"
        "Title: <short academic topic title>\n"
        "Explanation: <2-3 sentences explaining the topic based on its keywords>\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": batched_prompt},
    ]
    generated, used_model = _complete_with_meta(llm_client, cfg, messages)
    per_topic_blocks = _split_batched_topic_blocks(generated, topic_ids)

    interpretations: List[TopicInterpretation] = []
    for topic_id, keywords, weights in normalized_items:
        block = per_topic_blocks.get(topic_id, generated)
        parsed = _parse_topic_interpretation(block, topic_id, keywords, weights)
        parsed.model_id = used_model
        interpretations.append(parsed)
    return interpretations


# Backward/alternative names for callers and tests.
interpret_topic = interpret_topics
generate_topic_interpretation = interpret_topics
generate_topic_interpretations = interpret_topics
interpret_lda_topics = interpret_topics


def generate_summary(
    text: str,
    summary_type: str = "short",
    config: Optional[LLMConfig] = None,
    client: Any = None,
    chunk_size: int = 4000,
    overlap: int = 400,
    system_prompt: str = SYSTEM_PROMPT,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    chunk_cache: Optional[Dict[str, Any]] = None,
) -> LLMSummaryResult:
    """Generate an educational summary using Gemini or configured LLM.

    Optimization:
    1. Short document optimization: Uses official SDK token counting where supported.
       If the full document fits within the conservative token budget (default 16000),
       it is processed in a single Gemini request without splitting into unnecessary chunks.
    2. Genuinely long documents retain hierarchical chunk-based summarization with
       chunk resumability (retaining successful chunks and retrying only failed ones).
    """
    text = _validate_text(text)
    cfg = _resolve_config(config)
    llm_client = _create_client(cfg, client)
    logger = _logger()

    warnings: List[str] = []
    max_single_tokens = getattr(cfg, "max_single_request_tokens", 16000)

    # 1. Short Document Optimization using official SDK token counting
    exact_tokens = count_tokens(cfg, text, client=llm_client) if llm_client is not None else None
    if exact_tokens is not None and exact_tokens <= max_single_tokens:
        logger.info(
            "Short document optimization: exact token count %d fits budget %d; processing in 1 request.",
            exact_tokens,
            max_single_tokens,
        )
        if progress_callback:
            progress_callback({
                "current_chunk": 1,
                "total_chunks": 1,
                "completed_chunks": 0,
                "failed_chunks": 0,
                "retry_status": "Running (single request)",
            })
        prompt = build_prompt(summary_type, text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            generated, used_model = _complete_with_meta(llm_client, cfg, messages)
        except Exception as exc:
            if progress_callback:
                progress_callback({
                    "current_chunk": 1,
                    "total_chunks": 1,
                    "completed_chunks": 0,
                    "failed_chunks": 1,
                    "retry_status": "Failed",
                    "error": str(exc),
                })
            raise
        if progress_callback:
            progress_callback({
                "current_chunk": 1,
                "total_chunks": 1,
                "completed_chunks": 1,
                "failed_chunks": 0,
                "retry_status": "Completed",
            })
        return LLMSummaryResult(
            summary_type=summary_type,
            summary_text=generated,
            chunk_count=1,
            chunks_completed=1,
            chunks_failed=0,
            warnings=warnings,
            model_id=used_model,
        )

    # 2. Hierarchical Chunk-based Summarization for genuinely long documents
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    logger.info("LLM summarization: %d chunk(s), type=%s", len(chunks), summary_type)

    if len(chunks) == 1:
        if progress_callback:
            progress_callback({
                "current_chunk": 1,
                "total_chunks": 1,
                "completed_chunks": 0,
                "failed_chunks": 0,
                "retry_status": "Running (single chunk)",
            })
        prompt = build_prompt(summary_type, chunks[0])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            generated, used_model = _complete_with_meta(llm_client, cfg, messages)
        except Exception as exc:
            if progress_callback:
                progress_callback({
                    "current_chunk": 1,
                    "total_chunks": 1,
                    "completed_chunks": 0,
                    "failed_chunks": 1,
                    "retry_status": "Failed",
                    "error": str(exc),
                })
            raise
        if progress_callback:
            progress_callback({
                "current_chunk": 1,
                "total_chunks": 1,
                "completed_chunks": 1,
                "failed_chunks": 0,
                "retry_status": "Completed",
            })
        return LLMSummaryResult(
            summary_type=summary_type,
            summary_text=generated,
            chunk_count=1,
            chunks_completed=1,
            chunks_failed=0,
            warnings=warnings,
            model_id=used_model,
        )

    # 3. Resumable Chunk Processing for multi-chunk documents
    cache = chunk_cache if chunk_cache is not None else _CHUNK_CACHE
    cache_key = hashlib.sha256(
        f"{text}::{summary_type}::{cfg.model_id}::{system_prompt}".encode("utf-8")
    ).hexdigest()
    cached_chunks = cache.setdefault(cache_key, {})

    intermediate: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if index in cached_chunks:
            intermediate.append(cached_chunks[index])
            if progress_callback:
                progress_callback({
                    "current_chunk": index,
                    "total_chunks": len(chunks),
                    "completed_chunks": len(intermediate),
                    "failed_chunks": 0,
                    "retry_status": f"Chunk {index}/{len(chunks)} loaded from cache",
                })
            continue

        try:
            if progress_callback:
                progress_callback({
                    "current_chunk": index,
                    "total_chunks": len(chunks),
                    "completed_chunks": len(intermediate),
                    "failed_chunks": 0,
                    "retry_status": f"Summarizing chunk {index}/{len(chunks)}",
                })
            prompt = build_chunk_prompt(chunk)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            partial = _complete(llm_client, cfg, messages)
            cached_chunks[index] = partial
            intermediate.append(partial)
            if progress_callback:
                progress_callback({
                    "current_chunk": index,
                    "total_chunks": len(chunks),
                    "completed_chunks": len(intermediate),
                    "failed_chunks": 0,
                    "retry_status": f"Chunk {index}/{len(chunks)} completed",
                })
        except Exception as exc:
            if progress_callback:
                progress_callback({
                    "current_chunk": index,
                    "total_chunks": len(chunks),
                    "completed_chunks": len(intermediate),
                    "failed_chunks": len(chunks) - len(intermediate),
                    "retry_status": f"Failed on chunk {index}/{len(chunks)}",
                    "error": str(exc),
                })
            raise LLMError(
                f"LLM request failed for chunk {index} of {len(chunks)}: {exc}"
            ) from exc

    combined = "\n\n".join(intermediate)
    final_prompt = build_prompt(summary_type, combined)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_prompt},
    ]
    try:
        if progress_callback:
            progress_callback({
                "current_chunk": len(chunks),
                "total_chunks": len(chunks),
                "completed_chunks": len(chunks),
                "failed_chunks": 0,
                "retry_status": "Combining chunk summaries into final summary",
            })
        generated, used_model = _complete_with_meta(llm_client, cfg, messages)
    except Exception as exc:
        if progress_callback:
            progress_callback({
                "current_chunk": len(chunks),
                "total_chunks": len(chunks),
                "completed_chunks": len(chunks),
                "failed_chunks": 0,
                "retry_status": "Failed on final summary combination",
                "error": str(exc),
            })
        raise LLMError(f"Final LLM summarization failed: {exc}") from exc

    if not intermediate:
        warnings.append("No intermediate summaries were generated.")

    if progress_callback:
        progress_callback({
            "current_chunk": len(chunks),
            "total_chunks": len(chunks),
            "completed_chunks": len(chunks),
            "failed_chunks": 0,
            "retry_status": "Completed",
        })

    return LLMSummaryResult(
        summary_type=summary_type,
        summary_text=generated,
        chunk_count=len(chunks),
        chunks_completed=len(chunks),
        chunks_failed=0,
        warnings=warnings,
        model_id=used_model,
    )
