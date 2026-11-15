"""Versioned prompt templates for LLM educational summarization."""

from __future__ import annotations

from typing import Dict, List

PROMPT_VERSION = "2.0"

SYSTEM_PROMPT = (
    "You are an educational assistant. Use only the supplied source material. "
    "Preserve important technical terminology. Do not add unsupported information. "
    "Do not invent definitions or facts. If information is missing, say so clearly. "
    "Use clear headings and explain difficult concepts in understandable language. "
    "Avoid unnecessary repetition."
)

OUTPUT_FORMAT = (
    "## Main Summary\n"
    "<concise summary>\n\n"
    "## Key Concepts\n"
    "- ...\n\n"
    "## Important Definitions\n"
    "- Term: definition\n\n"
    "## Important Points\n"
    "- ...\n\n"
    "## Revision Notes\n"
    "- ...\n"
)

_TEMPLATES: Dict[str, str] = {
    "short": (
        "Create a short educational summary of the source material below.\n"
        "Focus on the most important ideas and keep it concise.\n\n"
        "Source material:\n{content}\n\n"
        "Return the result using this format:\n" + OUTPUT_FORMAT
    ),
    "detailed": (
        "Create a detailed educational summary of the source material below.\n"
        "Cover the main ideas, supporting details, and relationships between concepts.\n\n"
        "Source material:\n{content}\n\n"
        "Return the result using this format:\n" + OUTPUT_FORMAT
    ),
    "bullets": (
        "Convert the source material below into clear bullet-point study notes.\n"
        "Group related ideas and keep each bullet focused.\n\n"
        "Source material:\n{content}\n\n"
        "Return the result using this format:\n" + OUTPUT_FORMAT
    ),
    "definitions": (
        "Extract and explain the important definitions and technical terms "
        "from the source material below. Only define terms actually present.\n\n"
        "Source material:\n{content}\n\n"
        "Return the result using this format:\n" + OUTPUT_FORMAT
    ),
    "revision": (
        "Create exam revision notes from the source material below. "
        "Highlight key facts, formulas, and concepts a student should remember.\n\n"
        "Source material:\n{content}\n\n"
        "Return the result using this format:\n" + OUTPUT_FORMAT
    ),
}

CHUNK_SUMMARY_TEMPLATE = (
    "Summarize the following chunk of educational source material concisely, "
    "preserving technical terms and key facts. Do not add information not present.\n\n"
    "Chunk:\n{content}\n"
)

VALID_SUMMARY_TYPES = tuple(_TEMPLATES.keys())


def list_summary_types() -> List[str]:
    return list(VALID_SUMMARY_TYPES)


def build_prompt(summary_type: str, content: str) -> str:
    """Build the user prompt for a given summary type."""
    if summary_type not in _TEMPLATES:
        raise ValueError(
            f"Unknown summary type '{summary_type}'. "
            f"Valid types: {', '.join(VALID_SUMMARY_TYPES)}"
        )
    if not isinstance(content, str):
        raise ValueError("Content must be a string.")
    return _TEMPLATES[summary_type].format(content=content)


def build_chunk_prompt(content: str) -> str:
    """Build the prompt used to summarize an individual chunk."""
    if not isinstance(content, str):
        raise ValueError("Content must be a string.")
    return CHUNK_SUMMARY_TEMPLATE.format(content=content)
