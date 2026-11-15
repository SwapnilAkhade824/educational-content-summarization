"""Tests for prompt templates."""

from __future__ import annotations

import pytest

from summarization.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    VALID_SUMMARY_TYPES,
    build_chunk_prompt,
    build_prompt,
    list_summary_types,
)


def test_list_summary_types() -> None:
    types = list_summary_types()
    assert set(types) == {"short", "detailed", "bullets", "definitions", "revision"}


def test_build_prompt_each_type() -> None:
    for t in VALID_SUMMARY_TYPES:
        prompt = build_prompt(t, "Sample content about machine learning.")
        assert "Sample content about machine learning." in prompt
        assert "Main Summary" in prompt
        assert "Key Concepts" in prompt


def test_build_prompt_invalid_type_raises() -> None:
    with pytest.raises(ValueError):
        build_prompt("unknown", "content")


def test_build_prompt_invalid_content_raises() -> None:
    with pytest.raises(ValueError):
        build_prompt("short", 123)  # type: ignore[arg-type]


def test_system_prompt_present() -> None:
    assert "only the supplied source material" in SYSTEM_PROMPT.lower()
    assert PROMPT_VERSION


def test_build_chunk_prompt() -> None:
    prompt = build_chunk_prompt("chunk text")
    assert "chunk text" in prompt
    assert "concisely" in prompt.lower()


def test_prompts_are_distinct() -> None:
    prompts = {build_prompt(t, "same content") for t in VALID_SUMMARY_TYPES}
    assert len(prompts) == len(VALID_SUMMARY_TYPES)
