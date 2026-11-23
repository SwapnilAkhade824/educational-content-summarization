"""Tests for configuration loading."""

from __future__ import annotations

import os

import pytest

from utils.config import GeminiConfig, load_config


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL_ID",
        "GEMINI_THINKING_LEVEL",
        "GEMINI_MAX_OUTPUT_TOKENS",
        "GEMINI_TEMPERATURE",
        "GEMINI_TIMEOUT_SECONDS",
        "GEMINI_MAX_SINGLE_REQUEST_TOKENS",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    config = load_config()
    assert isinstance(config, GeminiConfig)
    assert config.api_key is None
    assert config.model_id == "gemini-3.5-flash-lite"
    assert config.thinking_level == "low"
    assert config.max_output_tokens == 8192
    assert config.max_tokens == 8192
    assert config.temperature == 0.2
    assert config.timeout_seconds == 120
    assert config.max_single_request_tokens == 16000
    assert config.is_configured is False


def test_load_config_reads_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setenv("GEMINI_MODEL_ID", "gemini-3.5-flash-lite")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "HIGH")
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "256")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.7")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "30")
    config = load_config()
    assert config.api_key == "secret"
    assert config.model_id == "gemini-3.5-flash-lite"
    assert config.thinking_level == "high"
    assert config.max_output_tokens == 256
    assert config.max_tokens == 256
    assert config.temperature == 0.7
    assert config.timeout_seconds == 30
    assert config.is_configured is True


def test_load_config_empty_strings_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setenv("GEMINI_MODEL_ID", "")
    config = load_config()
    assert config.api_key is None
    assert config.model_id == "gemini-3.5-flash-lite"
    assert config.is_configured is False


def test_load_config_invalid_int_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "not-a-number")
    with pytest.raises(ValueError):
        load_config()


def test_load_config_invalid_float_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_TEMPERATURE", "abc")
    with pytest.raises(ValueError):
        load_config()


def test_load_config_invalid_thinking_level_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "extreme")
    with pytest.raises(ValueError):
        load_config()