"""Application configuration loaded from environment variables.

Gemini and optional local Ollama fallback settings are read centrally from
environment variables (typically provided through a local .env file or the
deployment environment). The Streamlit UI never edits these values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]


DEFAULT_MODEL_ID = "gemini-3.8-flash"

# Ordered strictly from live API verification on this key (fastest/most reliable first).
DEFAULT_FALLBACK_MODELS: Tuple[str, ...] = (
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b-instruct"

_VALID_THINKING_LEVELS = {"low", "minimal", "medium", "high", "xhigh"}


def _env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return None


def _get_int(name_options: tuple[str, ...], default: int) -> int:
    raw = _env(*name_options)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name_options[0]} must be an integer, got {raw!r}"
        ) from exc


def _get_float(name_options: tuple[str, ...], default: float) -> float:
    raw = _env(*name_options)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name_options[0]} must be a number, got {raw!r}"
        ) from exc


def _get_bool(name_options: tuple[str, ...], default: bool) -> bool:
    raw = _env(*name_options)
    if raw is None:
        return default
    lower = raw.lower()
    if lower in {"1", "true", "yes", "on"}:
        return True
    if lower in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable {name_options[0]} must be a boolean, got {raw!r}"
    )


def _parse_model_list(raw: Optional[str], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if raw is None:
        return default
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(dict.fromkeys(items))


@dataclass(frozen=True)
class GeminiConfig:
    """Configuration for Google Gemini cloud inference and optional local Ollama fallback.

    All values come from environment variables. The Streamlit UI never edits
    them.
    """

    api_key: Optional[str] = None
    model_id: Optional[str] = None
    thinking_level: str = "low"
    max_output_tokens: int = 8192
    max_tokens: Optional[int] = None
    temperature: float = 0.2
    timeout_seconds: int = 120
    max_single_request_tokens: int = 16000

    # Cloud fallback chain & local-only Ollama fallback (safe no-op on cloud hosts).
    # Direct constructor defaults stay minimal so isolated unit tests remain deterministic;
    # load_config() populates verified runtime defaults from environment variables.
    fallback_models: Tuple[str, ...] = ()
    ollama_enabled: bool = False
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL

    # Backward-compatible fields retained for older callers/tests.
    provider: str = "gemini"
    base_url: Optional[str] = None

    def __post_init__(self) -> None:
        # Synchronise the legacy ``max_tokens`` alias with ``max_output_tokens``.
        if self.max_tokens is not None:
            object.__setattr__(self, "max_output_tokens", self.max_tokens)
        object.__setattr__(self, "max_tokens", self.max_output_tokens)

        level = (self.thinking_level or "").strip().lower()
        if level not in _VALID_THINKING_LEVELS:
            raise ValueError(
                f"Environment variable GEMINI_THINKING_LEVEL must be one of "
                f"{sorted(_VALID_THINKING_LEVELS)}, got {self.thinking_level!r}"
            )
        object.__setattr__(self, "thinking_level", level)

        # Normalize fallback_models to an immutable tuple of non-empty strings.
        if isinstance(self.fallback_models, str):
            normalized_fb = _parse_model_list(self.fallback_models, ())
        else:
            normalized_fb = tuple(
                dict.fromkeys(str(m).strip() for m in (self.fallback_models or ()) if str(m).strip())
            )
        object.__setattr__(self, "fallback_models", normalized_fb)

        base = (self.ollama_base_url or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
        object.__setattr__(self, "ollama_base_url", base)

        ollama_m = (self.ollama_model or DEFAULT_OLLAMA_MODEL).strip()
        object.__setattr__(self, "ollama_model", ollama_m)

    @property
    def is_configured(self) -> bool:
        """Whether the minimum required Gemini cloud settings are present."""
        return bool(self.api_key and self.model_id)

    # Backward-compatible aliases.
    @property
    def max_completion_tokens(self) -> int:
        return self.max_output_tokens


# Backward-compatible aliases for earlier provider-specific names.
LLMConfig = GeminiConfig
GoogleGeminiConfig = GeminiConfig
LocalLLMConfig = GeminiConfig
OpenAICompatibleConfig = GeminiConfig
GroqConfig = GeminiConfig


def load_config() -> GeminiConfig:
    """Load Gemini and optional local Ollama fallback configuration from env vars."""
    api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    model_id = _env("GEMINI_MODEL_ID") or DEFAULT_MODEL_ID
    thinking_level = _env("GEMINI_THINKING_LEVEL") or "low"
    max_output_tokens = _get_int(("GEMINI_MAX_OUTPUT_TOKENS",), 8192)
    temperature = _get_float(("GEMINI_TEMPERATURE",), 0.2)
    timeout_seconds = _get_int(("GEMINI_TIMEOUT_SECONDS",), 120)
    max_single_request_tokens = _get_int(("GEMINI_MAX_SINGLE_REQUEST_TOKENS",), 16000)

    fallback_models = _parse_model_list(
        _env("GEMINI_FALLBACK_MODELS"),
        DEFAULT_FALLBACK_MODELS,
    )
    ollama_enabled = _get_bool(("OLLAMA_ENABLED",), True)
    ollama_base_url = _env("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
    ollama_model = _env("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL

    return GeminiConfig(
        api_key=api_key,
        model_id=model_id,
        thinking_level=thinking_level,
        max_output_tokens=max_output_tokens,
        max_tokens=None,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_single_request_tokens=max_single_request_tokens,
        fallback_models=fallback_models,
        ollama_enabled=ollama_enabled,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        provider="gemini",
        base_url=None,
    )


# Backward-compatible aliases.
GeminiSettings = GeminiConfig
