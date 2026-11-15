"""Utility helpers for the Educational Content Summarization System."""

from .config import GeminiConfig, LLMConfig, load_config
from .logging import configure_logging

__all__ = ["GeminiConfig", "LLMConfig", "load_config", "configure_logging"]

# Backward-compatible aliases for earlier provider-specific names.
GroqConfig = GeminiConfig
LocalLLMConfig = GeminiConfig
