"""Smoke tests for package importability."""

from __future__ import annotations

import importlib

import src  # noqa: F401


def test_src_package_imports() -> None:
    assert importlib.import_module("src") is not None


def test_subpackages_import() -> None:
    for name in (
        "src.extraction",
        "src.preprocessing",
        "src.summarization",
        "src.information_extraction",
        "src.sentiment",
        "src.topic_modeling",
        "src.utils",
    ):
        module = importlib.import_module(name)
        assert module is not None


def test_utils_exports() -> None:
    from src.utils import LLMConfig, configure_logging, load_config

    assert LLMConfig is not None
    assert callable(load_config)
    assert callable(configure_logging)
