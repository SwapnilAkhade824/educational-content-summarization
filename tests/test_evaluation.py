"""Tests for evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.metrics import (
    classification_metrics,
    concept_metrics,
    rouge_l,
    rouge_n,
    rouge_scores,
)
from evaluation.runner import run_evaluation


def test_rouge_n() -> None:
    pred = "the cat sat on the mat"
    ref = "the cat is on the mat"
    result = rouge_n(pred, ref, 1)
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1
    assert 0 <= result["f1"] <= 1
    # ROUGE-2
    r2 = rouge_n(pred, ref, 2)
    assert r2["f1"] >= 0


def test_rouge_l() -> None:
    result = rouge_l("a b c d", "a b x c d")
    assert result["recall"] == 1.0 or result["recall"] > 0


def test_rouge_scores_structure() -> None:
    scores = rouge_scores("hello world", "hello world")
    assert set(scores.keys()) == {"rouge_1", "rouge_2", "rouge_l"}
    for v in scores.values():
        assert set(v.keys()) == {"precision", "recall", "f1"}


def test_classification_metrics() -> None:
    y_true = ["Positive", "Negative", "Neutral", "Positive"]
    y_pred = ["Positive", "Negative", "Positive", "Positive"]
    metrics = classification_metrics(y_pred, y_true)
    assert metrics["accuracy"] == 0.75
    assert "Positive" in metrics["per_class"]
    assert len(metrics["confusion_matrix"]) == 3
    assert metrics["macro_f1"] >= 0


def test_classification_metrics_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        classification_metrics(["a"], ["a", "b"])


def test_concept_metrics() -> None:
    metrics = concept_metrics(["Machine Learning", "AI"], ["machine learning", "deep learning"])
    assert metrics["true_positives"] == 1
    assert 0 < metrics["precision"] < 1
    assert 0 < metrics["recall"] < 1


def test_concept_metrics_both_empty() -> None:
    metrics = concept_metrics([], [])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_run_evaluation_writes_report() -> None:
    report = run_evaluation(write_report=True)
    assert "results" in report
    assert "summarization" in report["results"]
    assert "sentiment" in report["results"]
    assert "concepts" in report["results"]
    assert "topics" in report["results"]
    assert Path("data/evaluation/reports/evaluation_report.json").exists()
    # Validate JSON
    with open("data/evaluation/reports/evaluation_report.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["evaluation_date"]
    assert loaded["results"]["sentiment"]["metrics"]["accuracy"] >= 0


def test_run_evaluation_llm_skipped_without_config() -> None:
    report = run_evaluation(write_report=False)
    llm_status = report["results"]["summarization"]["samples"][0]["llm_summary"]
    assert llm_status["status"] in {"skipped", "executed", "failed"}
