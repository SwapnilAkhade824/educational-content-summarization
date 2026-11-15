"""Evaluation runner for the NLP pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from ..extraction.text_extractor import extract_document
    from ..information_extraction.concept_extractor import extract_concepts
    from ..sentiment.sentiment_analyzer import analyze_feedback
    from ..summarization.textrank import summarize_text
    from ..topic_modeling.lda_model import run_lda
    from ..utils.config import load_config
except ImportError:  # pragma: no cover - supports src-layout imports
    from extraction.text_extractor import extract_document
    from information_extraction.concept_extractor import extract_concepts
    from sentiment.sentiment_analyzer import analyze_feedback
    from summarization.textrank import summarize_text
    from topic_modeling.lda_model import run_lda
    from utils.config import load_config

from .metrics import classification_metrics, concept_metrics, rouge_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
DEMO_DIR = EVAL_DIR / "demo"
REPORT_PATH = EVAL_DIR / "reports" / "evaluation_report.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summarization_evaluation() -> Dict[str, Any]:
    data = _load_json(DEMO_DIR / "summarization_demo.json")
    samples_out: List[Dict[str, Any]] = []
    for sample in data["samples"]:
        text = sample["source_text"]
        reference = sample["reference_summary"]
        tr_result = summarize_text(text, num_sentences=2)
        generated = tr_result.summary_text
        rouge = rouge_scores(generated, reference)

        llm_status: Dict[str, Any]
        cfg = load_config()
        if cfg.is_configured:
            try:
                from summarization.llm_summarizer import generate_summary

                llm_result = generate_summary(text, summary_type="short")
                llm_rouge = rouge_scores(llm_result.summary_text, reference)
                llm_status = {
                    "status": "executed",
                    "summary": llm_result.summary_text,
                    "rouge": llm_rouge,
                }
            except Exception as exc:  # pragma: no cover - depends on network/API
                llm_status = {"status": "failed", "reason": str(exc)}
        else:
            llm_status = {
                "status": "skipped",
                "reason": "Gemini not configured (GEMINI_API_KEY missing).",
            }

        samples_out.append(
            {
                "id": sample["id"],
                "reference_summary": reference,
                "textrank_summary": generated,
                "textrank_rouge": rouge,
                "llm_summary": llm_status,
            }
        )

    return {
        "dataset_id": data["dataset_id"],
        "description": data["description"],
        "sample_count": len(samples_out),
        "samples": samples_out,
    }


def _sentiment_evaluation() -> Dict[str, Any]:
    data = _load_json(DEMO_DIR / "sentiment_demo.json")
    samples = data["samples"]
    predictions = [analyze_feedback(s["feedback"]).sentiment_label for s in samples]
    references = [s["label"] for s in samples]
    metrics = classification_metrics(predictions, references)
    per_sample = [
        {
            "id": s["id"],
            "feedback": s["feedback"],
            "predicted": p,
            "reference": r,
        }
        for s, p, r in zip(samples, predictions, references)
    ]
    return {
        "dataset_id": data["dataset_id"],
        "description": data["description"],
        "sample_count": len(samples),
        "metrics": metrics,
        "per_sample": per_sample,
    }


def _concept_evaluation() -> Dict[str, Any]:
    data = _load_json(DEMO_DIR / "concepts_demo.json")
    text = data["text"]
    result = extract_concepts(text)
    predicted_entities = [e.text for e in result.entities]
    predicted_definitions = [d.concept for d in result.definitions]

    entity_metrics = concept_metrics(predicted_entities, data["reference_entities"])
    definition_metrics = concept_metrics(
        predicted_definitions, [d["concept"] for d in data["reference_definitions"]]
    )

    return {
        "dataset_id": data["dataset_id"],
        "description": data["description"],
        "entity_metrics": entity_metrics,
        "definition_metrics": definition_metrics,
        "predicted_entities": predicted_entities,
        "predicted_definitions": predicted_definitions,
        "reference_entities": data["reference_entities"],
        "reference_definitions": data["reference_definitions"],
        "warnings": result.warnings,
        "matching_criteria": (
            "Case-insensitive, whitespace-normalized exact string match on entity text "
            "and definition concept."
        ),
    }


def _topic_evaluation() -> Dict[str, Any]:
    docs = [
        "Machine learning is a subset of artificial intelligence. It uses data and algorithms.",
        "Natural language processing deals with text and human language. It includes tokenization.",
        "Deep learning uses neural networks with many layers. It is applied to images.",
        "Data preprocessing cleans raw data. It is essential for machine learning pipelines.",
    ]
    r1 = run_lda(docs, n_topics=3, random_state=42)
    r2 = run_lda(docs, n_topics=3, random_state=42)

    topics = [
        {"topic_id": t.topic_id, "keywords": t.keywords, "weights": [float(w) for w in t.weights]}
        for t in r1.topics
    ]
    doc_topics = [
        {
            "chunk_id": d.chunk_id,
            "source_index": d.source_index,
            "dominant_topic": d.dominant_topic,
            "topic_distribution": [float(v) for v in d.topic_distribution],
        }
        for d in r1.document_topics
    ]

    return {
        "dataset_id": "demo_topics_v1",
        "description": "DEMONSTRATION DATA ONLY. Topics discovered from a small demo corpus.",
        "topics": topics,
        "document_topics": doc_topics,
        "vocabulary_size": r1.vocabulary_size,
        "perplexity": (float(r1.perplexity) if r1.perplexity is not None else None),
        "reproducible": [t.keywords for t in r1.topics] == [t.keywords for t in r2.topics],
        "warnings": r1.warnings,
        "note": "Perplexity does not establish human interpretability or semantic coherence.",
    }


def run_evaluation(write_report: bool = True) -> Dict[str, Any]:
    report = {
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "project": "Educational Content Summarization System",
        "sample_counts": {},
        "results": {
            "summarization": _summarization_evaluation(),
            "sentiment": _sentiment_evaluation(),
            "concepts": _concept_evaluation(),
            "topics": _topic_evaluation(),
        },
        "missing_evaluation_data": [
            "LLM evaluation is skipped unless an OpenAI-compatible endpoint is configured.",
            "Large-scale human-annotated datasets are not included; demo data is small.",
        ],
        "limitations": [
            "Demo datasets are small and author-created.",
            "ROUGE uses whitespace tokenization and does not use stemming.",
            "Concept matching uses exact normalized string match.",
            "Perplexity is not semantic coherence.",
        ],
    }

    report["sample_counts"] = {
        "summarization": report["results"]["summarization"]["sample_count"],
        "sentiment": report["results"]["sentiment"]["sample_count"],
        "concepts": 1,
        "topics": len(report["results"]["topics"]["document_topics"]),
    }

    if write_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

    return report
