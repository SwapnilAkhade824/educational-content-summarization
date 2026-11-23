"""Evaluation runner for the NLP pipeline (supports demo and benchmark suites)."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ..information_extraction.concept_extractor import extract_concepts
    from ..sentiment.sentiment_analyzer import analyze_feedback
    from ..summarization.textrank import summarize_text
    from ..topic_modeling.lda_model import run_lda
    from ..utils.config import load_config
except ImportError:  # pragma: no cover - supports src-layout imports
    from information_extraction.concept_extractor import extract_concepts
    from sentiment.sentiment_analyzer import analyze_feedback
    from summarization.textrank import summarize_text
    from topic_modeling.lda_model import run_lda
    from utils.config import load_config

from .metrics import classification_metrics, concept_metrics, rouge_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
DEMO_DIR = EVAL_DIR / "demo"
BENCHMARK_DIR = EVAL_DIR / "benchmark"
REPORT_PATH = EVAL_DIR / "reports" / "evaluation_report.json"
BENCHMARK_REPORT_PATH = EVAL_DIR / "reports" / "benchmark_report.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_main_summary(markdown_text: str) -> str:
    """Extract the '## Main Summary' section from structured Gemini output.

    Falls back to the full cleaned text when no heading is present so ROUGE
    evaluates the actual concise summary rather than extra study-note sections.
    """
    if not markdown_text or not markdown_text.strip():
        return ""
    match = re.search(
        r"##\s*Main\s+Summary\s*\n+(.*?)(?:\n##\s+|\Z)",
        markdown_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        extracted = re.sub(r"\s+", " ", match.group(1)).strip()
        if extracted:
            return extracted
    return re.sub(r"\s+", " ", markdown_text).strip()


def _macro_average_rouge(
    score_list: List[Dict[str, Dict[str, float]]],
) -> Optional[Dict[str, Dict[str, float]]]:
    """Compute macro-averaged ROUGE-1, ROUGE-2, and ROUGE-L over a list of scores."""
    if not score_list:
        return None
    n = len(score_list)
    averaged: Dict[str, Dict[str, float]] = {}
    for metric_key in ("rouge_1", "rouge_2", "rouge_l"):
        averaged[metric_key] = {
            sub_key: sum(item[metric_key][sub_key] for item in score_list) / n
            for sub_key in ("precision", "recall", "f1")
        }
    return averaged


def _summarization_evaluation(
    suite: str = "demo",
    skip_llm: bool = False,
) -> Dict[str, Any]:
    data_path = (
        BENCHMARK_DIR / "summarization_benchmark.json"
        if suite == "benchmark"
        else DEMO_DIR / "summarization_demo.json"
    )
    data = _load_json(data_path)
    samples_out: List[Dict[str, Any]] = []
    textrank_rouges: List[Dict[str, Dict[str, float]]] = []
    textrank_latencies: List[float] = []
    llm_rouges: List[Dict[str, Dict[str, float]]] = []
    llm_full_rouges: List[Dict[str, Dict[str, float]]] = []
    llm_latencies: List[float] = []

    cfg = load_config()
    llm_model_id: Optional[str] = cfg.model_id if cfg.is_configured else None

    for idx, sample in enumerate(data["samples"]):
        text = sample["source_text"]
        reference = sample["reference_summary"]

        t0 = time.perf_counter()
        tr_result = summarize_text(text, num_sentences=2)
        tr_latency = time.perf_counter() - t0
        generated = tr_result.summary_text
        rouge = rouge_scores(generated, reference)
        textrank_rouges.append(rouge)
        textrank_latencies.append(tr_latency)

        llm_status: Dict[str, Any]
        if skip_llm:
            llm_status = {"status": "skipped", "reason": "LLM evaluation skipped by caller."}
        elif cfg.is_configured:
            try:
                from summarization.llm_summarizer import generate_summary

                if idx > 0:
                    time.sleep(0.3)
                t_llm = time.perf_counter()
                llm_result = generate_summary(text, summary_type="short", config=cfg)
                llm_latency = time.perf_counter() - t_llm
                main_summary = _extract_main_summary(llm_result.summary_text)
                llm_rouge = rouge_scores(main_summary, reference)
                llm_full_rouge = rouge_scores(llm_result.summary_text, reference)
                llm_rouges.append(llm_rouge)
                llm_full_rouges.append(llm_full_rouge)
                llm_latencies.append(llm_latency)
                llm_status = {
                    "status": "executed",
                    "model_id": llm_result.model_id,
                    "main_summary": main_summary,
                    "summary": llm_result.summary_text,
                    "latency_seconds": round(llm_latency, 4),
                    "rouge": llm_rouge,
                    "full_output_rouge": llm_full_rouge,
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
                "domain": sample.get("domain"),
                "reference_summary": reference,
                "textrank_summary": generated,
                "textrank_latency_seconds": round(tr_latency, 4),
                "textrank_rouge": rouge,
                "llm_summary": llm_status,
            }
        )

    return {
        "dataset_id": data["dataset_id"],
        "description": data["description"],
        "sample_count": len(samples_out),
        "textrank_macro_rouge": _macro_average_rouge(textrank_rouges),
        "textrank_avg_latency_seconds": (
            round(sum(textrank_latencies) / len(textrank_latencies), 4)
            if textrank_latencies
            else None
        ),
        "llm_model_id": llm_model_id,
        "llm_evaluated_count": len(llm_rouges),
        "llm_macro_rouge": _macro_average_rouge(llm_rouges),
        "llm_full_output_macro_rouge": _macro_average_rouge(llm_full_rouges),
        "llm_avg_latency_seconds": (
            round(sum(llm_latencies) / len(llm_latencies), 4)
            if llm_latencies
            else None
        ),
        "samples": samples_out,
    }


def _sentiment_evaluation(suite: str = "demo") -> Dict[str, Any]:
    data_path = (
        BENCHMARK_DIR / "sentiment_benchmark.json"
        if suite == "benchmark"
        else DEMO_DIR / "sentiment_demo.json"
    )
    data = _load_json(data_path)
    samples = data["samples"]
    analyzed = [analyze_feedback(s["feedback"]) for s in samples]
    predictions = [r.sentiment_label for r in analyzed]
    references = [s["label"] for s in samples]
    metrics = classification_metrics(predictions, references)
    cm = metrics["confusion_matrix"]
    cm_total = sum(sum(row) for row in cm)  # type: ignore[arg-type]
    if cm_total != len(samples):
        raise ValueError(
            f"Confusion matrix total ({cm_total}) does not match sample count ({len(samples)})."
        )

    per_sample = [
        {
            "id": s["id"],
            "feedback": s["feedback"],
            "predicted": p,
            "reference": r,
            "compound": round(res.compound_score, 4),
        }
        for s, p, r, res in zip(samples, predictions, references, analyzed)
    ]
    return {
        "dataset_id": data["dataset_id"],
        "description": data["description"],
        "sample_count": len(samples),
        "confusion_matrix_total": cm_total,
        "metrics": metrics,
        "per_sample": per_sample,
    }


def _concept_evaluation(suite: str = "demo") -> Dict[str, Any]:
    data_path = (
        BENCHMARK_DIR / "concepts_benchmark.json"
        if suite == "benchmark"
        else DEMO_DIR / "concepts_demo.json"
    )
    data = _load_json(data_path)

    # Single-document schema (demo_concepts_v1)
    if "samples" not in data:
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
            "sample_count": 1,
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

    # Multi-document schema (academic_concepts_benchmark_v1)
    per_sample: List[Dict[str, Any]] = []
    ent_tp = ent_pred_total = ent_ref_total = 0
    def_tp = def_pred_total = def_ref_total = 0
    ent_macro_p: List[float] = []
    ent_macro_r: List[float] = []
    ent_macro_f1: List[float] = []
    def_macro_p: List[float] = []
    def_macro_r: List[float] = []
    def_macro_f1: List[float] = []
    all_warnings: List[str] = []

    for sample in data["samples"]:
        result = extract_concepts(sample["text"])
        all_warnings.extend(result.warnings)
        pred_ents = [e.text for e in result.entities]
        ref_ents = list(sample["reference_entities"])
        pred_defs = [d.concept for d in result.definitions]
        ref_defs = [d["concept"] for d in sample["reference_definitions"]]

        e_m = concept_metrics(pred_ents, ref_ents)
        d_m = concept_metrics(pred_defs, ref_defs)

        pred_ent_set = {" ".join(x.lower().split()) for x in pred_ents}
        ref_ent_set = {" ".join(x.lower().split()) for x in ref_ents}
        pred_def_set = {" ".join(x.lower().split()) for x in pred_defs}
        ref_def_set = {" ".join(x.lower().split()) for x in ref_defs}

        ent_tp += int(e_m["true_positives"])
        ent_pred_total += len(pred_ent_set)
        ent_ref_total += len(ref_ent_set)
        def_tp += int(d_m["true_positives"])
        def_pred_total += len(pred_def_set)
        def_ref_total += len(ref_def_set)

        ent_macro_p.append(e_m["precision"])
        ent_macro_r.append(e_m["recall"])
        ent_macro_f1.append(e_m["f1"])
        def_macro_p.append(d_m["precision"])
        def_macro_r.append(d_m["recall"])
        def_macro_f1.append(d_m["f1"])

        per_sample.append(
            {
                "id": sample["id"],
                "predicted_entities": pred_ents,
                "reference_entities": ref_ents,
                "entity_metrics": e_m,
                "predicted_definitions": pred_defs,
                "reference_definitions": ref_defs,
                "definition_metrics": d_m,
            }
        )

    def _micro(tp: int, pred_n: int, ref_n: int) -> Dict[str, Any]:
        p = tp / pred_n if pred_n else 0.0
        r = tp / ref_n if ref_n else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "true_positives": tp,
            "false_positives": pred_n - tp,
            "false_negatives": ref_n - tp,
            "predicted_total": pred_n,
            "reference_total": ref_n,
        }

    n = max(1, len(per_sample))
    return {
        "dataset_id": data["dataset_id"],
        "description": data["description"],
        "sample_count": len(per_sample),
        "entity_metrics": _micro(ent_tp, ent_pred_total, ent_ref_total),
        "entity_macro_metrics": {
            "precision": sum(ent_macro_p) / n,
            "recall": sum(ent_macro_r) / n,
            "f1": sum(ent_macro_f1) / n,
        },
        "definition_metrics": _micro(def_tp, def_pred_total, def_ref_total),
        "definition_macro_metrics": {
            "precision": sum(def_macro_p) / n,
            "recall": sum(def_macro_r) / n,
            "f1": sum(def_macro_f1) / n,
        },
        "per_sample": per_sample,
        "warnings": sorted(set(all_warnings)),
        "matching_criteria": (
            "Case-insensitive, whitespace-normalized exact string match on entity text "
            "and definition concept (micro-averaged across all benchmark paragraphs)."
        ),
    }


def _topic_evaluation(suite: str = "demo") -> Dict[str, Any]:
    if suite == "benchmark":
        data = _load_json(BENCHMARK_DIR / "topics_benchmark.json")
        dataset_id = data["dataset_id"]
        description = data["description"]
        docs = [d["text"] for d in data["documents"]]
        domains = [d["domain"] for d in data["documents"]]
        n_topics = int(data.get("n_topics", 3))
    else:
        dataset_id = "demo_topics_v1"
        description = "DEMONSTRATION DATA ONLY. Topics discovered from a small demo corpus."
        docs = [
            "Machine learning is a subset of artificial intelligence. It uses data and algorithms.",
            "Natural language processing deals with text and human language. It includes tokenization.",
            "Deep learning uses neural networks with many layers. It is applied to images.",
            "Data preprocessing cleans raw data. It is essential for machine learning pipelines.",
        ]
        domains = None
        n_topics = 3

    r1 = run_lda(docs, n_topics=n_topics, random_state=42)
    r2 = run_lda(docs, n_topics=n_topics, random_state=42)

    topics = [
        {"topic_id": t.topic_id, "keywords": t.keywords, "weights": [float(w) for w in t.weights]}
        for t in r1.topics
    ]
    doc_topics = [
        {
            "chunk_id": d.chunk_id,
            "source_index": d.source_index,
            "domain": (domains[d.source_index] if domains and d.source_index < len(domains) else None),
            "dominant_topic": d.dominant_topic,
            "topic_distribution": [float(v) for v in d.topic_distribution],
        }
        for d in r1.document_topics
    ]

    # Optional clustering purity when ground-truth sub-domains exist
    purity: Optional[float] = None
    if domains:
        by_topic: Dict[int, Dict[str, int]] = {}
        for item in doc_topics:
            dom_topic = item["dominant_topic"]
            dom_label = item["domain"] or "unknown"
            by_topic.setdefault(dom_topic, {})
            by_topic[dom_topic][dom_label] = by_topic[dom_topic].get(dom_label, 0) + 1
        correct_majority = sum(max(counts.values()) for counts in by_topic.values() if counts)
        purity = correct_majority / len(doc_topics) if doc_topics else 0.0

    return {
        "dataset_id": dataset_id,
        "description": description,
        "document_count": len(docs),
        "n_topics": n_topics,
        "bow_matrix_shape": list(r1.bow_matrix_shape),
        "topics": topics,
        "document_topics": doc_topics,
        "vocabulary_size": r1.vocabulary_size,
        "perplexity": (float(r1.perplexity) if r1.perplexity is not None else None),
        "cluster_purity": purity,
        "reproducible": [t.keywords for t in r1.topics] == [t.keywords for t in r2.topics],
        "warnings": r1.warnings,
        "note": "Perplexity does not establish human interpretability or semantic coherence.",
    }


def run_evaluation(
    write_report: bool = True,
    suite: str = "demo",
    skip_llm: bool = False,
) -> Dict[str, Any]:
    if suite not in {"demo", "benchmark"}:
        raise ValueError(f"Unsupported evaluation suite {suite!r}. Choose 'demo' or 'benchmark'.")

    report = {
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "project": "Educational Content Summarization System",
        "suite": suite,
        "sample_counts": {},
        "results": {
            "summarization": _summarization_evaluation(suite=suite, skip_llm=skip_llm),
            "sentiment": _sentiment_evaluation(suite=suite),
            "concepts": _concept_evaluation(suite=suite),
            "topics": _topic_evaluation(suite=suite),
        },
        "missing_evaluation_data": [
            "LLM evaluation is skipped unless GEMINI_API_KEY is configured.",
        ],
        "limitations": [
            "Evaluation datasets are curated academic benchmarks rather than published third-party corpora.",
            "ROUGE uses whitespace tokenization and does not use stemming.",
            "Concept matching uses case-insensitive exact normalized string match.",
            "Perplexity is not semantic coherence.",
        ],
    }

    report["sample_counts"] = {
        "summarization": report["results"]["summarization"]["sample_count"],
        "sentiment": report["results"]["sentiment"]["sample_count"],
        "concepts": report["results"]["concepts"]["sample_count"],
        "topics": len(report["results"]["topics"]["document_topics"]),
    }

    if write_report:
        target_path = BENCHMARK_REPORT_PATH if suite == "benchmark" else REPORT_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NLP pipeline evaluation suite.")
    parser.add_argument(
        "--suite",
        choices=("demo", "benchmark"),
        default="demo",
        help="Dataset suite to evaluate (default: demo).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write JSON report to disk.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip live Gemini LLM summarization calls.",
    )
    args = parser.parse_args()
    report = run_evaluation(
        write_report=not args.no_write,
        suite=args.suite,
        skip_llm=args.skip_llm,
    )
    print(json.dumps(report["sample_counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
