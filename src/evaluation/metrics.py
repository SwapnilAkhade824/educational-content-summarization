"""Evaluation metrics (ROUGE, classification, concept overlap)."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


def _tokens(text: str) -> List[str]:
    return str(text).lower().split()


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    # Dynamic programming LCS.
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_n(prediction: str, reference: str, n: int) -> Dict[str, float]:
    """Compute ROUGE-N precision/recall/F1."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    pred_ngrams = Counter(_ngrams(pred_tokens, n))
    ref_ngrams = Counter(_ngrams(ref_tokens, n))

    overlap = sum((pred_ngrams & ref_ngrams).values())
    pred_total = sum(pred_ngrams.values())
    ref_total = sum(ref_ngrams.values())

    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-L (LCS-based) precision/recall/F1."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_scores(prediction: str, reference: str) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L."""
    return {
        "rouge_1": rouge_n(prediction, reference, 1),
        "rouge_2": rouge_n(prediction, reference, 2),
        "rouge_l": rouge_l(prediction, reference),
    }


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> List[List[int]]:
    """Build a confusion matrix (rows=true, cols=predicted)."""
    label_index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0] * size for _ in range(size)]
    for t, p in zip(y_true, y_pred):
        if t in label_index and p in label_index:
            matrix[label_index[t]][label_index[p]] += 1
    return matrix


def classification_metrics(
    y_pred: Sequence[str], y_true: Sequence[str]
) -> Dict[str, object]:
    """Compute accuracy, precision, recall, F1, and confusion matrix."""
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred and y_true must have the same length.")

    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
    accuracy = correct / total if total else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for label in labels:
        tp = sum(1 for p, t in zip(y_pred, y_true) if p == label and t == label)
        fp = sum(1 for p, t in zip(y_pred, y_true) if p == label and t != label)
        fn = sum(1 for p, t in zip(y_pred, y_true) if p != label and t == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == label),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    cm = confusion_matrix(y_true, y_pred, labels)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels,
    }


def _normalize_concept(value: str) -> str:
    return " ".join(str(value).lower().split())


def concept_metrics(predicted: Iterable[str], reference: Iterable[str]) -> Dict[str, float]:
    """Compute precision/recall/F1 over normalized concept sets."""
    pred_set = {_normalize_concept(x) for x in predicted}
    ref_set = {_normalize_concept(x) for x in reference}

    if not pred_set and not ref_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "true_positives": 0}

    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(ref_set) if ref_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
    }
