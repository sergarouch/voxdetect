"""Binary classification metrics for voice vs non-voice."""

from __future__ import annotations

import torch


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, precision, recall, and F1. Positive class is voice (label 1)."""
    predictions = (torch.sigmoid(logits) >= threshold).to(dtype=torch.int64)
    truth = labels.to(dtype=torch.int64)
    true_positive = int(((predictions == 1) & (truth == 1)).sum())
    true_negative = int(((predictions == 0) & (truth == 0)).sum())
    false_positive = int(((predictions == 1) & (truth == 0)).sum())
    false_negative = int(((predictions == 0) & (truth == 1)).sum())
    total = true_positive + true_negative + false_positive + false_negative
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count": float(total),
    }


def multiclass_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Accuracy plus macro precision, recall, and F1 across the classes present."""
    predictions = logits.argmax(dim=-1)
    truth = labels.to(dtype=torch.int64)
    total = int(truth.numel())
    correct = int((predictions == truth).sum())
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    classes = sorted(set(int(value) for value in truth.tolist()) | set(int(value) for value in predictions.tolist()))
    for class_id in classes:
        true_positive = int(((predictions == class_id) & (truth == class_id)).sum())
        false_positive = int(((predictions == class_id) & (truth != class_id)).sum())
        false_negative = int(((predictions != class_id) & (truth == class_id)).sum())
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return {
        "accuracy": correct / total if total else 0.0,
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "count": float(total),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"accuracy {metrics['accuracy']:.3f}  "
        f"precision {metrics['precision']:.3f}  "
        f"recall {metrics['recall']:.3f}  "
        f"f1 {metrics['f1']:.3f}"
    )
