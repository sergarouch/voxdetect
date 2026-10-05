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


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"accuracy {metrics['accuracy']:.3f}  "
        f"precision {metrics['precision']:.3f}  "
        f"recall {metrics['recall']:.3f}  "
        f"f1 {metrics['f1']:.3f}"
    )
