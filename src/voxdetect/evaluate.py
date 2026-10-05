"""Score a held-out manifest with a saved checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from voxdetect.config import AudioConfig, ModelConfig
from voxdetect.dataset import VoiceDataset
from voxdetect.metrics import binary_metrics, format_metrics, multiclass_metrics
from voxdetect.model import VoiceCRNN


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[VoiceCRNN, AudioConfig, dict]:
    try:
        blob = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    except TypeError:
        blob = torch.load(Path(path), map_location=device or "cpu")
    audio_config = AudioConfig.from_dict(blob["audio_config"])
    model_config = ModelConfig.from_dict(blob["model_config"])
    model = VoiceCRNN(model_config)
    model.load_state_dict(blob["model_state"])
    model.to(device or "cpu")
    model.eval()
    return model, audio_config, blob


def evaluate(checkpoint: str | Path, manifest: str | Path, batch_size: int = 16, threshold: float | None = None) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, audio_config, blob = load_checkpoint(checkpoint, device)
    cutoff = blob.get("threshold", 0.5) if threshold is None else threshold
    dataset = VoiceDataset(manifest, audio_config, train=False)
    if (model.config.num_classes > 1) != dataset.multiclass:
        raise ValueError("The checkpoint label scheme does not match this manifest.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss() if dataset.multiclass else nn.BCEWithLogitsLoss()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    losses: list[float] = []
    with torch.no_grad():
        for features, batch_labels in loader:
            features = features.to(device)
            batch_labels = batch_labels.to(device)
            batch_logits = model(features)
            losses.append(float(criterion(batch_logits, batch_labels)))
            logits.append(batch_logits.cpu())
            labels.append(batch_labels.cpu())
    stacked = torch.cat(logits)
    truth = torch.cat(labels)
    if dataset.multiclass:
        metrics = multiclass_metrics(stacked, truth)
    else:
        metrics = binary_metrics(stacked, truth, threshold=cutoff)
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    print(f"{manifest}: loss {metrics['loss']:.4f}  {format_metrics(metrics)}  n {int(metrics['count'])}")
    return metrics
