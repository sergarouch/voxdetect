"""Training loop for the voice-activity CRNN."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from voxdetect.config import AudioConfig, ModelConfig
from voxdetect.dataset import VoiceDataset
from voxdetect.importer import read_manifest
from voxdetect.metrics import binary_metrics, format_metrics
from voxdetect.model import VoiceCRNN


def fit(
    train_manifest: str | Path,
    out_path: str | Path,
    val_manifest: str | Path | None = None,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
    audio_config: AudioConfig | None = None,
) -> dict:
    """Train and write the checkpoint with the best validation F1 (or the last epoch)."""
    torch.manual_seed(seed)
    config = audio_config or AudioConfig()
    model_config = ModelConfig.from_audio(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = VoiceDataset(train_manifest, config, train=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_manifest is not None and Path(val_manifest).is_file() and read_manifest(val_manifest):
        val_set = VoiceDataset(val_manifest, config, train=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    labels = [row.label for row in train_set.rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training manifest needs both speech and non_speech clips.")
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)

    model = VoiceCRNN(model_config).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_f1 = -1.0
    best_payload: dict | None = None
    history: list[dict] = []
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Device {device}  train {len(train_set)} "
        f"(speech {positives}, non_speech {negatives})  "
        f"val {0 if val_loader is None else len(val_loader.dataset)}"
    )
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        row = {"epoch": epoch, "train_loss": train_loss}
        message = f"epoch {epoch:02d}  train_loss {train_loss:.4f}"
        save = val_loader is None
        if val_loader is not None:
            val_loss, val_metrics = _evaluate_loader(model, val_loader, criterion, device)
            row["val_loss"] = val_loss
            row["val_metrics"] = val_metrics
            message += f"  val_loss {val_loss:.4f}  {format_metrics(val_metrics)}"
            save = val_metrics["f1"] >= best_f1
            if save:
                best_f1 = val_metrics["f1"]
        if save:
            best_payload = _checkpoint(model, config, model_config, epoch, row)
            torch.save(best_payload, destination)
            message += "  saved"
        print(message)
        history.append(row)

    if best_payload is None:
        raise RuntimeError("Training produced no checkpoint")
    return {"checkpoint": str(destination.resolve()), "history": history, "device": str(device)}


def _run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total = 0.0
    seen = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * labels.shape[0]
        seen += int(labels.shape[0])
    return total / max(seen, 1)


@torch.no_grad()
def _evaluate_loader(model, loader, criterion, device) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for features, batch_labels in loader:
        features = features.to(device)
        batch_labels = batch_labels.to(device)
        batch_logits = model(features)
        losses.append(criterion(batch_logits, batch_labels).detach().cpu())
        logits.append(batch_logits.detach().cpu())
        labels.append(batch_labels.detach().cpu())
    loss = float(torch.stack(losses).mean()) if losses else 0.0
    metrics = binary_metrics(torch.cat(logits), torch.cat(labels))
    return loss, metrics


def _checkpoint(model, audio_config: AudioConfig, model_config: ModelConfig, epoch: int, row: dict) -> dict:
    return {
        "model_state": model.state_dict(),
        "audio_config": audio_config.to_dict(),
        "model_config": model_config.to_dict(),
        "epoch": epoch,
        "metrics": row,
        "threshold": 0.5,
    }
