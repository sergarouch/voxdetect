"""Score one file or every audio file in a folder."""

from __future__ import annotations

from pathlib import Path

import torch

from voxdetect.audio import AUDIO_EXTENSIONS, iter_windows, load_audio
from voxdetect.evaluate import load_checkpoint
from voxdetect.features import LogMelExtractor
from voxdetect.labels import CLASS_ORDER


def score_file(
    model: torch.nn.Module,
    extractor: LogMelExtractor,
    path: Path,
    device: torch.device,
) -> torch.Tensor:
    """Mean probability for each class across 1-second windows with a 0.5 second hop."""
    config = extractor.config
    waveform = load_audio(path, config.sample_rate)
    windows = iter_windows(waveform, config.clip_samples, max(1, config.clip_samples // 2))
    batch = torch.stack([extractor(window).squeeze(0) for window in windows]).to(device)
    with torch.no_grad():
        logits = model(batch)
    if getattr(model.config, "num_classes", 1) == 1:
        return torch.sigmoid(logits).mean().reshape(1)
    return torch.softmax(logits, dim=-1).mean(dim=0)


def infer_paths(
    checkpoint: str | Path,
    audio: str | Path,
    threshold: float | None = None,
) -> list[tuple[Path, float, str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, audio_config, blob = load_checkpoint(checkpoint, device)
    cutoff = float(blob.get("threshold", 0.5) if threshold is None else threshold)
    extractor = LogMelExtractor(audio_config)
    extractor.eval()
    paths = _audio_paths(Path(audio))
    if not paths:
        raise FileNotFoundError(f"No audio files at {audio}")

    names = list(blob.get("label_names") or CLASS_ORDER)
    multiclass = model.config.num_classes > 1
    results: list[tuple[Path, float, str]] = []
    for path in paths:
        probabilities = score_file(model, extractor, path, device)
        if multiclass:
            index = int(probabilities.argmax())
            label = names[index] if index < len(names) else str(index)
            probability = float(probabilities[index])
            breakdown = "  ".join(f"{name} {float(probabilities[i]):.2f}" for i, name in enumerate(names))
            print(f"{label:12}  {probability:.3f}  {breakdown}  {path}")
        else:
            probability = float(probabilities[0])
            label = "speech" if probability >= cutoff else "non_speech"
            print(f"{probability:.3f}  {label:10}  {path}")
        results.append((path, probability, label))
    if multiclass:
        counts = {name: sum(label == name for _path, _prob, label in results) for name in names}
        rendered = ", ".join(f"{name} {count}" for name, count in counts.items())
        print(rendered)
    else:
        speech = sum(label == "speech" for _path, _prob, label in results)
        print(f"{speech} speech, {len(results) - speech} non_speech, threshold {cutoff:.2f}")
    return results


def _audio_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"Audio path not found: {target}")
    return sorted(
        path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
