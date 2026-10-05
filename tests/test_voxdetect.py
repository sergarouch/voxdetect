"""Unit and pipeline tests for import, features, and a short training run."""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from voxdetect.audio import crop_or_pad, iter_windows
from voxdetect.cli import main
from voxdetect.config import AudioConfig, ModelConfig
from voxdetect.evaluate import evaluate
from voxdetect.features import LogMelExtractor
from voxdetect.importer import DatasetImportError, import_dataset, parse_label, read_manifest
from voxdetect.infer import infer_paths
from voxdetect.metrics import binary_metrics
from voxdetect.model import VoiceCRNN
from voxdetect.synthetic import write_synthetic
from voxdetect.train import fit

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


def write_wav(path: Path, seconds: float, frequency: float, sample_rate: int = 16000) -> None:
    count = int(seconds * sample_rate)
    timeline = np.arange(count, dtype=np.float32) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * frequency * timeline)).astype(np.float32)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_parse_label_accepts_names_and_bits():
    assert parse_label("speech") == 1
    assert parse_label("Voice") == 1
    assert parse_label("1") == 1
    assert parse_label("non-speech") == 0
    assert parse_label("noise") == 0
    assert parse_label("0") == 0
    with pytest.raises(ValueError):
        parse_label("maybe")


def test_folder_import_splits_and_rejects(tmp_path: Path):
    root = tmp_path / "raw"
    for index in range(5):
        write_wav(root / "speech" / f"s{index}.wav", 1.0, 180 + index * 20)
        write_wav(root / "noise" / f"n{index}.wav", 1.0, 40)
    write_wav(root / "speech" / "short.wav", 0.1, 200)
    write_wav(root / "loose.wav", 1.0, 220)
    (root / "speech" / "notes.txt").write_text("not audio", encoding="utf-8")
    (root / "speech" / "broken.wav").write_text("not a wav", encoding="utf-8")

    report = import_dataset(source=root, out_dir=tmp_path / "manifests", seed=7)
    assert len(report.accepted()) == 10
    assert {row.label for row in report.train} == {0, 1}
    assert len(report.val) == 2
    assert len(report.test) == 2
    reasons = " ".join(reason for _path, reason in report.rejected)
    assert "too_short" in reasons
    assert "unlabeled" in reasons
    assert "unreadable" in reasons

    again = import_dataset(source=root, out_dir=tmp_path / "manifests2", seed=7)
    assert [row.path for row in report.train] == [row.path for row in again.train]
    stored = read_manifest(tmp_path / "manifests" / "train.csv")
    assert stored
    assert Path(stored[0].path).is_absolute()


def test_csv_import_resolves_paths_relative_to_the_csv(tmp_path: Path):
    clips = tmp_path / "clips"
    write_wav(clips / "hello.wav", 1.0, 200)
    write_wav(clips / "room.wav", 1.0, 50)
    write_wav(clips / "street.mp3", 1.0, 60)
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "path,label\n"
        "clips/hello.wav,speech\n"
        "clips/room.wav,0\n"
        "clips/missing.wav,speech\n"
        "clips/street.mp3,banana\n",
        encoding="utf-8",
    )
    report = import_dataset(csv_path=labels, out_dir=tmp_path / "manifests")
    assert len(report.accepted()) == 2
    assert any(reason.startswith("missing") for _path, reason in report.rejected)
    assert any(reason.startswith("bad_label") for _path, reason in report.rejected)


def test_import_requires_both_classes(tmp_path: Path):
    root = tmp_path / "only_speech"
    write_wav(root / "speech" / "a.wav", 1.0, 180)
    with pytest.raises(DatasetImportError):
        import_dataset(source=root, out_dir=tmp_path / "out")


def test_log_mel_shape_and_model_forward():
    config = AudioConfig()
    waveform = torch.sin(torch.linspace(0, 80, config.clip_samples))
    features = LogMelExtractor(config)(waveform)
    assert features.shape == (1, config.n_mels, config.n_frames)
    model = VoiceCRNN(ModelConfig.from_audio(config))
    logits = model(features.unsqueeze(0))
    assert logits.shape == (1,)
    assert torch.isfinite(logits)


def test_windows_cover_a_longer_clip_without_duplicating_the_tail():
    window = 8
    waveform = torch.arange(20)
    chunks = iter_windows(waveform, window, hop=window // 2)
    assert torch.equal(chunks[0], waveform[:8])
    assert torch.equal(chunks[-1], waveform[-8:])
    assert len(chunks) == len({int(chunk[0]) for chunk in chunks})
    padded = crop_or_pad(torch.ones(3), 8, randomize=False)
    assert padded.shape == (8,)
    assert padded.sum() == 3


def test_binary_metrics_perfect_split():
    logits = torch.tensor([4.0, 4.0, -4.0, -4.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    metrics = binary_metrics(logits, labels)
    assert metrics["accuracy"] == 1
    assert metrics["f1"] == 1


def test_cli_trains_on_synthetic_audio_and_scores_a_file(tmp_path: Path):
    raw = write_synthetic(tmp_path / "raw", clips_per_class=6, seed=1)
    assert main(["import", "--input", str(raw), "--out", str(tmp_path / "manifests"), "--seed", "1"]) == 0
    checkpoint = tmp_path / "best.pt"
    result = fit(
        train_manifest=tmp_path / "manifests" / "train.csv",
        val_manifest=tmp_path / "manifests" / "val.csv",
        out_path=checkpoint,
        epochs=3,
        batch_size=4,
        seed=1,
    )
    assert Path(result["checkpoint"]).is_file()
    metrics = evaluate(checkpoint, tmp_path / "manifests" / "test.csv", batch_size=4)
    assert metrics["count"] > 0
    scored = infer_paths(checkpoint, raw / "speech" / "speech_00.wav")
    assert len(scored) == 1
    assert 0.0 <= scored[0][1] <= 1.0
    assert main(["infer", "--checkpoint", str(checkpoint), "--audio", str(raw / "non_speech")]) == 0
