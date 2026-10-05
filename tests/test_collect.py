"""Collection schema, anonymous storage, and four-way training manifests."""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from voxdetect.collect.app import create_app
from voxdetect.config import AudioConfig, ModelConfig
from voxdetect.importer import import_collected
from voxdetect.labels import CLASS_ORDER, LabelError, validate_submission
from voxdetect.model import VoiceCRNN
from voxdetect.train import fit


def wav_bytes(seconds: float, frequency: float = 220.0, sample_rate: int = 16000) -> bytes:
    count = max(1, int(seconds * sample_rate))
    timeline = np.arange(count, dtype=np.float32) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * frequency * timeline)).astype(np.float32)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def post_clip(client: TestClient, label: str, **fields) -> dict:
    payload = {"label": label, "consent": "yes", **fields}
    response = client.post(
        "/api/submissions",
        data=payload,
        files={"audio": ("Jane Doe.wav", wav_bytes(1.0, 180 + len(label) * 40), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_schema_keeps_english_out_of_other_languages(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    schema = client.get("/api/schema").json()
    assert [item["id"] for item in schema["labels"]] == list(CLASS_ORDER)
    codes = {language["code"] for language in schema["languages"]}
    assert "en" not in codes
    assert schema["unknown_language"]["code"] == "und"
    by_id = {item["id"]: item for item in schema["labels"]}
    assert len(by_id["silence"]["steps"]) >= 4
    assert any("do not speak" in step.lower() for step in by_id["silence"]["steps"])
    assert any("words" in step.lower() for step in by_id["noise"]["steps"])
    assert any("only english" in step.lower() for step in by_id["english"]["steps"])
    assert any("not sure" in step.lower() for step in by_id["non_english"]["steps"])
    assert "fan" in next(kind["summary"].lower() for kind in schema["noise_kinds"] if kind["id"] == "room")
    page = client.get("/")
    assert page.status_code == 200
    assert "Silence" in page.text
    assert "Other language" in page.text
    assert "no name field" in page.text


def test_label_rules_reject_mixed_fields():
    silence = validate_submission("silence", None, None, None, True)
    assert silence.language is None and silence.noise_kind is None
    english = validate_submission("english", "", None, "en-03", True)
    assert english.language == "en" and english.prompt_id == "en-03"
    other = validate_submission("non_english", "ES", None, "es-01", True)
    assert other.language == "es"
    unknown = validate_submission("non_english", "und", None, None, True)
    assert unknown.language == "und" and unknown.prompt_id is None

    with pytest.raises(LabelError):
        validate_submission("non_english", "en", None, None, True)
    with pytest.raises(LabelError):
        validate_submission("noise", None, None, None, True)
    with pytest.raises(LabelError):
        validate_submission("silence", "es", None, None, True)
    with pytest.raises(LabelError):
        validate_submission("english", None, None, None, False)


def test_submission_is_anonymous_and_filed_by_label(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    short = client.post(
        "/api/submissions",
        data={"label": "silence", "consent": "yes"},
        files={"audio": ("short.wav", wav_bytes(0.1), "audio/wav")},
    )
    assert short.status_code == 400
    refused = client.post(
        "/api/submissions",
        data={"label": "non_english", "consent": "yes"},
        files={"audio": ("clip.wav", wav_bytes(1.0), "audio/wav")},
    )
    assert refused.status_code == 400
    assert "language" in refused.json()["detail"].lower()

    stored = post_clip(client, "non_english", language="zh", prompt_id="zh-01")
    assert stored["label"] == "non_english"
    assert stored["language"] == "zh"
    assert "path" not in stored
    journal = (tmp_path / "submissions.jsonl").read_text(encoding="utf-8")
    assert "Jane" not in journal
    assert "Doe" not in journal
    record = json.loads(journal.strip())
    audio = tmp_path / record["path"]
    assert audio.is_file()
    assert audio.parent.name == "zh"
    assert audio.parent.parent.name == "non_english"

    noise = post_clip(client, "noise", noise_kind="traffic")
    assert (tmp_path / "audio" / "noise" / "traffic" / f"{noise['submission_id']}.wav").is_file()
    stats = client.get("/api/stats").json()
    assert stats["by_label"]["non_english"] == 1
    assert stats["by_label"]["noise"] == 1
    assert stats["by_language"] == {"zh": 1}
    assert "submission_id" not in stats

    deleted = client.delete(f"/api/submissions/{stored['submission_id']}")
    assert deleted.status_code == 200
    assert not audio.exists()
    assert stored["submission_id"] not in (tmp_path / "submissions.jsonl").read_text(encoding="utf-8")


def test_collected_catalog_imports_without_collapsing_labels(tmp_path: Path):
    root = tmp_path / "collected"
    client = TestClient(create_app(root))
    post_clip(client, "silence")
    post_clip(client, "silence")
    post_clip(client, "english", prompt_id="en-02")
    post_clip(client, "english", prompt_id="en-05")
    post_clip(client, "noise", noise_kind="music")
    post_clip(client, "non_english", language="es", prompt_id="es-01")

    report = import_collected(root, out_dir=tmp_path / "manifests", seed=3)
    names = {row.label_name for row in report.accepted()}
    assert names == {"silence", "noise", "english", "non_english"}
    spanish = [row for row in report.accepted() if row.language == "es"]
    assert len(spanish) == 1
    assert spanish[0].label == CLASS_ORDER.index("non_english")
    assert spanish[0].label != CLASS_ORDER.index("noise")
    header = (tmp_path / "manifests" / "train.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "label_name" in header and "language" in header

    result = fit(
        train_manifest=tmp_path / "manifests" / "train.csv",
        val_manifest=tmp_path / "manifests" / "val.csv",
        out_path=tmp_path / "best.pt",
        epochs=1,
        batch_size=4,
        seed=3,
    )
    blob = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert blob["label_names"] == list(CLASS_ORDER)
    assert blob["model_config"]["num_classes"] == 4


def test_four_class_head_shape():
    config = ModelConfig.from_audio(AudioConfig(), num_classes=4)
    model = VoiceCRNN(config)
    features = torch.zeros(2, 1, config.n_mels, config.n_frames)
    logits = model(features)
    assert logits.shape == (2, 4)

