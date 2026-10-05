"""Write anonymous clips into a catalog the trainer can import.

Files are named by submission id. The contributor's original filename is ignored.
"""

from __future__ import annotations

import csv
import fcntl
import json
import subprocess
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

from voxdetect.audio import require_ffmpeg
from voxdetect.labels import (
    CLASS_ORDER,
    MAX_SECONDS,
    MIN_SECONDS,
    SAMPLE_RATE,
    SCHEMA_VERSION,
    LabelError,
    NormalizedSubmission,
    public_schema,
)

CSV_FIELDS = (
    "submission_id",
    "label",
    "language",
    "noise_kind",
    "prompt_id",
    "duration_sec",
    "sample_rate",
    "created_at",
    "path",
    "schema_version",
)


def ensure_layout(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "audio").mkdir(exist_ok=True)
    schema_path = root / "schema.json"
    schema_path.write_text(json.dumps(public_schema(), indent=2) + "\n", encoding="utf-8")
    journal = root / "submissions.jsonl"
    if not journal.exists():
        journal.write_text("", encoding="utf-8")
    _rewrite_csv(root, _read_jsonl(journal))


def save_submission(root: Path, audio_bytes: bytes, normalized: NormalizedSubmission) -> dict:
    """Decode audio to 16 kHz mono wav and append one catalog record."""
    if not audio_bytes:
        raise LabelError("The recording was empty.")
    ensure_layout(root)
    submission_id = str(uuid.uuid4())
    relative = _relative_path(submission_id, normalized)
    destination = root / relative
    if not destination.resolve().is_relative_to(root.resolve()):
        raise LabelError("Could not place that clip in the catalog.")

    temp_dir = root / ".tmp"
    temp_dir.mkdir(exist_ok=True)
    source = temp_dir / f"{submission_id}.upload"
    source.write_bytes(audio_bytes)
    committed = False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _transcode(source, destination)
        duration = _wav_duration(destination)
        if duration < MIN_SECONDS:
            raise LabelError(f"Clip is {duration:.2f}s. Record at least {MIN_SECONDS:.1f}s.")
        if duration > MAX_SECONDS:
            raise LabelError(f"Clip is {duration:.1f}s. Keep recordings under {MAX_SECONDS:.0f}s.")
        record = {
            "schema_version": SCHEMA_VERSION,
            "submission_id": submission_id,
            "label": normalized.label,
            "language": normalized.language,
            "noise_kind": normalized.noise_kind,
            "prompt_id": normalized.prompt_id,
            "duration_sec": round(duration, 4),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "format": "wav",
            "consent": True,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "path": relative.as_posix(),
        }
        _append_record(root, record)
        committed = True
        return record
    except Exception:
        if not committed:
            destination.unlink(missing_ok=True)
        raise
    finally:
        source.unlink(missing_ok=True)


def delete_submission(root: Path, submission_id: str) -> bool:
    """Remove one clip by the id returned at submit time. Unknown ids do nothing."""
    if not _is_uuid(submission_id):
        raise LabelError("That submission id is not valid.")
    ensure_layout(root)
    with _locked(root):
        records = _read_jsonl(root / "submissions.jsonl")
        kept = [record for record in records if record.get("submission_id") != submission_id]
        if len(kept) == len(records):
            return False
        removed = next(record for record in records if record.get("submission_id") == submission_id)
        audio = root / str(removed.get("path", ""))
        if audio.is_file() and audio.resolve().is_relative_to(root.resolve()):
            audio.unlink()
        _write_jsonl(root / "submissions.jsonl", kept)
        _rewrite_csv(root, kept)
        return True


def collection_stats(root: Path) -> dict:
    records = _read_jsonl(root / "submissions.jsonl") if (root / "submissions.jsonl").is_file() else []
    by_label = {label: 0 for label in CLASS_ORDER}
    by_language: dict[str, int] = {}
    by_noise: dict[str, int] = {}
    for record in records:
        label = record.get("label")
        if label in by_label:
            by_label[label] += 1
        if label == "non_english" and record.get("language"):
            code = str(record["language"])
            by_language[code] = by_language.get(code, 0) + 1
        if label == "noise" and record.get("noise_kind"):
            kind = str(record["noise_kind"])
            by_noise[kind] = by_noise.get(kind, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "total": len(records),
        "by_label": by_label,
        "by_language": by_language,
        "by_noise_kind": by_noise,
    }


def load_catalog(root: Path) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Return valid records plus (record, reason) pairs for rows that cannot train."""
    journal = root / "submissions.jsonl"
    if not journal.is_file():
        raise FileNotFoundError(
            f"No collection catalog at {root}. Start the site with `voxdetect collect` and gather clips first."
        )
    valid: list[dict] = []
    problems: list[tuple[dict, str]] = []
    seen: set[str] = set()
    for record in _read_jsonl(journal):
        submission_id = str(record.get("submission_id") or "")
        if submission_id in seen:
            problems.append((record, "duplicate: submission already counted"))
            continue
        seen.add(submission_id)
        label = record.get("label")
        if label not in CLASS_ORDER:
            problems.append((record, f"bad_label: {label}"))
            continue
        relative = str(record.get("path") or "")
        audio = (root / relative).resolve()
        if not relative or not audio.is_file() or not audio.is_relative_to(root.resolve()):
            problems.append((record, "missing: audio file not in the catalog"))
            continue
        item = dict(record)
        item["absolute_path"] = str(audio)
        valid.append(item)
    return valid, problems


def _relative_path(submission_id: str, normalized: NormalizedSubmission) -> Path:
    filename = f"{submission_id}.wav"
    if normalized.label == "silence":
        return Path("audio") / "silence" / filename
    if normalized.label == "noise":
        return Path("audio") / "noise" / str(normalized.noise_kind) / filename
    if normalized.label == "english":
        return Path("audio") / "english" / filename
    return Path("audio") / "non_english" / str(normalized.language) / filename


def _transcode(source: Path, destination: Path) -> None:
    ffmpeg = require_ffmpeg()
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "wav",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0 or not destination.is_file():
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LabelError(detail or "Could not read that audio. Try wav, flac, mp3, or a browser recording.")


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate() or SAMPLE_RATE
        return handle.getnframes() / rate


def _append_record(root: Path, record: dict) -> None:
    with _locked(root):
        journal = root / "submissions.jsonl"
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        records = _read_jsonl(journal)
        _rewrite_csv(root, records)


def _rewrite_csv(root: Path, records: list[dict]) -> None:
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: "" if record.get(field) is None else record.get(field) for field in CSV_FIELDS})


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def _locked(root: Path):
    class _Lock:
        def __enter__(self):
            self.handle = (root / ".lock").open("a", encoding="utf-8")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
            return self.handle

        def __exit__(self, exc_type, exc, tb):
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    return _Lock()


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def public_record(record: dict) -> dict:
    """Fields safe to show back to the contributor. The server path stays local."""
    return {
        "schema_version": record["schema_version"],
        "submission_id": record["submission_id"],
        "label": record["label"],
        "language": record["language"],
        "noise_kind": record["noise_kind"],
        "prompt_id": record["prompt_id"],
        "duration_sec": record["duration_sec"],
        "sample_rate": record["sample_rate"],
        "created_at": record["created_at"],
    }

