"""Scan a folder or a label CSV and write train, validation, and test manifests."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

from voxdetect.audio import AUDIO_EXTENSIONS, AudioReadError, load_audio
from voxdetect.config import AudioConfig

SPEECH_NAMES = {"speech", "voice", "positive", "vad"}
NON_SPEECH_NAMES = {"non_speech", "nonspeech", "noise", "silence", "negative", "background"}
MANIFEST_FIELDS = ("path", "label", "duration_sec", "sample_rate")


@dataclass
class ClipRow:
    path: str
    label: int
    duration_sec: float
    sample_rate: int


class DatasetImportError(Exception):
    """The folder or CSV did not produce a usable two-class dataset."""


@dataclass
class ImportReport:
    train: list[ClipRow] = field(default_factory=list)
    val: list[ClipRow] = field(default_factory=list)
    test: list[ClipRow] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    out_dir: Path | None = None

    def accepted(self) -> list[ClipRow]:
        return [*self.train, *self.val, *self.test]

    def summary(self) -> str:
        lines = [
            f"Accepted {len(self.accepted())} clips "
            f"(speech { _count(self.accepted(), 1) }, non_speech {_count(self.accepted(), 0)}).",
            f"Rejected {len(self.rejected)} files.",
        ]
        reasons: dict[str, int] = {}
        for _path, reason in self.rejected:
            category = reason.split(":", 1)[0]
            reasons[category] = reasons.get(category, 0) + 1
        for reason, count in sorted(reasons.items()):
            lines.append(f"  {reason}: {count}")
        for name, rows in (("train", self.train), ("val", self.val), ("test", self.test)):
            lines.append(
                f"{name}: {len(rows)} (speech {_count(rows, 1)}, non_speech {_count(rows, 0)})"
            )
        if self.out_dir is not None:
            lines.append(f"Manifests written to {self.out_dir}")
        accepted = self.accepted()
        if accepted:
            majority = max(_count(accepted, 1), _count(accepted, 0)) / len(accepted)
            if majority >= 0.9:
                lines.append(
                    "Warning: one class is at least 90% of the clips. "
                    "Accuracy can look high while the model ignores the smaller class."
                )
        for name, rows in (("val", self.val), ("test", self.test)):
            if rows and (_count(rows, 0) == 0 or _count(rows, 1) == 0):
                lines.append(f"Note: {name} contains only one class, so metrics on it will be thin.")
        return "\n".join(lines)


def import_dataset(
    source: str | Path | None = None,
    csv_path: str | Path | None = None,
    out_dir: str | Path = "data/manifests",
    config: AudioConfig | None = None,
    seed: int = 42,
) -> ImportReport:
    """Decode audio, drop unusable files, and write `train.csv`, `val.csv`, and `test.csv`.

    Pass `source` for a folder whose class is the directory name (`speech` or
    `non_speech`), or `csv_path` for a file with `path` and `label` columns.
    Paths in the CSV may be absolute or relative to the CSV itself.
    """
    if (source is None) == (csv_path is None):
        raise ValueError("Pass exactly one of source folder or csv_path")

    audio_config = config or AudioConfig()
    labeled, rejected = _collect(source, csv_path)
    accepted: list[ClipRow] = []
    for path, label in labeled:
        try:
            waveform = load_audio(path, audio_config.sample_rate)
        except AudioReadError as exc:
            rejected.append((str(path), f"unreadable: {exc}"))
            continue
        duration = waveform.numel() / audio_config.sample_rate
        if duration < audio_config.min_seconds:
            rejected.append((str(path), f"too_short: {duration:.3f}s < {audio_config.min_seconds:.3f}s"))
            continue
        accepted.append(
            ClipRow(
                path=str(Path(path).resolve()),
                label=label,
                duration_sec=round(duration, 4),
                sample_rate=audio_config.sample_rate,
            )
        )

    if not accepted:
        report = ImportReport(rejected=rejected)
        raise DatasetImportError(
            "No usable clips. Every file was rejected or no audio was found.\n" + report.summary()
        )
    if _count(accepted, 0) == 0 or _count(accepted, 1) == 0:
        report = ImportReport(rejected=rejected)
        present = "speech" if _count(accepted, 1) else "non_speech"
        raise DatasetImportError(
            f"Need both classes. Only {present} clips were accepted.\n" + report.summary()
        )

    splits = _split(accepted, seed)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_manifest(destination / "train.csv", splits["train"])
    _write_manifest(destination / "val.csv", splits["val"])
    _write_manifest(destination / "test.csv", splits["test"])
    _write_rejected(destination / "rejected.csv", rejected)
    return ImportReport(
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        rejected=rejected,
        out_dir=destination.resolve(),
    )


def read_manifest(path: str | Path) -> list[ClipRow]:
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}. Import audio first with `voxdetect import`."
        )
    rows: list[ClipRow] = []
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{manifest} must have path and label columns")
        for record in reader:
            rows.append(
                ClipRow(
                    path=record["path"],
                    label=int(record["label"]),
                    duration_sec=float(record.get("duration_sec") or 0),
                    sample_rate=int(record.get("sample_rate") or 0),
                )
            )
    return rows


def parse_label(value: str) -> int:
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    if text in SPEECH_NAMES or text in {"1", "yes", "true"}:
        return 1
    if text in NON_SPEECH_NAMES or text in {"0", "no", "false"}:
        return 0
    raise ValueError(
        f"Unknown label {value!r}. Use speech/non_speech or 1/0."
    )


def _collect(
    source: str | Path | None,
    csv_path: str | Path | None,
) -> tuple[list[tuple[Path, int]], list[tuple[str, str]]]:
    if csv_path is not None:
        return _collect_csv(Path(csv_path))
    return _collect_folder(Path(source).expanduser().resolve())  # type: ignore[arg-type]


def _collect_folder(root: Path) -> tuple[list[tuple[Path, int]], list[tuple[str, str]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {root}")
    labeled: list[tuple[Path, int]] = []
    rejected: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        label = _label_from_parents(path, root)
        if label is None:
            rejected.append((str(path), "unlabeled: place the file under a speech/ or non_speech/ folder"))
            continue
        labeled.append((path, label))
    return labeled, rejected


def _collect_csv(csv_file: Path) -> tuple[list[tuple[Path, int]], list[tuple[str, str]]]:
    if not csv_file.is_file():
        raise FileNotFoundError(f"Label CSV not found: {csv_file}")
    labeled: list[tuple[Path, int]] = []
    rejected: list[tuple[str, str]] = []
    with csv_file.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{csv_file} must have a header with path and label columns")
        for record in reader:
            raw_path = (record.get("path") or "").strip()
            raw_label = record.get("label")
            if not raw_path:
                rejected.append(("", "missing: blank path"))
                continue
            path = _resolve_csv_path(raw_path, csv_file)
            if raw_label is None or not str(raw_label).strip():
                rejected.append((str(path), "bad_label: empty"))
                continue
            try:
                label = parse_label(str(raw_label))
            except ValueError as exc:
                rejected.append((str(path), f"bad_label: {exc}"))
                continue
            if not path.is_file():
                rejected.append((str(path), "missing: file not found"))
                continue
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                rejected.append((str(path), f"unsupported: {path.suffix or 'no extension'}"))
                continue
            labeled.append((path, label))
    return labeled, rejected


def _resolve_csv_path(raw: str, csv_file: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = csv_file.parent / path
    return path.resolve()


def _label_from_parents(path: Path, root: Path) -> int | None:
    current = path.parent
    while True:
        name = current.name.lower().replace("-", "_").replace(" ", "_")
        if name in SPEECH_NAMES:
            return 1
        if name in NON_SPEECH_NAMES:
            return 0
        if current.resolve() == root.resolve() or current.parent == current:
            return None
        current = current.parent


def _split(rows: list[ClipRow], seed: int) -> dict[str, list[ClipRow]]:
    groups: dict[int, list[ClipRow]] = {0: [], 1: []}
    for row in rows:
        groups[row.label].append(row)
    rng = random.Random(seed)
    splits: dict[str, list[ClipRow]] = {"train": [], "val": [], "test": []}
    for group in groups.values():
        rng.shuffle(group)
        n_train, n_val, n_test = _split_counts(len(group))
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val : n_train + n_val + n_test])
    for bucket in splits.values():
        rng.shuffle(bucket)
    return splits


def _split_counts(count: int) -> tuple[int, int, int]:
    if count <= 2:
        n_val = 1 if count == 2 else 0
        return count - n_val, n_val, 0
    n_val = max(1, round(count * 0.1))
    n_test = max(1, round(count * 0.1))
    n_train = count - n_val - n_test
    if n_train >= 1:
        return n_train, n_val, n_test
    leftover = count - 1
    n_val = leftover // 2
    n_test = leftover - n_val
    return 1, n_val, n_test


def _write_manifest(path: Path, rows: list[ClipRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row.path,
                    "label": row.label,
                    "duration_sec": f"{row.duration_sec:.4f}",
                    "sample_rate": row.sample_rate,
                }
            )


def _write_rejected(path: Path, rejected: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "reason"))
        writer.writeheader()
        for item_path, reason in rejected:
            writer.writerow({"path": item_path, "reason": reason})


def _count(rows: list[ClipRow], label: int) -> int:
    return sum(row.label == label for row in rows)
