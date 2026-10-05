"""Manifest-backed dataset of one-second log-mel clips."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from voxdetect.audio import crop_or_pad, load_audio
from voxdetect.config import AudioConfig
from voxdetect.features import LogMelExtractor
from voxdetect.importer import ClipRow, read_manifest


class VoiceDataset(Dataset):
    """Load a manifest row, crop or pad it to one second, and return log-mel features."""

    def __init__(
        self,
        manifest: str | Path,
        config: AudioConfig | None = None,
        train: bool = False,
    ) -> None:
        self.rows = read_manifest(manifest)
        if not self.rows:
            raise ValueError(f"Manifest is empty: {manifest}")
        self.config = config or AudioConfig()
        self.train = train
        self.extractor = LogMelExtractor(self.config)
        self.extractor.eval()
        named = [bool(row.label_name) for row in self.rows]
        if any(named) and not all(named):
            raise ValueError("Manifest mixes speech/non-speech labels with silence/noise/english/non_english labels.")
        self.multiclass = all(named)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row: ClipRow = self.rows[index]
        waveform = load_audio(row.path, self.config.sample_rate)
        waveform = crop_or_pad(waveform, self.config.clip_samples, randomize=self.train)
        with torch.no_grad():
            features = self.extractor(waveform)
        dtype = torch.long if self.multiclass else torch.float32
        label = torch.tensor(row.label, dtype=dtype)
        return features.detach(), label
