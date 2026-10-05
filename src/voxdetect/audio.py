"""Decode audio to a mono 16 kHz waveform via ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"}


class AudioReadError(RuntimeError):
    """ffmpeg could not decode a file, or the file is empty."""


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioReadError(
            "ffmpeg is not on PATH. Install it (for example `sudo apt install ffmpeg`) "
            "and retry. wav, flac, and mp3 are decoded with ffmpeg."
        )
    return ffmpeg


def load_audio(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    """Return a mono float32 waveform at `sample_rate`, shape `(samples,)`."""
    ffmpeg = require_ffmpeg()
    source = Path(path)
    if not source.is_file():
        raise AudioReadError(f"file not found: {source}")

    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AudioReadError(detail or f"ffmpeg failed on {source}")
    if not completed.stdout:
        raise AudioReadError(f"no audio samples in {source}")

    samples = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise AudioReadError(f"no audio samples in {source}")
    return torch.from_numpy(samples)


def crop_or_pad(waveform: torch.Tensor, num_samples: int, randomize: bool) -> torch.Tensor:
    """Fit a waveform to a fixed length. Training crops randomly; eval uses the center."""
    if waveform.numel() == num_samples:
        return waveform
    if waveform.numel() > num_samples:
        extra = waveform.numel() - num_samples
        start = int(torch.randint(0, extra + 1, ()).item()) if randomize else extra // 2
        return waveform[start : start + num_samples]

    out = torch.zeros(num_samples, dtype=waveform.dtype)
    extra = num_samples - waveform.numel()
    start = int(torch.randint(0, extra + 1, ()).item()) if randomize else extra // 2
    out[start : start + waveform.numel()] = waveform
    return out


def iter_windows(waveform: torch.Tensor, window: int, hop: int) -> list[torch.Tensor]:
    """Slice a waveform into fixed windows. A short clip is padded once."""
    if waveform.numel() <= window:
        return [crop_or_pad(waveform, window, randomize=False)]

    chunks: list[torch.Tensor] = []
    start = 0
    last_start = 0
    while start + window <= waveform.numel():
        chunks.append(waveform[start : start + window])
        last_start = start
        start += hop
    tail_start = waveform.numel() - window
    if tail_start != last_start:
        chunks.append(waveform[tail_start:])
    return chunks
