"""Write a tiny speech / non-speech folder so the pipeline can be checked."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np


def write_synthetic(out_dir: str | Path, clips_per_class: int = 8, sample_rate: int = 16000, seed: int = 0) -> Path:
    """Create `speech/` tones and `non_speech/` noise as 1-second wav files.

    These clips only confirm that import and training run. They are not a
    substitute for real recordings.
    """
    root = Path(out_dir)
    speech = root / "speech"
    noise = root / "non_speech"
    speech.mkdir(parents=True, exist_ok=True)
    noise.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
    for index in range(clips_per_class):
        fundamental = 110 + 15 * index
        voiced = np.zeros(sample_rate, dtype=np.float32)
        for harmonic in (1, 2, 3, 4):
            voiced += (0.35 / harmonic) * np.sin(2 * math.pi * fundamental * harmonic * timeline)
        envelope = 0.5 * (1 + np.sin(2 * math.pi * 4 * timeline))
        voiced = (voiced * envelope).astype(np.float32)
        _write_wav(speech / f"speech_{index:02d}.wav", voiced, sample_rate)
        rumble = 0.2 * np.sin(2 * math.pi * 40 * timeline) + 0.15 * rng.standard_normal(sample_rate)
        _write_wav(noise / f"noise_{index:02d}.wav", rumble.astype(np.float32), sample_rate)
    return root


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
