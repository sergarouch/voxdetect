"""Log-mel spectrogram used by both training and inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio

from voxdetect.config import AudioConfig


class LogMelExtractor(torch.nn.Module):
    """Convert a mono waveform into a normalized log-mel spectrogram.

    Output shape is `(1, n_mels, n_frames)` for a single clip and
    `(batch, n_mels, n_frames)` when the input is already batched.
    """

    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0,
            center=True,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=config.top_db)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        single = waveform.dim() == 1
        if single:
            waveform = waveform.unsqueeze(0)
        spec = self.to_db(self.mel(waveform))
        spec = _fix_time(spec, self.config.n_frames)
        mean = spec.mean(dim=(-2, -1), keepdim=True)
        std = spec.std(dim=(-2, -1), keepdim=True)
        spec = (spec - mean) / (std + 1e-5)
        if single:
            return spec.squeeze(0).unsqueeze(0)
        return spec


def _fix_time(spec: torch.Tensor, n_frames: int) -> torch.Tensor:
    frames = spec.shape[-1]
    if frames == n_frames:
        return spec
    if frames > n_frames:
        start = (frames - n_frames) // 2
        return spec[..., start : start + n_frames]
    return F.pad(spec, (0, n_frames - frames))
