"""Small convolutional-recurrent network for voice activity detection."""

from __future__ import annotations

import torch
from torch import nn

from voxdetect.config import ModelConfig


class VoiceCRNN(nn.Module):
    """Log-mel CRNN that outputs one voice logit per clip.

    Three conv blocks pool frequency and time, a bidirectional GRU reads the
    remaining sequence, and a linear layer produces a single logit. Apply
    sigmoid to that logit for a voice probability.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.n_mels % 8 != 0:
            raise ValueError("n_mels must be divisible by 8 so three pooling layers fit")

        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in config.channels:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*blocks)

        with torch.no_grad():
            probe = self.conv(torch.zeros(1, 1, config.n_mels, config.n_frames))
        channels, freq, _time = probe.shape[1:]
        self.gru = nn.GRU(
            input_size=channels * freq,
            hidden_size=config.gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.gru_hidden * 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            features = features.unsqueeze(1)
        encoded = self.conv(features)
        batch, channels, freq, time = encoded.shape
        sequence = encoded.permute(0, 3, 1, 2).reshape(batch, time, channels * freq)
        recurrent, _hidden = self.gru(sequence)
        pooled = recurrent.mean(dim=1)
        return self.head(pooled).squeeze(-1)
