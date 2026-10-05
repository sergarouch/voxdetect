"""Audio and model settings shared by training and inference."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AudioConfig:
    """Waveform and log-mel settings. Inference reloads these from the checkpoint."""

    sample_rate: int = 16000
    clip_seconds: float = 1.0
    min_seconds: float = 0.3
    n_mels: int = 64
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    f_min: float = 50.0
    f_max: float = 7600.0
    top_db: float = 80.0

    @property
    def clip_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)

    @property
    def n_frames(self) -> int:
        # torch.stft frame count with center=True.
        return self.clip_samples // self.hop_length + 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "AudioConfig":
        known = {field: payload[field] for field in cls.__dataclass_fields__ if field in payload}
        return cls(**known)


@dataclass(frozen=True)
class ModelConfig:
    """CRNN shape. n_mels and n_frames must match AudioConfig."""

    n_mels: int = 64
    n_frames: int = 101
    channels: tuple[int, ...] = (16, 32, 64)
    gru_hidden: int = 64
    dropout: float = 0.2

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["channels"] = list(self.channels)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelConfig":
        data = dict(payload)
        if "channels" in data:
            data["channels"] = tuple(data["channels"])
        known = {field: data[field] for field in cls.__dataclass_fields__ if field in data}
        return cls(**known)

    @classmethod
    def from_audio(cls, audio: AudioConfig) -> "ModelConfig":
        return cls(n_mels=audio.n_mels, n_frames=audio.n_frames)
