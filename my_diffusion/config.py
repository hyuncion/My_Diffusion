from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _as_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    raise TypeError(f"Unsupported type for list[int]: {type(value)}")


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 32
    in_channels: int = 3
    out_channels: int = 3
    model_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (16,)
    dropout: float = 0.0
    num_heads: int = 4

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["channel_mult"] = list(self.channel_mult)
        d["attention_resolutions"] = list(self.attention_resolutions)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(
            image_size=int(d.get("image_size", 32)),
            in_channels=int(d.get("in_channels", 3)),
            out_channels=int(d.get("out_channels", d.get("in_channels", 3))),
            model_channels=int(d.get("model_channels", 128)),
            channel_mult=tuple(_as_list(d.get("channel_mult", (1, 2, 2, 4)))),
            num_res_blocks=int(d.get("num_res_blocks", 2)),
            attention_resolutions=tuple(_as_list(d.get("attention_resolutions", (16,)))),
            dropout=float(d.get("dropout", 0.0)),
            num_heads=int(d.get("num_heads", 4)),
        )


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "linear"  # linear | cosine
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiffusionConfig":
        return cls(
            timesteps=int(d.get("timesteps", 1000)),
            beta_schedule=str(d.get("beta_schedule", "linear")),
            beta_start=float(d.get("beta_start", 1e-4)),
            beta_end=float(d.get("beta_end", 2e-2)),
        )


def build_model_config_from_args(args: Any) -> ModelConfig:
    return ModelConfig(
        image_size=int(args.image_size),
        in_channels=int(getattr(args, "in_channels", 3)),
        out_channels=int(getattr(args, "out_channels", getattr(args, "in_channels", 3))),
        model_channels=int(args.model_channels),
        channel_mult=tuple(_as_list(args.channel_mult)),
        num_res_blocks=int(args.num_res_blocks),
        attention_resolutions=tuple(_as_list(args.attention_resolutions)),
        dropout=float(args.dropout),
        num_heads=int(args.num_heads),
    )


def build_diffusion_config_from_args(args: Any) -> DiffusionConfig:
    return DiffusionConfig(
        timesteps=int(args.timesteps),
        beta_schedule=str(args.beta_schedule),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
    )

