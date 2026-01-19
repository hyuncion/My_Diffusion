from __future__ import annotations

import torch

from .config import DiffusionConfig, ModelConfig
from .diffusion import DDPM
from .models import UNet


def build_model(config: ModelConfig) -> UNet:
    return UNet(
        image_size=config.image_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        model_channels=config.model_channels,
        channel_mult=config.channel_mult,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        dropout=config.dropout,
        num_heads=config.num_heads,
    )


def build_diffusion(
    config: DiffusionConfig,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> DDPM:
    return DDPM(config, device=device, dtype=dtype)

