from __future__ import annotations

import math

import torch


def make_beta_schedule(
    schedule: str,
    timesteps: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    schedule = schedule.lower().strip()
    if timesteps <= 0:
        raise ValueError(f"timesteps must be > 0, got {timesteps}")

    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

    if schedule == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)

    raise ValueError(f"Unknown beta schedule: {schedule} (expected linear|cosine)")

