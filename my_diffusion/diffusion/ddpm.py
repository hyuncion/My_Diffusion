from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F

from ..config import DiffusionConfig
from .schedules import make_beta_schedule


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _randn_like(x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


class DDPM:
    def __init__(
        self,
        config: DiffusionConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.timesteps = int(config.timesteps)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        betas = make_beta_schedule(
            config.beta_schedule,
            self.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )
        betas = betas.to(device=self.device, dtype=self.dtype)

        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device, dtype=self.dtype), self.alphas_cumprod[:-1]],
            dim=0,
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def to_config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        ) * noise

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (xt - _extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape) * eps) / _extract(
            self.sqrt_alphas_cumprod, t, xt.shape
        )

    def p_mean_variance(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps_pred = model(xt, t)
        x0_pred = self.predict_x0_from_eps(xt, t, eps_pred)
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        mean = _extract(self.posterior_mean_coef1, t, xt.shape) * x0_pred + _extract(
            self.posterior_mean_coef2, t, xt.shape
        ) * xt
        var = _extract(self.posterior_variance, t, xt.shape)
        return mean, var, x0_pred

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        *,
        clip_denoised: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        mean, var, _ = self.p_mean_variance(model, xt, t, clip_denoised=clip_denoised)
        noise = _randn_like(xt, generator=generator)
        nonzero = (t != 0).float().reshape(xt.shape[0], *((1,) * (xt.ndim - 1)))
        return mean + nonzero * torch.sqrt(var) * noise

    def training_loss(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        *,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = x0.shape[0]
        if t is None:
            t = self.sample_timesteps(b)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)
        eps_pred = model(xt, t)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample_loop(
        self,
        model: torch.nn.Module,
        shape: tuple[int, int, int, int],
        *,
        clip_denoised: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=self.device, dtype=self.dtype, generator=generator)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, clip_denoised=clip_denoised, generator=generator)
        return x

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: torch.nn.Module,
        shape: tuple[int, int, int, int],
        *,
        steps: int,
        eta: float = 0.0,
        clip_denoised: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")
        if steps > self.timesteps:
            raise ValueError(f"steps must be <= timesteps({self.timesteps}), got {steps}")

        times = torch.linspace(0, self.timesteps - 1, steps, device=self.device)
        times = times.round().long()
        times = torch.unique_consecutive(times)
        times = times.flip(0)  # descending

        x = torch.randn(shape, device=self.device, dtype=self.dtype, generator=generator)

        for i, t_val in enumerate(times):
            t = torch.full((shape[0],), int(t_val.item()), device=self.device, dtype=torch.long)
            eps = model(x, t)
            alpha_t = _extract(self.alphas_cumprod, t, x.shape)
            x0 = (x - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t)
            if clip_denoised:
                x0 = x0.clamp(-1.0, 1.0)

            if i == len(times) - 1:
                x = x0
                break

            t_prev_val = int(times[i + 1].item())
            t_prev = torch.full((shape[0],), t_prev_val, device=self.device, dtype=torch.long)
            alpha_prev = _extract(self.alphas_cumprod, t_prev, x.shape)

            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t))
                * torch.sqrt(1.0 - (alpha_t / alpha_prev))
            )
            noise = _randn_like(x, generator=generator)
            dir_xt = torch.sqrt((1.0 - alpha_prev) - sigma**2) * eps
            x = torch.sqrt(alpha_prev) * x0 + dir_xt + sigma * noise

        return x
