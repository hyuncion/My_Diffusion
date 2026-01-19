from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _num_groups(channels: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must be 1D (batch,), got shape={tuple(t.shape)}")

        device = t.device
        half = self.dim // 2
        if half == 0:
            return t[:, None].to(torch.float32)

        t = t.to(torch.float32)
        freqs = torch.exp(
            torch.linspace(0, math.log(10000.0), half, device=device, dtype=torch.float32) * (-1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device, dtype=emb.dtype)], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, *, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels({channels}) must be divisible by num_heads({num_heads})")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        qkv = self.qkv(y)
        q, k, v = qkv.chunk(3, dim=1)

        tokens = h * w
        q = q.view(b, self.num_heads, self.head_dim, tokens).permute(0, 1, 3, 2)
        k = k.view(b, self.num_heads, self.head_dim, tokens).permute(0, 1, 3, 2)
        v = v.view(b, self.num_heads, self.head_dim, tokens).permute(0, 1, 3, 2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = self.proj(attn)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        *,
        use_attn: bool,
        dropout: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, time_emb_dim, dropout=dropout)
        self.attn = AttentionBlock(out_channels, num_heads=num_heads) if use_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, t_emb)
        return self.attn(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_emb_dim: int,
        *,
        use_attn: bool,
        dropout: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.res = ResBlock(in_channels + skip_channels, out_channels, time_emb_dim, dropout=dropout)
        self.attn = AttentionBlock(out_channels, num_heads=num_heads) if use_attn else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        return self.attn(x)


class UNet(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int = 3,
        out_channels: int | None = None,
        model_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.0,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels if out_channels is not None else in_channels)
        self.model_channels = int(model_channels)
        self.channel_mult = tuple(int(x) for x in channel_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.attention_resolutions = tuple(int(x) for x in attention_resolutions)
        self.dropout = float(dropout)
        self.num_heads = int(num_heads)

        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.conv_in = nn.Conv2d(self.in_channels, model_channels, kernel_size=3, padding=1)

        ch = model_channels
        resolution = self.image_size
        down: list[nn.Module] = []
        skip_channels: list[int] = [ch]

        for level, mult in enumerate(self.channel_mult):
            out_ch = model_channels * mult
            for _ in range(self.num_res_blocks):
                down.append(
                    DownBlock(
                        ch,
                        out_ch,
                        time_emb_dim,
                        use_attn=resolution in self.attention_resolutions,
                        dropout=self.dropout,
                        num_heads=self.num_heads,
                    )
                )
                ch = out_ch
                skip_channels.append(ch)

            if level != len(self.channel_mult) - 1:
                down.append(Downsample(ch))
                skip_channels.append(ch)
                resolution //= 2

        self.down = nn.ModuleList(down)

        self.mid1 = ResBlock(ch, ch, time_emb_dim, dropout=self.dropout)
        self.mid_attn = AttentionBlock(ch, num_heads=self.num_heads)
        self.mid2 = ResBlock(ch, ch, time_emb_dim, dropout=self.dropout)

        up: list[nn.Module] = []
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            out_ch = model_channels * mult
            for i in range(self.num_res_blocks + 1):
                if not skip_channels:
                    raise RuntimeError("Skip channel stack underflow while building UNet.")
                sc = skip_channels.pop()
                up.append(
                    UpBlock(
                        ch,
                        sc,
                        out_ch,
                        time_emb_dim,
                        use_attn=resolution in self.attention_resolutions,
                        dropout=self.dropout,
                        num_heads=self.num_heads,
                    )
                )
                ch = out_ch

                if level != 0 and i == self.num_res_blocks:
                    up.append(Upsample(ch))
                    resolution *= 2

        if skip_channels:
            raise RuntimeError("Skip channel stack not empty after building UNet.")

        self.up = nn.ModuleList(up)

        self.out_norm = nn.GroupNorm(_num_groups(ch), ch)
        self.out_conv = nn.Conv2d(ch, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        h = self.conv_in(x)
        hs: list[torch.Tensor] = [h]

        for module in self.down:
            if isinstance(module, Downsample):
                h = module(h)
            else:
                h = module(h, t_emb)
            hs.append(h)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        for module in self.up:
            if isinstance(module, Upsample):
                h = module(h)
            else:
                skip = hs.pop()
                h = module(h, skip, t_emb)

        if hs:
            raise RuntimeError("Skip tensor stack not empty after UNet forward.")

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)

