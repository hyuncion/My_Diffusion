from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    *,
    nrow: int = 8,
) -> None:
    if images.ndim != 4:
        raise ValueError(f"images must be 4D (B,C,H,W), got shape={tuple(images.shape)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = images.detach().cpu().float().clamp(-1.0, 1.0)
    images = (images + 1.0) / 2.0
    images = (images * 255.0).round().to(torch.uint8)

    b, c, h, w = images.shape
    nrow = max(1, min(int(nrow), b))
    ncol = int(math.ceil(b / nrow))

    if c == 1:
        images = images.repeat(1, 3, 1, 1)
        c = 3

    grid = np.zeros((ncol * h, nrow * w, c), dtype=np.uint8)
    for idx in range(b):
        r = idx // nrow
        col = idx % nrow
        img = images[idx].permute(1, 2, 0).numpy()
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = img

    Image.fromarray(grid).save(path)

