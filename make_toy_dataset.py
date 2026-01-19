from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a simple toy image dataset (no download needed).")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_images", type=int, default=20000)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _rand_color(rng: random.Random) -> tuple[int, int, int]:
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def _make_one(size: int, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (size, size), _rand_color(rng))
    draw = ImageDraw.Draw(img)

    n_shapes = rng.randint(1, 4)
    for _ in range(n_shapes):
        kind = rng.choice(["rect", "circle", "tri"])
        color = _rand_color(rng)
        x0 = rng.randint(0, size - 2)
        y0 = rng.randint(0, size - 2)
        x1 = rng.randint(x0 + 1, size - 1)
        y1 = rng.randint(y0 + 1, size - 1)

        if kind == "rect":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        elif kind == "circle":
            draw.ellipse([x0, y0, x1, y1], fill=color)
        else:
            x2 = rng.randint(0, size - 1)
            y2 = rng.randint(0, size - 1)
            draw.polygon([(x0, y0), (x1, y1), (x2, y2)], fill=color)

    # Small pixel noise for diversity
    arr = np.asarray(img, dtype=np.int16)
    noise = rng.randint(0, 12)
    if noise > 0:
        arr = arr + rng.randint(-noise, noise)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for i in tqdm(range(int(args.num_images)), desc="make_toy_dataset", dynamic_ncols=True):
        img = _make_one(int(args.image_size), rng)
        img.save(out_dir / f"{i:07d}.png")


if __name__ == "__main__":
    main()
