from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_diffusion.build import build_diffusion, build_model
from my_diffusion.config import DiffusionConfig, ModelConfig
from my_diffusion.data import ImageFolderDataset, build_image_transform, list_image_paths
from my_diffusion.utils import get_device, load_checkpoint, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate noise-prediction MSE on a dataset.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_batches", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    config = ckpt.get("config") or {}
    model_cfg = ModelConfig.from_dict(config.get("model", {}))
    diff_cfg = DiffusionConfig.from_dict(config.get("diffusion", {}))

    model = build_model(model_cfg).to(device)
    state = ckpt.get("ema") if (args.use_ema and ckpt.get("ema") is not None) else ckpt["model"]
    model.load_state_dict(state)
    model.eval()

    diffusion = build_diffusion(diff_cfg, device=device, dtype=torch.float32)

    transform = build_image_transform(image_size=model_cfg.image_size, random_flip=False, center_crop=True)
    paths = list_image_paths(args.data_dir)
    ds = ImageFolderDataset(paths, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    total_loss = 0.0
    total_seen = 0
    start = time.perf_counter()

    for i, x0 in enumerate(tqdm(dl, desc="eval", dynamic_ncols=True)):
        if i >= args.num_batches:
            break
        x0 = x0.to(device, non_blocking=True)
        loss = diffusion.training_loss(model, x0)
        bs = x0.shape[0]
        total_loss += loss.item() * bs
        total_seen += bs

    elapsed = max(1e-9, time.perf_counter() - start)
    mean_loss = total_loss / max(1, total_seen)
    imgs_per_sec = total_seen / elapsed

    print(f"mean_mse_loss: {mean_loss:.6f}")
    print(f"images_per_sec: {imgs_per_sec:.1f}")


if __name__ == "__main__":
    main()

