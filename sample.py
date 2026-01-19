from __future__ import annotations

import argparse
from pathlib import Path

import torch

from my_diffusion.build import build_diffusion, build_model
from my_diffusion.config import DiffusionConfig, ModelConfig
from my_diffusion.utils import get_device, load_checkpoint, save_image_grid, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample images from a trained DDPM checkpoint.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--nrow", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", type=str, default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_tensor", action="store_true")
    return p.parse_args()


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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=device).manual_seed(args.seed)

    samples_all: list[torch.Tensor] = []
    remaining = int(args.num_samples)
    while remaining > 0:
        bs = min(int(args.batch_size), remaining)
        shape = (bs, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size)
        if args.method == "ddpm":
            x = diffusion.sample_loop(model, shape, generator=gen)
        else:
            x = diffusion.ddim_sample_loop(
                model,
                shape,
                steps=int(args.sample_steps),
                eta=float(args.ddim_eta),
                generator=gen,
            )
        samples_all.append(x.detach().cpu())
        remaining -= bs

    samples = torch.cat(samples_all, dim=0)
    save_image_grid(samples, out_dir / "samples.png", nrow=args.nrow)

    if args.save_tensor:
        torch.save(samples, out_dir / "samples.pt")


if __name__ == "__main__":
    main()
