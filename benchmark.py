from __future__ import annotations

import argparse
import time

import torch

from my_diffusion.build import build_diffusion, build_model
from my_diffusion.config import DiffusionConfig, ModelConfig
from my_diffusion.utils import get_device, load_checkpoint, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark sampling speed (and CUDA memory if available).")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", type=str, default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


@torch.no_grad()
def _sample_batch(
    diffusion,
    model,
    *,
    shape: tuple[int, int, int, int],
    method: str,
    sample_steps: int,
    ddim_eta: float,
    gen: torch.Generator,
) -> None:
    if method == "ddpm":
        diffusion.sample_loop(model, shape, generator=gen)
    else:
        diffusion.ddim_sample_loop(model, shape, steps=sample_steps, eta=ddim_eta, generator=gen)


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

    gen = torch.Generator(device=device).manual_seed(args.seed)

    # Warmup
    for _ in range(max(0, args.warmup)):
        _sample_batch(
            diffusion,
            model,
            shape=(min(args.batch_size, args.num_samples), model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size),
            method=args.method,
            sample_steps=args.sample_steps,
            ddim_eta=args.ddim_eta,
            gen=gen,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    remaining = int(args.num_samples)
    while remaining > 0:
        bs = min(int(args.batch_size), remaining)
        _sample_batch(
            diffusion,
            model,
            shape=(bs, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size),
            method=args.method,
            sample_steps=args.sample_steps,
            ddim_eta=args.ddim_eta,
            gen=gen,
        )
        remaining -= bs

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = max(1e-9, time.perf_counter() - start)
    samples_per_sec = int(args.num_samples) / elapsed

    print(f"elapsed_sec: {elapsed:.4f}")
    print(f"samples_per_sec: {samples_per_sec:.2f}")
    if device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"cuda_max_memory_allocated_mb: {max_mem:.1f}")


if __name__ == "__main__":
    main()

