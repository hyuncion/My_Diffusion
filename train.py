from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_diffusion.build import build_diffusion, build_model
from my_diffusion.config import (
    build_diffusion_config_from_args,
    build_model_config_from_args,
)
from my_diffusion.data import ImageFolderDataset, build_image_transform, list_image_paths, split_paths
from my_diffusion.utils import EMA, get_device, save_checkpoint, save_image_grid, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a basic DDPM (unconditional).")

    # Data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--val_ratio", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)

    # Model
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--out_channels", type=int, default=3)
    p.add_argument("--model_channels", type=int, default=128)
    p.add_argument("--channel_mult", type=str, default="1,2,2,4")
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--attention_resolutions", type=str, default="16")
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    # Diffusion
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)

    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.9999)

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--sample_num", type=int, default=64)
    p.add_argument("--sample_method", type=str, default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=0.0)

    # Runtime
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default="")

    # Output
    p.add_argument("--out_dir", type=str, default="outputs/ddpm_run")

    return p.parse_args()


@torch.no_grad()
def _generate_samples(
    *,
    diffusion,
    model: torch.nn.Module,
    model_config,
    out_path: Path,
    num_samples: int,
    method: str,
    sample_steps: int,
    ddim_eta: float,
) -> None:
    model.eval()
    b = int(num_samples)
    shape = (b, model_config.in_channels, model_config.image_size, model_config.image_size)
    if method == "ddpm":
        samples = diffusion.sample_loop(model, shape)
    else:
        samples = diffusion.ddim_sample_loop(model, shape, steps=sample_steps, eta=ddim_eta)
    save_image_grid(samples, out_path, nrow=int(round(b**0.5)) or 8)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    sample_dir = out_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device(args.device)

    # Data
    transform = build_image_transform(image_size=args.image_size, random_flip=True, center_crop=True)
    all_paths = list_image_paths(args.data_dir)
    train_paths, _ = split_paths(all_paths, val_ratio=args.val_ratio, seed=args.seed)
    train_ds = ImageFolderDataset(train_paths, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Model / diffusion
    model_config = build_model_config_from_args(args)
    diff_config = build_diffusion_config_from_args(args)
    model = build_model(model_config).to(device)
    diffusion = build_diffusion(diff_config, device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    ema = None
    if args.ema_decay > 0.0:
        ema = EMA(model, decay=args.ema_decay)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") and optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if ema is not None and ckpt.get("ema"):
            ema.load_state_dict(ckpt["ema"])
        step = int(ckpt.get("step", 0))

    config_for_ckpt = {
        "model": model_config.to_dict(),
        "diffusion": diff_config.to_dict(),
        "train": {
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "ema_decay": float(args.ema_decay),
            "amp": bool(args.amp),
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config_for_ckpt, indent=2), encoding="utf-8")

    model.train()
    last_log_time = time.perf_counter()
    last_log_step = step
    pbar = tqdm(total=args.epochs * len(train_loader), initial=step, desc="train", dynamic_ncols=True)

    for _epoch in range(args.epochs):
        for x0 in train_loader:
            x0 = x0.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = diffusion.training_loss(model, x0)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            step += 1
            pbar.update(1)

            if step % args.log_every == 0:
                now = time.perf_counter()
                dt = max(1e-9, now - last_log_time)
                steps_done = step - last_log_step
                imgs_done = steps_done * args.batch_size
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    img_s=f"{imgs_done / dt:.1f}",
                )
                last_log_time = now
                last_log_step = step

            if step % args.sample_every == 0:
                sample_model = ema.ema_model if ema is not None else model
                out_path = sample_dir / f"step_{step:07d}.png"
                _generate_samples(
                    diffusion=diffusion,
                    model=sample_model,
                    model_config=model_config,
                    out_path=out_path,
                    num_samples=args.sample_num,
                    method=args.sample_method,
                    sample_steps=args.sample_steps,
                    ddim_eta=args.ddim_eta,
                )

            if step % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{step:07d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config_for_ckpt,
                    ema_state=(ema.state_dict() if ema is not None else None),
                    scaler_state=(scaler.state_dict() if scaler is not None else None),
                )
                save_checkpoint(
                    ckpt_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config_for_ckpt,
                    ema_state=(ema.state_dict() if ema is not None else None),
                    scaler_state=(scaler.state_dict() if scaler is not None else None),
                )

    save_checkpoint(
        ckpt_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        step=step,
        config=config_for_ckpt,
        ema_state=(ema.state_dict() if ema is not None else None),
        scaler_state=(scaler.state_dict() if scaler is not None else None),
    )
    pbar.close()


if __name__ == "__main__":
    main()
