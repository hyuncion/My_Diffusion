from __future__ import annotations

import warnings

import torch


def get_device(device: str = "auto") -> torch.device:
    device = device.strip().lower()
    if device == "auto":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"CUDA initialization:.*")
            cuda_ok = torch.cuda.is_available()
        return torch.device("cuda" if cuda_ok else "cpu")
    return torch.device(device)
