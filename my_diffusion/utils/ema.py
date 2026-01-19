from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, *, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters(), strict=True):
            ema_p.data.mul_(d).add_(p.data, alpha=1.0 - d)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.ema_model.load_state_dict(state)

