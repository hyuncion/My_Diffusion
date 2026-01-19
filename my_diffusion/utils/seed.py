from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"CUDA initialization:.*")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
