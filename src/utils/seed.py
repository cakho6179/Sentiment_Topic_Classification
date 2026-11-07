"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds.

    The environment variable ``PYTHONHASHSEED`` must be set before the
    interpreter starts to fully control hash-based ordering.  This function
    still records the requested seed for experiment metadata.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
