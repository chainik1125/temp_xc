import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
