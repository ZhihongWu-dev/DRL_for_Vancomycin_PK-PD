"""Utility helpers for IQL: seeding, device detection, and weight initialization."""
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(m: nn.Module):
    """Apply weight initialization to common layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
