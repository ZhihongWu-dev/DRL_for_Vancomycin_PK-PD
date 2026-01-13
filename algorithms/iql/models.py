"""Neural network models for IQL: Q, V, Policy (simple MLPs)
"""
from typing import Tuple
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: Tuple[int, ...] = (256, 256), activation: nn.Module = nn.ReLU()):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(dim, h))
            layers.append(activation)
            dim = h
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, hidden=(256, 256)):
        super().__init__()
        self.model = MLP(state_dim + action_dim, 1, hidden=hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.model(x)


class VNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden=(256, 256)):
        super().__init__()
        self.model = MLP(state_dim, 1, hidden=hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.model(s)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, hidden=(256, 256)):
        super().__init__()
        self.net = MLP(state_dim, action_dim * 2, hidden=hidden)  # mean and logstd

    def forward(self, s: torch.Tensor):
        out = self.net(s)
        mean, logstd = out.chunk(2, dim=-1)
        std = logstd.clamp(-20, 2).exp()
        return mean, std

    def sample(self, s: torch.Tensor):
        mean, std = self.forward(s)
        eps = torch.randn_like(mean)
        return mean + eps * std

# convenience helper to initialize model weights using utilities
from algorithms.iql.utils import init_weights

def init_model_weights(model: nn.Module):
    """Apply standard initializations to a model in-place."""
    model.apply(init_weights)
    return model