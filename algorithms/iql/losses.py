"""Loss functions for IQL: expectile loss, q loss, policy loss (AW-regression)
"""
import torch
import torch.nn.functional as F


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Expectile loss for a difference tensor (Q - V)."""
    weight = torch.where(diff > 0, tau, (1 - tau))
    return (weight * (diff ** 2)).mean()


def q_mse_loss(q_values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(q_values, targets)


def policy_aw_loss(pi_actions: torch.Tensor, behavior_actions: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted MSE between policy action and behavior action with advantage weights.
    weight should be non-negative and shaped (batch, 1)
    """
    loss = (weight * ((pi_actions - behavior_actions) ** 2)).mean()
    return loss
