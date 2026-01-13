"""Training utilities for IQL: single-step update function.

Provides:
- iql_update_step: runs one Q/V/Policy update given a batch and optimizers
"""
from typing import Dict
import torch
import torch.nn.functional as F
from algorithms.iql.losses import expectile_loss


def iql_update_step(batch: Dict[str, torch.Tensor],
                    q_net: torch.nn.Module,
                    v_net: torch.nn.Module,
                    pi_net: torch.nn.Module,
                    q_opt: torch.optim.Optimizer,
                    v_opt: torch.optim.Optimizer,
                    pi_opt: torch.optim.Optimizer,
                    gamma: float = 0.99,
                    tau: float = 0.7,
                    beta: float = 3.0,
                    weight_clip: float = 1e2) -> Dict[str, torch.Tensor]:
    """Perform a single optimization step for IQL.

    Args:
        batch: dict with keys 's','a','r','s_next','done' (tensors on same device)
        networks and optimizers
        gamma, tau, beta as algorithm hyperparameters
        weight_clip: upper clamp for advantage weights

    Returns:
        dict of losses: q_loss, v_loss, pi_loss
    """
    s = batch['s']
    a = batch['a']
    r = batch['r']
    s_next = batch['s_next']
    done = batch['done']

    # Q update
    q_opt.zero_grad()
    q_vals = q_net(s, a)
    with torch.no_grad():
        v_next = v_net(s_next)
        q_targets = r + gamma * v_next * (1 - done)
    q_loss = F.mse_loss(q_vals, q_targets)
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    q_opt.step()

    # V expectile update
    v_opt.zero_grad()
    with torch.no_grad():
        q_vals_detach = q_net(s, a).detach()
    v_vals = v_net(s)
    v_loss = expectile_loss(q_vals_detach - v_vals, tau)
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(v_net.parameters(), 10.0)
    v_opt.step()

    # Policy update: advantage-weighted regression
    pi_opt.zero_grad()
    with torch.no_grad():
        adv = (q_net(s, a).detach() - v_net(s).detach())
        weights = torch.exp(adv / beta)
        weights = torch.clamp(weights, 0.0, weight_clip)
        weights = weights / (weights.sum() + 1e-8)
    pi_actions = pi_net.sample(s)
    pi_loss = (weights * (pi_actions - a) ** 2).mean()
    pi_loss.backward()
    torch.nn.utils.clip_grad_norm_(pi_net.parameters(), 10.0)
    pi_opt.step()

    return {"q_loss": q_loss.detach(), "v_loss": v_loss.detach(), "pi_loss": pi_loss.detach()}
