"""Training utilities for BCQ: complete training step implementation.

Implements all four BCQ training steps:
1. Train Behavior VAE (action model)
2. Train critics (Double Q) with constrained next-action selection
3. Train perturbation actor (bounded adjustment)
4. Soft update target networks
"""
from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def vae_loss(a_recon: torch.Tensor, a: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss: reconstruction + KL divergence.
    
    Args:
        a_recon: reconstructed action [batch, action_dim]
        a: original action [batch, action_dim]
        mu: latent mean [batch, latent_dim]
        log_sigma: latent log std [batch, latent_dim]
    
    Returns:
        total_loss: reconstruction + KL loss
        recon_loss: MSE reconstruction loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss: MSE
    recon_loss = F.mse_loss(a_recon, a)
    
    # KL divergence: D_KL(q(z|s,a) || N(0, I))
    sigma = log_sigma.exp()
    kl_loss = -0.5 * (1 + 2 * log_sigma - mu.pow(2) - sigma.pow(2)).sum(dim=1).mean()
    
    beta = 0.5  # or 0.1–1.0
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def bcq_update_step(batch: Dict[str, torch.Tensor],
                    vae: torch.nn.Module,
                    actor: torch.nn.Module,
                    q1: torch.nn.Module,
                    q2: torch.nn.Module,
                    q1_target: torch.nn.Module,
                    q2_target: torch.nn.Module,
                    vae_opt: torch.optim.Optimizer,
                    actor_opt: torch.optim.Optimizer,
                    q1_opt: torch.optim.Optimizer,
                    q2_opt: torch.optim.Optimizer,
                    gamma: float = 0.99,
                    tau: float = 0.005,
                    xi: float = 0.05,
                    num_candidates: int = 10) -> Dict[str, torch.Tensor]:
    """Perform a single BCQ optimization step.
    
    Implements all four training steps:
    1. Train VAE on reconstruction + KL
    2. Train critics with constrained next-action selection
    3. Train perturbation actor to maximize Q
    4. Soft update target networks
    
    Args:
        batch: dict with keys 's','a','r','s_next','done' (tensors on same device)
        vae, actor, q1, q2, q1_target, q2_target: network modules
        optimizers for each component
        gamma: discount factor
        tau: soft update coefficient
        xi: perturbation limit
        num_candidates: number of candidate actions to sample
    
    Returns:
        dict of losses: vae_loss, recon_loss, kl_loss, q1_loss, q2_loss, actor_loss
    """
    s = batch['s']
    a = batch['a']
    r = batch['r']
    s_next = batch['s_next']
    done = batch['done']
    
    batch_size = s.shape[0]
    device = s.device
    
    # ========== Step 1: Train Behavior VAE ==========
    vae_opt.zero_grad()
    a_recon, mu, log_sigma = vae(s, a)
    vae_loss_total, recon_loss, kl_loss = vae_loss(a_recon, a, mu, log_sigma)
    vae_loss_total.backward()
    torch.nn.utils.clip_grad_norm_(vae.parameters(), 10.0)
    vae_opt.step()
    
    # ========== Step 2: Train Critics (Double Q) with Constrained Next-Action Selection ==========
    
    # Sample K candidate actions for next state
    with torch.no_grad():
        a_candidates = vae.sample(s_next, num_samples=num_candidates)  # [batch * num_candidates, action_dim]
        
        # Expand states for all candidates
        s_next_expanded = s_next.unsqueeze(1).expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, s_next.shape[1])
        
        # Perturb each candidate
        delta = actor(s_next_expanded, a_candidates)
        a_perturbed = a_candidates + delta
        
        # Clamp perturbed actions to valid range (assuming [-1, 1] after normalization)
        a_perturbed = torch.clamp(a_perturbed, -1.0, 1.0)
        
        # Evaluate with target Q networks
        q1_next = q1_target(s_next_expanded, a_perturbed).reshape(batch_size, num_candidates)
        q2_next = q2_target(s_next_expanded, a_perturbed).reshape(batch_size, num_candidates)
        
        # Pick best by minimum of two Q networks (double Q)
        q_next = torch.min(q1_next, q2_next)
        
        # Select best action per state
        best_idx = q_next.argmax(dim=1)
        
        # Compute target Q values
        q_target_vals = q_next[torch.arange(batch_size), best_idx].unsqueeze(1)  # [B, 1]
        
        td_target = r + gamma * q_target_vals * (1 - done)
    
    # Update Q1
    q1_opt.zero_grad()
    q1_vals = q1(s, a)
    q1_loss = F.mse_loss(q1_vals, td_target)
    q1_loss.backward()
    torch.nn.utils.clip_grad_norm_(q1.parameters(), 10.0)
    q1_opt.step()
    
    # Update Q2
    q2_opt.zero_grad()
    q2_vals = q2(s, a)
    q2_loss = F.mse_loss(q2_vals, td_target)
    q2_loss.backward()
    torch.nn.utils.clip_grad_norm_(q2.parameters(), 10.0)
    q2_opt.step()
    
    # ========== Step 3: Train Perturbation Actor ==========
    # Objective: maximize Q1(s, a + xi * phi(s, a)) where a is sampled from VAE
    
    actor_opt.zero_grad()
    
    # Sample actions from VAE at current state
    with torch.no_grad():
        a_vae = vae.sample(s, num_samples=1).reshape(batch_size, a.shape[1])
    
    # Compute perturbation
    delta = actor(s, a_vae)
    a_perturbed = a_vae + delta
    a_perturbed = torch.clamp(a_perturbed, -1.0, 1.0)
    
    # Maximize Q1 (negative loss for gradient ascent)
    q1_val = q1(s, a_perturbed)
    q2_val = q2(s, a_perturbed)
    actor_loss = -torch.min(q1_val, q2_val).mean()
    
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
    actor_opt.step()
    
    # ========== Step 4: Soft Update Target Networks ==========
    # theta_target <- tau * theta + (1 - tau) * theta_target
    for param, target_param in zip(q1.parameters(), q1_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    for param, target_param in zip(q2.parameters(), q2_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    return {
        "vae_loss": vae_loss_total.detach(),
        "recon_loss": recon_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "q1_loss": q1_loss.detach(),
        "q2_loss": q2_loss.detach(),
        "actor_loss": actor_loss.detach(),
    }


def bcq_update_step_with_action_scaling(batch: Dict[str, torch.Tensor],
                                        vae: torch.nn.Module,
                                        actor: torch.nn.Module,
                                        q1: torch.nn.Module,
                                        q2: torch.nn.Module,
                                        q1_target: torch.nn.Module,
                                        q2_target: torch.nn.Module,
                                        vae_opt: torch.optim.Optimizer,
                                        actor_opt: torch.optim.Optimizer,
                                        q1_opt: torch.optim.Optimizer,
                                        q2_opt: torch.optim.Optimizer,
                                        gamma: float = 0.99,
                                        tau: float = 0.005,
                                        xi: float = 0.05,
                                        num_candidates: int = 10,
                                        action_min: float = -1.0,
                                        action_max: float = 1.0) -> Dict[str, torch.Tensor]:
    """BCQ update step with explicit action range clamping.
    
    This version is useful when actions have been normalized to [-1, 1] range
    and we want to ensure they stay within that range.
    
    Args:
        Same as bcq_update_step, plus:
        action_min, action_max: valid action range
    
    Returns:
        dict of losses
    """
    s = batch['s']
    a = batch['a']
    r = batch['r']
    s_next = batch['s_next']
    done = batch['done']
    
    batch_size = s.shape[0]
    device = s.device
    
    # ========== Step 1: Train Behavior VAE ==========
    vae_opt.zero_grad()
    a_recon, mu, log_sigma = vae(s, a)
    vae_loss_total, recon_loss, kl_loss = vae_loss(a_recon, a, mu, log_sigma)
    vae_loss_total.backward()
    torch.nn.utils.clip_grad_norm_(vae.parameters(), 10.0)
    vae_opt.step()
    
    # ========== Step 2: Train Critics ==========
    with torch.no_grad():
        a_candidates = vae.sample(s_next, num_samples=num_candidates)
        a_candidates = torch.clamp(a_candidates, action_min, action_max)
        
        s_next_expanded = s_next.unsqueeze(1).expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, s_next.shape[1])
        
        delta = actor(s_next_expanded, a_candidates)
        a_perturbed = a_candidates + delta
        a_perturbed = torch.clamp(a_perturbed, action_min, action_max)
        
        q1_next = q1_target(s_next_expanded, a_perturbed).reshape(batch_size, num_candidates)
        q2_next = q2_target(s_next_expanded, a_perturbed).reshape(batch_size, num_candidates)
        
        q_next = torch.min(q1_next, q2_next)
        best_idx = q_next.argmax(dim=1)
        a_best_perturbed = a_perturbed.reshape(batch_size, num_candidates, a.shape[1])[torch.arange(batch_size), best_idx]
        
        q1_target_vals = q1_target(s_next, a_best_perturbed)
        q2_target_vals = q2_target(s_next, a_best_perturbed)
        q_target_vals = torch.min(q1_target_vals, q2_target_vals)
        
        td_target = r + gamma * q_target_vals * (1 - done)
    
    q1_opt.zero_grad()
    q1_vals = q1(s, a)
    q1_loss = F.mse_loss(q1_vals, td_target)
    q1_loss.backward()
    torch.nn.utils.clip_grad_norm_(q1.parameters(), 10.0)
    q1_opt.step()
    
    q2_opt.zero_grad()
    q2_vals = q2(s, a)
    q2_loss = F.mse_loss(q2_vals, td_target)
    q2_loss.backward()
    torch.nn.utils.clip_grad_norm_(q2.parameters(), 10.0)
    q2_opt.step()
    
    # ========== Step 3: Train Perturbation Actor ==========
    actor_opt.zero_grad()
    
    with torch.no_grad():
        a_vae = vae.sample(s, num_samples=1).reshape(batch_size, a.shape[1])
        a_vae = torch.clamp(a_vae, action_min, action_max)
    
    delta = actor(s, a_vae)
    a_perturbed = a_vae + delta
    a_perturbed = torch.clamp(a_perturbed, action_min, action_max)
    
    q1_actor = q1(s, a_perturbed)
    actor_loss = -q1_actor.mean()
    
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
    actor_opt.step()
    
    # ========== Step 4: Soft Update Target Networks ==========
    for param, target_param in zip(q1.parameters(), q1_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    for param, target_param in zip(q2.parameters(), q2_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    return {
        "vae_loss": vae_loss_total.detach(),
        "recon_loss": recon_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "q1_loss": q1_loss.detach(),
        "q2_loss": q2_loss.detach(),
        "actor_loss": actor_loss.detach(),
    }
