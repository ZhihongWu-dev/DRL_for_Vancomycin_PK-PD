"""Neural network models for BCQ: VAE, Perturbation Actor, and Double Q-networks.

BCQ (Batch-Constrained Q-Learning) uses:
1. ActionVAE: learns the distribution of actions in the batch (encoder + decoder)
2. PerturbationActor: learns bounded perturbations to VAE-sampled actions
3. Q1, Q2: double Q-networks for value estimation
4. Target copies: soft-updated target networks
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Standard multi-layer perceptron."""
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


class ActionVAE(nn.Module):
    """Variational Autoencoder for action distribution modeling.
    
    Encodes (state, action) pairs to latent space, then decodes (state, latent) to action.
    This learns the distribution of actions in the batch given state.
    """
    def __init__(self, state_dim: int, action_dim: int = 1, latent_dim: int = 2, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Encoder: (s, a) -> (mu, log_sigma)
        self.encoder = MLP(state_dim + action_dim, latent_dim * 2, hidden=hidden)
        
        # Decoder: (s, z) -> a
        self.decoder = MLP(state_dim + latent_dim, action_dim, hidden=hidden)
    
    def encode(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode (s, a) to latent distribution parameters.
        
        Args:
            s: state tensor [batch, state_dim]
            a: action tensor [batch, action_dim]
        
        Returns:
            mu: mean of latent distribution [batch, latent_dim]
            log_sigma: log std of latent distribution [batch, latent_dim]
        """
        x = torch.cat([s, a], dim=-1)
        out = self.encoder(x)
        mu, log_sigma = out.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, -20, 2)  # Numerical stability
        return mu, log_sigma
    
    def decode(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode (s, z) to action.
        
        Args:
            s: state tensor [batch, state_dim]
            z: latent sample [batch, latent_dim]
        
        Returns:
            a: reconstructed action [batch, action_dim]
        """
        x = torch.cat([s, z], dim=-1)
        return self.decoder(x)
    
    def sample(self, s: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample actions from the VAE given states.
        
        Args:
            s: state tensor [batch, state_dim]
            num_samples: number of action samples per state
        
        Returns:
            actions: sampled actions [batch * num_samples, action_dim]
        """
        batch = s.size(0)
        
        # Sample from latent distribution
        z = torch.randn(batch * num_samples, self.latent_dim, device=s.device, dtype=s.dtype)
        
        # Expand states for all samples
        s_expanded = s.unsqueeze(1).expand(-1, num_samples, -1).reshape(batch * num_samples, self.state_dim)
        
        # Decode to actions
        a = torch.tanh(self.decoder(x))
        return a
    
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode, sample, decode.
        
        Returns:
            a_recon: reconstructed action [batch, action_dim]
            mu: latent mean [batch, latent_dim]
            log_sigma: latent log std [batch, latent_dim]
        """
        mu, log_sigma = self.encode(s, a)
        sigma = log_sigma.exp()
        
        # Reparameterization trick
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        
        a_recon = self.decode(s, z)
        return a_recon, mu, log_sigma


class PerturbationActor(nn.Module):
    """Perturbation actor that learns bounded adjustments to VAE-sampled actions.
    
    Given (state, action), outputs a small perturbation delta constrained to [-xi, xi].
    """
    def __init__(self, state_dim: int, action_dim: int = 1, hidden: Tuple[int, ...] = (256, 256), xi: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.xi = xi  # Perturbation limit
        
        self.net = MLP(state_dim + action_dim, action_dim, hidden=hidden)
    
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Compute perturbation for action.
        
        Args:
            s: state tensor [batch, state_dim]
            a: action tensor [batch, action_dim]
        
        Returns:
            delta: perturbation [batch, action_dim], bounded to [-xi, xi]
        """
        x = torch.cat([s, a], dim=-1)
        delta = torch.tanh(self.net(x)) * self.xi
        return delta


class QNetwork(nn.Module):
    """Q-network for state-action value estimation."""
    def __init__(self, state_dim: int, action_dim: int = 1, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.model = MLP(state_dim + action_dim, 1, hidden=hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a).
        
        Args:
            s: state tensor [batch, state_dim]
            a: action tensor [batch, action_dim]
        
        Returns:
            q: Q-value [batch, 1]
        """
        x = torch.cat([s, a], dim=-1)
        return self.model(x)


class BCQAgent(nn.Module):
    """Complete BCQ agent combining VAE, perturbation actor, and double Q-networks."""
    def __init__(self, state_dim: int, action_dim: int = 1, latent_dim: int = 8,
                 hidden: Tuple[int, ...] = (256, 256), xi: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.xi = xi
        
        # VAE for action distribution
        self.vae = ActionVAE(state_dim, action_dim, latent_dim, hidden)
        
        # Perturbation actor
        self.actor = PerturbationActor(state_dim, action_dim, hidden, xi)
        
        # Double Q-networks
        self.q1 = QNetwork(state_dim, action_dim, hidden)
        self.q2 = QNetwork(state_dim, action_dim, hidden)
        
        # Target networks
        self.q1_target = QNetwork(state_dim, action_dim, hidden)
        self.q2_target = QNetwork(state_dim, action_dim, hidden)
        
        # Copy initial weights to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
    
    def soft_update_targets(self, tau: float = 0.005):
        """Soft update target networks: theta_target <- tau * theta + (1 - tau) * theta_target."""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def select_action(self, s: torch.Tensor, num_candidates: int = 10) -> torch.Tensor:
        """Select action using BCQ: sample from VAE, perturb, pick best by Q1.
        
        Args:
            s: state tensor [batch, state_dim]
            num_candidates: number of candidate actions to sample
        
        Returns:
            a: selected action [batch, action_dim]
        """
        batch_size = s.shape[0]
        
        # Sample candidate actions from VAE
        a_candidates = self.vae.sample(s, num_samples=num_candidates)  # [batch * num_candidates, action_dim]
        
        # Expand states for all candidates
        s_expanded = s.unsqueeze(1).expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, self.state_dim)
        
        # Perturb candidates
        delta = self.actor(s_expanded, a_candidates)
        a_perturbed = torch.clamp(a_candidates + delta, -1.0, 1.0)  # Assuming action space is normalized to [-1, 1]
        
        # Evaluate with Q values
        q1 = self.q1(s_expanded, a_perturbed)
        q2 = self.q2(s_expanded, a_perturbed)
        q_values = torch.min(q1, q2).reshape(batch_size, num_candidates)
        
        # Select best action per state
        best_idx = q_values.argmax(dim=1)
        best_actions = a_perturbed.reshape(batch_size, num_candidates, self.action_dim)[torch.arange(batch_size), best_idx]
        
        return best_actions


def init_model_weights(model: nn.Module):
    """Apply standard initializations to a model in-place."""
    def init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    model.apply(init_weights)
    return model
