"""Evaluation script for trained BCQ model.

Functionality:
1. Load checkpoint
2. Compute offline metrics:
   - Average Q values (Q1, Q2)
   - Q value statistics
   - Behavior policy performance
   - BCQ policy performance (with action selection)
   - Greedy policy performance (max Q)
3. Save evaluation results
"""
import argparse
import yaml
import torch
import numpy as np
import json
from pathlib import Path

from dataset import ReadyDataset
from bcq_models import ActionVAE, PerturbationActor, QNetwork
from utils import get_device


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(ckpt_path: str, state_dim: int, action_dim: int, latent_dim: int, hidden: tuple, xi: float):
    """Load checkpoint and return BCQ networks."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    vae = ActionVAE(state_dim, action_dim, latent_dim, hidden)
    actor = PerturbationActor(state_dim, action_dim, hidden, xi)
    q1 = QNetwork(state_dim, action_dim, hidden)
    q2 = QNetwork(state_dim, action_dim, hidden)
    
    vae.load_state_dict(ckpt['vae_state'])
    actor.load_state_dict(ckpt['actor_state'])
    q1.load_state_dict(ckpt['q1_state'])
    q2.load_state_dict(ckpt['q2_state'])
    
    vae.eval()
    actor.eval()
    q1.eval()
    q2.eval()
    
    # Extract action scaling if available
    action_min = ckpt.get('action_min', -1.0)
    action_max = ckpt.get('action_max', 1.0)
    
    return vae, actor, q1, q2, action_min, action_max


def denormalize_actions(actions, action_min, action_max):
    """Denormalize actions from [-1, 1] to [min, max]."""
    return (actions + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min


def evaluate_offline(dataset: ReadyDataset, vae, actor, q1, q2, 
                     action_min: float, action_max: float,
                     gamma: float = 0.99, num_candidates: int = 10):
    """Compute offline evaluation metrics.
    
    Metrics:
    - Q value statistics (mean, std, min, max)
    - Behavior policy Q values (dataset actions)
    - BCQ policy Q values (VAE + perturb + select)
    - Greedy policy Q values (argmax Q)
    - MC returns from dataset
    """
    device = next(q1.parameters()).device
    
    # Get transitions (normalized states, normalized actions)
    trans_df = dataset.to_transitions(normalize=True)
    
    # Convert to tensors
    states = torch.FloatTensor(np.stack(trans_df['s'].values)).to(device)
    actions = torch.FloatTensor(trans_df['a'].values).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(trans_df['r'].values).to(device)
    next_states = torch.FloatTensor(np.stack(trans_df['s_next'].values)).to(device)
    dones = torch.FloatTensor(trans_df['done'].values).to(device)
    
    batch_size = len(states)
    
    with torch.no_grad():
        # ========== Q Value Statistics ==========
        q1_values = q1(states, actions)
        q2_values = q2(states, actions)
        q_min = torch.min(q1_values, q2_values)
        
        q1_mean = q1_values.mean().item()
        q1_std = q1_values.std().item()
        q1_min = q1_values.min().item()
        q1_max = q1_values.max().item()
        
        q2_mean = q2_values.mean().item()
        q2_std = q2_values.std().item()
        q2_min = q2_values.min().item()
        q2_max = q2_values.max().item()
        
        q_min_mean = q_min.mean().item()
        
        # Behavior policy: Q values for dataset actions
        behavior_q = q_min.mean().item()
        
        # ========== BCQ Policy (VAE + Perturb + Select) ==========
        a_candidates = vae.sample(states, num_samples=num_candidates)
        a_candidates = torch.clamp(a_candidates, -1.0, 1.0)
        
        states_expanded = states.unsqueeze(1).expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, states.shape[1])
        
        delta = actor(states_expanded, a_candidates)
        a_perturbed = a_candidates + delta
        a_perturbed = torch.clamp(a_perturbed, -1.0, 1.0)
        
        q1_perturbed = q1(states_expanded, a_perturbed).reshape(batch_size, num_candidates)
        q2_perturbed = q2(states_expanded, a_perturbed).reshape(batch_size, num_candidates)
        q_perturbed = torch.min(q1_perturbed, q2_perturbed)
        
        best_idx = q_perturbed.argmax(dim=1)
        a_bcq = a_perturbed.reshape(batch_size, num_candidates, 1)[torch.arange(batch_size), best_idx]
        
        q1_bcq = q1(states, a_bcq)
        q2_bcq = q2(states, a_bcq)
        q_bcq = torch.min(q1_bcq, q2_bcq)
        
        bcq_q_mean = q_bcq.mean().item()
        bcq_q_std = q_bcq.std().item()
        
        # ========== Greedy Policy (max Q over action samples) ==========
        # Sample many actions and pick best by Q1
        n_greedy_samples = 50
        a_greedy_candidates = vae.sample(states, num_samples=n_greedy_samples)
        a_greedy_candidates = torch.clamp(a_greedy_candidates, -1.0, 1.0)
        
        states_greedy_expanded = states.unsqueeze(1).expand(-1, n_greedy_samples, -1).reshape(batch_size * n_greedy_samples, states.shape[1])
        
        q1_greedy = q1(states_greedy_expanded, a_greedy_candidates).reshape(batch_size, n_greedy_samples)
        best_greedy_idx = q1_greedy.argmax(dim=1)
        a_greedy = a_greedy_candidates.reshape(batch_size, n_greedy_samples, 1)[torch.arange(batch_size), best_greedy_idx]
        
        q1_greedy_final = q1(states, a_greedy)
        q2_greedy_final = q2(states, a_greedy)
        q_greedy = torch.min(q1_greedy_final, q2_greedy_final)
        
        greedy_q_mean = q_greedy.mean().item()
        greedy_q_std = q_greedy.std().item()
        
        # ========== MC Returns (Actual dataset returns) ==========
        mc_returns = []
        current_return = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            current_return = rewards[i].item() + gamma * current_return * (1 - dones[i].item())
            mc_returns.insert(0, current_return)
        
        mc_return_mean = np.mean(mc_returns)
        mc_return_std = np.std(mc_returns)
        
        # ========== Action Statistics ==========
        # Denormalize for reporting
        actions_original = denormalize_actions(actions.cpu().numpy(), action_min, action_max)
        a_bcq_original = denormalize_actions(a_bcq.cpu().numpy(), action_min, action_max)
        a_greedy_original = denormalize_actions(a_greedy.cpu().numpy(), action_min, action_max)
        
        behavior_action_mean = actions_original.mean()
        behavior_action_std = actions_original.std()
        
        bcq_action_mean = a_bcq_original.mean()
        bcq_action_std = a_bcq_original.std()
        
        greedy_action_mean = a_greedy_original.mean()
        greedy_action_std = a_greedy_original.std()
    
    results = {
        'num_transitions': len(trans_df),
        'num_episodes': len(dataset.episodes()),
        
        'q1_stats': {
            'mean': q1_mean,
            'std': q1_std,
            'min': q1_min,
            'max': q1_max,
        },
        'q2_stats': {
            'mean': q2_mean,
            'std': q2_std,
            'min': q2_min,
            'max': q2_max,
        },
        'q_min_stats': {
            'mean': q_min_mean,
        },
        
        'behavior_policy': {
            'q_mean': behavior_q,
            'action_mean': behavior_action_mean,
            'action_std': behavior_action_std,
        },
        
        'bcq_policy': {
            'q_mean': bcq_q_mean,
            'q_std': bcq_q_std,
            'action_mean': bcq_action_mean,
            'action_std': bcq_action_std,
        },
        
        'greedy_policy': {
            'q_mean': greedy_q_mean,
            'q_std': greedy_q_std,
            'action_mean': greedy_action_mean,
            'action_std': greedy_action_std,
        },
        
        'mc_returns': {
            'mean': mc_return_mean,
            'std': mc_return_std,
        },
        
        'dataset_rewards': {
            'mean': rewards.mean().item(),
            'std': rewards.std().item(),
            'min': rewards.min().item(),
            'max': rewards.max().item(),
        },
    }
    
    # Compute improvements
    results['improvements'] = {
        'bcq_vs_behavior': ((bcq_q_mean - behavior_q) / (abs(behavior_q) + 1e-8)) * 100,
        'greedy_vs_behavior': ((greedy_q_mean - behavior_q) / (abs(behavior_q) + 1e-8)) * 100,
        'greedy_vs_bcq': ((greedy_q_mean - bcq_q_mean) / (abs(bcq_q_mean) + 1e-8)) * 100,
    }
    
    return results


def print_evaluation(results: dict):
    """Print evaluation results in a readable format."""
    print("=" * 80)
    print("BCQ OFFLINE EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\nDataset Information:")
    print(f"  Transitions: {results['num_transitions']}")
    print(f"  Episodes: {results['num_episodes']}")
    
    print(f"\nDataset Rewards:")
    print(f"  Mean: {results['dataset_rewards']['mean']:10.4f}")
    print(f"  Std:  {results['dataset_rewards']['std']:10.4f}")
    print(f"  Min:  {results['dataset_rewards']['min']:10.4f}")
    print(f"  Max:  {results['dataset_rewards']['max']:10.4f}")
    
    print(f"\nQ-Network Statistics:")
    print(f"  Q1: mean={results['q1_stats']['mean']:10.4f}, std={results['q1_stats']['std']:10.4f}, "
          f"min={results['q1_stats']['min']:10.4f}, max={results['q1_stats']['max']:10.4f}")
    print(f"  Q2: mean={results['q2_stats']['mean']:10.4f}, std={results['q2_stats']['std']:10.4f}, "
          f"min={results['q2_stats']['min']:10.4f}, max={results['q2_stats']['max']:10.4f}")
    print(f"  Min(Q1,Q2): mean={results['q_min_stats']['mean']:10.4f}")
    
    print(f"\nBehavior Policy (Dataset Actions):")
    print(f"  Q-value: {results['behavior_policy']['q_mean']:10.4f}")
    print(f"  Action mean: {results['behavior_policy']['action_mean']:10.4f}")
    print(f"  Action std:  {results['behavior_policy']['action_std']:10.4f}")
    
    print(f"\nBCQ Policy (VAE + Perturb + Select):")
    print(f"  Q-value: {results['bcq_policy']['q_mean']:10.4f} (±{results['bcq_policy']['q_std']:10.4f})")
    print(f"  Action mean: {results['bcq_policy']['action_mean']:10.4f}")
    print(f"  Action std:  {results['bcq_policy']['action_std']:10.4f}")
    
    print(f"\nGreedy Policy (Max Q over VAE samples):")
    print(f"  Q-value: {results['greedy_policy']['q_mean']:10.4f} (±{results['greedy_policy']['q_std']:10.4f})")
    print(f"  Action mean: {results['greedy_policy']['action_mean']:10.4f}")
    print(f"  Action std:  {results['greedy_policy']['action_std']:10.4f}")
    
    print(f"\nMonte Carlo Returns (from dataset):")
    print(f"  Mean: {results['mc_returns']['mean']:10.4f}")
    print(f"  Std:  {results['mc_returns']['std']:10.4f}")
    
    print(f"\nPolicy Improvements:")
    print(f"  BCQ vs Behavior:    {results['improvements']['bcq_vs_behavior']:+7.2f}%")
    print(f"  Greedy vs Behavior: {results['improvements']['greedy_vs_behavior']:+7.2f}%")
    print(f"  Greedy vs BCQ:      {results['improvements']['greedy_vs_bcq']:+7.2f}%")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BCQ model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    parser.add_argument("--output", type=str, default=None, help="Path to save results (optional)")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load config
    cfg = load_config(args.config)
    print(f"Config loaded from: {args.config}")
    
    # Load data
    data_cfg = cfg.get("data", {})
    path = data_cfg.get("path")
    state_cols = data_cfg.get("state_cols")
    
    print(f"Loading data from: {path}")
    import pandas as pd
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=data_cfg.get("sheet", 0))
    
    dataset = ReadyDataset(df=df, state_cols=state_cols)
    dataset.fit_normalizer()
    
    state_dim = len(state_cols)
    action_dim = 1
    
    # Load model
    model_cfg = cfg.get("model", {})
    latent_dim = model_cfg.get("latent_dim", 2)
    hidden = tuple(model_cfg.get("hidden", (256, 256)))
    xi = model_cfg.get("xi", 0.05)
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    vae, actor, q1, q2, action_min, action_max = load_checkpoint(
        args.checkpoint, state_dim, action_dim, latent_dim, hidden, xi
    )
    
    # Move to device
    vae = vae.to(device)
    actor = actor.to(device)
    q1 = q1.to(device)
    q2 = q2.to(device)
    
    print(f"Action scaling: [{action_min:.4f}, {action_max:.4f}]")
    
    # Evaluate
    print("\nStarting evaluation...")
    results = evaluate_offline(
        dataset, vae, actor, q1, q2,
        action_min, action_max,
        gamma=model_cfg.get("gamma", 0.99),
        num_candidates=model_cfg.get("num_candidates", 10)
    )
    
    # Print results
    print_evaluation(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
