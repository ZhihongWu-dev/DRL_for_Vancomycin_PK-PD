"""Training scaffold for BCQ (Batch-Constrained Q-Learning).

Usage: python train_bcq.py --config configs/bcq_base.yaml

BCQ is an offline RL algorithm that:
1. Learns action distribution with VAE
2. Uses double Q-networks with constrained action selection
3. Learns bounded perturbations to improve actions
"""
import argparse
import yaml
import torch
import os
import time
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    # Fallback minimal writer if tensorboard is not installed
    class SummaryWriter:
        def __init__(self, log_dir=None):
            self.log_dir = log_dir
        def add_scalar(self, *args, **kwargs):
            return
        def close(self):
            return

from dataset import ReadyDataset, ReplayBuffer
from bcq_models import ActionVAE, PerturbationActor, QNetwork, init_model_weights
from bcq_train_utils import bcq_update_step, bcq_update_step_with_action_scaling
from utils import set_seed, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to config file")
    p.add_argument("--workdir", type=str, default=None, help="Working directory for logs/checkpoints")
    return p.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_action_stats(transitions_df):
    """Compute action min/max/mean/std for normalization."""
    actions = transitions_df['a'].values
    return {
        'min': float(np.min(actions)),
        'max': float(np.max(actions)),
        'mean': float(np.mean(actions)),
        'std': float(np.std(actions)),
    }


def normalize_actions(actions, action_min, action_max):
    """Normalize actions from [min, max] to [-1, 1]."""
    return 2.0 * (actions - action_min) / (action_max - action_min + 1e-8) - 1.0


def denormalize_actions(actions, action_min, action_max):
    """Denormalize actions from [-1, 1] to [min, max]."""
    return (actions + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min


def run_training(cfg: dict, workdir: str = None) -> str:
    """Run BCQ training job. Returns checkpoint path."""
    # Config parsing with defaults
    seed = cfg.get("train", {}).get("seed", 0)
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    # ========== Data Loading ==========
    data_cfg = cfg.get("data", {})
    if "dataframe" in data_cfg and data_cfg["dataframe"] is not None:
        df = data_cfg["dataframe"]
        state_cols = data_cfg.get("state_cols")
        ds = ReadyDataset.from_dataframe(df, state_cols=state_cols)
    else:
        path = data_cfg.get("path")
        sheet = data_cfg.get("sheet", 0)
        state_cols = data_cfg.get("state_cols")
        if path is None:
            raise ValueError("No data.path or data.dataframe provided in cfg")
        
        import pandas as pd
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            ds = ReadyDataset.from_dataframe(df, state_cols=state_cols)
        elif path.endswith(".xlsx") or path.endswith(".xls"):
            df = pd.read_excel(path, sheet_name=sheet)
            ds = ReadyDataset.from_dataframe(df, state_cols=state_cols)
        else:
            raise ValueError("Unsupported data file type: must be .csv or .xlsx")

    # Fit normalizer on states
    ds.fit_normalizer()
    trans = ds.to_transitions(normalize=True)
    
    print(f"Loaded {len(trans)} transitions")
    print(f"State dimension: {len(ds.state_cols)}")
    
    # Compute action statistics for normalization
    action_stats = compute_action_stats(trans)
    print(f"Action statistics (original): min={action_stats['min']:.4f}, max={action_stats['max']:.4f}, "
          f"mean={action_stats['mean']:.4f}, std={action_stats['std']:.4f}")
    
    # Normalize actions to [-1, 1]
    action_min = action_stats['min']
    action_max = action_stats['max']
    trans['a'] = trans['a'].apply(lambda x: normalize_actions(x, action_min, action_max))
    
    print(f"Action statistics (normalized): min={trans['a'].min():.4f}, max={trans['a'].max():.4f}, "
          f"mean={trans['a'].mean():.4f}, std={trans['a'].std():.4f}")

    # Replay buffer
    buf = ReplayBuffer(capacity=cfg.get("train", {}).get("buffer_capacity", max(1000, len(trans) * 10)))
    buf.add_batch(trans)

    # ========== Model Setup ==========
    model_cfg = cfg.get("model", {})
    state_dim = len(ds.state_cols)
    action_dim = 1
    latent_dim = model_cfg.get("latent_dim", 2)
    hidden = tuple(model_cfg.get("hidden", (256, 256)))
    xi = model_cfg.get("xi", 0.05)  # Perturbation limit
    
    print(f"\nModel configuration:")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}, Latent dim: {latent_dim}")
    print(f"  Hidden layers: {hidden}")
    print(f"  Perturbation limit (xi): {xi}")

    # Create networks
    vae = ActionVAE(state_dim, action_dim, latent_dim, hidden).to(device)
    actor = PerturbationActor(state_dim, action_dim, hidden, xi).to(device)
    q1 = QNetwork(state_dim, action_dim, hidden).to(device)
    q2 = QNetwork(state_dim, action_dim, hidden).to(device)
    q1_target = QNetwork(state_dim, action_dim, hidden).to(device)
    q2_target = QNetwork(state_dim, action_dim, hidden).to(device)

    # Initialize weights
    init_model_weights(vae)
    init_model_weights(actor)
    init_model_weights(q1)
    init_model_weights(q2)
    init_model_weights(q1_target)
    init_model_weights(q2_target)

    # Copy initial weights to target networks
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    # Optimizers
    lr = model_cfg.get("lr", 3e-4)
    vae_opt = torch.optim.Adam(vae.parameters(), lr=lr)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=lr)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=lr)

    # Training hyperparameters
    total_steps = cfg.get("train", {}).get("total_steps", 100)
    batch_size = cfg.get("train", {}).get("batch_size", 256)
    log_interval = cfg.get("train", {}).get("log_interval", 10)
    ckpt_interval = cfg.get("train", {}).get("ckpt_interval", 50)
    gamma = model_cfg.get("gamma", 0.99)
    tau = model_cfg.get("tau", 0.005)
    num_candidates = model_cfg.get("num_candidates", 10)

    # Logging
    workdir = workdir or cfg.get("workdir", f"runs/bcq_{int(time.time())}")
    os.makedirs(workdir, exist_ok=True)
    writer = SummaryWriter(log_dir=workdir)
    
    print(f"\nTraining configuration:")
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Gamma: {gamma}, Tau: {tau}")
    print(f"  Num candidates: {num_candidates}")
    print(f"  Workdir: {workdir}\n")

    # ========== Training Loop ==========
    print("Starting training...")
    for step in range(1, total_steps + 1):
        batch = buf.sample(min(batch_size, len(buf)), seed=int(time.time()) % (2 ** 31))
        
        # Move to device
        for k in batch:
            batch[k] = batch[k].to(device)

        # BCQ update step
        losses = bcq_update_step_with_action_scaling(
            batch,
            vae, actor, q1, q2, q1_target, q2_target,
            vae_opt, actor_opt, q1_opt, q2_opt,
            gamma=gamma,
            tau=tau,
            xi=xi,
            num_candidates=num_candidates,
            action_min=-1.0,
            action_max=1.0
        )

        # Logging
        if step % log_interval == 0 or step == 1:
            log_msg = f"[step {step:5d}] "
            log_msg += f"vae_loss={losses['vae_loss'].item():.6f} "
            log_msg += f"recon={losses['recon_loss'].item():.6f} "
            log_msg += f"kl={losses['kl_loss'].item():.6f} "
            log_msg += f"q1={losses['q1_loss'].item():.6f} "
            log_msg += f"q2={losses['q2_loss'].item():.6f} "
            log_msg += f"actor={losses['actor_loss'].item():.6f}"
            print(log_msg)
            
            writer.add_scalar("loss/vae", losses['vae_loss'].item(), step)
            writer.add_scalar("loss/recon", losses['recon_loss'].item(), step)
            writer.add_scalar("loss/kl", losses['kl_loss'].item(), step)
            writer.add_scalar("loss/q1", losses['q1_loss'].item(), step)
            writer.add_scalar("loss/q2", losses['q2_loss'].item(), step)
            writer.add_scalar("loss/actor", losses['actor_loss'].item(), step)

        # Checkpointing
        if step % ckpt_interval == 0 or step == total_steps:
            ckpt_path = os.path.join(workdir, f"ckpt_step{step}.pt")
            torch.save({
                "step": step,
                "vae_state": vae.state_dict(),
                "actor_state": actor.state_dict(),
                "q1_state": q1.state_dict(),
                "q2_state": q2.state_dict(),
                "q1_target_state": q1_target.state_dict(),
                "q2_target_state": q2_target.state_dict(),
                "vae_opt": vae_opt.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "q1_opt": q1_opt.state_dict(),
                "q2_opt": q2_opt.state_dict(),
                "cfg": cfg,
                "action_min": action_min,
                "action_max": action_max,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    writer.close()
    print(f"\nTraining finished!")
    print(f"Last checkpoint: {ckpt_path}")
    print(f"Logs saved to: {workdir}")
    
    return ckpt_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print("=" * 70)
    print("BCQ Training")
    print("=" * 70)
    print("Config loaded:")
    print(yaml.dump(cfg, default_flow_style=False))
    print("=" * 70)
    
    ckpt = run_training(cfg, workdir=args.workdir)
    print(f"\nTraining completed. Final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
