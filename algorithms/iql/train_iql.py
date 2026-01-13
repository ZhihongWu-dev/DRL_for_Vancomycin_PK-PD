"""Training scaffold for IQL (skeleton)

Usage: python train_iql.py --config configs/iql_base.yaml
"""
import argparse
import yaml
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


import os
import time
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    # Fallback minimal writer if tensorboard is not installed (keeps tests runnable)
    class SummaryWriter:
        def __init__(self, log_dir=None):
            self.log_dir = log_dir
        def add_scalar(self, *args, **kwargs):
            return
        def close(self):
            return
from algorithms.iql.dataset import ReadyDataset, ReplayBuffer
from algorithms.iql.models import QNetwork, VNetwork, GaussianPolicy, init_model_weights
from algorithms.iql.train_utils import iql_update_step
from algorithms.iql.utils import set_seed, get_device


def run_training(cfg: dict, workdir: str = None) -> str:
    """Run a short training job using cfg dict. Returns checkpoint path."""
    # config parsing with defaults
    seed = cfg.get("train", {}).get("seed", 0)
    set_seed(seed)
    device = get_device()

    # data
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

    ds.fit_normalizer()
    trans = ds.to_transitions(normalize=True)

    # replay buffer
    buf = ReplayBuffer(capacity=cfg.get("train", {}).get("buffer_capacity", max(1000, len(trans) * 10)))
    buf.add_batch(trans)

    # model and optimizer setup
    model_cfg = cfg.get("model", {})
    state_dim = len(ds.state_cols)
    hidden = tuple(model_cfg.get("hidden", (256, 256)))
    q_net = QNetwork(state_dim, action_dim=1, hidden=hidden).to(device)
    v_net = VNetwork(state_dim, hidden=hidden).to(device)
    pi = GaussianPolicy(state_dim, action_dim=1, hidden=hidden).to(device)

    init_model_weights(q_net)
    init_model_weights(v_net)
    init_model_weights(pi)

    lr = model_cfg.get("lr", 3e-4)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=lr)
    pi_opt = torch.optim.Adam(pi.parameters(), lr=lr)

    total_steps = cfg.get("train", {}).get("total_steps", 100)
    batch_size = cfg.get("train", {}).get("batch_size", 256)
    log_interval = cfg.get("train", {}).get("log_interval", 10)
    ckpt_interval = cfg.get("train", {}).get("ckpt_interval", 50)

    # logging
    workdir = workdir or cfg.get("workdir", f"runs/iql_{int(time.time())}")
    os.makedirs(workdir, exist_ok=True)
    writer = SummaryWriter(log_dir=workdir)

    # training loop
    for step in range(1, total_steps + 1):
        batch = buf.sample(min(batch_size, len(buf)), seed=int(time.time()) % (2 ** 31))
        # move to device
        for k in batch:
            batch[k] = batch[k].to(device)

        losses = iql_update_step(batch, q_net, v_net, pi, q_opt, v_opt, pi_opt,
                                 gamma=model_cfg.get("gamma", 0.99),
                                 tau=model_cfg.get("tau", 0.7),
                                 beta=model_cfg.get("beta", 3.0),
                                 weight_clip=model_cfg.get("weight_clip", 1e2))

        if step % log_interval == 0 or step == 1:
            print(f"[step {step}] q_loss={losses['q_loss'].item():.6f} v_loss={losses['v_loss'].item():.6f} pi_loss={losses['pi_loss'].item():.6f}")
            writer.add_scalar("loss/q", losses['q_loss'].item(), step)
            writer.add_scalar("loss/v", losses['v_loss'].item(), step)
            writer.add_scalar("loss/pi", losses['pi_loss'].item(), step)

        if step % ckpt_interval == 0 or step == total_steps:
            ckpt_path = os.path.join(workdir, f"ckpt_step{step}.pt")
            torch.save({
                "step": step,
                "q_state": q_net.state_dict(),
                "v_state": v_net.state_dict(),
                "pi_state": pi.state_dict(),
                "q_opt": q_opt.state_dict(),
                "v_opt": v_opt.state_dict(),
                "pi_opt": pi_opt.state_dict(),
                "cfg": cfg,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    writer.close()
    return ckpt_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print("Config loaded:", cfg)
    ckpt = run_training(cfg)
    print("Training finished, last checkpoint:", ckpt)


if __name__ == "__main__":
    main()
