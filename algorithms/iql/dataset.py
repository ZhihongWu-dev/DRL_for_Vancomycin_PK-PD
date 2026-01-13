"""Dataset and ReplayBuffer for READY dataset.

Provides:
- ReadyDataset: load from Excel/CSV or DataFrame and converts episodes to transitions
- ReplayBuffer: simple in-memory circular buffer that samples transitions
"""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ReadyDataset:
    """Wraps a DataFrame of ready data and yields transitions.

    Expected columns: stay_id, step_4hr, totalamount_mg, step_reward, <state features...>

    Features:
      - supports fitting a simple normalizer (mean/std) on state columns
      - provides numeric transforms for states
    """

    def __init__(self, df: pd.DataFrame, state_cols: List[str], action_col: str = "totalamount_mg", reward_col: str = "step_reward"):
        self.df = df.copy()
        self.state_cols = state_cols
        self.action_col = action_col
        self.reward_col = reward_col
        self._state_mean = None
        self._state_std = None

    @staticmethod
    def from_excel(path: str, sheet_name: str = 0, state_cols: List[str] = None, **kwargs) -> "ReadyDataset":
        df = pd.read_excel(path, sheet_name=sheet_name)
        if state_cols is None:
            state_cols = [c for c in df.columns if c not in ["stay_id", "step_4hr", "totalamount_mg", "step_reward"]]
        return ReadyDataset(df, state_cols)

    @staticmethod
    def from_dataframe(df: pd.DataFrame, state_cols: List[str], **kwargs) -> "ReadyDataset":
        return ReadyDataset(df, state_cols, **kwargs)

    def episodes(self) -> Dict[str, pd.DataFrame]:
        """Return dict of stay_id -> df for that episode (sorted by step)."""
        eps = {}
        for sid, g in self.df.groupby("stay_id"):
            g2 = g.sort_values("step_4hr")
            eps[sid] = g2
        return eps

    def fit_normalizer(self, center: bool = True, scale: bool = True):
        """Compute mean/std on state columns and store them."""
        arr = self.df[self.state_cols].to_numpy(dtype=float)
        # handle constant columns safely
        mean = np.nanmean(arr, axis=0) if center else np.zeros(arr.shape[1])
        std = np.nanstd(arr, axis=0) if scale else np.ones(arr.shape[1])
        std[std == 0] = 1.0
        self._state_mean = mean
        self._state_std = std
        return self._state_mean, self._state_std

    def transform_state(self, s: np.ndarray) -> np.ndarray:
        """Apply stored normalization to a single state or array of states."""
        if self._state_mean is None or self._state_std is None:
            raise RuntimeError("Normalizer not fitted. Call fit_normalizer() first.")
        return (s - self._state_mean) / self._state_std

    def to_transitions(self, normalize: bool = False) -> pd.DataFrame:
        """Convert episodes to transition rows: s, a, r, s', done"""
        rows = []
        for sid, ep in self.episodes().items():
            ep = ep.reset_index(drop=True)
            for i in range(len(ep)):
                s = ep.loc[i, self.state_cols].to_numpy(dtype=float)
                # Skip transitions with NaN in state
                if np.isnan(s).any():
                    continue
                a = float(ep.loc[i, self.action_col]) if not pd.isna(ep.loc[i, self.action_col]) else 0.0
                r = float(ep.loc[i, self.reward_col]) if not pd.isna(ep.loc[i, self.reward_col]) else 0.0
                if i + 1 < len(ep):
                    s_next = ep.loc[i + 1, self.state_cols].to_numpy(dtype=float)
                    # Skip if next state has NaN
                    if np.isnan(s_next).any():
                        continue
                    done = 0
                else:
                    s_next = np.zeros_like(s)
                    done = 1
                if normalize:
                    if self._state_mean is None or self._state_std is None:
                        self.fit_normalizer()
                    s = self.transform_state(s)
                    s_next = self.transform_state(s_next)
                rows.append({"s": s, "a": a, "r": r, "s_next": s_next, "done": done})
        return pd.DataFrame(rows)


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000):
        self.capacity = int(capacity)
        self._storage = []
        self._pos = 0

    def add(self, transition: Dict):
        """Add a single transition. transition should have keys: s, a, r, s_next, done"""
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._pos] = transition
            self._pos = (self._pos + 1) % self.capacity

    def add_batch(self, transitions: pd.DataFrame):
        """Add multiple transitions from DataFrame (columns: s,a,r,s_next,done)."""
        for idx in range(len(transitions)):
            row = transitions.iloc[idx]
            # Explicitly convert state arrays to float32
            s = np.asarray(row["s"], dtype=np.float32)
            s_next = np.asarray(row["s_next"], dtype=np.float32)
            transition = {
                "s": s,
                "a": float(row["a"]),
                "r": float(row["r"]),
                "s_next": s_next,
                "done": int(row["done"])
            }
            self.add(transition)

    def sample(self, batch_size: int, seed: int = None) -> Dict[str, torch.Tensor]:
        if len(self._storage) == 0:
            raise RuntimeError("ReplayBuffer is empty")
        if batch_size > len(self._storage):
            raise ValueError("batch_size larger than buffer size")
        rng = np.random.RandomState(seed) if seed is not None else np.random
        idx = rng.choice(len(self._storage), size=batch_size, replace=False)
        batch = [self._storage[i] for i in idx]
        
        s = torch.tensor(np.stack([b["s"] for b in batch]), dtype=torch.float32)
        a = torch.tensor([b["a"] for b in batch], dtype=torch.float32).unsqueeze(-1)
        r = torch.tensor([b["r"] for b in batch], dtype=torch.float32).unsqueeze(-1)
        s_next = torch.tensor(np.stack([b["s_next"] for b in batch]), dtype=torch.float32)
        done = torch.tensor([b["done"] for b in batch], dtype=torch.float32).unsqueeze(-1)
        return {"s": s, "a": a, "r": r, "s_next": s_next, "done": done}

    def __len__(self):
        return len(self._storage)
