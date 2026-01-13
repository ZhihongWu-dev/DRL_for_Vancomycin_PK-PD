import pandas as pd
import yaml
import os
from tempfile import TemporaryDirectory
from algorithms.iql.train_iql import run_training


def test_run_short_training_and_checkpoint_created():
    # prepare tiny ready dataset as CSV
    df = pd.DataFrame({
        "stay_id": [1,1,1],
        "step_4hr": [1,2,3],
        "totalamount_mg": [100, 0, 50],
        "step_reward": [0.0, 1.0, 0.5],
        "vanco_level": [10.0, 12.0, 9.0],
        "creatinine": [1.0, 1.1, 1.0]
    })
    with TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "ready.csv")
        df.to_csv(data_path, index=False)

        cfg = {
            "data": {"path": data_path, "state_cols": ["vanco_level", "creatinine"]},
            "train": {"seed": 0, "total_steps": 3, "batch_size": 2, "log_interval": 1, "ckpt_interval": 2, "buffer_capacity": 100},
            "model": {"hidden": [16, 16], "lr": 1e-3, "gamma": 0.99, "tau": 0.7, "beta": 3.0},
            "workdir": tmp
        }
        ckpt = run_training(cfg, workdir=tmp)
        assert os.path.exists(ckpt)
        # checkpoint file should contain keys
        import torch
        d = torch.load(ckpt)
        assert "q_state" in d and "v_state" in d and "pi_state" in d
