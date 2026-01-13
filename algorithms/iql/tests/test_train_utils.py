import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from algorithms.iql.dataset import ReadyDataset, ReplayBuffer
from algorithms.iql.models import QNetwork, VNetwork, GaussianPolicy, init_model_weights
from algorithms.iql.train_utils import iql_update_step
from algorithms.iql.utils import set_seed


def test_iql_update_changes_params_and_returns_finite_losses():
    set_seed(0)
    # tiny synthetic dataset
    df = pd.DataFrame({
        "stay_id": [1,1,1,1],
        "step_4hr": [1,2,3,4],
        "totalamount_mg": [100, 0, 50, 0],
        "step_reward": [0.0, 1.0, 0.5, 0.0],
        "vanco_level": [10.0, 12.0, 9.0, 11.0],
        "creatinine": [1.0, 1.1, 1.0, 1.2]
    })
    state_cols = ["vanco_level","creatinine"]
    ds = ReadyDataset.from_dataframe(df, state_cols=state_cols)
    ds.fit_normalizer()
    trans = ds.to_transitions(normalize=True)

    buf = ReplayBuffer(capacity=100)
    buf.add_batch(trans)

    batch = buf.sample(4, seed=2)

    state_dim = batch['s'].shape[1]
    q_net = QNetwork(state_dim, action_dim=1, hidden=(32,32))
    v_net = VNetwork(state_dim, hidden=(32,32))
    pi = GaussianPolicy(state_dim, action_dim=1, hidden=(32,32))

    init_model_weights(q_net)
    init_model_weights(v_net)
    init_model_weights(pi)

    q_opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=1e-3)
    pi_opt = torch.optim.Adam(pi.parameters(), lr=1e-3)

    # record copies of parameters (detach to produce leaf tensors so deepcopy works)
    q_params_before = deepcopy([p.detach().clone() for p in q_net.parameters()])
    v_params_before = deepcopy([p.detach().clone() for p in v_net.parameters()])
    pi_params_before = deepcopy([p.detach().clone() for p in pi.parameters()])

    losses = iql_update_step(batch, q_net, v_net, pi, q_opt, v_opt, pi_opt, gamma=0.99, tau=0.7, beta=3.0)

    # losses finite
    assert torch.isfinite(losses['q_loss'])
    assert torch.isfinite(losses['v_loss'])
    assert torch.isfinite(losses['pi_loss'])

    # parameters changed
    q_changed = any(not torch.allclose(p0, p1) for p0, p1 in zip(q_params_before, q_net.parameters()))
    v_changed = any(not torch.allclose(p0, p1) for p0, p1 in zip(v_params_before, v_net.parameters()))
    pi_changed = any(not torch.allclose(p0, p1) for p0, p1 in zip(pi_params_before, pi.parameters()))

    assert q_changed and v_changed and pi_changed