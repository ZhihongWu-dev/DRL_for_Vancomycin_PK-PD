import torch
import numpy as np
import pandas as pd
from algorithms.iql.dataset import ReadyDataset, ReplayBuffer
from algorithms.iql.models import QNetwork, VNetwork, GaussianPolicy
from algorithms.iql.losses import expectile_loss


def test_smoke_iql_step():
    torch.manual_seed(0)
    np.random.seed(0)
    # create tiny dataset
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

    # sample a batch
    batch = buf.sample(4, seed=1)
    s = batch['s']
    a = batch['a']
    r = batch['r']
    s_next = batch['s_next']
    done = batch['done']

    state_dim = s.shape[1]
    q_net = QNetwork(state_dim, action_dim=1, hidden=(32,32))
    v_net = VNetwork(state_dim, hidden=(32,32))
    pi = GaussianPolicy(state_dim, action_dim=1, hidden=(32,32))

    q_opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=1e-3)
    pi_opt = torch.optim.Adam(pi.parameters(), lr=1e-3)

    gamma = 0.99
    tau = 0.7
    beta = 3.0

    # Q update
    q_vals = q_net(s, a)
    with torch.no_grad():
        v_next = v_net(s_next)
        q_targets = r + gamma * v_next * (1 - done)
    q_loss = torch.nn.functional.mse_loss(q_vals, q_targets)
    q_opt.zero_grad(); q_loss.backward(); q_opt.step()

    # V expectile update
    q_vals_detach = q_net(s, a).detach()
    v_vals = v_net(s)
    v_loss = expectile_loss(q_vals_detach - v_vals, tau)
    v_opt.zero_grad(); v_loss.backward(); v_opt.step()

    # policy update (adv-weighted regression)
    with torch.no_grad():
        adv = (q_net(s, a).detach() - v_net(s).detach())
        weights = torch.exp(adv / beta)
        weights = torch.clamp(weights, 0.0, 1e2)
        weights = weights / (weights.sum() + 1e-8)
    pi_actions = pi.sample(s)
    pi_loss = (weights * (pi_actions - a) ** 2).mean()
    pi_opt.zero_grad(); pi_loss.backward(); pi_opt.step()

    # losses should be finite
    assert torch.isfinite(q_loss)
    assert torch.isfinite(v_loss)
    assert torch.isfinite(pi_loss)
