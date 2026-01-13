import torch
import numpy as np
from algorithms.iql.models import QNetwork, VNetwork, GaussianPolicy, init_model_weights
from algorithms.iql.utils import set_seed


def test_model_forward_shapes_and_init():
    set_seed(0)
    batch = 8
    state_dim = 5

    s = torch.randn(batch, state_dim)
    a = torch.randn(batch, 1)

    q = QNetwork(state_dim, action_dim=1, hidden=(32,32))
    v = VNetwork(state_dim, hidden=(32,32))
    pi = GaussianPolicy(state_dim, action_dim=1, hidden=(32,32))

    # apply weight init
    init_model_weights(q)
    init_model_weights(v)
    init_model_weights(pi)

    q_out = q(s, a)
    v_out = v(s)
    pi_mean, pi_std = pi(s)
    pi_sample = pi.sample(s)

    assert q_out.shape == (batch, 1)
    assert v_out.shape == (batch, 1)
    assert pi_mean.shape == (batch, 1)
    assert pi_std.shape == (batch, 1)
    assert pi_sample.shape == (batch, 1)

    # params are finite
    for p in list(q.parameters()) + list(v.parameters()) + list(pi.parameters()):
        assert torch.isfinite(p).all()