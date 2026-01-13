import numpy as np
from algorithms.iql.dataset import ReplayBuffer


def make_dummy_transition(i):
    return {"s": np.array([i, i+0.1]), "a": float(i), "r": float(i%2), "s_next": np.array([i+1, i+1.1]), "done": 0}


def test_deterministic_sampling():
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        buf.add(make_dummy_transition(i))
    b1 = buf.sample(5, seed=42)
    b2 = buf.sample(5, seed=42)
    # Sampling with same seed should be identical
    assert (b1['a'] == b2['a']).all()


def test_add_batch_and_len():
    import pandas as pd
    df = pd.DataFrame([make_dummy_transition(i) for i in range(10)])
    buf = ReplayBuffer(capacity=50)
    buf.add_batch(df)
    assert len(buf) == 10
