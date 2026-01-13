import pandas as pd
import numpy as np
from algorithms.iql.dataset import ReadyDataset


def test_fit_transform():
    df = pd.DataFrame({
        "stay_id": [1,1,2],
        "step_4hr": [1,2,1],
        "totalamount_mg": [100, 0, 50],
        "step_reward": [0.0, 1.0, 0.5],
        "vanco_level": [10.0, 12.0, 8.0],
        "creatinine": [1.0, 1.1, 0.9]
    })
    state_cols = ["vanco_level","creatinine"]
    ds = ReadyDataset.from_dataframe(df, state_cols=state_cols)
    mean, std = ds.fit_normalizer()
    trans = ds.to_transitions(normalize=True)
    s0 = trans.loc[0,'s']
    # after normalization, s should be finite and shaped correctly
    assert s0.shape[0] == 2
    assert np.isfinite(s0).all()
