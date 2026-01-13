import pandas as pd
import numpy as np
from algorithms.iql.dataset import ReadyDataset


def make_small_df():
    data = {
        "stay_id": [1,1,2],
        "step_4hr": [1,2,1],
        "totalamount_mg": [500, 0, 250],
        "step_reward": [0.0, 1.0, 0.5],
        "vanco_level": [10.0, 12.0, 8.0],
        "creatinine": [1.0, 1.1, 0.9]
    }
    return pd.DataFrame(data)


def test_transitions_shape():
    df = make_small_df()
    state_cols = ["vanco_level", "creatinine"]
    ds = ReadyDataset.from_dataframe(df, state_cols=state_cols)
    trans = ds.to_transitions()
    assert set(["s","a","r","s_next","done"]) <= set(trans.columns)
    # number of transitions should be rows in dataset
    assert len(trans) == len(df)
    # s shape
    assert isinstance(trans.loc[0, "s"], np.ndarray)