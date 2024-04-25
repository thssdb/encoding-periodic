import os
import numpy as np
import pandas as pd

points = {"dianwang": 4, "guoshou": 0, "hangxin": -1, "liantong": 0, "yinlian": 1}

files = os.listdir("origin_labeled")
for file in files:
    dataset = file.split("_")[0]
    if points[dataset] == -1:
        continue
    df = pd.read_csv(os.path.join("origin_labeled", file))
    new_df = pd.DataFrame(
        data={
            "value": (df["value"] * 10 ** points[dataset]).astype(np.int64),
            "label": df["label"].astype(np.int64),
        }
    )
    new_df.to_csv(os.path.join("data_labeled", file), index=False)
