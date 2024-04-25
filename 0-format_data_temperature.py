import os
import numpy as np
import pandas as pd

df = pd.read_csv(os.path.join("origin", "temperature", "city_temperature.csv"))
# df = df[df["Country"] == "China"]

dfs = [(key, group) for key, group in df.groupby("City")]
for key, df in dfs:
    data = pd.DataFrame(data={"value": (df["AvgTemperature"] * 10).astype(np.int64)})
    data.to_csv(os.path.join("data", f"temperature_{key}.csv"), index=False)
