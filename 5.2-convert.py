import numpy as np
import pandas as pd
import os

INPUT_PATH = os.path.join("exp_result", "prediction.csv")
OUTPUT_PATH = os.path.join("exp_result", "prediction_draw.csv")

df = pd.read_csv(INPUT_PATH)
new_df = df.groupby(["dataset", "type"])[["mse", "time"]].mean().reset_index()
new_df.to_csv(OUTPUT_PATH, index=False)
