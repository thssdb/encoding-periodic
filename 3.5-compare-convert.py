import numpy as np
import pandas as pd
import os

INPUT_PATH = os.path.join("exp_result", "exp_compare.csv")
RESULT_PATH = os.path.join("exp_result", "exp_compare_draw.csv")

df = pd.read_csv(INPUT_PATH)
new_df = df.groupby(["dataset", "algorithm"])["compress_ratio"].mean().reset_index()
new_df.to_csv(RESULT_PATH, index=False)
