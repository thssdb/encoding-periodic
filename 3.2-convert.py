import os
import pandas as pd
import matplotlib.pyplot as plt

exp_result_path = os.path.join("exp_result", "exp.csv")
df = pd.read_csv(exp_result_path)
algorithms = df["algorithm"].drop_duplicates().tolist()

compress_ratio_df = pd.DataFrame(df["dataset"].drop_duplicates().reset_index())
del compress_ratio_df["index"]
encoding_time_df = pd.DataFrame(df["dataset"].drop_duplicates().reset_index())
del encoding_time_df["index"]
decoding_time_df = pd.DataFrame(df["dataset"].drop_duplicates().reset_index())
del decoding_time_df["index"]

for algorithm in algorithms:
    current_df = df[df["algorithm"] == algorithm]

    result_df = current_df.groupby("dataset")["compress_ratio"].mean().reset_index()
    result_df[algorithm] = result_df["compress_ratio"]
    del result_df["compress_ratio"]
    compress_ratio_df = compress_ratio_df.merge(
        result_df,
        on="dataset",
    )

    result_df = current_df.groupby("dataset")["encoding_time"].mean().reset_index()
    result_df[algorithm] = result_df["encoding_time"]
    del result_df["encoding_time"]
    encoding_time_df = encoding_time_df.merge(
        result_df,
        on="dataset",
    )

    result_df = current_df.groupby("dataset")["decoding_time"].mean().reset_index()
    result_df[algorithm] = result_df["decoding_time"]
    del result_df["decoding_time"]
    decoding_time_df = decoding_time_df.merge(
        result_df,
        on="dataset",
    )

compress_ratio_df.to_csv(os.path.join("exp_result", "compress_ratio.csv"), index=False)
encoding_time_df.to_csv(os.path.join("exp_result", "encoding_time.csv"), index=False)
decoding_time_df.to_csv(os.path.join("exp_result", "decoding_time.csv"), index=False)
