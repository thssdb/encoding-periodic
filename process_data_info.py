import numpy as np
import pandas as pd
from config import DATA_DIR, DATA_NO_PERIOD_DIR, EXP_RESULTS_DIR
from exp_utils.loader import load_datasets_from_directory
from pathlib import Path
from typing import List


def process_data_info(dir: Path) -> pd.DataFrame:
    datasets = load_datasets_from_directory(dir, "value")
    data_info: List[pd.DataFrame] = []

    for dataset_name, dataset in datasets.items():
        series_count = len(dataset)
        total_length = sum(len(data) for data in dataset.values())
        data_info.append(
            pd.DataFrame(
                {
                    "dataset": [dataset_name],
                    "series_count": [series_count],
                    "total_length": [total_length],
                }
            )
        )

    return pd.concat(data_info, ignore_index=True)


if __name__ == "__main__":
    data_info = process_data_info(DATA_DIR)
    data_info_no_period = process_data_info(DATA_NO_PERIOD_DIR)
    data_info.to_latex(EXP_RESULTS_DIR / "data_info.tex", index=False)
    data_info_no_period.to_latex(
        EXP_RESULTS_DIR / "data_info_no_period.tex", index=False
    )
    print(data_info["total_length"].sum() + data_info_no_period["total_length"].sum())
