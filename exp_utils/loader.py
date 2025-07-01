import os
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pathlib import Path
from typing import Dict


def load_csv(file_path: Path, value_column: str) -> NDArray[np.int64]:
    """
    Load a CSV file and return the specified column as a NumPy array.

    Args:
        file_path (Path): The path to the CSV file.
        value_column (str): The name of the column to extract values from.

    Returns:
        NDArray[np.int64]: A NumPy array containing the values from the specified column.
    """
    df = pd.read_csv(file_path)
    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in {file_path}")
    return df[value_column].to_numpy(dtype=np.int64)


def load_datasets_from_directory(
    directory: Path, value_column: str
) -> Dict[str, Dict[str, NDArray[np.int64]]]:
    datasets: Dict[str, Dict[str, NDArray[np.int64]]] = {}
    for file in directory.glob("*.csv"):
        dataset_name = file.stem.split("_")[0]
        if dataset_name not in datasets:
            datasets[dataset_name] = {}
        datasets[dataset_name][file.stem] = load_csv(file, value_column)
    return datasets
