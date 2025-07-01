import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Dict, List
from encoder import Encoder
from custom_tools.period import find_period


def exp_dataset(
    encoder: Encoder, dataset: Dict[str, NDArray[np.int64]]
) -> pd.DataFrame:
    """
    Run the encoder on a dataset and return the results as a DataFrame.

    Args:
        encoder (Encoder): The encoder to use.
        dataset (Dict[str, NDArray[np.int64]]): A dictionary where keys are dataset names
            and values are the corresponding data arrays.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the encoding.
    """
    results: List[pd.DataFrame] = []
    for name, data in dataset.items():
        print(f"Processing dataset: {name} with encoder: {encoder.get_name()}")
        result = encoder.exp(data)
        result["file"] = name
        results.append(result)

    return pd.concat(results, ignore_index=True)


def exp_datasets(
    encoder: Encoder, datasets: Dict[str, Dict[str, NDArray[np.int64]]]
) -> pd.DataFrame:
    """
    Run the encoder on multiple datasets and return the results as a DataFrame.

    Args:
        encoder (Encoder): The encoder to use.
        datasets (Dict[str, Dict[str, NDArray[np.int64]]]): A dictionary where keys are dataset names
            and values are dictionaries of data arrays.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the encoding for all datasets.
    """
    results: List[pd.DataFrame] = []
    for dataset_name, dataset in datasets.items():
        result = exp_dataset(encoder, dataset)
        result["dataset"] = dataset_name
        results.append(result)

    return pd.concat(results, ignore_index=True)


def exp_datasets_encoders(
    encoders: List[Encoder], datasets: Dict[str, Dict[str, NDArray[np.int64]]]
) -> pd.DataFrame:
    """
    Run multiple encoders on multiple datasets and return the results as a DataFrame.

    Args:
        encoders (List[Encoder]): A list of encoders to use.
        datasets (Dict[str, Dict[str, NDArray[np.int64]]]): A dictionary where keys are dataset names
            and values are dictionaries of data arrays.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the encoding for all encoders and datasets.
    """
    results: List[pd.DataFrame] = []
    for encoder in encoders:
        result = exp_datasets(encoder, datasets)
        results.append(result)

    return pd.concat(results, ignore_index=True)
