import os
from config import DATA_DIR, DATA_NO_PERIOD_DIR, EXP_RESULTS_DIR
from exp_utils.loader import load_datasets_from_directory
from exp_utils.benchmark import exp_datasets_encoders
from typing import List
from encoder import Encoder
from encoder.flea import FLEAEncoder
from encoder.gorilla import GorillaEncoder
from encoder.chimp import ChimpEncoder
from encoder.rle import RLEEncoder
from encoder.sprintz import SprintzEncoder
from encoder.buff import BuffEncoder
from encoder.hire import HireEncoder

RESULT_MAIN = "results_main.csv"
RESULT_NO_PERIOD_MAIN = "results_no_period_main.csv"

if __name__ == "__main__":
    os.makedirs(EXP_RESULTS_DIR, exist_ok=True)
    datasets = load_datasets_from_directory(DATA_DIR, "value")
    datasets_no_period = load_datasets_from_directory(DATA_NO_PERIOD_DIR, "value")
    encoders: List[Encoder] = [
        FLEAEncoder(),
        GorillaEncoder(),
        ChimpEncoder(),
        RLEEncoder(),
        SprintzEncoder(),
        BuffEncoder(),
        HireEncoder(),
    ]
    results = exp_datasets_encoders(encoders, datasets)
    results.to_csv(EXP_RESULTS_DIR / RESULT_MAIN, index=False)
    results_no_period = exp_datasets_encoders(encoders, datasets_no_period)
    results_no_period.to_csv(EXP_RESULTS_DIR / RESULT_NO_PERIOD_MAIN, index=False)
