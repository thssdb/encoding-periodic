import os
from config import DATA_DIR, DATA_NO_PERIOD_DIR, EXP_RESULTS_DIR
from exp_utils.loader import load_datasets_from_directory
from exp_utils.benchmark import exp_datasets_encoders
from typing import List
from encoder import Encoder
from encoder.flea_ablation import FLEAEncoder

RESULT_ABLATION_P = "results_ablation_p.csv"
RESULT_NO_PERIOD_ABLATION_P = "results_no_period_ablation_p.csv"

if __name__ == "__main__":
    os.makedirs(EXP_RESULTS_DIR, exist_ok=True)
    datasets = load_datasets_from_directory(DATA_DIR, "value")
    datasets_no_period = load_datasets_from_directory(DATA_NO_PERIOD_DIR, "value")
    encoders: List[Encoder] = [
        FLEAEncoder(),
        FLEAEncoder(is_given_p=True, given_p=0),
        FLEAEncoder(is_given_p=True, given_p=0.2),
        FLEAEncoder(is_given_p=True, given_p=0.4),
        FLEAEncoder(is_given_p=True, given_p=0.6),
        FLEAEncoder(is_given_p=True, given_p=0.8),
        FLEAEncoder(is_given_p=True, given_p=1.0),
    ]
    results = exp_datasets_encoders(encoders, datasets)
    results.to_csv(EXP_RESULTS_DIR / RESULT_ABLATION_P, index=False)
    results_no_period = exp_datasets_encoders(encoders, datasets_no_period)
    results_no_period.to_csv(EXP_RESULTS_DIR / RESULT_NO_PERIOD_ABLATION_P, index=False)
