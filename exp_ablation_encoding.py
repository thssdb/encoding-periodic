import os
from config import DATA_DIR, DATA_NO_PERIOD_DIR, EXP_RESULTS_DIR
from exp_utils.loader import load_datasets_from_directory
from exp_utils.benchmark import exp_datasets_encoders
from typing import List
from encoder import Encoder
from encoder.flea_ablation import FLEAEncoder

RESULT_ABLATION_ENCODING = "results_ablation_encoding.csv"
RESULT_NO_PERIOD_ABLATION_ENCODING = "results_no_period_ablation_encoding.csv"

if __name__ == "__main__":
    os.makedirs(EXP_RESULTS_DIR, exist_ok=True)
    datasets = load_datasets_from_directory(DATA_DIR, "value")
    datasets_no_period = load_datasets_from_directory(DATA_NO_PERIOD_DIR, "value")
    encoders: List[Encoder] = [
        FLEAEncoder(simple_residual=True),
        FLEAEncoder(simple_frequency=True),
        FLEAEncoder(),
    ]
    results = exp_datasets_encoders(encoders, datasets)
    results.to_csv(EXP_RESULTS_DIR / RESULT_ABLATION_ENCODING, index=False)
    results_no_period = exp_datasets_encoders(encoders, datasets_no_period)
    results_no_period.to_csv(
        EXP_RESULTS_DIR / RESULT_NO_PERIOD_ABLATION_ENCODING, index=False
    )
