import numpy as np
import pandas as pd
import os
import time
from sklearn.metrics import f1_score
from algorithm.period_encode import period_direct_result
from algorithm.period_encode import period_result

DATA_LABELED_POS = "data_labeled"
RESULT_PATH = os.path.join("exp_result", "abnormal_detect.csv")


def abnormal_direct(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return ((data < mu - 3 * sigma) | (data > mu + 3 * sigma)).astype(int)


result = {"dataset": [], "name": [], "type": [], "score": [], "time": []}

files = os.listdir(DATA_LABELED_POS)
for file in files:
    dataset = file.split("_")[0]
    file_full = os.path.join(DATA_LABELED_POS, file)
    df = pd.read_csv(file_full)
    data = df["value"].to_numpy()

    y_true = df["label"].to_numpy()

    start = time.time()
    y_pred1 = abnormal_direct(data)
    end = time.time()
    result["dataset"].append(dataset)
    result["name"].append(file)
    result["type"].append("origin")
    result["score"].append(f1_score(y_true, y_pred1))
    result["time"].append(end - start)

    _, __, res = period_result(data.tolist())
    start = time.time()
    y_pred3 = abnormal_direct(res)
    end = time.time()
    result["dataset"].append(dataset)
    result["name"].append(file)
    result["type"].append("period")
    result["score"].append(f1_score(y_true, y_pred3))
    result["time"].append(end - start)

    start = time.time()
    _, __, res = period_direct_result(data.tolist())
    y_pred2 = abnormal_direct(res)
    end = time.time()
    result["dataset"].append(dataset)
    result["name"].append(file)
    result["type"].append("period_no_quan")
    result["score"].append(f1_score(y_true, y_pred2))
    result["time"].append(end - start)

df = pd.DataFrame(data=result)
df.to_csv(RESULT_PATH, index=False)
