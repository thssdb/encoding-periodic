import os
import pandas as pd

if not os.path.exists("exp_result"):
    os.makedirs("exp_result")

count = {}
length = {}

files = os.listdir("data")
for file in files:
    dataset = file.split("_")[0]
    if dataset not in count.keys():
        count[dataset] = 0
        length[dataset] = 0
    count[dataset] += 1
    file_pull = os.path.join("data", file)
    df = pd.read_csv(file_pull)
    length[dataset] += len(df["value"].tolist())

result = {"dataset": [], "count": [], "length": []}

for dataset in count.keys():
    result["dataset"].append(dataset)
    result["count"].append(count[dataset])
    result["length"].append(length[dataset])

df = pd.DataFrame(data=result)
df.to_csv(os.path.join("exp_result", "stat.csv"), index=False)
