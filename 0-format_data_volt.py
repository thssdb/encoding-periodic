import os
import pandas as pd

if not os.path.exists("data"):
    os.makedirs("data")

data_dir = os.path.join("origin", "volt", "datanormal")
folders = os.listdir(data_dir)

for folder in folders:
    folder_full = os.path.join(data_dir, folder)
    files = os.listdir(folder_full)
    for file in files:
        filename = file.split(".")[0]
        file_full = os.path.join(folder_full, file)
        with open(file_full, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = [int(float(x) * 10**5) for x in lines]
        df = pd.DataFrame(data={"value": data})
        df.to_csv(os.path.join("data", f"volt_{folder}_{filename}.csv"), index=False)
