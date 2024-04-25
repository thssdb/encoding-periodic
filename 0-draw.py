import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import os
import shutil

file_list = os.listdir("data")

if os.path.exists("fig"):
    shutil.rmtree("fig")

os.makedirs("fig")

for file in file_list:
    data = pd.read_csv(os.path.join("data", file))
    plt.plot(data["value"])
    plt.savefig(os.path.join("fig", file.replace("csv", "png")))
    plt.clf()
