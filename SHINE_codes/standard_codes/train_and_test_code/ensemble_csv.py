# Ensemble models
from itertools import groupby

import os
import pandas as pd
import glob
import numpy as np
import torch
import h5py

result_csvs = []
result_dir = './result_csv/'

for csv_file in glob.glob(os.path.join(result_dir, '*.csv')):
    result_csvs.append(csv_file)

result_csvs = sorted(result_csvs)
# read data
datas = []
for csv_file in result_csvs:
    df = pd.read_csv(csv_file)
    # read data
    data = df['speech_prob'].values
    datas.append(data)


datas = np.array(datas)
# calculate mean
mean_data = np.mean(datas, axis=0)
mean_data2 = np.mean(datas, axis=1)

# Some models may fail to learn the result.
for i in range(len(mean_data2)):
    if mean_data2[i] <0.7:
        print('error')

# Binarization
mean_data[mean_data >= 0.5] = 1
mean_data[mean_data < 0.5] = 0

final_pred = torch.tensor(mean_data, dtype=int)


import csv
output_path = "final_v4_25.csv"
with open(output_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["idx", "speech_prob"])

    for idx, tensor in enumerate(final_pred):
        # Ensure we extract the scalar float from tensor
        pred = tensor.item() if isinstance(
            tensor, torch.Tensor) else float(tensor)
        writer.writerow([idx, pred])
