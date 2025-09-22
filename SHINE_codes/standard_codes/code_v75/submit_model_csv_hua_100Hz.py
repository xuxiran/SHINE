# output the csv

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import glob
import os
import h5py
import numpy as np
import config as cfg
import csv
import torch.nn.functional as F
import argparse

def moving_average(signal, window_size):
    signal_mean = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
    # padding
    padding = (window_size - 1) // 2
    signal_mean = np.pad(signal_mean, (padding, padding), mode='edge')
    return signal_mean

def myupsample(predictions, output_len):
    # predictions to torch
    predictions = torch.tensor(predictions, dtype=torch.float32).unsqueeze(0)

    predictions = F.interpolate(
        predictions,
        scale_factor=2.5,
        mode='linear',
        align_corners=False
    )

    predictions = predictions.squeeze(0).cpu().detach().numpy()
    predictions = predictions[0][:output_len]

    # avg pooling
    predictions = moving_average(predictions, 31)

    return predictions

def gen_csv(output_path,final_pred):
    final_pred = torch.tensor(final_pred, dtype=int)
    # 560638
    final_pred = final_pred[:560638]  # Ensure we only keep the first 560638 elements

    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["idx", "speech_prob"])

        for idx, tensor in enumerate(final_pred):
            # Ensure we extract the scalar float from tensor
            pred = tensor.item() if isinstance(
                tensor, torch.Tensor) else float(tensor)
            writer.writerow([idx, pred])

    print(output_path)

parser = argparse.ArgumentParser()
parser.add_argument('--eeglen', type=int, default=3000)
eeglen = parser.parse_args().eeglen

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

meg_dir = '/home/xxr/NIPS2025/libribrain_test/holdout_new/lowpass100/'

meglist = [x for x in glob.glob(os.path.join(meg_dir, "*"))]

for megfile in meglist:
    with h5py.File(megfile, "r") as f:
        variable_path = "data"
        data = f[variable_path][()]

test_meg = torch.tensor(data, dtype=torch.float32)

if test_meg.shape[1] < 75*eeglen:
    padding = torch.zeros((test_meg.shape[0], 75*eeglen - test_meg.shape[1]), dtype=torch.float32)
    test_meg = torch.cat((test_meg, padding), dim=1)

# load the model
from newmodel_v75_100Hz import SHINE
model = SHINE().to(device)

# load the model

output_dir = './result_csv/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_dir = './model/'
modellist = [x for x in glob.glob(os.path.join(model_dir, "*"))]
# modellist = modellist[0:3]
predlist = []
print('model1',len(modellist))

for model_path in modellist:
    print(f'Loading model from {model_path}')
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    predictions = []
    for i in range(test_meg.shape[1]//(eeglen-500)):
        # print(i)
        test_meg_i = test_meg[:, i*(eeglen-500):i*(eeglen-500)+eeglen]
        test_meg_i_mean = torch.mean(test_meg_i, dim=1, keepdim=True)
        test_meg_i_std = torch.std(test_meg_i, dim=1, keepdim=True)
        test_meg_i = (test_meg_i - test_meg_i_mean) / (test_meg_i_std)

        test_meg_i = test_meg_i.to(device)
        test_meg_i = test_meg_i.unsqueeze(0)  # Add batch dimension

        seg01 = model(test_meg_i)
        seg01 = seg01.cpu().detach().numpy()

        if i == 0:
            predictions.append(seg01[:,:eeglen-250])
        else:
            predictions.append(seg01[:,250:eeglen-250])

        # final_pred[i*eeglen:i*eeglen+eeglen] = seg01[0]
    predictions = np.concatenate(predictions, axis=1)
    final_pred = myupsample(predictions, 560638)

    small_value = 0.22
    mean_value = 0.23
    large_value = 0.24

    pred_all_sort = np.sort(final_pred)

    pred_all_sort_1_4_small = pred_all_sort[int(len(pred_all_sort) * small_value)]
    # Binarization
    # 25%
    pred_final_small = final_pred.copy()
    pred_final_small[final_pred < pred_all_sort_1_4_small] = 0
    pred_final_small[final_pred >= pred_all_sort_1_4_small] = 1
    path_small = model_path.replace('.ckpt', '_small.csv').replace('./model',output_dir)
    gen_csv(path_small, pred_final_small)

    # mean
    pred_all_sort_1_4_mean = pred_all_sort[int(len(pred_all_sort) * mean_value)]
    # Binarization
    pred_final_mean = final_pred.copy()
    pred_final_mean[final_pred < pred_all_sort_1_4_mean] = 0
    pred_final_mean[final_pred >= pred_all_sort_1_4_mean] = 1
    path_mean = model_path.replace('.ckpt', '_mean.csv').replace('./model',output_dir)
    gen_csv(path_mean, pred_final_mean)

    # 75%
    pred_all_sort_1_4_large = pred_all_sort[int(len(pred_all_sort) * large_value)]
    # Binarization
    pred_final_large = final_pred.copy()
    pred_final_large[final_pred < pred_all_sort_1_4_large] = 0
    pred_final_large[final_pred >= pred_all_sort_1_4_large] = 1
    path_large = model_path.replace('.ckpt', '_large.csv').replace('./model',output_dir)
    gen_csv(path_large, pred_final_large)





