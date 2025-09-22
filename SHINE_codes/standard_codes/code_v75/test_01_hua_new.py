from scipy.ndimage import gaussian_filter, laplace
import glob
import torch
import os
import h5py
import numpy as np
from EEG_Dataset import EEG_Dataset

import config as cfg
import tqdm
import pandas as pd
from sklearn.metrics import f1_score
import argparse
from itertools import groupby


def correlation(y_pred, y_true):

    x = y_pred
    y = y_true

    xmean = torch.mean(x, 1, keepdim=True)
    ymean = torch.mean(y, 1, keepdim=True)
    vx = x - xmean
    vy = y - ymean
    cov = torch.sum(vx * vy, 1, keepdim=True)
    fenmu = (torch.sqrt(torch.sum(vx ** 2, 1, keepdim=True)) * torch.sqrt(
        torch.sum(vy ** 2, 1, keepdim=True)) + 1e-6)
    corr = cov / fenmu

    return corr

def moving_average(signal, window_size):
    signal_mean = np.convolve(signal[0], np.ones(window_size)/window_size, mode='valid')
    # padding
    padding = (window_size - 1) // 2
    signal_mean = np.pad(signal_mean, (padding, padding), mode='edge')
    return signal_mean

def thresholding(signal, threshold):
    return (signal > threshold).astype(int)

def sharpen_signal(signal):
    return signal - laplace(signal)

def optimize_predictions(predictions, env_final, small_value=0.22, mean_value=0.23, large_value=0.24):
    predlist = []

    # ƽ������
    window_size = 31
    smoothed_predictions = moving_average(predictions, window_size)
    # smoothed_predictions = sharpen_signal(predictions)[0]
    # smoothed_predictions = predictions[0]

    # ���ɶ�ֵ�����
    pred_all_sort = np.sort(smoothed_predictions, axis=0)
    pred_all_sort_1_4_small = pred_all_sort[int(len(pred_all_sort) * small_value)]
    pred_all_sort_1_4_mean = pred_all_sort[int(len(pred_all_sort) * mean_value)]
    pred_all_sort_1_4_large = pred_all_sort[int(len(pred_all_sort) * large_value)]

    # 25%
    pred_final_small = smoothed_predictions.copy()
    pred_final_small[smoothed_predictions < pred_all_sort_1_4_small] = 0
    pred_final_small[smoothed_predictions >= pred_all_sort_1_4_small] = 1
    predlist.append(pred_final_small)

    # ����25%���
    final_acc = np.sum(pred_final_small == env_final) / len(env_final)
    print('Final acc (25%):', final_acc)
    f1_macro = f1_score(env_final, pred_final_small, average='macro')
    print(f"F1-macro score (25%): {f1_macro}")

    # 50%
    pred_final_mean = smoothed_predictions.copy()
    pred_final_mean[smoothed_predictions < pred_all_sort_1_4_mean] = 0
    pred_final_mean[smoothed_predictions >= pred_all_sort_1_4_mean] = 1
    predlist.append(pred_final_mean)

    # ����50%���
    final_acc = np.sum(pred_final_mean == env_final) / len(env_final)
    print('Final acc (50%):', final_acc)
    f1_macro = f1_score(env_final, pred_final_mean, average='macro')
    print(f"F1-macro score (50%): {f1_macro}")

    # 75%
    pred_final_large = smoothed_predictions.copy()
    pred_final_large[smoothed_predictions < pred_all_sort_1_4_large] = 0
    pred_final_large[smoothed_predictions >= pred_all_sort_1_4_large] = 1
    predlist.append(pred_final_large)

    # ����75%���
    final_acc = np.sum(pred_final_large == env_final) / len(env_final)
    print('Final acc (75%):', final_acc)
    f1_macro = f1_score(env_final, pred_final_large, average='macro')
    print(f"F1-macro score (75%): {f1_macro}")

    return predlist




def flip_short_segments(seq, v, threshold):
    result = []
    for val, group in groupby(seq):
        group_list = list(group)
        length = len(group_list)

        if val == v and length < threshold:
            result.extend([1 - val] * length)
        else:
            result.extend(group_list)

    return np.array(result)
parser = argparse.ArgumentParser()
parser.add_argument('--eeglen', type=int, default=3000)
parser.add_argument('--valid_num', type=int, default=0)
parser.add_argument('--seed', type=int, default=2025)

eeglen = parser.parse_args().eeglen

cdir = os.getcwd()
cd_ = cdir.split('/')[-1].split('_')[-1]
if 'gpfs' in cdir:
    model_saveckpt = './model/seed' + str(parser.parse_args().seed) + '_' + cd_ + '_online.ckpt'
else:
    model_saveckpt = './model/seed_' + str(parser.parse_args().seed) + '_' + cd_ + '_loc.ckpt'

model_saveckpt = './model/seed10043_v20_online.ckpt'
# This is for ensemble model, you could change the seedstart to get different results using one GPU.
# So you could run "train_and_valid.py" for 100 times to get 100 results, and then run "ensemble_model.py" to get the final result.
# If you have many GPUs, you could run this code in each GPU by changing the seedstart, and then run "ensemble_model.py" to get the final result.
# For example, We have 20 GPUs, so we run this code in each GPU by changing the seedstart from 1*100 to 20*100, and then run "ensemble_model.py" to get the final result.
# We ensembled 520 models to get the final result.
seedstart = 0*100
seedend = seedstart + 100

if 'gpfs' in cdir:
    rootdir = '../../../../NIPS2025/libribrain/data/'
else:
    rootdir = '../../libribrain/data/'

megfiles = []
eventsfiles = []
envfiles = []
for Sherlock in range(1,8):
    megdir = rootdir + 'Sherlock' + str(Sherlock) + '/derivatives/lowpass100/'
    events_dir = rootdir + 'Sherlock' + str(Sherlock) + '/derivatives/events/'
    seq01_dir = rootdir + 'Sherlock' + str(Sherlock) + '/derivatives/seq01_100Hz/'
    if not os.path.exists(seq01_dir):
        os.makedirs(seq01_dir)

    meglist = [x for x in glob.glob(os.path.join(megdir, "*"))]
    eventslist = [x for x in glob.glob(os.path.join(events_dir, "*"))]
    seq01list = [x for x in glob.glob(os.path.join(seq01_dir, "*"))]

    megfiles = megfiles + meglist
    eventsfiles = eventsfiles + eventslist
    envfiles = envfiles + seq01list

meglist = sorted(megfiles)
envlist = sorted(envfiles)
tsvlist = sorted(eventsfiles)

meglist = meglist[0:3]
envlist = envlist[0:3]
tsvlist = tsvlist[0:3]

myseed = 42
device = cfg.device

torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(myseed)

megdata = []
for megfile in meglist:
    with h5py.File(megfile, "r") as f:
        variable_path = "data"
        data = f[variable_path][()]
        megdata.append(data)

for i in range(len(tsvlist)):
    tsvpath = tsvlist[i]
    df = pd.read_csv(tsvpath, sep='\t')
    meg_start = df.loc[0, 'timemeg']
    megdata_i = megdata[i]
    megdata_i = megdata_i[:, int(meg_start * 100):]
    megdata[i] = megdata_i

envdata = []
for envfile in envlist:
    # npy
    env_datai = np.load(envfile)
    envdata.append(env_datai[0])

# aling the meg and env

for i in range(len(envdata)):
    megdata_i = megdata[i]
    megdata_i = megdata_i[:, :envdata[i].shape[0]]
    megdata[i] = megdata_i


# split training,validation,and test


test_meg = megdata[2]
test_env = envdata[2]


test_meg = torch.tensor((test_meg), dtype=torch.float32)
test_env = torch.tensor((test_env), dtype=torch.float32)



# load the model
model_dir = './model/'
modellist = [x for x in glob.glob(os.path.join(model_dir, "*"))]
predlist = []

print('model list:', len(modellist))

# modellist = modellist[0:1]
for model_saveckpt in modellist:
    # if 'v10' not in model_saveckpt:
    #     continue
    with torch.no_grad():
        if 'v75' or'sota' in model_saveckpt:
            from newmodel_v75_100Hz import SHINE
            model = SHINE().to(cfg.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            model.load_state_dict(torch.load(model_saveckpt, map_location=cfg.device))

        # elif 'v20' in model_saveckpt:
        #     from newmodel_v20 import SHINE
        #     model = SHINE().to(cfg.device)
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        #     model.load_state_dict(torch.load(model_saveckpt, map_location=cfg.device))
        #
        # elif 'v10' in model_saveckpt:
        #     from newmodel_v10 import SHINE
        #     model = SHINE().to(cfg.device)
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        #     model.load_state_dict(torch.load(model_saveckpt, map_location=cfg.device))
        #
        # elif 'v24' in model_saveckpt:
        #     from newmodel_v9 import SHINE
        #     model = SHINE().to(cfg.device)
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        #     model.load_state_dict(torch.load(model_saveckpt, map_location=cfg.device))
        else:
            continue

        # test
        model.eval()
        test_loss = 0
        corr_sum = 0
        env_all = []
        pred_all = []

        predictions = []

        envs = []
        test_meg_0 = test_meg[:, :eeglen]
        test_meg_0_mean = torch.mean(test_meg_0, dim=1, keepdim=True)
        test_meg_0_std = torch.std(test_meg_0, dim=1, keepdim=True)
        test_meg_i = (test_meg_0 - test_meg_0_mean) / (test_meg_0_std)
        test_meg_i = test_meg_i.to(device)
        test_meg_i = test_meg_i.unsqueeze(0)  # Add batch dimension

        seg01 = model(test_meg_i)
        seg01 = seg01.cpu().detach().numpy()
        # 0-1 norm
        seg01 = (seg01 - np.min(seg01)) / (np.max(seg01) - np.min(seg01))

        predictions.append(seg01[:,:250])
        test_env_0 = test_env[:250]
        envs.append(test_env_0.cpu().numpy())
        print(model_saveckpt)
        for i in range(test_meg.shape[1]//(eeglen-500)+1):

            test_meg_i = test_meg[:, i*(eeglen-500):i*(eeglen-500)+eeglen]
            test_meg_i_mean = torch.mean(test_meg_i, dim=1, keepdim=True)
            test_meg_i_std = torch.std(test_meg_i, dim=1, keepdim=True)
            test_meg_i = (test_meg_i - test_meg_i_mean) / (test_meg_i_std)

            test_meg_i = test_meg_i.to(device)
            test_meg_i = test_meg_i.unsqueeze(0)  # Add batch dimension

            seg01 = model(test_meg_i)
            seg01 = seg01.cpu().detach().numpy()
            seg01_norm = (seg01 - np.min(seg01)) / (np.max(seg01) - np.min(seg01))
            predictions.append(seg01_norm[:,250:eeglen-250])
            test_env_i = test_env[i*(eeglen-500)+250:i*(eeglen-500)+(eeglen-250)]
            envs.append(test_env_i.cpu().numpy())

        predictions = np.concatenate(predictions, axis=1)
        # plot predictions

        envs = np.concatenate(envs, axis=0)

        env_final = envs
        pred_final = predictions[0]

        predictions_sharp = sharpen_signal(pred_final)

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(15, 5))
        # plt.plot(predictions[0][0:10000], label='Predictions')
        # plt.plot(env_final[0:10000], label='Predictions')
        # predictions_sharp_norm = (predictions_sharp - np.min(predictions_sharp)) / (np.max(predictions_sharp) - np.min(predictions_sharp))
        # plt.plot(predictions_sharp_norm[0:10000], label='Predictions_sharp')
        # plt.show()

        small_value = 0.22
        mean_value = 0.23
        large_value = 0.24


        predlist_one = optimize_predictions(predictions, env_final)
        predlist.extend(predlist_one)
        print(1)

# Concatenate all predictions
pred_final_np = np.array(predlist)
# Average the predictions
pred_final_np_mean = np.mean(pred_final_np, axis=0)
# Binarization
pred_final_np_mean[pred_final_np_mean < 0.5] = 0
pred_final_np_mean[pred_final_np_mean >= 0.5] = 1

# flip_short_segments
pred_final_np_mean_flip = flip_short_segments(pred_final_np_mean, 1, 60)
pred_final_np_mean_flip = flip_short_segments(pred_final_np_mean_flip, 0, 20)

# final pred accuracy
final_acc = np.sum(pred_final_np_mean == env_final) / len(env_final)
print('ensemble_Final acc:', final_acc)

# F1-macro score
f1_macro = f1_score(env_final, pred_final_np_mean, average='macro')
print(f"ensemble_F1-macro score: {f1_macro}")

final_acc = np.sum(pred_final_np_mean_flip == env_final) / len(env_final)
print('ensemble_Final acc:', final_acc)

# F1-macro score
f1_macro = f1_score(env_final, pred_final_np_mean_flip, average='macro')
print(f"ensemble_F1-macro score: {f1_macro}")


print(1)