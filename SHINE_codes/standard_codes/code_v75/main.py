import glob
import torch
import os
import h5py
import numpy as np
from EEG_Dataset import EEG_Dataset
from newmodel_v75_100Hz import SHINE
import config as cfg
import tqdm
import pandas as pd
from sklearn.metrics import f1_score
import argparse
import random
print(1)

# nb_blocks=6,emb = 64, output_dim=1,norm_params=True
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=199)
parser.add_argument('--nb_blocks', type=int, default=12)
parser.add_argument('--emb', type=int, default=64)
parser.add_argument('--conv_len', type=int, default=6)
parser.add_argument('--valid_num', type=int, default=0)
parser.add_argument('--eeglen', type=int, default=3000)

session_num = [
    [0,10,11],
    [1,2],
    [3,4,5,6,25],
    [7,8,9],
    [12,22,23],
    [13,14],
    [15,16,17,26,31],
    [18,20,24,29,35],
    [19,21],
    [27,38,54,55,56,57,66,67,68],
    [28,30,32,33,34,39],
    [36,37,46,47],
    [40,41],
    [42,43,44,45],
    [48,61,62],
    [49,50,51],
    [52,53],
    [58,59,60],
    [63,64,65],
    [69,72],
    [70,71,73,78],
    [74,75,76],
    [77,86,88,89,90],
    [79,80,81],
    [82,83,84],
    [85,87],
]



eeglen = parser.parse_args().eeglen
# saveckpt name
cdir = os.getcwd()
cd_ = cdir.split('/')[-1].split('_')[-1]
if 'gpfs' in cdir:
    model_saveckpt = './model/seed' + str(parser.parse_args().seed) + '_' + cd_ + '_online.ckpt'
else:
    model_saveckpt = './model/seed_' + str(parser.parse_args().seed) + '_' + cd_ + '_loc.ckpt'

if not os.path.exists(os.path.dirname(model_saveckpt)):
    os.makedirs(os.path.dirname(model_saveckpt))

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


# This is for ensemble model, you could change the seedstart to get different results using one GPU.
# So you could run "train_and_valid.py" for 100 times to get 100 results, and then run "ensemble_model.py" to get the final result.
# If you have many GPUs, you could run this code in each GPU by changing the seedstart, and then run "ensemble_model.py" to get the final result.
# For example, We have 20 GPUs, so we run this code in each GPU by changing the seedstart from 1*100 to 20*100, and then run "ensemble_model.py" to get the final result.
# We ensembled 520 models to get the final result.
myseed = parser.parse_args().seed

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

cdir = os.getcwd()

# for loc debug, we do not use all the data, just for debug
# if 'gpfs' not in cdir:
#     meglist = meglist[0:30]
#     envlist = envlist[0:30]
#     tsvlist = tsvlist[0:30]

torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
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
    print('megdata_i shape:', megdata_i.shape)
    print(megdata_i.shape[1]-envdata[i].shape[0])


day_index = [-1] * 91

# Populate the day_index list based on session_num
for day, sessions in enumerate(session_num):
    for session in sessions:
        day_index[session] = day

validation_set = [random.choice(sublist) for sublist in session_num]
validation_set = [val for val in validation_set if val not in [1, 2]]
validation_indices = random.sample(validation_set, 7)
validation_indices = validation_indices + [2]
validation_indices = sorted(validation_indices)

# else in 0-91 is trian
train_indices = [i for i in range(91) if i not in validation_indices]

train_days = [day_index[i] for i in train_indices]
valid_days = [day_index[i] for i in validation_indices]

cdir = os.getcwd()


train_meg = [megdata[i] for i in train_indices]
train_env = [envdata[i] for i in train_indices]
valid_meg = [megdata[i] for i in validation_indices]
valid_env = [envdata[i] for i in validation_indices]

train_dataset = EEG_Dataset(train_meg, train_env,eeglen)
valid_dataset = EEG_Dataset(valid_meg, valid_env,eeglen)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)


model = SHINE().to(cfg.device)
# model = SHINE().to(cfg.device)


optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
valid_corr_sum = -100
for epoch in range(cfg.epoch_num):

    # train the model
    num_correct = 0
    num_samples = 0
    train_loss = 0
    corr_sum = 0
    env_corr_sum = 0
    model.train()
    # ---------------------train---------------------
    for iter, (eeg,env) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):
        running_loss = 0.0
        # get the input
        eeg = eeg.to(cfg.device)
        env = env.to(cfg.device)

        pred = model(eeg)

        env_corr = correlation(pred, env)
        env_corr_sum += torch.mean(env_corr)

        loss = 1 - env_corr.mean()
        train_loss += loss
        # backward
        optimizer.zero_grad()  # clear the grad
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print('epoch:', epoch)
    print('train_loss:', train_loss / iter)
    print('env_corr_sum:', env_corr_sum / iter)


    # ---------------------valid---------------------
    model.eval()
    env_all = []
    pred_all = []
    env_corr_sum = 0

    for iter, (eeg,env) in enumerate(tqdm.tqdm(valid_loader, position=0, leave=True), start=1):
        with torch.no_grad():
            running_loss = 0.0
            # get the input
            eeg = eeg.to(cfg.device)
            env = env.to(cfg.device)
            pred = model(eeg)

            tmp_corr = correlation(pred, env)
            env_corr = tmp_corr
            loss = 1 - tmp_corr.mean()

            env_corr_sum += torch.mean(env_corr)

            corr_sum += 1 - torch.mean(loss)

    # save the model
    if env_corr_sum > valid_corr_sum:
        valid_corr_sum = env_corr_sum

        if not os.path.exists(os.path.dirname(model_saveckpt)):
            os.makedirs(os.path.dirname(model_saveckpt))
        torch.save(model.state_dict(), model_saveckpt)

    print('epoch:', epoch)
    print('env_corr_sum:', env_corr_sum / iter)
    print('max valid:', valid_corr_sum / iter)
    print('valid_num:', parser.parse_args().valid_num)

    print('\n')