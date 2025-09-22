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
import random
import importlib


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=199)
parser.add_argument('--valid_num', type=int, default=0)
parser.add_argument('--eeglen', type=int, default=3000)


eeglen = parser.parse_args().eeglen
# saveckpt name
cdir = os.getcwd()
cd_ = cdir.split('/')[-1].split('_')[-1]


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

envfiles = [x for x in envfiles if x.endswith('seq01_100Hz.npy')]
meglist = sorted(megfiles)
envlist = sorted(envfiles)
tsvlist = sorted(eventsfiles)

cdir = os.getcwd()

modellist = os.listdir('./SHINEs/')

# random select one model
modelname = random.choice(modellist)
modelname = modelname.replace('.py', '')
version = modelname.split('_')[1]  # Assuming the version is the last part of the model name
module_name = f"SHINEs.{modelname}"
class_name = f"SHINE"  # Assuming the class name follows this pattern
# Dynamically import the module
module = importlib.import_module(module_name)
model_class = getattr(module, class_name)
# Instantiate the model
model = model_class()
model = model.to(cfg.device)
model_saveckpt = './model/seed_' + str(myseed) + '_' + version + '_online.ckpt'
if not os.path.exists(os.path.dirname(model_saveckpt)):
    os.makedirs(os.path.dirname(model_saveckpt))
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





validation_indices = [random.choice(range(91)) for _ in range(6)]
validation_indices = sorted(validation_indices)

# else in 0-91 is trian
train_indices = [i for i in range(91) if i not in validation_indices]
train_indices = sorted(train_indices)

cdir = os.getcwd()


train_meg = [megdata[i] for i in train_indices]
train_env = [envdata[i] for i in train_indices]
valid_meg = [megdata[i] for i in validation_indices]
valid_env = [envdata[i] for i in validation_indices]

train_dataset = EEG_Dataset(train_meg, train_env,eeglen)
valid_dataset = EEG_Dataset(valid_meg, valid_env,eeglen)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)


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