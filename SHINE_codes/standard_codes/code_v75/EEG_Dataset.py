import torch
from torch.utils.data import Dataset
import numpy as np
import random


class EEG_Dataset(Dataset):
    def __init__(self, eegs, envs,eeglen):
        self.eegs = eegs
        self.envs = envs
        self.seg = np.zeros((len(self.envs),1))
        self.eeglen = eeglen
        for index in range(len(self.envs)):
            env = self.envs[index]
            if len(env)<eeglen:
                self.seg[index] = 0
            else:
                self.seg[index] = int((len(env))/self.eeglen)

    def __len__(self):
        return int(np.sum(self.seg))

    def find_index(self, target):
        acc = 0
        for i, num in enumerate(self.seg):
            acc += num
            if acc > target:
                return i, int(target - acc + num)

    def __getitem__(self, index):

        random_value = random.randint(-2, 2)
        nparry_index, time_index = self.find_index(index)

        alleeg = self.eegs[nparry_index]
        allenv = self.envs[nparry_index]
        eeg = alleeg[:,self.eeglen*time_index:self.eeglen*time_index+self.eeglen]
        if random_value+self.eeglen * time_index<0 or random_value+self.eeglen * time_index+self.eeglen>len(allenv):
            random_value = 0
        env = allenv[random_value+self.eeglen * time_index:random_value+self.eeglen * time_index+self.eeglen]

        eeg = torch.tensor(eeg, dtype=torch.float32)
        eeg_mean = torch.mean(eeg, dim=1, keepdim=True)
        eeg_std = torch.std(eeg, dim=1, keepdim=True)
        eeg = (eeg - eeg_mean) / eeg_std
        env = torch.tensor(env, dtype=torch.float32)

        return eeg,env

