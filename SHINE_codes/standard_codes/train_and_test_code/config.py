import torch
import os

device_ids = 0
cdir = os.getcwd()

if 'gpfs' in cdir:
    device_ids = 0
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
epoch_num = 15
batch_size = 4


lr=1e-3
weight_decay=0.01





