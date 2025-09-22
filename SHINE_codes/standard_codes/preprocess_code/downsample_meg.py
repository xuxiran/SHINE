import glob
import torch
import os
import h5py
import mne
import argparse

def lh_meg(meg_data):
    info = mne.create_info(ch_names=['MEG' + str(i) for i in range(306)], sfreq=250, ch_types='mag')
    # Create a RawArray object to hold the MEG data
    raw = mne.io.RawArray(meg_data, info)

    # Apply a 50Hz low-pass filter
    raw_lowpass = raw.copy().filter(l_freq=None, h_freq=50, fir_design='firwin')

    # Downsample to 100Hz
    raw_lowpass.resample(100, npad="auto")

    # Get the processed data
    lowpass_data = raw_lowpass.get_data()

    return lowpass_data



# saveckpt name
cdir = os.getcwd()
cd_ = cdir.split('/')[-1].split('_')[-1]


rootdir = '../../libribrain/data/'

megfiles = []
eventsfiles = []
envfiles = []
for Sherlock in range(1,8):
    megdir = rootdir + 'Sherlock' + str(Sherlock) + '/derivatives/serialised/'
    meglist = [x for x in glob.glob(os.path.join(megdir, "*"))]
    megfiles = megfiles + meglist


meglist = sorted(megfiles)


megcnt = 0
for megfile in meglist:
    with h5py.File(megfile, "r") as f:
        print(f"Loading MEG file {megcnt + 1}/{len(meglist)}: {megfile}")
        variable_path = "data"
        data = f[variable_path][()]
    lowpass_data = lh_meg(data)
    low_meg_file = megfile.replace('serialised', 'lowpass100')
    # create directory if not exists
    low_meg_dir = os.path.dirname(low_meg_file)
    if not os.path.exists(low_meg_dir):
        os.makedirs(low_meg_dir)

    with h5py.File(low_meg_file, "w") as f:
        variable_path = "data"
        lowpass_data = lowpass_data.astype('float32')
        f[variable_path] = lowpass_data

    megcnt += 1
