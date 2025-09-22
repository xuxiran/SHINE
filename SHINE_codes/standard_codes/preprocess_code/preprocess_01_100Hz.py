import glob
import os
import numpy as np
import pandas as pd

cdir = os.getcwd()

rootdir = '../../libribrain/data/'

megfiles = []
eventsfiles = []
for Sherlock in range(1,8):

    events_dir = rootdir + 'Sherlock' + str(Sherlock) + '/derivatives/events/'
    seq01_dir = rootdir + 'Sherlock' + str(Sherlock) + '/derivatives/seq01_100Hz/'
    if not os.path.exists(seq01_dir):
        os.makedirs(seq01_dir)

    eventslist = [x for x in glob.glob(os.path.join(events_dir, "*"))]
    eventsfiles = eventsfiles + eventslist

megfiles = sorted(megfiles)
eventsfiles = sorted(eventsfiles)


if 1:
    for tsvfile in eventsfiles:
        df = pd.read_csv(tsvfile, sep='\t')
        sr = 100
        meg_start = round(df.loc[0, 'timemeg'] * sr)
        meg_end = round((df.iloc[-1]['timemeg']  + df.iloc[-1]['duration']) * sr)
        seq_01 = np.ones(meg_end)
        last_silence_index = df.loc[0, 'timemeg']
        # Iterate through each row in the df table
        for index, row in df.iterrows():
            if row['kind'] != 'silence':
                continue
            silent_start = row['timemeg']
            silent_end = silent_start + row['duration']
            silent_start_index = int(silent_start * sr+0.5)
            silent_end_index = int(silent_end * sr+0.5)

            seq_01[silent_start_index:silent_end_index] = 0
        # save seq_01 to npy file
        seq_01 = seq_01[meg_start:meg_end]
        seq_01 = np.expand_dims(seq_01, axis=0)
        # if 'sub-0_ses-12_task-Sherlock1_run-2' in tsvfile:
        #     print(f"Skipping smoothing for {tsvfile} due to test locally.")
        # else:
        #     seq_01 = smooth_sequence(seq_01, threshold=50)
        seq_01_name = tsvfile.replace('events.tsv', 'seq01_100Hz.npy')
        seq_01_name = seq_01_name.replace('events', 'seq01_100Hz')


        np.save(seq_01_name, seq_01)


# cal the silence proportion
seq01files = []
for Sherlock in range(1,8):
    seq01_dir = rootdir + '/Sherlock' + str(Sherlock) + '/derivatives/seq01_100Hz/'
    seq01list = [x for x in glob.glob(os.path.join(seq01_dir, "*"))]
    seq01files = seq01files + seq01list

seq01files = sorted(seq01files)

silence_proportions = np.zeros(len(seq01files))
cnt = 0
for file in seq01files:
    seq_01 = np.load(file)
    seq_01 = seq_01[0]
    silence_proportion = np.sum(seq_01 == 0) / len(seq_01)
    print(f"{file}: {silence_proportion:.4f}")
    silence_proportions[cnt] = silence_proportion
    cnt += 1

print("Silence proportions:")
print(silence_proportions.mean())
# min and max
print("Min silence proportion:", silence_proportions.min())
print("Max silence proportion:", silence_proportions.max())






