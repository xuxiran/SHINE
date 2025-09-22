#!/bin/bash
#SBATCH -o ./log/%j.out
#SBATCH -e ./log/%j.err
#SBATCH --ntasks-per-node=8
#SBATCH --partition=GPUA800
#SBATCH --exclude=gpua800n22
#SBATCH -J pre
#SBATCH --gres=gpu:1

python preprocess_01_100Hz.py
python downsample_meg.py
python downsample_meg_test.py



