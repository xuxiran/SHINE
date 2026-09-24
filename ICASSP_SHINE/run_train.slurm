#!/bin/bash
#SBATCH -p GPUA800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 7
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o shine_recon_%j.out
#SBATCH -e shine_recon_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
experiment_number="${EXPERIMENT_NUMBER:-55}"
"${PYTHON_BIN:-python}" -u main.py \
  --experiment "${experiment_number}" \
  --dataset "${DATASET:?Set DATASET before sbatch}" \
  --seed "${SEED:-0}" \
  --workers 0 \
  --data-root "${SHINE_DATA_ROOT:?Set SHINE_DATA_ROOT before sbatch}" \
  --output "${OUTPUT_DIR:-results}/e${experiment_number}_${DATASET}_s${SEED:-0}.json"
