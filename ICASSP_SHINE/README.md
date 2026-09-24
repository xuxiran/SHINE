# SHINE-Recon experiment code

This directory contains the SHINE-Recon model, the comparison models, and the historical component variants. `experiments.py` maps each experiment number to its model and release training protocol; `spec.py` retains the experiment-55 constants. These are common-protocol adaptations, not official upstream implementations of the cited methods. The code defaults to one run with seed 0; the paper's existing tables report means over seeds 0, 1, and 2.

## Data

The shared preprocessed dataset is not included. Obtain authorized access to `ICASSP_shared_v1` and arrange it in this layout:

```text
ICASSP_shared_v1/
  neural_lp30_64hz/<dataset>/*.npy
  stimuli/<dataset>/envelope1_lp8_64Hz/*.npy
  stimuli/<dataset>/mel10_64Hz/*.npy
```

Supported dataset keys are `SparKULee`, `PKUEEG`, `LibriBrain`, and `SEM4Lang`. The first and last are internal folder names for the paper's SparrKULee and SMN4Lang datasets. [PKUEEG version 1.0.3](https://openneuro.org/datasets/ds008834/versions/1.0.3) is public, but this loader expects the preprocessed NumPy layout above; preprocessing from the raw archive is not included. The loader reads each split fully into RAM (without memory mapping), standardizes each recording after common-length truncation, and keeps training entities separate from held-out entities. Validation and test entities can overlap under the priority rule in `dataset.py`. Historical jobs used a two-recording cache and four loader workers; the release loader uses full RAM and zero workers. Data layout, source version, and split statistics still need end-to-end verification before claiming exact numerical reproduction.

Set `SHINE_DATA_ROOT` or pass `--data-root /path/to/ICASSP_shared_v1`. No dataset, pretrained checkpoint, result, or model weight is bundled.

## Environment

Use Python 3.10 or newer, a PyTorch build compatible with the available CUDA runtime for formal training, and the packages in `requirements.txt`. Install the CUDA-specific PyTorch build separately when needed.

## Run

Formal training on Huairou must be submitted through Slurm; see `run_train.slurm`. Its GPUA800 request mirrors the original experiment. Activate a compatible environment before submission or set `PYTHON_BIN` to its interpreter. Check the full-RAM dataset size and request enough Slurm memory with `sbatch --mem=...`; if one GPU allocation cannot provide enough host memory, request a larger allocation rather than reverting to memory mapping. The script writes one JSON result per run, prints every epoch to standard output, and saves the best validation checkpoint under the project-level `checkpoints/` folder. No jobs are submitted by this package.

Submit from the `ICASSP_SHINE` directory so `SLURM_SUBMIT_DIR` resolves `main.py`:

```bash
cd /path/to/SHINE/ICASSP_SHINE
EXPERIMENT_NUMBER=55 DATASET=PKUEEG SEED=0 \
  SHINE_DATA_ROOT=/path/to/ICASSP_shared_v1 sbatch run_train.slurm
```

```bash
python main.py --experiment 55 --dataset PKUEEG --seed 0 \
  --data-root /path/to/ICASSP_shared_v1 --output results/e55_PKUEEG_s0.json
```

The direct Python command shows the CLI arguments; use the Slurm script for formal Huairou runs. `EXPERIMENT_NUMBER` selects a row in the matrix below, and `SEED` defaults to 0. The Slurm script writes `results/e<experiment>_<dataset>_s<seed>.json`; the runner saves the corresponding best checkpoint in `checkpoints/`.

## Experiment matrix

| Experiments | Model family | Release training protocol |
|---|---|---|
| 40–42 | BrainMagick-MEG, VLAAI-MEG, HappyQuokka-MEG | 60 epochs; batch 8; learning rate 3e-4; minimum 60 epochs; patience 60 |
| 43, 45–47 | ConvConcatNet, AWaveNet, CNN–LSTM, DilatedConv | 60 epochs; batch 8; learning rate 3e-4; minimum 60 epochs; patience 60 |
| 48 | SSM2Mel-adapted | 60 epochs; batch 4; learning rate 5e-4; minimum 60 epochs; patience 60 |
| 49 | DMF2Mel-adapted | 60 epochs; batch 4; learning rate 3e-4; minimum 60 epochs; patience 60 |
| 53–58 | Historical components and SHINE-Recon (55) | 60 epochs; batch 8; learning rate 3e-4; minimum 60 epochs; patience 60; auxiliary loss weight 0.2 |

All entries use the same loader, AdamW with weight decay 0.01, validation envelope-plus-mean-Mel checkpoint selection, and pooled test correlations. Experiments 43–49 retain their historical loss-correlation denominator clamp; the other entries use the stabilized per-energy clamp. See `baselines/README.md` and `ablations/README.md` for model identities and limits.

Experiment 45 is **AWaveNet**, not CAT-WaveNet: this implementation has no cross-attention. Experiments 53 and 54 contain envelope-guided Mel decoding in addition to their context/fusion changes, so they are mixed variants rather than isolated ablations of experiment 55. The current manuscript must not attribute their score differences to just one component. Experiments 48 and 49 are architectural protocol adaptations, not official reproductions.

## Summaries and tables

After four dataset runs finish for one experiment, validate their results with `python summarize.py --experiment 55 --results-dir results`. After all 15 experiments have four results each, run `python make_tables.py --results-dir results --output tables.md` to generate envelope and mean-Mel comparison/component tables. The table builder rejects missing runs, protocol mismatches, and runs that did not complete all 60 epochs. Both commands default to seed 0. To summarize **already existing** three-seed results, pass `--seeds 0,1,2` explicitly. No per-seed result JSON is bundled, so the manuscript's numerical means cannot be independently recomputed from this package alone.

## Local model smoke check

`python smoke_test.py` checks the main architecture. `python experiment_smoke_test.py` checks every registered model and backward pass. `python data_smoke_test.py` checks common-length normalization and RAM-resident window access. `python runner_integration_test.py` runs a synthetic one-epoch CPU path for representative models; `python results_pipeline_test.py` checks aggregation and rejection of missing or mismatched runs. All tests use synthetic arrays and start no formal training.
