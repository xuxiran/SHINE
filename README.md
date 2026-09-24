# SHINE

This repository contains **two distinct codebases** for decoding speech-related information from noninvasive brain recordings. Choose the directory that matches your task; the competition and ICASSP experiments use different targets and protocols.

## Competition result

The SHINE team placed **second in the Speech Detection Extended Track** of the **NeurIPS 2025 PNPL Competition**. See the [official leaderboard](https://libribrain.com/editions/2025/leaderboard/) and [prize listing](https://libribrain.com/editions/2025/prizes/). This result pertains to the competition entry in `SHINE_codes/`.

## Codebases

| Directory | Purpose | Data and output | Documentation |
| --- | --- | --- | --- |
| [`SHINE_codes/`](SHINE_codes/) | NeurIPS 2025 PNPL competition entry, including preprocessing, training, holdout prediction, and ensembling. | LibriBrain MEG to speech/non-speech detection. | [Competition guide](SHINE_codes/README.md) |
| [`ICASSP_SHINE/`](ICASSP_SHINE/) | Code accompanying the SHINE ICASSP manuscript, including the main model, comparison implementations, and component variants. | EEG/MEG to a continuous speech envelope and ten Mel bands. | [Reconstruction guide](ICASSP_SHINE/README.md) |

The competition ranking is **not** a ranking of the continuous-reconstruction system: the two directories address different tasks and use different evaluation protocols.

## Getting started

For the competition entry, start with [`SHINE_codes/README.md`](SHINE_codes/README.md). Its scripts and requirements are organized under `SHINE_codes/standard_codes/`.

For the ICASSP reconstruction code, start with [`ICASSP_SHINE/README.md`](ICASSP_SHINE/README.md). Its formal experiment configurations use **60 epochs**. A quick synthetic code check is:

```bash
cd ICASSP_SHINE
pip install -r requirements.txt
python experiment_smoke_test.py
```

The datasets, pretrained weights, and result files are not included. Follow the relevant directory guide for data layout and training instructions.
