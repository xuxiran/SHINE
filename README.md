## SHINE Team Code Repository
Welcome to the SHINE team's code repository for processing and analyzing the LibriBrain dataset. This README provides comprehensive instructions for setting up the dataset, organizing the directory structure, preprocessing the MEG data, training, testing, and generating predictions.
# Overview
This repository contains code developed by the SHINE team for MEG-based speech reconstruction using the LibriBrain dataset. The code was originally developed for EEG reconstruction, so some variable names and files use eeg instead of meg, which may slightly affect readability. We apologize for any confusion.
# Prerequisites
1.	Download the LibriBrain Dataset:
•	The dataset can be downloaded from the link provided in the official paper: https://arxiv.org/abs/2506.02098.
•	Ensure sufficient storage space and follow the paper's instructions for downloading.
2.	Software Requirements:
•	Python (version 3.9 or higher recommended).
•	Required Python packages (install via pip):
bash
pip install -r requirements.txt
3.	Hardware:
•	For faster training, access to multiple GPUs is recommended. The gen_run.py script facilitates multi-GPU execution.
Directory Structure
After downloading the LibriBrain dataset, organize the files into the following structure under the SHINE_codes directory:

SHINE_codes/

├── libribrain/

│   └── data/

│       ├── Sherlock1/
│       │   └── derivatives/
│       │       ├── events/
│       │       └── serialised/
│       ├── ... (Sherlock2 to Sherlock7, same structure)
├── libribrain_test/
│   └── data/
│       └── serialised/
├── standard_codes/
│   ├── preprocess_code/
│   ├── train_and_test_code/
│   └── code_v75/



# Data Preprocessing
The preprocessing scripts in standard_codes/preprocess_code/ prepare the MEG data for training and testing. The scripts include:
•	downsample_meg.py: Downsamples the training MEG data to 100Hz.
•	downsample_meg_test.py: Downsamples the holdout (test) data to 100Hz.
•	preprocess_01_100Hz.py: Extracts silent and speech segment labels from the training set's events folder, generating 0-1 binary labels for each session's MEG data.
•	run0.slurm: A SLURM script for running preprocessing jobs on a cluster.

# Training and Testing
The scripts in standard_codes/train_and_test_code/ handle model training, validation, and prediction generation. Key files include:
•	config.py: Contains hyperparameters tuned via grid search. The differences between hyperparameter settings are minimal.
•	EEG_Dataset.py: Generates a DataLoader for MEG data to facilitate training and validation.
•	main.py: The core script for training and validation. It reconstructs 0-1 sequences from MEG data using 1 minus the Pearson correlation coefficient as the loss function. Key hyperparameters include:
•	seed: Controls random initialization for reproducibility and enables training multiple model versions.
•	valid_num: Initially used to select specific sessions for testing but found to have minimal impact.
•	eeglen: Specifies the length of data segments used for reconstruction.
•	Models are saved in the model/ directory after each run.
•	test_holdout_model.py: Generates predictions for the holdout set, producing CSV files in the result_csv/ directory. The script:
•	Applies thresholds (22%, 23%, and 24% quantiles of the reconstructed sequence) to binarize the 0-1 sequence.
•	Uses average pooling to smooth the results.
•	Focuses on central portions of the sequence to avoid poor reconstruction at sequence edges.
•	(Note: The script's readability may be limited, but it is fully functional.)
•	ensemble_csv.py: Combines multiple CSV prediction files from the result_csv/ directory to produce an ensemble result.
•	BM.py: A modified version of the BrainMagic model (Défossez et al., 2023).
•	WavNet.py: A modified version of the WavNet model (Oord et al., 2016).
•	gen_run.py: Generates scripts for multi-GPU training, enabling simultaneous training of multiple models across servers for efficient ensemble creation.
Models The SHINEs/ directory contains 20 of all 100 versions of our trained models during the competition, including the best-performing ones. The best model (v75) combines:
•	Modified BrainMagic model (Défossez et al., 2023).
•	Modified WavNet model (Oord et al., 2016).
•	Our proposed SHINE model.
•	An LSTM model to integrate global information.
# code_v75
During the competition, we trained 16 versions of each model daily for ensembling. However, to improve readability, we have streamlined the codebase and avoided including manual processes. You could reproduce this operation by using codes in “standard_codes/code_v75”.Specifically, use main.py for training, test_01_hua_new.py for local testing, submit_model_csv_hua_100Hz.py for testing on the holdout test set, and ensemble_csv.py for ensembling. You can use run{i}.slurm (where i ranges from 0 to 11) to directly train 12 models on the server.

# Additional Notes
•	Data Augmentation: Techniques such as temporal reversal of MEG data and labels for some sessions were explored but omitted from the codebase to maintain clarity.
•	Normalization: Alternative normalization methods (e.g., normalizing MEG data along another dimension) were tested but not included to keep the code clean.

# References
•	Défossez, A., Caucheteux, C., Rapin, J., Kabeli, O., & King, J.-R. (2023). Decoding speech perception from non-invasive brain recordings. Nature Machine Intelligence, 5(10), 1097–1107. https://doi.org/10.1038/s42256-023-00714-5
•	Oord, A. van den, Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., & Kavukcuoglu, K. (2016). WaveNet: A Generative Model for Raw Audio. arXiv. https://doi.org/10.48550/arXiv.1609.03499
Contact For questions, issues, or further clarification, please contact the SHINE team or open an issue in this repository. We welcome feedback and are happy to assist!
 
