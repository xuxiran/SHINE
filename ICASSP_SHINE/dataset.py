#!/usr/bin/env python3
"""Frozen data protocol shared by all ICASSP reconstruction experiments."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SHARED_ROOT = Path(os.environ.get("SHINE_DATA_ROOT", "data/ICASSP_shared_v1"))
DATASETS = ("SparKULee", "PKUEEG", "LibriBrain", "SEM4Lang")
EXPECTED_NEURAL = {"SparKULee": 662, "PKUEEG": 1250, "LibriBrain": 70, "SEM4Lang": 720}
EXPECTED_STIMULI = {"SparKULee": 72, "PKUEEG": 50, "LibriBrain": 70, "SEM4Lang": 60}
FEATURES = ("envelope1_lp8_64Hz", "mel10_64Hz")
MEG_DATASETS = ("LibriBrain", "SEM4Lang")
# Neuromag 306 arrays are ordered as 102 triplets. Empirical sensor-scale
# inspection and the IEEEtrans grad_in_meg protocol identify triplet offsets
# 1 and 2 as the two planar gradiometers; offset 0 is the magnetometer.
MEG_GRADIENT_INDICES = np.asarray(
    [index for index in range(306) if index % 3 in (1, 2)], dtype=np.int64,
)
assert len(MEG_GRADIENT_INDICES) == 204


@dataclass(frozen=True)
class Recording:
    neural: Path
    envelope: Path
    mel: Path
    subject: str
    stimulus: str
    split: str


def _subject_and_stimulus(path: Path) -> tuple[str, str]:
    stem = path.stem
    match = re.match(r"(sub-[^_]+)_(.+)_LP-30_64Hz$", stem)
    if match is None:
        raise ValueError(f"unrecognized neural filename: {path.name}")
    return match.group(1), match.group(2)


def _ranked_split(values: Iterable[str], namespace: str) -> dict[str, str]:
    values = sorted(set(values), key=lambda value: hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest())
    if len(values) < 4:
        return {value: "train" for value in values}
    n_test = max(1, round(0.15 * len(values)))
    n_valid = max(1, round(0.15 * len(values)))
    result = {value: "train" for value in values}
    for value in values[:n_test]:
        result[value] = "test"
    for value in values[n_test : n_test + n_valid]:
        result[value] = "valid"
    return result


def discover_recordings(dataset: str, root: Path = SHARED_ROOT) -> list[Recording]:
    if dataset not in DATASETS:
        raise ValueError(dataset)
    neural_paths = sorted((root / "neural_lp30_64hz" / dataset).glob("*.npy"))
    parsed = [(path, *_subject_and_stimulus(path)) for path in neural_paths]
    envelope_dir = root / "stimuli" / dataset / FEATURES[0]
    mel_dir = root / "stimuli" / dataset / FEATURES[1]
    available_envelopes = {path.stem for path in envelope_dir.glob("*.npy")}
    available_mels = {path.stem for path in mel_dir.glob("*.npy")}
    subject_split = _ranked_split((row[1] for row in parsed), f"{dataset}:subject")
    stimulus_split = _ranked_split((row[2] for row in parsed), f"{dataset}:stimulus")
    recordings: list[Recording] = []
    for neural, subject, stimulus in parsed:
        envelope = envelope_dir / f"{stimulus}.npy"
        mel = mel_dir / f"{stimulus}.npy"
        if stimulus not in available_envelopes or stimulus not in available_mels:
            raise FileNotFoundError(f"missing targets for {neural.name}: {envelope}, {mel}")
        # A train recording shares neither subject nor stimulus with held-out data.
        if subject_split[subject] == "test" or stimulus_split[stimulus] == "test":
            split = "test"
        elif subject_split[subject] == "valid" or stimulus_split[stimulus] == "valid":
            split = "valid"
        else:
            split = "train"
        recordings.append(Recording(neural, envelope, mel, subject, stimulus, split))
    for split in ("train", "valid", "test"):
        if not any(recording.split == split for recording in recordings):
            raise RuntimeError(f"{dataset} has no {split} recordings")
    return recordings


def validate_shared_contract(root: Path = SHARED_ROOT, datasets: Iterable[str] = DATASETS) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(root)
    report = {}
    for dataset in datasets:
        recordings = discover_recordings(dataset, root)
        stimulus_count = len({recording.stimulus for recording in recordings})
        if len(recordings) != EXPECTED_NEURAL[dataset]:
            raise RuntimeError(f"{dataset}: {len(recordings)} neural files, expected {EXPECTED_NEURAL[dataset]}")
        if stimulus_count > EXPECTED_STIMULI[dataset]:
            raise RuntimeError(f"{dataset}: unexpected stimulus count {stimulus_count}")
        split_counts = {split: sum(r.split == split for r in recordings) for split in ("train", "valid", "test")}
        report[dataset] = {"neural": len(recordings), "stimuli_used": stimulus_count, "splits": split_counts, "sample_rate_hz": 64}
    return report


def _time_first(array: np.ndarray, expected_channels: int | None = None) -> np.ndarray:
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got {array.shape}")
    if expected_channels is not None:
        if array.shape[1] == expected_channels:
            return array
        if array.shape[0] == expected_channels:
            return array.T
    # Neural time axes are always much longer than channel axes in this corpus.
    return array if array.shape[0] > array.shape[1] else array.T


class WindowDataset(Dataset):
    def __init__(self, recordings: list[Recording], split: str, window: int = 1920, stride: int | None = None,
                 max_windows_per_recording: int | None = None) -> None:
        self.recordings = [recording for recording in recordings if recording.split == split]
        self.split = split
        self.window = window
        self.stride = stride or (window // 2 if split == "train" else window)
        self.max_windows_per_recording = max_windows_per_recording
        self._arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.index: list[tuple[int, int]] = []
        target_lengths: dict[Path, int] = {}
        target_arrays: dict[Path, np.ndarray] = {}
        for record_index, recording in enumerate(self.recordings):
            neural = np.load(recording.neural, allow_pickle=False)
            time_length = max(neural.shape)
            if recording.envelope not in target_lengths:
                target_arrays[recording.envelope] = _time_first(
                    np.load(recording.envelope, allow_pickle=False), 1)
                target_lengths[recording.envelope] = len(target_arrays[recording.envelope])
            if recording.mel not in target_lengths:
                target_arrays[recording.mel] = _time_first(
                    np.load(recording.mel, allow_pickle=False), 10)
                target_lengths[recording.mel] = len(target_arrays[recording.mel])
            envelope_length = target_lengths[recording.envelope]
            mel_length = target_lengths[recording.mel]
            length = min(time_length, envelope_length, mel_length)
            starts = list(range(0, max(1, length - window + 1), self.stride))
            if max_windows_per_recording is not None:
                starts = starts[:max_windows_per_recording]
            self.index.extend((record_index, start) for start in starts)
            neural = _time_first(neural)
            dataset_name = recording.neural.parent.name
            if dataset_name in MEG_DATASETS:
                if neural.shape[1] != 306:
                    raise RuntimeError(f"{dataset_name}: expected 306-channel input, got {neural.shape}")
                neural = neural[:, MEG_GRADIENT_INDICES]
            arrays = (neural, target_arrays[recording.envelope], target_arrays[recording.mel])
            length = min(len(array) for array in arrays)
            self._arrays.append(tuple(self._standardize(array[:length]) for array in arrays))
        del target_arrays
        if not self.index:
            raise RuntimeError(f"no windows for {split}")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _standardize(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        mean = array.mean(axis=0, keepdims=True)
        std = array.std(axis=0, keepdims=True)
        return (array - mean) / np.maximum(std, 1e-5)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record_index, start = self.index[item]
        neural, envelope, mel = self._arrays[record_index]
        stop = start + self.window
        return (
            torch.from_numpy(neural[start:stop].T.copy()),
            torch.from_numpy(envelope[start:stop].T.copy()),
            torch.from_numpy(mel[start:stop].T.copy()),
        )


def input_channels(dataset: str, root: Path = SHARED_ROOT) -> int:
    recording = discover_recordings(dataset, root)[0]
    array = np.load(recording.neural, allow_pickle=False)
    channels = min(array.shape)
    if dataset in MEG_DATASETS:
        if channels != 306:
            raise RuntimeError(f"{dataset}: expected 306 source channels, got {channels}")
        return len(MEG_GRADIENT_INDICES)
    return channels


def build_loaders(dataset: str, batch_size: int, workers: int, integration: bool = False,
                  root: Path = SHARED_ROOT) -> tuple[dict[str, DataLoader], dict]:
    recordings = discover_recordings(dataset, root)
    if integration:
        recordings = [recording for split in ("train", "valid", "test")
                      for recording in [row for row in recordings if row.split == split][:2]]
    max_windows = 2 if integration else None
    datasets = {
        split: WindowDataset(recordings, split, max_windows_per_recording=max_windows)
        for split in ("train", "valid", "test")
    }
    if integration:
        for split in datasets:
            datasets[split].index = datasets[split].index[: max(2, batch_size)]
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            drop_last=(split == "train" and len(ds) >= batch_size),
        )
        for split, ds in datasets.items()
    }
    metadata = {
        "input_channels": input_channels(dataset, root),
        "channel_protocol": (
            "neuromag_planar_gradients_204_triplet_offsets_1_2"
            if dataset in MEG_DATASETS else "all_eeg_channels"
        ),
        "recordings": {split: len(datasets[split].recordings) for split in datasets},
        "windows": {split: len(datasets[split]) for split in datasets},
        "window_samples": datasets["train"].window,
        "sample_rate_hz": 64,
        "split_protocol": "subject_and_stimulus_disjoint_sha256_v1",
    }
    return loaders, metadata


if __name__ == "__main__":
    print(validate_shared_contract())
