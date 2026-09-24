#!/usr/bin/env python3
"""Check normalization and RAM-resident window access with synthetic arrays."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dataset import Recording, WindowDataset


def main() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        neural_path = root / "neural_lp30_64hz" / "PKUEEG" / "sub-01_stimulus_LP-30_64Hz.npy"
        envelope_path = root / "stimuli" / "PKUEEG" / "envelope1_lp8_64Hz" / "stimulus.npy"
        mel_path = root / "stimuli" / "PKUEEG" / "mel10_64Hz" / "stimulus.npy"
        for path in (neural_path, envelope_path, mel_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        neural = np.arange(90, dtype=np.float32).reshape(30, 3)
        envelope = np.arange(40, dtype=np.float32).reshape(40, 1)
        envelope[30:] += 1000
        mel = np.arange(400, dtype=np.float32).reshape(40, 10)
        mel[30:] += 1000
        np.save(neural_path, neural)
        np.save(envelope_path, envelope)
        np.save(mel_path, mel)

        recording = Recording(neural_path, envelope_path, mel_path, "sub-01", "stimulus", "train")
        windows = WindowDataset([recording], "train", window=20, stride=20)
        expected_envelope = (envelope[:30] - envelope[:30].mean(axis=0)) / envelope[:30].std(axis=0)
        np.testing.assert_allclose(windows._arrays[0][1], expected_envelope, rtol=1e-5)

        for path in (neural_path, envelope_path, mel_path):
            path.unlink()
        neural_window, envelope_window, mel_window = windows[0]
        assert neural_window.shape == (3, 20)
        assert envelope_window.shape == (1, 20)
        assert mel_window.shape == (10, 20)
    print("PASS full-RAM windows and common-length normalization")


if __name__ == "__main__":
    main()
