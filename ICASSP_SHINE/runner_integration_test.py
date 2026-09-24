"""CPU runner checks using mocked in-memory batches only."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import main
from experiments import EXPERIMENTS


class RunnerIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        generator = torch.Generator().manual_seed(123)
        self.batch = (
            torch.randn(1, 64, 16, generator=generator),
            torch.randn(1, 1, 16, generator=generator),
            torch.randn(1, 10, 16, generator=generator),
        )
        self.loaders = {
            "train": [self.batch],
            "valid": [self.batch],
            "test": [self.batch],
        }
        self.metadata = {
            "input_channels": 64,
            "split_protocol": "synthetic-test-only",
        }

    def tearDown(self) -> None:
        torch.set_num_threads(self.previous_threads)
        self.temporary.cleanup()

    def run_experiment(self, number: int) -> tuple[dict, dict, str]:
        experiment = EXPERIMENTS[number]
        output_path = self.root / "results" / f"e{number}_synthetic_s0.json"
        arguments = [
            "main.py", "--experiment", str(number), "--dataset", "SparKULee",
            "--output", str(output_path), "--data-root", str(self.root), "--integration",
        ]
        output = io.StringIO()
        with (
            patch("sys.argv", arguments),
            patch.object(main.torch.cuda, "is_available", return_value=False),
            patch.object(main, "validate_shared_contract",
                         return_value={"SparKULee": self.metadata}) as validate,
            patch.object(main, "build_loaders",
                         return_value=(self.loaders, self.metadata)) as build,
            contextlib.redirect_stdout(output),
        ):
            main.main()

        self.assertTrue(output_path.is_file())
        result = json.loads(output_path.read_text(encoding="utf-8"))
        checkpoint_path = Path(result["checkpoint"])
        self.assertTrue(checkpoint_path.is_file())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        validate.assert_called_once()
        build.assert_called_once()
        self.assertIn('"epoch": 0, "train_loss":', output.getvalue())
        self.assertEqual(result["experiment"], number)
        self.assertEqual(result["experiment_name"], f"experiment{number}")
        self.assertEqual(result["model"], experiment.model_name)
        self.assertEqual(result["selection_epoch"], 0)
        self.assertEqual(result["trained_epochs"], 1)
        self.assertEqual(checkpoint["epoch"], 0)
        self.assertEqual(checkpoint["config"]["epochs"], 1)
        self.assertEqual(checkpoint["config"]["min_epochs"], 1)
        return result, checkpoint, output.getvalue()

    def test_baseline_mixed_ablation_and_main_model(self) -> None:
        for number in (43, 53, 55):
            with self.subTest(experiment=number):
                result, checkpoint, _ = self.run_experiment(number)
                self.assertGreater(result["parameter_count"], 0)
                self.assertTrue(checkpoint["model"])
                self.assertEqual(result["data"]["split_protocol"], "synthetic-test-only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
