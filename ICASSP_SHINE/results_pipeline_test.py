"""Temporary synthetic-result checks for the summary and table pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dataset
import make_tables
import summarize
from experiments import EXPERIMENTS


class ResultsPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results_dir = self.root / "synthetic_results"
        self.results_dir.mkdir()
        self.numbers = tuple(dict.fromkeys(
            make_tables.MAIN_ORDER + make_tables.VARIANT_ORDER
        ))
        self._write_synthetic_results()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_synthetic_results(self) -> None:
        for number in self.numbers:
            experiment = EXPERIMENTS[number]
            for dataset_index, dataset_name in enumerate(dataset.DATASETS):
                envelope = (number + dataset_index) / 1000
                mean_mel = (number + dataset_index + 1) / 2000
                result = {
                    "experiment": number,
                    "model": experiment.model_name,
                    "dataset": dataset_name,
                    "seed": 0,
                    "envelope_r": envelope,
                    "mel_band_r": [mean_mel] * 10,
                    "mean_mel_r": mean_mel,
                    "integration": False,
                    "trained_epochs": 60,
                    "config": {
                        "epochs": experiment.epochs,
                        "batch_size": experiment.batch_size,
                        "learning_rate": experiment.learning_rate,
                        "patience": experiment.patience,
                        "min_epochs": experiment.min_epochs,
                        "weight_decay": 1e-2,
                        "mel_weight": 1.0,
                    },
                }
                path = self.results_dir / f"e{number}_{dataset_name}_s0.json"
                path.write_text(json.dumps(result), encoding="utf-8")

    @staticmethod
    def _run_summary(results_dir: Path, output: Path, experiment: int = 55) -> None:
        with patch("sys.argv", ["summarize.py", "--results-dir", str(results_dir),
                                "--output", str(output), "--experiment", str(experiment),
                                "--seeds", "0"]):
            summarize.main()

    @staticmethod
    def _run_tables(results_dir: Path, output: Path) -> None:
        with patch("sys.argv", ["make_tables.py", "--results-dir", str(results_dir),
                                "--output", str(output), "--seeds", "0"]):
            make_tables.main()

    def test_single_seed_complete_summary_and_tables(self) -> None:
        summary_path = self.root / "summary.md"
        table_path = self.root / "tables.md"
        self._run_summary(self.results_dir, summary_path)
        self._run_tables(self.results_dir, table_path)

        summary = summary_path.read_text(encoding="utf-8")
        self.assertIn("Status: complete", summary)
        self.assertIn("Seeds: 0", summary)
        self.assertIn("| SparKULee | 0.0550 | 0.0280 |", summary)
        self.assertIn("| SEM4Lang | 0.0580 | 0.0295 |", summary)

        tables = table_path.read_text(encoding="utf-8")
        self.assertIn("Seeds: 0.", tables)
        self.assertEqual(tables.count("## "), 4)
        self.assertIn("| SHINE-Recon |", tables)
        self.assertIn("| AWaveNet |", tables)

    def test_missing_result_is_rejected(self) -> None:
        missing = self.results_dir / "e55_SEM4Lang_s0.json"
        missing.unlink()
        summary_path = self.root / "incomplete_summary.md"
        with self.assertRaisesRegex(RuntimeError, "missing e55_SEM4Lang_s0.json"):
            self._run_summary(self.results_dir, summary_path)
        self.assertIn("Status: incomplete", summary_path.read_text(encoding="utf-8"))

        with self.assertRaisesRegex(SystemExit, "Missing result files"):
            self._run_tables(self.results_dir, self.root / "missing_tables.md")

    def test_wrong_model_identity_is_rejected(self) -> None:
        path = self.results_dir / "e40_SparKULee_s0.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        result["model"] = "wrong_model_identity"
        path.write_text(json.dumps(result), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            self._run_summary(self.results_dir, self.root / "wrong_identity_summary.md", 40)

        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self._run_tables(self.results_dir, self.root / "wrong_identity_tables.md")

    def test_protocol_override_is_rejected_for_tables(self) -> None:
        path = self.results_dir / "e55_SparKULee_s0.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        result["config"]["batch_size"] = 4
        path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "release protocol"):
            self._run_tables(self.results_dir, self.root / "wrong_protocol_tables.md")

    def test_incomplete_training_is_rejected_for_tables(self) -> None:
        path = self.results_dir / "e55_SparKULee_s0.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        result["trained_epochs"] = 59
        path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "trained_epochs=59; expected 60"):
            self._run_tables(self.results_dir, self.root / "incomplete_run_tables.md")


if __name__ == "__main__":
    unittest.main(verbosity=2)
