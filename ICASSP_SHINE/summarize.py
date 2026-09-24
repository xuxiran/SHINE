#!/usr/bin/env python3
"""Validate requested runs and summarize their envelope and Mel scores."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from dataset import DATASETS
from experiments import EXPERIMENTS, Experiment, get_experiment


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("provide distinct seed values")
    return seeds


def result_path(results_dir: Path, experiment: Experiment, dataset: str, seed: int) -> Path:
    path = results_dir / f"e{experiment.number}_{dataset}_s{seed}.json"
    legacy_path = results_dir / f"e{experiment.number}_{dataset}_{experiment.model_name}_s{seed}.json"
    return path if path.is_file() or not legacy_path.is_file() else legacy_path


def load_result(results_dir: Path, experiment: Experiment, dataset: str, seed: int) -> dict:
    path = result_path(results_dir, experiment, dataset, seed)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["experiment"] != experiment.number or result["model"] != experiment.model_name:
        raise ValueError(f"identity mismatch in {path.name}")
    if result["dataset"] != dataset or result["seed"] != seed:
        raise ValueError(f"dataset or seed mismatch in {path.name}")
    if len(result["mel_band_r"]) != 10:
        raise ValueError(f"expected ten Mel bands in {path.name}")
    if not np.isfinite([result["envelope_r"], result["mean_mel_r"],
                        *result["mel_band_r"]]).all():
        raise ValueError(f"non-finite metric in {path.name}")
    if not np.isclose(result["mean_mel_r"], np.mean(result["mel_band_r"]), atol=1e-8):
        raise ValueError(f"mean-Mel mismatch in {path.name}")
    if result.get("integration", False):
        raise ValueError(f"integration run in {path.name}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize selected runs")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--experiment", type=int, choices=sorted(EXPERIMENTS), default=55)
    parser.add_argument("--seeds", type=parse_seeds, default=(0,))
    args = parser.parse_args()
    experiment = get_experiment(args.experiment)
    output = args.output or Path(f"summary_e{experiment.number}.md")
    rows = []
    errors = []
    for dataset in DATASETS:
        for seed in args.seeds:
            try:
                rows.append(load_result(args.results_dir, experiment, dataset, seed))
            except FileNotFoundError:
                errors.append(f"missing {result_path(args.results_dir, experiment, dataset, seed).name}")
            except Exception as exc:
                errors.append(f"invalid {dataset} seed {seed}: {exc}")
    lines = [f"# experiment{experiment.number}: {experiment.model_name}", "",
             f"Label: {experiment.label}", "",
             f"Seeds: {', '.join(str(seed) for seed in args.seeds)}", "", "## Results", ""]
    if errors:
        lines.extend(["Status: incomplete", "", *[f"- {error}" for error in errors]])
    else:
        lines.extend(["Status: complete", "", "| Dataset | Envelope r | Mean Mel r |", "|---|---:|---:|"])
        for dataset in DATASETS:
            selected = [row for row in rows if row["dataset"] == dataset]
            envelope = np.asarray([row["envelope_r"] for row in selected])
            mel = np.asarray([row["mean_mel_r"] for row in selected])
            if len(args.seeds) == 1:
                envelope_text = f"{envelope.mean():.4f}"
                mel_text = f"{mel.mean():.4f}"
            else:
                envelope_text = f"{envelope.mean():.4f} +/- {envelope.std(ddof=1):.4f}"
                mel_text = f"{mel.mean():.4f} +/- {mel.std(ddof=1):.4f}"
            lines.append(f"| {dataset} | {envelope_text} | {mel_text} |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("; ".join(errors))
    print(f"validated {len(DATASETS) * len(args.seeds)} results and wrote {output}")


if __name__ == "__main__":
    main()
