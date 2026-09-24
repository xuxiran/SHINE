"""Build comparison tables from validated experiment result files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset import DATASETS
from experiments import EXPERIMENTS
from summarize import load_result, parse_seeds, result_path


DISPLAY_DATASETS = ("SparrKULee", "PKUEEG", "LibriBrain", "SMN4Lang")
MAIN_ORDER = (55, 40, 41, 42, 43, 45, 48, 49, 46, 47)
VARIANT_ORDER = (55, 57, 58, 56, 53, 54)


def validate_paper_schedule(result: dict, experiment) -> None:
    config = result.get("config", {})
    if result.get("trained_epochs") != 60:
        raise ValueError(
            f"experiment{experiment.number} {result['dataset']} seed {result['seed']}: "
            f"trained_epochs={result.get('trained_epochs')}; expected 60 completed epochs")
    expected = {
        "epochs": experiment.epochs,
        "batch_size": experiment.batch_size,
        "learning_rate": experiment.learning_rate,
        "patience": experiment.patience,
        "min_epochs": experiment.min_epochs,
        "weight_decay": 1e-2,
        "mel_weight": 1.0,
    }
    for key, value in expected.items():
        actual = config.get(key)
        if actual is None or not np.isclose(actual, value):
            raise ValueError(
                f"experiment{experiment.number} {result['dataset']} seed {result['seed']}: "
                f"{key}={actual} differs from release protocol {value}")


def render_table(title: str, numbers: tuple[int, ...], metric: str,
                 results: dict[tuple[int, str], list[dict]]) -> list[str]:
    lines = [f"## {title}", "",
             "| Model | " + " | ".join(DISPLAY_DATASETS) + " |",
             "|---|" + "---:|" * len(DATASETS)]
    for number in numbers:
        experiment = EXPERIMENTS[number]
        values = []
        for dataset in DATASETS:
            scores = [row[metric] for row in results[number, dataset]]
            values.append(f"{float(np.mean(scores)):.4f}")
        lines.append(f"| {experiment.label} | " + " | ".join(values) + " |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all four paper-style result tables")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("tables.md"))
    parser.add_argument("--seeds", type=parse_seeds, default=(0,))
    args = parser.parse_args()
    results: dict[tuple[int, str], list[dict]] = {}
    missing = []
    for number in dict.fromkeys(MAIN_ORDER + VARIANT_ORDER):
        experiment = EXPERIMENTS[number]
        for dataset in DATASETS:
            results[number, dataset] = []
            for seed in args.seeds:
                try:
                    result = load_result(args.results_dir, experiment, dataset, seed)
                    validate_paper_schedule(result, experiment)
                    results[number, dataset].append(result)
                except FileNotFoundError:
                    missing.append(result_path(args.results_dir, experiment, dataset, seed).name)
    if missing:
        raise SystemExit("Missing result files:\n" + "\n".join(missing))

    seeds_text = ", ".join(str(seed) for seed in args.seeds)
    lines = ["# SHINE-Recon result tables", "",
             f"Seeds: {seeds_text}. Values are arithmetic means across the listed seeds.", "",
             "Experiments 53 and 54 also contain envelope-guided Mel decoding; they are mixed variants, not isolated ablations of experiment 55.", ""]
    lines += render_table("Main comparison: envelope Pearson r", MAIN_ORDER,
                          "envelope_r", results)
    lines += render_table("Main comparison: mean-Mel Pearson r", MAIN_ORDER,
                          "mean_mel_r", results)
    lines += render_table("Components and mixed variants: envelope Pearson r",
                          VARIANT_ORDER, "envelope_r", results)
    lines += render_table("Components and mixed variants: mean-Mel Pearson r",
                          VARIANT_ORDER, "mean_mel_r", results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"validated {len(results) * len(args.seeds)} runs and wrote {args.output}")


if __name__ == "__main__":
    main()
