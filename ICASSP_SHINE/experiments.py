"""Experiment identities and release training protocols."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Experiment:
    number: int
    model_name: str
    label: str
    module: str
    family: str
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    min_epochs: int
    auxiliary_loss_weight: float = 0.0
    correlation_style: str = "stable"


def _baseline(number: int, model_name: str, label: str, *,
              epochs: int = 60, batch_size: int = 8,
              learning_rate: float = 3e-4, patience: int = 60,
              min_epochs: int = 60, correlation_style: str = "legacy") -> Experiment:
    return Experiment(number, model_name, label, f"baselines.experiment{number}",
                      "baseline", epochs, batch_size, learning_rate, patience,
                      min_epochs, correlation_style=correlation_style)


def _ablation(number: int, model_name: str, label: str) -> Experiment:
    return Experiment(number, model_name, label, f"ablations.experiment{number}",
                      "ablation", 60, 8, 3e-4, 60, 60, 0.2)


EXPERIMENTS = {
    40: _baseline(40, "brainmagic_meg", "BrainMagick-MEG", epochs=60,
                  patience=60, min_epochs=60, correlation_style="stable"),
    41: _baseline(41, "vlaai_meg", "VLAAI-MEG", epochs=60,
                  patience=60, min_epochs=60, correlation_style="stable"),
    42: _baseline(42, "happyquokka_meg", "HappyQuokka-MEG", epochs=60,
                  patience=60, min_epochs=60, correlation_style="stable"),
    43: _baseline(43, "convconcatnet", "ConvConcatNet"),
    45: _baseline(45, "awavenet", "AWaveNet"),
    46: _baseline(46, "cnnlstm", "CNN-LSTM"),
    47: _baseline(47, "dilatedconv", "DilatedConv"),
    48: _baseline(48, "ssm2mel_adapted", "SSM2Mel-adapted",
                  batch_size=4, learning_rate=5e-4),
    49: _baseline(49, "dmf2mel_adapted", "DMF2Mel-adapted",
                  batch_size=4),
    53: _ablation(53, "fc_uc_shine_nocontext", "No pooled context + EGD"),
    54: _ablation(54, "fc_uc_shine_fixedfusion", "Fixed 0.5/0.5 fusion + EGD"),
    55: Experiment(55, "fc_uc_shine_recon_60", "SHINE-Recon", "model",
                   "full", 60, 8, 3e-4, 60, 60, 0.2),
    56: _ablation(56, "fc_uc_shine_nohier", "No hierarchy concatenation"),
    57: _ablation(57, "fc_uc_shine_simplespatial", "Single spatial projection"),
    58: _ablation(58, "fc_uc_shine_nogate", "No temporal gate"),
}


def get_experiment(number: int) -> Experiment:
    try:
        return EXPERIMENTS[number]
    except KeyError as exc:
        raise ValueError(f"unknown experiment {number}; choose from {sorted(EXPERIMENTS)}") from exc


def create_experiment_model(experiment: Experiment, input_channels: int):
    return import_module(experiment.module).create_model(input_channels=input_channels)
