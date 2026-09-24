#!/usr/bin/env python3
"""Train and evaluate one selected reconstruction experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from dataset import DATASETS, build_loaders, validate_shared_contract
from experiments import EXPERIMENTS, create_experiment_model, get_experiment

try:
    from spec import DEFAULT_MEL_WEIGHT
except ImportError:
    DEFAULT_MEL_WEIGHT = 1.0

try:
    from spec import GATE_LR_MULTIPLIER
except ImportError:
    GATE_LR_MULTIPLIER = 1.0

try:
    from spec import PRETRAINED_AWAVE_ROOT
except ImportError:
    PRETRAINED_AWAVE_ROOT = None

try:
    from spec import ADAPTIVE_ANCHOR_ROOT
except ImportError:
    ADAPTIVE_ANCHOR_ROOT = None

try:
    from spec import DEFAULT_EPOCHS, DEFAULT_MIN_EPOCHS, DEFAULT_PATIENCE, LR_RESTART_EPOCHS
except ImportError:
    DEFAULT_EPOCHS = 60
    DEFAULT_MIN_EPOCHS = 60
    DEFAULT_PATIENCE = 60
    LR_RESTART_EPOCHS = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SHINE-Recon paper experiment")
    parser.add_argument("--experiment", type=int, choices=sorted(EXPERIMENTS), default=55)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="shared dataset root (or set SHINE_DATA_ROOT)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mel-weight", type=float, default=DEFAULT_MEL_WEIGHT)
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def channel_correlation(prediction: torch.Tensor, target: torch.Tensor,
                        style: str = "stable") -> torch.Tensor:
    prediction = prediction.float() - prediction.float().mean(dim=-1, keepdim=True)
    target = target.float() - target.float().mean(dim=-1, keepdim=True)
    numerator = (prediction * target).sum(dim=-1)
    if style == "legacy":
        denominator = prediction.square().sum(dim=-1).sqrt() * target.square().sum(dim=-1).sqrt()
        return numerator / denominator.clamp_min(1e-6)
    if style != "stable":
        raise ValueError(f"unknown correlation style: {style}")
    # Clamp each energy before sqrt. Clamping only their product leaves the
    # derivative of sqrt(0) unbounded and can turn an otherwise finite batch
    # into NaNs during residual-path training.
    prediction_norm = prediction.square().sum(dim=-1).clamp_min(1e-6).sqrt()
    target_norm = target.square().sum(dim=-1).clamp_min(1e-6).sqrt()
    return numerator / (prediction_norm * target_norm)


class CorrelationAccumulator:
    def __init__(self, channels: int) -> None:
        self.n = 0
        self.sx = torch.zeros(channels, dtype=torch.float64)
        self.sy = torch.zeros(channels, dtype=torch.float64)
        self.sxx = torch.zeros(channels, dtype=torch.float64)
        self.syy = torch.zeros(channels, dtype=torch.float64)
        self.sxy = torch.zeros(channels, dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().double().transpose(0, 1).reshape(prediction.shape[1], -1).cpu()
        target = target.detach().double().transpose(0, 1).reshape(target.shape[1], -1).cpu()
        self.n += prediction.shape[1]
        self.sx += prediction.sum(1)
        self.sy += target.sum(1)
        self.sxx += prediction.square().sum(1)
        self.syy += target.square().sum(1)
        self.sxy += (prediction * target).sum(1)

    def compute(self) -> list[float]:
        numerator = self.n * self.sxy - self.sx * self.sy
        denominator = ((self.n * self.sxx - self.sx.square()) * (self.n * self.syy - self.sy.square())).clamp_min(1e-12).sqrt()
        return (numerator / denominator).tolist()


def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    envelope_metrics = CorrelationAccumulator(1)
    mel_metrics = CorrelationAccumulator(10)
    with torch.inference_mode():
        for neural, envelope, mel in loader:
            neural = neural.to(device, non_blocking=True)
            envelope = envelope.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)
            prediction = model(neural)
            envelope_metrics.update(prediction["envelope"], envelope)
            mel_metrics.update(prediction["mel"], mel)
    envelope_r = envelope_metrics.compute()[0]
    mel_band_r = mel_metrics.compute()
    return {"envelope_r": envelope_r, "mel_band_r": mel_band_r, "mean_mel_r": float(np.mean(mel_band_r))}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    experiment = get_experiment(args.experiment)
    args.batch_size = experiment.batch_size if args.batch_size is None else args.batch_size
    args.learning_rate = experiment.learning_rate if args.learning_rate is None else args.learning_rate
    seed_everything(args.seed)
    data_root = args.data_root or Path(os.environ.get("SHINE_DATA_ROOT", "data/ICASSP_shared_v1"))
    contract = validate_shared_contract(data_root, (args.dataset,))
    if args.check_data:
        print(json.dumps(contract[args.dataset], indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available() and not args.integration:
        raise RuntimeError("formal training requires a CUDA GPU")
    if args.integration:
        args.epochs = 1
        args.min_epochs = 1
        args.batch_size = min(args.batch_size, 2)
        args.workers = 0
        args.patience = 1
    else:
        args.epochs = DEFAULT_EPOCHS
        args.min_epochs = DEFAULT_MIN_EPOCHS
        args.patience = DEFAULT_PATIENCE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, data_metadata = build_loaders(
        args.dataset, args.batch_size, args.workers, args.integration, data_root)
    model = create_experiment_model(experiment, data_metadata["input_channels"])
    pretrained_checkpoint = None
    if ADAPTIVE_ANCHOR_ROOT and hasattr(model, "load_best_anchor"):
        pretrained_checkpoint = Path(model.load_best_anchor(ADAPTIVE_ANCHOR_ROOT,
                                                            args.dataset, args.seed))
    elif PRETRAINED_AWAVE_ROOT and hasattr(model, "load_awave") and not args.integration:
        pretrained_checkpoint = (Path(PRETRAINED_AWAVE_ROOT)
                                 / f"e7_{args.dataset}_awavenet_s{args.seed}.pt")
        if not pretrained_checkpoint.is_file():
            raise FileNotFoundError(f"missing AWave anchor: {pretrained_checkpoint}")
        model.load_awave(str(pretrained_checkpoint))
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad)
    gate_parameters = [parameter for name, parameter in model.named_parameters()
                       if name == "fusion_logits"]
    gate_ids = {id(parameter) for parameter in gate_parameters}
    base_parameters = [parameter for parameter in model.parameters()
                       if id(parameter) not in gate_ids]
    parameter_groups = [{"params": base_parameters, "lr": args.learning_rate,
                         "weight_decay": args.weight_decay}]
    if gate_parameters:
        parameter_groups.append({"params": gate_parameters,
                                 "lr": args.learning_rate * GATE_LR_MULTIPLIER,
                                 "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(parameter_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = args.output.parent.parent / "checkpoints" / args.output.with_suffix(".pt").name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["data_root"] = str(data_root)
    print(json.dumps({"experiment": f"experiment{experiment.number}", "model": experiment.model_name,
                      "device": str(device), "parameters": parameter_count,
                      "data": data_metadata, "config": config}, indent=2, sort_keys=True), flush=True)

    best_score = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    started = time.time()
    for epoch in range(args.epochs):
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        if epoch in LR_RESTART_EPOCHS:
            for index, group in enumerate(optimizer.param_groups):
                group["lr"] = args.learning_rate * (GATE_LR_MULTIPLIER if index == 1 else 1.0)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2)
            stale_epochs = 0
        model.train()
        total_loss = 0.0
        batches = 0
        for neural, envelope, mel in loaders["train"]:
            neural = neural.to(device, non_blocking=True)
            envelope = envelope.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(neural)
                envelope_loss = 1.0 - channel_correlation(
                    prediction["envelope"], envelope, experiment.correlation_style).mean()
                mel_loss = 1.0 - channel_correlation(
                    prediction["mel"], mel, experiment.correlation_style).mean()
                loss = envelope_loss + args.mel_weight * mel_loss
                # Multi-expert models expose auxiliary predictions so every
                # path remains predictive even when the fusion gate favors
                # another expert early in training.
                if "aux_envelopes" in prediction and "aux_mels" in prediction:
                    auxiliary_losses = []
                    for aux_envelope, aux_mel in zip(prediction["aux_envelopes"],
                                                     prediction["aux_mels"]):
                        auxiliary_losses.append(
                            1.0 - channel_correlation(aux_envelope, envelope, experiment.correlation_style).mean()
                            + args.mel_weight * (1.0 - channel_correlation(
                                aux_mel, mel, experiment.correlation_style).mean())
                        )
                    loss = loss + experiment.auxiliary_loss_weight * torch.stack(auxiliary_losses).mean()
                # Residual decoders learn only what the frozen primary path
                # has not explained, instead of duplicating its prediction.
                if ("aux_residual_envelopes" in prediction and
                        "aux_residual_mels" in prediction):
                    envelope_residual = envelope - prediction["base_envelope"].detach()
                    mel_residual = mel - prediction["base_mel"].detach()
                    residual_losses = []
                    for aux_envelope, aux_mel in zip(
                            prediction["aux_residual_envelopes"],
                            prediction["aux_residual_mels"]):
                        residual_losses.append(
                            1.0 - channel_correlation(aux_envelope,
                                                      envelope_residual,
                                                      experiment.correlation_style).mean()
                            + args.mel_weight *
                            (1.0 - channel_correlation(aux_mel, mel_residual,
                                                       experiment.correlation_style).mean())
                        )
                    loss = loss + experiment.auxiliary_loss_weight * torch.stack(
                        residual_losses).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            batches += 1
        valid = evaluate(model, loaders["valid"], device)
        score = valid["envelope_r"] + valid["mean_mel_r"]
        scheduler_enabled = (not hasattr(model, "scheduler_enabled") or
                             model.scheduler_enabled())
        if scheduler_enabled:
            scheduler.step(score)
        print(json.dumps({"epoch": epoch, "train_loss": total_loss / max(1, batches), "valid": valid,
                          "selection_score": score, "lr": optimizer.param_groups[0]["lr"]}), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            temporary = checkpoint.with_suffix(checkpoint.suffix + f".tmp.{os.getpid()}")
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score, "config": config}, temporary)
            os.replace(temporary, checkpoint)
        else:
            stale_epochs += 1
            if epoch + 1 >= args.min_epochs and stale_epochs >= args.patience:
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    if hasattr(model, "set_epoch"):
        model.set_epoch(saved["epoch"])
    valid = evaluate(model, loaders["valid"], device)
    test = evaluate(model, loaders["test"], device)
    result = {
        "schema_version": 1,
        "experiment": experiment.number,
        "experiment_name": f"experiment{experiment.number}",
        "model": experiment.model_name,
        "reported_label": experiment.label,
        "correlation_style": experiment.correlation_style,
        "dataset": args.dataset,
        "seed": args.seed,
        "split": data_metadata["split_protocol"],
        "config": config,
        "data": data_metadata,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "pretrained_checkpoint": str(pretrained_checkpoint) if pretrained_checkpoint else None,
        "anchor_name": getattr(model, "anchor_name", None),
        "anchor_validation_score": getattr(model, "anchor_validation_score", None),
        "checkpoint": str(checkpoint),
        "selection_epoch": best_epoch,
        "trained_epochs": epoch + 1,
        "validation": valid,
        "envelope_r": test["envelope_r"],
        "mel_band_r": test["mel_band_r"],
        "mean_mel_r": test["mean_mel_r"],
        "elapsed_seconds": time.time() - started,
        "integration": args.integration,
    }
    if hasattr(model, "fusion_logits"):
        weights = (model.effective_fusion_weights() if hasattr(model, "effective_fusion_weights")
                   else torch.sigmoid(model.fusion_logits))
        result["fusion_weights"] = weights.detach().cpu().reshape(-1).tolist()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
