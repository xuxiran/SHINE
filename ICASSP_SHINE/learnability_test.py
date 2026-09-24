#!/usr/bin/env python3
"""Overfit a fixed real LibriBrain batch before accepting a MEG repair."""
from __future__ import annotations

import json
import torch

from dataset import build_loaders
from main import channel_correlation, seed_everything
from model import create_model
from spec import MODEL_NAME


def score(prediction: dict[str, torch.Tensor], envelope: torch.Tensor,
          mel: torch.Tensor) -> torch.Tensor:
    return (channel_correlation(prediction["envelope"], envelope).mean()
            + channel_correlation(prediction["mel"], mel).mean())


def main() -> None:
    seed_everything(0)
    if not torch.cuda.is_available():
        raise RuntimeError("MEG learnability test requires CUDA")
    device = torch.device("cuda")
    loaders, metadata = build_loaders("LibriBrain", 2, workers=0, integration=True)
    neural, envelope, mel = next(iter(loaders["train"]))
    neural, envelope, mel = neural.to(device), envelope.to(device), mel.to(device)
    model = create_model(metadata["input_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    model.train()
    with torch.no_grad():
        initial = float(score(model(neural), envelope, mel))
    history = []
    for step in range(201):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(neural)
        current_score = score(prediction, envelope, mel)
        loss = 2.0 - current_score
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 25 == 0:
            history.append({"step": step, "score": float(current_score.detach())})
    model.eval()
    with torch.inference_mode():
        final = float(score(model(neural), envelope, mel))
    payload = {"model": MODEL_NAME, "initial_score": initial,
               "final_score": final, "history": history}
    print(json.dumps(payload, indent=2), flush=True)
    if not final > max(0.5, initial + 0.4):
        raise RuntimeError(f"real-batch learnability failed: {payload}")


if __name__ == "__main__":
    main()
