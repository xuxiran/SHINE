#!/usr/bin/env python3
"""Task-specific checks for SHINE-Recon shapes and gradients."""

import torch

from model import create_model
from spec import MODEL_NAME


def main() -> None:
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    model = create_model(input_channels=64)
    assert tuple(block.temporal.dilation[0] for block in model.local_blocks) == (
        1, 2, 4, 8, 16, 32, 64, 128,
    )
    assert 1 + 8 * sum(block.temporal.dilation[0]
                       for block in model.local_blocks) == 2041
    neural = torch.randn(1, 64, 32)
    prediction = model(neural)
    assert prediction["envelope"].shape == (1, 1, 32)
    assert prediction["mel"].shape == (1, 10, 32)
    assert torch.isfinite(prediction["envelope"]).all()
    assert torch.isfinite(prediction["mel"]).all()
    assert "gate" not in prediction
    # A Mel-only backward exercises fusion and global context.  With direct
    # heads the Mel branch is not routed through the predicted envelope, so
    # the local/context heads still receive gradient through shared features.
    prediction["mel"].square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    envelope_parameters = [
        parameter for name, parameter in model.named_parameters()
        if "local_head" in name or "context_head" in name
    ]
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in envelope_parameters
    ), "Mel loss must reach the shared fusion path"

    with torch.inference_mode():
        meg_prediction = create_model(input_channels=204)(torch.randn(1, 204, 32))
    assert meg_prediction["envelope"].shape == (1, 1, 32)
    assert meg_prediction["mel"].shape == (1, 10, 32)
    assert torch.isfinite(meg_prediction["mel"]).all()
    print(f"PASS {MODEL_NAME}: EEG/204-MEG shape, finite backward, fusion, direct heads")


if __name__ == "__main__":
    main()
