"""Synthetic import, shape, finiteness, and gradient checks for registered models."""

from __future__ import annotations

import gc

import torch

from experiments import EXPERIMENTS, create_experiment_model


EXPECTED_EXPERIMENTS = (set(range(40, 44)) | set(range(45, 50))
                        | {53, 54, 55, 56, 57, 58})
EXPECTED_PARAMETERS = {
    43: 291_275,
    45: 1_197_899,
}


def check_experiment(number: int) -> int:
    experiment = EXPERIMENTS[number]
    model = create_experiment_model(experiment, input_channels=64)
    model.train()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if number in EXPECTED_PARAMETERS:
        assert parameter_count == EXPECTED_PARAMETERS[number], (
            f"experiment {number}: expected {EXPECTED_PARAMETERS[number]} parameters, "
            f"got {parameter_count}"
        )

    neural = torch.randn(1, 64, 16)
    prediction = model(neural)
    assert {"envelope", "mel"}.issubset(prediction), (
        f"experiment {number}: missing primary outputs {set(prediction)}"
    )
    assert prediction["envelope"].shape == (1, 1, 16), number
    assert prediction["mel"].shape == (1, 10, 16), number
    for name, value in prediction.items():
        tensors = value if isinstance(value, (tuple, list)) else (value,)
        assert all(torch.isfinite(tensor).all() for tensor in tensors), (
            f"experiment {number}: non-finite {name} output"
        )

    loss = prediction["envelope"].square().mean() + prediction["mel"].square().mean()
    for name in ("aux_envelopes", "aux_mels"):
        loss = loss + sum(value.square().mean() for value in prediction.get(name, ()))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.grad is not None]
    assert gradients, f"experiment {number}: backward produced no gradients"
    assert all(torch.isfinite(gradient).all() for gradient in gradients), (
        f"experiment {number}: backward produced non-finite gradients"
    )
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients), (
        f"experiment {number}: backward produced only zero gradients"
    )

    del model, neural, prediction, loss, gradients
    gc.collect()
    return parameter_count


def main() -> None:
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    registered = set(EXPERIMENTS)
    assert registered == EXPECTED_EXPERIMENTS, (
        f"registry mismatch: missing={EXPECTED_EXPERIMENTS - registered}, "
        f"unexpected={registered - EXPECTED_EXPERIMENTS}"
    )
    for experiment in EXPERIMENTS.values():
        assert experiment.epochs == 60, experiment
        assert experiment.min_epochs == 60, experiment
        assert experiment.patience == 60, experiment

    for number in sorted(EXPECTED_EXPERIMENTS):
        parameter_count = check_experiment(number)
        print(f"PASS experiment{number}: {parameter_count:,} parameters")


if __name__ == "__main__":
    main()
