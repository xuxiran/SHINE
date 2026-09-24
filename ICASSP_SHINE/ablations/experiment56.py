"""Experiment 56: final local block only before context (no hierarchy)."""
from .variants import create_model as _create_model


def create_model(input_channels: int, **_: object):
    return _create_model(input_channels, 56)
