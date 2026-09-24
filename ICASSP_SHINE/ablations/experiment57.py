"""Experiment 57: simple 1x1 spatial projection."""
from .variants import create_model as _create_model


def create_model(input_channels: int, **_: object):
    return _create_model(input_channels, 57)
