"""Experiment 54: pooled context, fixed fusion, and envelope-guided decoding."""
from .variants import create_model as _create_model


def create_model(input_channels: int, **_: object):
    return _create_model(input_channels, 54)
