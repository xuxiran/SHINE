"""Experiment 58: ungated depthwise temporal blocks."""
from .variants import create_model as _create_model


def create_model(input_channels: int, **_: object):
    return _create_model(input_channels, 58)
