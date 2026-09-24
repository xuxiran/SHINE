
#!/usr/bin/env python3
"""MEG-ready VLAAI with shift-free iterative temporal refinement."""
from __future__ import annotations

import math
import torch
from torch import nn


class ChannelNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResidualSpatialAdapter(nn.Module):
    def __init__(self, input_channels: int, channels: int = 64):
        super().__init__()
        self.skip = nn.Conv1d(input_channels, channels, 1, bias=False)
        self.main = nn.Sequential(
            nn.Conv1d(input_channels, 2 * channels, 1, bias=False),
            nn.GELU(), nn.Conv1d(2 * channels, channels, 1, bias=False),
        )
        self.norm = ChannelNorm(channels)

    def forward(self, neural: torch.Tensor) -> torch.Tensor:
        return self.norm((self.skip(neural) + self.main(neural)) * math.sqrt(0.5))


class ShiftFreeExtractor(nn.Module):
    """VLAAI feature extractor with symmetric length-preserving convolutions."""
    def __init__(self, channels: int = 64, kernel: int = 9):
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = channels
        for output_channels in (128, 128, 96, 64):
            layers.extend([
                nn.Conv1d(input_channels, output_channels, kernel,
                          padding=kernel // 2),
                ChannelNorm(output_channels),
                nn.LeakyReLU(0.1),
            ])
            input_channels = output_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class IterativeVLAAIBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.extractor = ShiftFreeExtractor(channels)
        self.context = nn.Sequential(
            nn.Conv1d(channels, channels, 33, padding=16, groups=channels),
            nn.Conv1d(channels, channels, 1),
            ChannelNorm(channels),
            nn.LeakyReLU(0.1),
        )
        self.norm = ChannelNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + 0.5 * self.context(self.extractor(x)))


class MEGReadyVLAAI(nn.Module):
    def __init__(self, input_channels: int, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.spatial = ResidualSpatialAdapter(input_channels, channels)
        self.blocks = nn.ModuleList([IterativeVLAAIBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(nn.GELU(), nn.Conv1d(channels, 11, 1))

    def forward(self, neural: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.spatial(neural)
        for block in self.blocks:
            x = block(x)
        output = self.head(x)
        return {"envelope": output[:, :1], "mel": output[:, 1:]}


def create_model(input_channels: int, **_: object) -> nn.Module:
    return MEGReadyVLAAI(input_channels)
