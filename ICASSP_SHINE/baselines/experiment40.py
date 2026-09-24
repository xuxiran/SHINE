
#!/usr/bin/env python3
"""MEG-ready BrainMagic with a residual spatial adapter and bounded context."""
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
    """Short-gradient 204/64-channel to latent-channel projection."""
    def __init__(self, input_channels: int, channels: int = 64):
        super().__init__()
        self.skip = nn.Conv1d(input_channels, channels, 1, bias=False)
        self.main = nn.Sequential(
            nn.Conv1d(input_channels, 2 * channels, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(2 * channels, channels, 1, bias=False),
        )
        self.norm = ChannelNorm(channels)

    def forward(self, neural: torch.Tensor) -> torch.Tensor:
        return self.norm((self.skip(neural) + self.main(neural)) * math.sqrt(0.5))


class GatedDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int = 9):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.temporal = nn.Conv1d(
            channels, 2 * channels, kernel, padding=padding,
            dilation=dilation, groups=channels,
        )
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = ChannelNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.temporal(x).chunk(2, 1)
        update = self.mix(torch.tanh(value) * torch.sigmoid(gate))
        return self.norm(x + 0.5 * update)


class MEGReadyBrainMagic(nn.Module):
    def __init__(self, input_channels: int, channels: int = 64):
        super().__init__()
        self.spatial = ResidualSpatialAdapter(input_channels, channels)
        self.blocks = nn.ModuleList([
            GatedDilatedBlock(channels, dilation)
            for dilation in (1, 2, 4, 8, 1, 2)
        ])
        self.head = nn.Sequential(
            nn.GELU(), nn.Conv1d(channels, 2 * channels, 1),
            nn.GELU(), nn.Conv1d(2 * channels, 11, 1),
        )

    def forward(self, neural: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.spatial(neural)
        skip_sum = x
        for block in self.blocks:
            x = block(x)
            skip_sum = skip_sum + x
        output = self.head(skip_sum / math.sqrt(len(self.blocks) + 1))
        return {"envelope": output[:, :1], "mel": output[:, 1:]}


def create_model(input_channels: int, **_: object) -> nn.Module:
    return MEGReadyBrainMagic(input_channels)
