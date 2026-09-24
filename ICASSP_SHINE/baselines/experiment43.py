
#!/usr/bin/env python3
"""ConvConcatNet baseline adapted to arbitrary EEG/MEG channel counts."""

from __future__ import annotations

import torch
from torch import nn


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class DenseConcatBlock(nn.Module):
    """Four temporal layers with the extensive concatenation used by ConvConcatNet."""

    def __init__(self, channels: int = 64, growth: int = 32, kernel: int = 9) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for index in range(4):
            input_channels = channels + index * growth
            self.layers.append(nn.Sequential(
                nn.Conv1d(input_channels, input_channels, kernel, padding=kernel // 2, groups=input_channels),
                nn.Conv1d(input_channels, growth, 1),
                ChannelLayerNorm(growth),
                nn.LeakyReLU(0.1),
            ))
        self.project = nn.Conv1d(channels + 4 * growth, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return self.project(torch.cat(features, dim=1))


class ConvConcatNet(nn.Module):
    def __init__(self, input_channels: int, hidden: int = 64, blocks: int = 6) -> None:
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv1d(input_channels, hidden, 1), ChannelLayerNorm(hidden), nn.LeakyReLU(0.1))
        self.blocks = nn.ModuleList([DenseConcatBlock(hidden) for _ in range(blocks)])
        self.context = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, 33, padding=16, groups=hidden),
                nn.Conv1d(hidden, hidden, 1),
                ChannelLayerNorm(hidden),
                nn.LeakyReLU(0.1),
            ) for _ in range(blocks)
        ])
        self.fusion = nn.Sequential(nn.Conv1d(hidden * (blocks + 1), hidden * 2, 1), nn.GELU(), nn.Dropout(0.15))
        self.envelope_head = nn.Conv1d(hidden * 2, 1, 1)
        self.mel_head = nn.Conv1d(hidden * 2, 10, 1)

    def forward(self, neural: torch.Tensor) -> dict[str, torch.Tensor]:
        original = self.spatial(neural)
        x = original
        hierarchy = [original]
        for block, context in zip(self.blocks, self.context):
            x = x + context(block(x))
            hierarchy.append(x)
        features = self.fusion(torch.cat(hierarchy, dim=1))
        return {"envelope": self.envelope_head(features), "mel": self.mel_head(features)}


def create_model(input_channels: int, **_: object) -> nn.Module:
    return ConvConcatNet(input_channels=input_channels)


if __name__ == "__main__":
    model = create_model(64)
    output = model(torch.randn(2, 64, 256))
    print({key: tuple(value.shape) for key, value in output.items()})
