
#!/usr/bin/env python3
"""MEG-ready HappyQuokka with continuous position-aware attention."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
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


class ContinuousPreLNFFTBlock(nn.Module):
    def __init__(self, channels: int = 64, heads: int = 4, pool: int = 8):
        super().__init__()
        self.pool = pool
        self.position = nn.Conv1d(channels, channels, 31, padding=15,
                                  groups=channels, bias=False)
        self.attn_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=0.1, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn_in = nn.Conv1d(channels, 4 * channels, 9, padding=4)
        self.ffn_out = nn.Conv1d(4 * channels, channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        positioned = sequence + 0.2 * self.position(sequence.transpose(1, 2)).transpose(1, 2)
        normalized = self.attn_norm(positioned)
        key_value = F.avg_pool1d(
            normalized.transpose(1, 2), self.pool, self.pool, ceil_mode=True,
        ).transpose(1, 2)
        attended, _ = self.attention(
            normalized, key_value, key_value, need_weights=False,
        )
        sequence = positioned + self.dropout(attended)
        hidden = self.ffn_norm(sequence).transpose(1, 2)
        update = self.ffn_out(self.dropout(F.gelu(self.ffn_in(hidden)))).transpose(1, 2)
        return sequence + self.dropout(update)


class MEGReadyHappyQuokka(nn.Module):
    def __init__(self, input_channels: int, channels: int = 64, layers: int = 6):
        super().__init__()
        self.spatial = ResidualSpatialAdapter(input_channels, channels)
        self.layers = nn.ModuleList([
            ContinuousPreLNFFTBlock(channels) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, 11)

    def forward(self, neural: torch.Tensor) -> dict[str, torch.Tensor]:
        sequence = self.spatial(neural).transpose(1, 2)
        for block in self.layers:
            sequence = block(sequence)
        output = self.head(self.final_norm(sequence)).transpose(1, 2)
        return {"envelope": output[:, :1], "mel": output[:, 1:]}


def create_model(input_channels: int, **_: object) -> nn.Module:
    return MEGReadyHappyQuokka(input_channels)
