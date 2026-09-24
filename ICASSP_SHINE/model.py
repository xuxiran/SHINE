#!/usr/bin/env python3
"""Full-context UC-SHINE for 64-Hz EEG and 204-channel MEG."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ChannelNorm(nn.Module):
    """Layer-normalize channels without mixing time samples."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResidualSpatialAdapter(nn.Module):
    """Two-layer residual sensor projection shared by EEG and MEG."""

    def __init__(self, input_channels: int, channels: int):
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


class ShiftFreeHierarchicalBlock(nn.Module):
    """Symmetric gated temporal block with a stable residual gradient."""

    def __init__(self, channels: int, dilation: int, kernel: int = 9):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.temporal = nn.Conv1d(
            channels, 2 * channels, kernel, padding=padding,
            dilation=dilation, groups=channels,
        )
        self.mix = nn.Sequential(
            nn.Conv1d(channels, 2 * channels, 1),
            nn.GELU(),
            nn.Conv1d(2 * channels, channels, 1),
        )
        self.norm = ChannelNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.temporal(x).chunk(2, 1)
        update = self.mix(torch.tanh(value) * torch.sigmoid(gate))
        return self.norm(x + 0.5 * update)


class PooledContextBlock(nn.Module):
    """Position-aware global context with pooled keys and values."""

    def __init__(self, channels: int, heads: int = 4, pool: int = 8):
        super().__init__()
        self.pool = pool
        self.position = nn.Conv1d(
            channels, channels, 31, padding=15, groups=channels, bias=False,
        )
        self.attn_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=0.1, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * channels, channels),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        positioned = sequence + 0.2 * self.position(x).transpose(1, 2)
        normalized = self.attn_norm(positioned)
        key_value = F.avg_pool1d(
            normalized.transpose(1, 2), self.pool, self.pool, ceil_mode=True,
        ).transpose(1, 2)
        context, _ = self.attention(
            normalized, key_value, key_value, need_weights=False,
        )
        sequence = positioned + self.dropout(context)
        sequence = sequence + self.dropout(self.ffn(self.ffn_norm(sequence)))
        return sequence.transpose(1, 2)


class UnifiedContextSHINE(nn.Module):
    """Shared full-window hierarchy, pooled context, and gated fusion."""

    def __init__(self, input_channels: int, channels: int = 96):
        super().__init__()
        self.spatial = ResidualSpatialAdapter(input_channels, channels)
        self.local_blocks = nn.ModuleList([
            ShiftFreeHierarchicalBlock(channels, dilation)
            for dilation in (1, 2, 4, 8, 16, 32, 64, 128)
        ])
        self.hierarchy = nn.Sequential(
            nn.Conv1d(5 * channels, channels, 1),
            ChannelNorm(channels),
            nn.GELU(),
        )
        self.context_blocks = nn.ModuleList([
            PooledContextBlock(channels), PooledContextBlock(channels),
        ])
        self.local_head = nn.Conv1d(channels, 11, 1)
        self.context_head = nn.Conv1d(channels, 11, 1)
        self.fusion_gate = nn.Conv1d(2 * channels, 11, 1)

    def forward(self, neural: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.spatial(neural)
        levels = [x]
        for index, block in enumerate(self.local_blocks):
            x = block(x)
            if index % 2 == 1:
                levels.append(x)
        local_features = self.hierarchy(torch.cat(levels, dim=1))

        context_features = local_features
        for block in self.context_blocks:
            context_features = block(context_features)

        local_output = self.local_head(local_features)
        context_output = self.context_head(context_features)
        mixture = torch.sigmoid(
            self.fusion_gate(torch.cat([local_features, context_features], dim=1))
        )
        fused = local_output + mixture * (context_output - local_output)

        envelope = fused[:, :1]
        mel = fused[:, 1:]
        return {
            "envelope": envelope,
            "mel": mel,
            "aux_envelopes": (local_output[:, :1], context_output[:, :1]),
            "aux_mels": (local_output[:, 1:], context_output[:, 1:]),
        }


def create_model(input_channels: int, **_: object) -> nn.Module:
    return UnifiedContextSHINE(input_channels)
