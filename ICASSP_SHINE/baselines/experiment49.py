
#!/usr/bin/env python3
"""Protocol-adapted DMF2Mel with dynamic contrast, HAMS and spline attention."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ChannelNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class DynamicContrastAggregation(nn.Module):
    """Separates transient foreground from smoothed background and learns fusion."""
    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.gate = nn.Conv1d(2 * channels, channels, 1)
        self.project = nn.Conv1d(channels, channels, 1)
        self.norm = ChannelNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        background = F.avg_pool1d(x, 17, stride=1, padding=8)
        foreground = local - background
        gate = torch.sigmoid(self.gate(torch.cat([foreground, background], dim=1)))
        return self.norm(x + self.project(gate * foreground + (1.0 - gate) * background))


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, 2 * channels, 5, padding=2 * dilation,
                                   dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = ChannelNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.depthwise(x).chunk(2, dim=1)
        return self.norm(x + self.pointwise(F.gelu(value) * torch.sigmoid(gate)))


class HAMSNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.level1 = TemporalBlock(channels, 1)
        self.down1 = nn.Conv1d(channels, channels, 4, 2, 1)
        self.level2 = TemporalBlock(channels, 2)
        self.down2 = nn.Conv1d(channels, channels, 4, 2, 1)
        self.level3 = nn.Sequential(TemporalBlock(channels, 4), TemporalBlock(channels, 8))
        self.up2 = nn.Conv1d(2 * channels, channels, 1)
        self.up1 = nn.Conv1d(2 * channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.level1(x)
        b = self.level2(self.down1(a))
        c = self.level3(self.down2(b))
        c = F.interpolate(c, size=b.shape[-1], mode="linear", align_corners=False)
        b = self.level2(self.up2(torch.cat([b, c], dim=1)))
        b = F.interpolate(b, size=a.shape[-1], mode="linear", align_corners=False)
        return self.level1(self.up1(torch.cat([a, b], dim=1)))


class SplineGate(nn.Module):
    """Per-channel radial spline map, a stable lightweight AGKAN adaptation."""
    def __init__(self, channels: int, knots: int = 8):
        super().__init__()
        self.register_buffer("centers", torch.linspace(-2.0, 2.0, knots))
        self.weights = nn.Parameter(torch.zeros(channels, knots))
        self.base = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-((sequence.unsqueeze(-1) - self.centers) / 0.75).square())
        spline = (basis * self.weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return self.norm(self.base(sequence) + spline)


class SplineMapAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4, pool: int = 8):
        super().__init__()
        self.pool = pool
        self.spline = SplineGate(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        query = self.spline(sequence)
        key_value = F.avg_pool1d(query.transpose(1, 2), self.pool, self.pool,
                                 ceil_mode=True).transpose(1, 2)
        context, _ = self.attention(query, key_value, key_value, need_weights=False)
        return self.norm(sequence + context).transpose(1, 2)


class BiConvStateBlock(nn.Module):
    """Dependency-free bidirectional conv-state substitute for convMamba."""
    def __init__(self, channels: int):
        super().__init__()
        self.pre = TemporalBlock(channels, 2)
        self.state = nn.GRU(channels, channels // 2, batch_first=True, bidirectional=True)
        self.mix = nn.Conv1d(2 * channels, channels, 1)
        self.norm = ChannelNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.pre(x)
        state, _ = self.state(local.transpose(1, 2))
        return self.norm(x + self.mix(torch.cat([local, state.transpose(1, 2)], dim=1)))


class DMF2MelAdapted(nn.Module):
    def __init__(self, input_channels: int, channels: int = 96):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv1d(input_channels, channels, 1),
                                     ChannelNorm(channels), nn.GELU())
        self.dynamic = nn.Sequential(DynamicContrastAggregation(channels),
                                     DynamicContrastAggregation(channels))
        self.hams = HAMSNet(channels)
        self.spline_attention = SplineMapAttention(channels)
        self.state = nn.Sequential(BiConvStateBlock(channels), BiConvStateBlock(channels))
        self.head = nn.Conv1d(channels, 11, 1)

    def forward(self, neural: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.spatial(neural)
        x = self.spline_attention(self.dynamic(x) + self.hams(x))
        output = self.head(self.state(x))
        return {"envelope": output[:, :1], "mel": output[:, 1:]}


def create_model(input_channels: int, **_: object) -> nn.Module:
    return DMF2MelAdapted(input_channels)
