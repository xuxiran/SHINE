
#!/usr/bin/env python3
"""Protocol-adapted SSM2Mel: S4-like U-Net, external memory and SSM/MHSA.

The original paper uses subject IDs.  This implementation intentionally replaces
that path with self-conditioned feature strength modulation so an unseen-subject
test recording cannot leak its identity into the model.
"""
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


class S4LikeBlock(nn.Module):
    """Linear-complexity structured temporal filter used as an S4 surrogate."""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.filter = nn.Conv1d(channels, 2 * channels, 9, padding=4 * dilation,
                                dilation=dilation, groups=channels)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = ChannelNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.filter(x).chunk(2, dim=1)
        return self.norm(x + self.mix(torch.tanh(value) * torch.sigmoid(gate)))


class S4UNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.enc1 = S4LikeBlock(channels, 1)
        self.down1 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)
        self.enc2 = S4LikeBlock(channels, 2)
        self.down2 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)
        self.bottleneck = nn.Sequential(S4LikeBlock(channels, 4), S4LikeBlock(channels, 8))
        self.fuse2 = nn.Conv1d(2 * channels, channels, 1)
        self.dec2 = S4LikeBlock(channels, 2)
        self.fuse1 = nn.Conv1d(2 * channels, channels, 1)
        self.dec1 = S4LikeBlock(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        z = self.bottleneck(self.down2(e2))
        z = F.interpolate(z, size=e2.shape[-1], mode="linear", align_corners=False)
        z = self.dec2(self.fuse2(torch.cat([z, e2], dim=1)))
        z = F.interpolate(z, size=e1.shape[-1], mode="linear", align_corners=False)
        return self.dec1(self.fuse1(torch.cat([z, e1], dim=1)))


class ExternalMemoryAttention(nn.Module):
    def __init__(self, channels: int, slots: int = 32):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(slots, channels) / math.sqrt(channels))
        self.values = nn.Parameter(torch.randn(slots, channels) / math.sqrt(channels))
        self.query = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        weights = torch.softmax(self.query(sequence) @ self.keys.t() / math.sqrt(x.shape[1]), dim=-1)
        return self.norm(sequence + self.output(weights @ self.values)).transpose(1, 2)


class HybridSSMAttention(nn.Module):
    """Bidirectional state recurrence plus pooled-key MHSA, matching the paper ablation."""
    def __init__(self, channels: int, heads: int = 4, pool: int = 8):
        super().__init__()
        self.pool = pool
        self.pre = nn.LayerNorm(channels)
        self.state = nn.GRU(channels, channels // 2, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(4 * channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        normalized = self.pre(sequence)
        state, _ = self.state(normalized)
        key_value = F.avg_pool1d(normalized.transpose(1, 2), self.pool, self.pool,
                                 ceil_mode=True).transpose(1, 2)
        attended, _ = self.attn(normalized, key_value, key_value, need_weights=False)
        sequence = sequence + state + attended
        return (sequence + self.ffn(self.ffn_norm(sequence))).transpose(1, 2)


class SSM2MelAdapted(nn.Module):
    def __init__(self, input_channels: int, channels: int = 96):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv1d(input_channels, channels, 1),
                                     ChannelNorm(channels), nn.GELU())
        self.strength = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, channels, 1),
                                      nn.Sigmoid())
        self.s4_unet = S4UNet(channels)
        self.external_attention = ExternalMemoryAttention(channels)
        self.backbone = nn.Sequential(HybridSSMAttention(channels), HybridSSMAttention(channels))
        self.head = nn.Conv1d(channels, 11, 1)

    def forward(self, neural: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.spatial(neural)
        x = x * (1.0 + 0.5 * self.strength(x))
        x = self.external_attention(self.s4_unet(x))
        output = self.head(self.backbone(x))
        return {"envelope": output[:, :1], "mel": output[:, 1:]}


def create_model(input_channels: int, **_: object) -> nn.Module:
    return SSM2MelAdapted(input_channels)
