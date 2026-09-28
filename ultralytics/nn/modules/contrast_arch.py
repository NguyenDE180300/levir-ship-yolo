"""Contrast/SENetV2/EnSimAM/ASF architecture blocks for isolated ablations."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import ASFAttention, EnSimAM, ScalSeq
from .explicit_cues import explicit_cue


def _groups(channels: int) -> int:
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


class SENetV2(nn.Module):
    """Lightweight aggregated channel excitation with shape-preserving output."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.branches = nn.ModuleList(
            (
                nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.SiLU(), nn.Conv2d(hidden, channels, 1)),
                nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.SiLU(), nn.Conv2d(hidden, channels, 1)),
            )
        )
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, 1)
        excitation = sum(self.branches[i](pooled) for i in range(len(self.branches))) / len(self.branches)
        return x * torch.sigmoid(self.mix(excitation))


class _ContrastCueEncoder(nn.Module):
    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        hidden = max(8, min(hidden, channels))
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, cue: torch.Tensor) -> torch.Tensor:
        return self.net(cue)


class _ContrastStatsMixin:
    def _record_stats(self, gate: torch.Tensor) -> None:
        flat = gate.detach().float().flatten(1)
        q = torch.quantile(flat, flat.new_tensor((0.10, 0.50, 0.90)), dim=1)
        self.last_attention_stats = {
            "mean": float(flat.mean()), "std": float(flat.std()),
            "min": float(flat.min()), "max": float(flat.max()),
            "p10": float(q[0].mean()), "p50": float(q[1].mean()), "p90": float(q[2].mean()),
            "gamma": float(self.gamma.detach()),
        }


class ContrastSharedSENetV2EnSimAM(nn.Module, _ContrastStatsMixin):
    """Contrast residual guidance followed by SENetV2 and EnSimAM refinement."""

    def __init__(self, channels: list[int] | tuple[int, int], hidden: int = 32, cue: str = "contrast") -> None:
        super().__init__()
        feature_ch, rgb_ch = channels
        if rgb_ch != 3:
            raise ValueError("ContrastSharedSENetV2EnSimAM requires an RGB tap")
        self.cue = cue
        self.encoder = _ContrastCueEncoder(feature_ch, hidden)
        gate_hidden = max(8, min(hidden, feature_ch))
        self.gate = nn.Sequential(
            nn.Conv2d(2 * feature_ch, gate_hidden, 1, bias=False), nn.SiLU(),
            nn.Conv2d(gate_hidden, gate_hidden, 3, 1, 1, groups=gate_hidden, bias=False), nn.SiLU(),
            nn.Conv2d(gate_hidden, 1, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(()))
        self.senet = SENetV2(feature_ch)
        self.ensimam = EnSimAM()
        self.last_attention_stats: dict[str, float] = {}

    def forward(self, values: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feature, rgb = values
        cue = F.interpolate(explicit_cue(rgb, self.cue), size=feature.shape[-2:], mode="area")
        projected = self.encoder(cue)
        gate = torch.sigmoid(self.gate(torch.cat((feature, projected), dim=1)))
        self._record_stats(gate)
        return self.ensimam(self.senet(feature + self.gamma * gate * projected))


class ContrastClsGuidance(nn.Module, _ContrastStatsMixin):
    """Contrast-conditioned classification-only multiplicative guidance."""

    def __init__(self, channels: list[int] | tuple[int, int], hidden: int = 32) -> None:
        super().__init__()
        feature_ch, rgb_ch = channels
        if rgb_ch != 3:
            raise ValueError("ContrastClsGuidance requires an RGB tap")
        self.encoder = _ContrastCueEncoder(feature_ch, hidden)
        gate_hidden = max(8, min(hidden, feature_ch))
        self.gate = nn.Sequential(
            nn.Conv2d(2 * feature_ch, gate_hidden, 1, bias=False), nn.SiLU(),
            nn.Conv2d(gate_hidden, gate_hidden, 3, 1, 1, groups=gate_hidden, bias=False), nn.SiLU(),
            nn.Conv2d(gate_hidden, 1, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(()))
        self.last_attention_stats: dict[str, float] = {}

    def forward(self, values: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feature, rgb = values
        cue = F.interpolate(explicit_cue(rgb, "contrast"), size=feature.shape[-2:], mode="area")
        projected = self.encoder(cue)
        gate = torch.sigmoid(self.gate(torch.cat((feature, projected), dim=1)))
        self._record_stats(gate)
        return feature * (1.0 + self.gamma * gate)


class ASFHighResFusion(nn.Module):
    """ASF semantic enrichment for P2/P3 recipients only."""

    def __init__(self, channels: list[int] | tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"ASFHighResFusion expects detail + two semantic inputs, got {channels}")
        self.scale_fusion = ScalSeq(channels, out_channels)
        self.asf = ASFAttention(out_channels)

    def forward(self, values: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.asf(self.scale_fusion(values))


__all__ = (
    "SENetV2", "ContrastSharedSENetV2EnSimAM", "ContrastClsGuidance", "ASFHighResFusion",
)
