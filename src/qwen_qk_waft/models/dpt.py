from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ..official_waft import DPTHead


class QKLayerAdapter(nn.Module):
    """Per-layer Q/K projection into the official WAFT ViT embedding space."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.projection = nn.Linear(in_channels, embed_dim)

    def forward(self, value: Tensor) -> Tensor:
        value = value.permute(0, 2, 3, 1)
        value = self.projection(self.norm(value))
        return value.flatten(1, 2)


class OfficialQKDPT(nn.Module):
    """Independent Q or K stream backed by WAFT's official DPTHead."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        feature_channels: int,
        dpt_out_channels: list[int],
        layer_count: int = 4,
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            QKLayerAdapter(in_channels, embed_dim) for _ in range(layer_count)
        )
        self.dpt_head = DPTHead(
            embed_dim,
            feature_channels,
            out_channels=dpt_out_channels,
        )
        self.dpt_head.scratch.output_conv2.requires_grad_(False)
        self.dpt_head.scratch.refinenet4.resConfUnit1.requires_grad_(False)
        self.output_projection = nn.Sequential(
            nn.Conv2d(feature_channels // 2, feature_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, features: Tensor, output_size: tuple[int, int]) -> Tensor:
        patch_h, patch_w = features.shape[-2:]
        adapted = [
            [adapter(features[:, index])]
            for index, adapter in enumerate(self.adapters)
        ]
        dense, _, _, _, _ = self.dpt_head(
            adapted,
            patch_h,
            patch_w,
            return_intermediate=True,
        )
        dense = F.interpolate(
            dense,
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )
        return self.output_projection(dense)
