from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..geometry import (
    displacement_to_map,
    map_jacobian_determinant,
    map_to_displacement,
    resize_absolute_map,
    sample_by_map,
)
from ..official_waft import MODEL_CONFIGS
from .dpt import OfficialQKDPT
from .stage_a import StageA
from .waft import WAFTRefiner
from .waft_checkpoint import load_official_waft_initialization


class SourceLocalEncoder(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        half = max(out_channels // 2, 16)
        self.network = nn.Sequential(
            nn.Conv2d(3, half, 7, stride=2, padding=3),
            nn.GroupNorm(4, half),
            nn.GELU(),
            nn.Conv2d(half, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.network(image)


class QwenQKWAFT(nn.Module):
    """Qwen Q/K frontend and the official pretrained WAFT-A2 recurrent core."""

    def __init__(
        self,
        *,
        qk_channels: int,
        layer_count: int = 4,
        stage_a_base_channels: int = 32,
        stage_a_max_displacement_ratio: float = 0.35,
        stage_a_control_stride: int = 8,
        waft_model_name: str = "vits",
        waft_patch_size: int = 8,
        timm_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.stage_a = StageA(
            base_channels=stage_a_base_channels,
            max_displacement_ratio=stage_a_max_displacement_ratio,
            control_stride=stage_a_control_stride,
        )
        self.waft = WAFTRefiner(
            model_name=waft_model_name,
            patch_size=waft_patch_size,
            timm_checkpoint=timm_checkpoint,
        )
        config = MODEL_CONFIGS[waft_model_name]
        feature_channels = int(config["features"])
        embed_dim = int(self.waft.refine_net.embed_dim)
        dpt_out_channels = list(config["out_channels"])
        self.dpt_q = OfficialQKDPT(
            qk_channels,
            embed_dim,
            feature_channels,
            dpt_out_channels,
            layer_count,
        )
        self.dpt_k = OfficialQKDPT(
            qk_channels,
            embed_dim,
            feature_channels,
            dpt_out_channels,
            layer_count,
        )
        self.local_encoder = SourceLocalEncoder(feature_channels)
        self.source_fusion = nn.Conv2d(
            2 * feature_channels, feature_channels, 1
        )

    def load_waft_pretrained(
        self, checkpoint_path: str | Path
    ) -> dict[str, int | str]:
        return load_official_waft_initialization(
            checkpoint_path,
            self.waft,
            self.dpt_q,
            self.dpt_k,
        )

    def forward(
        self,
        warped: Tensor,
        target_queries: Tensor,
        source_keys: Tensor,
        *,
        iterations: int = 5,
        use_local_encoder: bool = True,
        use_gate: bool = True,
        render: bool = True,
    ) -> dict[str, Tensor | list[Tensor]]:
        _, _, height, width = warped.shape
        half_size = (height // 2, width // 2)
        with torch.no_grad():
            stage_a = self.stage_a(warped)
        target_feature = self.dpt_q(target_queries, half_size)
        diffusion_source = self.dpt_k(source_keys, half_size)
        if use_local_encoder:
            local_source = self.local_encoder(warped)
            source_feature = self.source_fusion(
                torch.cat((diffusion_source, local_source), dim=1)
            )
        else:
            source_feature = diffusion_source

        coarse_half_map = resize_absolute_map(
            stage_a["map"],
            half_size,
            source_size_from=(height, width),
            source_size_to=half_size,
        )
        initial_displacement = map_to_displacement(coarse_half_map)
        coarse_confidence = F.interpolate(
            stage_a["confidence"],
            half_size,
            mode="bilinear",
            align_corners=True,
        )
        refined = self.waft(
            target_feature,
            source_feature,
            initial_displacement,
            coarse_confidence,
            iterations=iterations,
            use_gate=use_gate,
        )
        displacements = refined["displacements"]
        maps = [displacement_to_map(value) for value in displacements]
        final_map = maps[-1]
        determinant = map_jacobian_determinant(final_map)
        result: dict[str, Tensor | list[Tensor]] = {
            "coarse_map": stage_a["map"],
            "coarse_confidence": stage_a["confidence"],
            "target_feature": target_feature,
            "source_feature": source_feature,
            "maps": maps,
            "displacements": displacements,
            "infos": refined["infos"],
            "gates": refined["gates"],
            "final_map": final_map,
            "jacobian_determinant": determinant,
            "fold_mask": determinant <= 0,
        }
        if render:
            result["rectified"] = sample_by_map(warped, final_map)
        return result
