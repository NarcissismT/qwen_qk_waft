from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..geometry import sample_feature_at_displacement
from ..official_waft import MODEL_CONFIGS, VisionTransformer


class ConfidenceGate(nn.Module):
    """Plan-specific protection gate kept outside the official updater."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2 * channels + 3, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        self.initialize_open()

    def initialize_open(self, probability: float = 0.99) -> None:
        output = self.network[-1]
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, math.log(probability / (1.0 - probability)))

    def forward(
        self,
        hidden: Tensor,
        coarse_confidence: Tensor,
        feature_difference: Tensor,
        uncertainty: Tensor,
    ) -> Tensor:
        mixture = torch.softmax(uncertainty[:, :2], dim=1)
        log_scale = torch.stack(
            (
                uncertainty[:, 2].clamp(0, 10),
                uncertainty[:, 3].clamp(-10, 0),
            ),
            dim=1,
        )
        expected_scale = (mixture * torch.exp(log_scale)).sum(dim=1, keepdim=True)
        entropy = -(mixture * torch.log(mixture.clamp_min(1.0e-6))).sum(
            dim=1, keepdim=True
        )
        value = torch.cat(
            (
                hidden,
                coarse_confidence,
                feature_difference.abs(),
                expected_scale,
                entropy,
            ),
            dim=1,
        )
        return torch.sigmoid(self.network(value))


class WAFTRefiner(nn.Module):
    """Official WAFT-A2 recurrent core with Stage-A warm-start adaptation."""

    def __init__(
        self,
        *,
        model_name: str = "vits",
        patch_size: int = 8,
        timm_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.iter_dim = MODEL_CONFIGS[model_name]["features"]
        self.refine_net = VisionTransformer(
            model_name,
            self.iter_dim,
            patch_size=patch_size,
            timm_checkpoint=timm_checkpoint,
        )
        self.refine_net.dpt_head.scratch.output_conv2.requires_grad_(False)
        self.refine_net.dpt_head.scratch.refinenet4.resConfUnit1.requires_grad_(False)
        self.hidden_conv = nn.Conv2d(
            2 * self.iter_dim, self.iter_dim, 1, 1, 0, bias=True
        )
        self.warp_linear = nn.Conv2d(
            3 * self.iter_dim + 2, self.iter_dim, 1, 1, 0, bias=True
        )
        self.refine_transform = nn.Conv2d(
            self.iter_dim // 2 * 3,
            self.iter_dim,
            1,
            1,
            0,
            bias=True,
        )
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(
                self.iter_dim,
                2 * self.iter_dim,
                3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.iter_dim, 4 * 9, 1, padding=0, bias=True),
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(
                self.iter_dim,
                2 * self.iter_dim,
                3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.iter_dim, 6, 1, padding=0, bias=True),
        )
        self.gate = ConfidenceGate(self.iter_dim)

    @staticmethod
    def upsample_data(flow: Tensor, info: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=flow.device.type, enabled=False):
            flow = flow.float()
            info = info.float()
            mask = mask.float()
            batch, channels, height, width = info.shape
            mask = mask.view(batch, 1, 9, 2, 2, height, width)
            mask = torch.softmax(mask, dim=2)
            up_flow = F.unfold(2 * flow, [3, 3], padding=1)
            up_flow = up_flow.view(batch, 2, 9, 1, 1, height, width)
            up_info = F.unfold(info, [3, 3], padding=1)
            up_info = up_info.view(batch, channels, 9, 1, 1, height, width)
            up_flow = torch.sum(mask * up_flow, dim=2)
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
            up_info = torch.sum(mask * up_info, dim=2)
            up_info = up_info.permute(0, 1, 4, 2, 5, 3)
            return (
                up_flow.reshape(batch, 2, 2 * height, 2 * width),
                up_info.reshape(batch, channels, 2 * height, 2 * width),
            )

    def forward(
        self,
        target_feature: Tensor,
        source_feature: Tensor,
        initial_displacement: Tensor,
        coarse_confidence: Tensor,
        *,
        iterations: int,
        use_gate: bool,
    ) -> dict[str, list[Tensor] | Tensor]:
        displacement = initial_displacement.float()
        coarse_confidence = coarse_confidence.float()
        aligned_source = sample_feature_at_displacement(source_feature, displacement)
        net = self.hidden_conv(torch.cat((target_feature, aligned_source), dim=1))
        displacements: list[Tensor] = []
        infos: list[Tensor] = []
        gates: list[Tensor] = []

        for _ in range(iterations):
            displacement = displacement.detach().float()
            aligned_source = sample_feature_at_displacement(source_feature, displacement)
            refine_input = self.warp_linear(
                torch.cat(
                    (target_feature, aligned_source, net, displacement),
                    dim=1,
                )
            )
            refine_outputs = self.refine_net(refine_input)
            net = self.refine_transform(
                torch.cat((refine_outputs["out"], net), dim=1)
            )
            flow_update = self.flow_head(net)
            residual, info = flow_update[:, :2].float(), flow_update[:, 2:].float()
            if use_gate:
                gate = self.gate(
                    net,
                    coarse_confidence,
                    target_feature - aligned_source,
                    info,
                ).float()
            else:
                gate = torch.ones_like(coarse_confidence)
            with torch.autocast(device_type=displacement.device.type, enabled=False):
                displacement = displacement + gate * residual
            mask = (0.25 * self.upsample_weight(net)).float()
            full_displacement, full_info = self.upsample_data(
                displacement, info, mask
            )
            displacements.append(full_displacement)
            infos.append(full_info)
            gates.append(gate)

        return {
            "half_displacement": displacement,
            "displacements": displacements,
            "infos": infos,
            "gates": gates,
        }
