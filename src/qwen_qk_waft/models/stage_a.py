from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..geometry import displacement_to_map


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.block(value))


class DecodeBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(in_channels + skip_channels, out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(
            value, size=skip.shape[-2:], mode="bilinear", align_corners=True
        )
        return self.fuse(torch.cat((value, skip), dim=1))


class StageA(nn.Module):
    """Deterministic coarse backward-map prior with calibrated confidence.

    The geometry branch intentionally retains the parameter names used by the
    existing v3.1 ``DocumentGeometryPrior`` checkpoint.  ``load_geometry`` can
    therefore import only its ``prior.*`` tensors while the new confidence
    head is trained separately.
    """

    def __init__(
        self,
        base_channels: int = 32,
        max_displacement_ratio: float = 0.35,
        control_stride: int = 8,
    ) -> None:
        super().__init__()
        channels = int(base_channels)
        self.max_displacement_ratio = float(max_displacement_ratio)
        self.control_stride = int(control_stride)
        self.stem = nn.Sequential(
            ConvNormAct(5, channels, kernel_size=5), ResidualBlock(channels)
        )
        self.down1 = nn.Sequential(
            ConvNormAct(channels, 2 * channels, stride=2),
            ResidualBlock(2 * channels),
        )
        self.down2 = nn.Sequential(
            ConvNormAct(2 * channels, 4 * channels, stride=2),
            ResidualBlock(4 * channels),
        )
        self.down3 = nn.Sequential(
            ConvNormAct(4 * channels, 8 * channels, stride=2),
            ResidualBlock(8 * channels),
        )
        self.bottleneck = nn.Sequential(
            ResidualBlock(8 * channels), ResidualBlock(8 * channels)
        )
        self.up2 = DecodeBlock(8 * channels, 4 * channels, 4 * channels)
        self.up1 = DecodeBlock(4 * channels, 2 * channels, 2 * channels)
        self.up0 = DecodeBlock(2 * channels, channels, channels)
        self.head = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, 2, kernel_size=3, padding=1),
        )
        self.confidence_head = nn.Sequential(
            ConvNormAct(channels + 2, channels),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.zeros_(self.confidence_head[-1].bias)

    @staticmethod
    def _coordinate_channels(image: Tensor) -> Tensor:
        batch, _, height, width = image.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        return torch.stack((x, y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, warped: Tensor) -> dict[str, Tensor]:
        _, _, height, width = warped.shape
        e0 = self.stem(torch.cat((warped, self._coordinate_channels(warped)), dim=1))
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.bottleneck(self.down3(e2))
        decoded = self.up0(self.up1(self.up2(e3, e2), e1), e0)
        raw = self.head(decoded)
        if self.control_stride > 1:
            coarse_size = (
                (height + self.control_stride - 1) // self.control_stride,
                (width + self.control_stride - 1) // self.control_stride,
            )
            raw = F.adaptive_avg_pool2d(raw, coarse_size)
            raw = F.interpolate(
                raw, size=(height, width), mode="bicubic", align_corners=True
            )
        scale = raw.new_tensor((width, height)).view(1, 2, 1, 1)
        displacement = torch.tanh(raw) * scale * self.max_displacement_ratio
        confidence = torch.sigmoid(
            self.confidence_head(torch.cat((decoded, displacement / scale), dim=1))
        )
        return {
            "displacement": displacement,
            "map": displacement_to_map(displacement),
            "confidence": confidence,
        }

    def load_geometry(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("state_dict", payload))
        geometry = {
            key.removeprefix("prior."): value
            for key, value in state.items()
            if key.startswith("prior.") and "global_pose_head" not in key
        }
        current = self.state_dict()
        current.update(geometry)
        self.load_state_dict(current)


def set_stage_a_trainable(stage_a: StageA, *, confidence_only: bool) -> None:
    for parameter in stage_a.parameters():
        parameter.requires_grad = False
    if confidence_only:
        for parameter in stage_a.confidence_head.parameters():
            parameter.requires_grad = True
