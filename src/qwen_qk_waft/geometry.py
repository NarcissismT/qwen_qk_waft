"""Geometry primitives for absolute target-to-source backward maps.

The complete project uses pixel ``(x, y)`` coordinates and
``align_corners=True``.  The conversion to a normalized sampling grid occurs
only at the native-RGB renderer boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


ALIGN_CORNERS = True


def pixel_grid(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def map_to_displacement(pixel_map: Tensor) -> Tensor:
    batch, _, height, width = pixel_map.shape
    return pixel_map - pixel_grid(
        batch, height, width, device=pixel_map.device, dtype=pixel_map.dtype
    )


def displacement_to_map(displacement: Tensor) -> Tensor:
    batch, _, height, width = displacement.shape
    return displacement + pixel_grid(
        batch, height, width, device=displacement.device, dtype=displacement.dtype
    )


def resize_absolute_map(
    pixel_map: Tensor,
    target_size: Sequence[int],
    *,
    source_size_from: Sequence[int] | None = None,
    source_size_to: Sequence[int] | None = None,
) -> Tensor:
    """Resize the target grid and, when requested, its source coordinates."""

    source_size_from = source_size_from or pixel_map.shape[-2:]
    source_size_to = source_size_to or target_size
    result = F.interpolate(
        pixel_map,
        size=tuple(int(v) for v in target_size),
        mode="bilinear",
        align_corners=ALIGN_CORNERS,
    )
    from_h, from_w = (int(v) for v in source_size_from)
    to_h, to_w = (int(v) for v in source_size_to)
    result[:, 0] *= (to_w - 1) / max(from_w - 1, 1)
    result[:, 1] *= (to_h - 1) / max(from_h - 1, 1)
    return result


def resize_displacement(
    displacement: Tensor,
    target_size: Sequence[int],
    *,
    source_size_from: Sequence[int] | None = None,
    source_size_to: Sequence[int] | None = None,
) -> Tensor:
    pixel_map = displacement_to_map(displacement)
    resized = resize_absolute_map(
        pixel_map,
        target_size,
        source_size_from=source_size_from,
        source_size_to=source_size_to,
    )
    return map_to_displacement(resized)


def normalized_grid(pixel_map: Tensor, source_size: Sequence[int]) -> Tensor:
    source_h, source_w = (int(v) for v in source_size)
    x = 2.0 * pixel_map[:, 0] / max(source_w - 1, 1) - 1.0
    y = 2.0 * pixel_map[:, 1] / max(source_h - 1, 1) - 1.0
    return torch.stack((x, y), dim=-1)


def sample_by_map(
    source: Tensor,
    pixel_map: Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> Tensor:
    grid = normalized_grid(pixel_map, source.shape[-2:])
    return F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=ALIGN_CORNERS,
    )


def sample_feature_at_displacement(feature: Tensor, displacement: Tensor) -> Tensor:
    batch, _, height, width = feature.shape
    current_map = displacement + pixel_grid(
        batch, height, width, device=feature.device, dtype=feature.dtype
    )
    return sample_by_map(feature, current_map)


def valid_map_mask(pixel_map: Tensor, source_size: Sequence[int]) -> Tensor:
    source_h, source_w = (int(v) for v in source_size)
    x, y = pixel_map[:, 0:1], pixel_map[:, 1:2]
    return (x >= 0) & (x <= source_w - 1) & (y >= 0) & (y <= source_h - 1)


def map_jacobian_determinant(pixel_map: Tensor) -> Tensor:
    dx = pixel_map[:, :, :, 1:] - pixel_map[:, :, :, :-1]
    dy = pixel_map[:, :, 1:, :] - pixel_map[:, :, :-1, :]
    dx = dx[:, :, :-1]
    dy = dy[:, :, :, :-1]
    return dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]

