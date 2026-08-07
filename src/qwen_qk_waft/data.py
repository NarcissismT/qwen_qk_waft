from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .geometry import (
    displacement_to_map,
    pixel_grid,
    resize_absolute_map,
    valid_map_mask,
)


def load_rgb(path: str | Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(image: Tensor) -> Image.Image:
    array = (
        image.detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _load_array(path: Path, key: str) -> np.ndarray:
    value = np.load(path)
    if isinstance(value, np.lib.npyio.NpzFile):
        return value[key] if key in value else value[value.files[0]]
    return value


def _flow_tensor(array: np.ndarray) -> Tensor:
    if array.shape[-1] == 2:
        array = np.moveaxis(array, -1, 0)
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).unsqueeze(0)


class DocumentMapDataset(Dataset[dict[str, Any]]):
    """Warped/target/absolute backward-map records from a JSONL manifest."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        work_size: tuple[int, int] = (512, 512),
        limit: int | None = None,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.work_size = tuple(int(v) for v in work_size)
        self.records = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit is not None:
            self.records = self.records[: int(limit)]

    def __len__(self) -> int:
        return len(self.records)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        warped = load_rgb(self._path(record["warped"]))
        target = load_rgb(self._path(record["target"]))
        flow_or_map = _flow_tensor(_load_array(self._path(record["flow"]), "flow"))
        flow_size = flow_or_map.shape[-2:]
        source_size = tuple(int(v) for v in record.get("flow_source_size", flow_size))
        flow_format = str(record.get("flow_format", "displacement")).lower()
        if flow_format in {"displacement", "backward_displacement", "flow"}:
            pixel_map = displacement_to_map(flow_or_map)
        elif flow_format in {"absolute_map", "backward_map", "map"}:
            pixel_map = flow_or_map
        else:
            source_h, source_w = source_size
            pixel_map = flow_or_map.clone()
            pixel_map[:, 0] = (pixel_map[:, 0] + 1) * (source_w - 1) / 2
            pixel_map[:, 1] = (pixel_map[:, 1] + 1) * (source_h - 1) / 2
        pixel_map = resize_absolute_map(
            pixel_map,
            self.work_size,
            source_size_from=source_size,
            source_size_to=self.work_size,
        ).squeeze(0)
        warped = F.interpolate(
            warped.unsqueeze(0),
            self.work_size,
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        target = F.interpolate(
            target.unsqueeze(0),
            self.work_size,
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        valid = valid_map_mask(pixel_map.unsqueeze(0), self.work_size).squeeze(0)
        valid_path = record.get("valid")
        if valid_path:
            mask = _load_array(self._path(valid_path), "valid").astype(np.float32)
            if mask.ndim == 2:
                mask = mask[None]
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
            mask_tensor = F.interpolate(mask_tensor, self.work_size, mode="nearest")
            valid &= mask_tensor.squeeze(0) > 0.5
        return {
            "id": str(record.get("id", index)),
            "warped": warped,
            "target": target,
            "map": pixel_map,
            "valid": valid,
        }

