from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwen_qk_waft.config import load_config, resolve_path
from qwen_qk_waft.geometry import (
    map_jacobian_determinant,
    pixel_grid,
    resize_absolute_map,
    sample_by_map,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        identity = pixel_grid(
            1, 512, 512, device=device, dtype=torch.bfloat16
        )
        determinant = map_jacobian_determinant(identity)
    identity_fold_rate = float((determinant <= 0).float().mean())
    unique_x = int(torch.unique(identity[0, 0, 0]).numel())
    unique_y = int(torch.unique(identity[0, 1, :, 0]).numel())
    if (
        identity.dtype != torch.float32
        or determinant.dtype != torch.float32
        or identity_fold_rate != 0.0
        or unique_x != 512
        or unique_y != 512
    ):
        raise RuntimeError("512px BF16-autocast coordinate audit failed")

    work_map = identity.clone()
    work_map[:, 0] += 0.375
    work_map[:, 1] -= 0.25
    native_results = []
    for side in (int(value) for value in config["audit"]["native_test_sizes"]):
        native_map = resize_absolute_map(
            work_map,
            (side, side),
            source_size_from=(512, 512),
            source_size_to=(side, side),
        )
        native_grid = pixel_grid(
            1, side, side, device=device, dtype=torch.bfloat16
        )
        source = torch.cat(
            (
                native_grid[:, 0:1] / (side - 1),
                native_grid[:, 1:2] / (side - 1),
                native_grid.mean(dim=1, keepdim=True) / (side - 1),
            ),
            dim=1,
        )
        sampled = sample_by_map(source, native_map)
        clamped_map = native_map.clamp(0, side - 1)
        expected = torch.cat(
            (
                clamped_map[:, 0:1] / (side - 1),
                clamped_map[:, 1:2] / (side - 1),
                clamped_map.mean(dim=1, keepdim=True) / (side - 1),
            ),
            dim=1,
        )
        determinant = map_jacobian_determinant(native_map)
        result = {
            "size": [side, side],
            "map_dtype": str(native_map.dtype).removeprefix("torch."),
            "renderer_dtype": str(sampled.dtype).removeprefix("torch."),
            "fold_rate": float((determinant <= 0).float().mean()),
            "maximum_linear_ramp_sampling_error": float(
                (sampled - expected).abs().max()
            ),
            "fractional_coordinate_fraction": float(
                ((native_map - native_map.round()).abs() > 1.0e-4)
                .float()
                .mean()
            ),
        }
        if (
            result["map_dtype"] != "float32"
            or result["fold_rate"] != 0.0
            or result["maximum_linear_ramp_sampling_error"] > 2.0e-5
            or result["fractional_coordinate_fraction"] <= 0.0
        ):
            raise RuntimeError(f"native coordinate audit failed at {side}px")
        native_results.append(result)
        del native_map, native_grid, source, sampled, expected, determinant

    report = {
        "device": str(device),
        "autocast_dtype": "bfloat16",
        "autocast_enabled": device.type == "cuda",
        "identity_size": [512, 512],
        "coordinate_dtype": str(identity.dtype).removeprefix("torch."),
        "jacobian_dtype": str(map_jacobian_determinant(identity).dtype).removeprefix(
            "torch."
        ),
        "identity_unique_x": unique_x,
        "identity_unique_y": unique_y,
        "identity_fold_rate": identity_fold_rate,
        "native_tests": native_results,
        "passed": True,
    }
    output = resolve_path(config, config["audit"]["coordinate_precision_report"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
