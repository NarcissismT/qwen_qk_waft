from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .config import load_config, resolve_path
from .data import DocumentMapDataset
from .geometry import sample_by_map


def run_audit(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    audit_config = config["audit"]
    dataset = DocumentMapDataset(
        resolve_path(config, config["data"]["val_manifest"]),
        work_size=tuple(config["data"]["work_size"]),
        limit=int(audit_config["samples"]),
    )
    l1_sum = 0.0
    mse_sum = 0.0
    valid_sum = 0.0
    valid_pixel_sum = 0.0
    pixel_sum = 0.0
    for sample in dataset:
        reconstruction = sample_by_map(
            sample["warped"].unsqueeze(0), sample["map"].unsqueeze(0)
        )[0]
        valid = sample["valid"].expand_as(reconstruction)
        difference = reconstruction - sample["target"]
        l1_sum += float(difference.abs()[valid].sum())
        mse_sum += float(difference.square()[valid].sum())
        valid_sum += float(valid.sum())
        valid_pixel_sum += float(sample["valid"].sum())
        pixel_sum += float(sample["valid"].numel())
    l1 = l1_sum / valid_sum
    mse = mse_sum / valid_sum
    valid_fraction = valid_pixel_sum / pixel_sum
    max_l1 = float(audit_config["max_gt_warp_l1"])
    min_psnr = float(audit_config["min_gt_warp_psnr"])
    min_valid_fraction = float(audit_config["min_valid_fraction"])
    passed = l1 <= max_l1 and -10.0 * math.log10(max(mse, 1.0e-12)) >= min_psnr
    passed = passed and valid_fraction >= min_valid_fraction
    result: dict[str, float | int | bool] = {
        "samples": len(dataset),
        "gt_warp_l1": l1,
        "gt_warp_psnr": -10.0 * math.log10(max(mse, 1.0e-12)),
        "valid_fraction": valid_fraction,
        "thresholds": {
            "max_gt_warp_l1": max_l1,
            "min_gt_warp_psnr": min_psnr,
            "min_valid_fraction": min_valid_fraction,
        },
        "passed": passed,
    }
    output = resolve_path(config, audit_config["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError(f"coordinate audit failed; see {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_audit(args.config)


if __name__ == "__main__":
    main()
