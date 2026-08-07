from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .config import load_config, resolve_path
from .data import DocumentMapDataset
from .geometry import sample_by_map


def run_audit(config_path: str | Path) -> dict[str, float | int]:
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
    for sample in dataset:
        reconstruction = sample_by_map(
            sample["warped"].unsqueeze(0), sample["map"].unsqueeze(0)
        )[0]
        valid = sample["valid"].expand_as(reconstruction)
        difference = reconstruction - sample["target"]
        l1_sum += float(difference.abs()[valid].sum())
        mse_sum += float(difference.square()[valid].sum())
        valid_sum += float(valid.sum())
    l1 = l1_sum / valid_sum
    mse = mse_sum / valid_sum
    result: dict[str, float | int] = {
        "samples": len(dataset),
        "gt_warp_l1": l1,
        "gt_warp_psnr": -10.0 * math.log10(max(mse, 1.0e-12)),
    }
    output = resolve_path(config, audit_config["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_audit(args.config)


if __name__ == "__main__":
    main()

