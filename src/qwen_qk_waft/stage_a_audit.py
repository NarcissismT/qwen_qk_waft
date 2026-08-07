from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from .config import load_config, resolve_path
from .data import DocumentMapDataset
from .geometry import displacement_to_map, pixel_grid
from .losses import endpoint_error, masked_mean
from .models.stage_a import StageA


def _reference_prior(project_root: Path, model_config: dict[str, Any]) -> torch.nn.Module:
    source_root = project_root / "src"
    sys.path.insert(0, str(source_root))
    module = importlib.import_module("diffusion2raft.models.prior")
    return module.DocumentGeometryPrior(
        base_channels=int(model_config["stage_a_base_channels"]),
        max_displacement_ratio=float(model_config["stage_a_max_displacement_ratio"]),
        control_stride=int(model_config["stage_a_control_stride"]),
        global_pose_enabled=False,
    )


def _prior_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload))
    result = {}
    for original_key, value in state.items():
        key = original_key
        while key.startswith(("module.", "model.")):
            key = key.split(".", 1)[1]
        if key.startswith("prior."):
            result[key.removeprefix("prior.")] = value
    return result


@torch.no_grad()
def run_stage_a_audit(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    model_config = config["model"]
    audit_config = config["audit"]
    checkpoint = resolve_path(config, model_config["stage_a_checkpoint"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_a = StageA(
        base_channels=int(model_config["stage_a_base_channels"]),
        max_displacement_ratio=float(model_config["stage_a_max_displacement_ratio"]),
        control_stride=int(model_config["stage_a_control_stride"]),
    )
    load_report = stage_a.load_geometry(checkpoint)
    reference = _reference_prior(
        resolve_path(config, model_config["stage_a_reference_project"]),
        model_config,
    )
    reference.load_state_dict(_prior_state(checkpoint), strict=True)
    stage_a.to(device).eval()
    reference.to(device).eval()
    dataset = DocumentMapDataset(
        resolve_path(config, config["data"]["val_manifest"]),
        work_size=tuple(config["data"]["work_size"]),
        limit=int(audit_config["stage_a_samples"]),
    )
    identity_epe = 0.0
    loaded_epe = 0.0
    reference_epe = 0.0
    maximum_difference = 0.0
    for sample in dataset:
        warped = sample["warped"].unsqueeze(0).to(device)
        target_map = sample["map"].unsqueeze(0).to(device)
        valid = sample["valid"].unsqueeze(0).to(device)
        loaded_map = stage_a(warped)["map"]
        reference_map = displacement_to_map(reference(warped))
        identity_map = pixel_grid(
            1,
            warped.shape[-2],
            warped.shape[-1],
            device=device,
            dtype=warped.dtype,
        )
        identity_epe += float(masked_mean(endpoint_error(identity_map, target_map), valid))
        loaded_epe += float(masked_mean(endpoint_error(loaded_map, target_map), valid))
        reference_epe += float(masked_mean(endpoint_error(reference_map, target_map), valid))
        maximum_difference = max(
            maximum_difference,
            float((loaded_map - reference_map).abs().max()),
        )
    count = max(len(dataset), 1)
    tolerance = float(audit_config["max_stage_a_parity_error"])
    passed = maximum_difference <= tolerance
    result: dict[str, object] = {
        "samples": len(dataset),
        "device": str(device),
        "load": load_report,
        "identity_epe": identity_epe / count,
        "loaded_stage_a_epe": loaded_epe / count,
        "reference_stage_a_epe": reference_epe / count,
        "maximum_map_difference": maximum_difference,
        "maximum_allowed_difference": tolerance,
        "passed": passed,
    }
    output = resolve_path(config, audit_config["stage_a_report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError(f"Stage-A parity audit failed; see {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_stage_a_audit(args.config)


if __name__ == "__main__":
    main()
