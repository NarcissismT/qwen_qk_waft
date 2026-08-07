from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dpt import OfficialQKDPT
from .waft import WAFTRefiner


def _component(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    start = prefix + "."
    return {
        key[len(start):]: value
        for key, value in state.items()
        if key.startswith(start)
    }


def load_official_waft_initialization(
    checkpoint_path: str | Path,
    refiner: WAFTRefiner,
    dpt_q: OfficialQKDPT,
    dpt_k: OfficialQKDPT,
) -> dict[str, int | str]:
    """Load every reusable official WAFT component with strict key matching."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    modules: tuple[tuple[str, nn.Module, str], ...] = (
        ("refine_net", refiner.refine_net, "refine_net"),
        ("hidden_conv", refiner.hidden_conv, "hidden_conv"),
        ("warp_linear", refiner.warp_linear, "warp_linear"),
        ("refine_transform", refiner.refine_transform, "refine_transform"),
        ("upsample_weight", refiner.upsample_weight, "upsample_weight"),
        ("flow_head", refiner.flow_head, "flow_head"),
    )
    report: dict[str, int | str] = {"checkpoint": str(Path(checkpoint_path))}
    for name, module, prefix in modules:
        values = _component(state, prefix)
        module.load_state_dict(values, strict=True)
        report[name] = len(values)

    dpt_values = _component(state, "refine_net.dpt_head")
    dpt_q.dpt_head.load_state_dict(dpt_values, strict=True)
    dpt_k.dpt_head.load_state_dict(dpt_values, strict=True)
    report["dpt_q"] = len(dpt_values)
    report["dpt_k"] = len(dpt_values)
    return report
