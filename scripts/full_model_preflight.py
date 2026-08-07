from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen_qk_waft.config import load_config, resolve_path
from qwen_qk_waft.data import DocumentMapDataset, tensor_to_pil
from qwen_qk_waft.geometry import map_jacobian_determinant
from qwen_qk_waft.models.model import QwenQKWAFT
from qwen_qk_waft.models.qwen_qk import QwenQKExtractor


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda")
    sample = DocumentMapDataset(
        resolve_path(config, config["data"]["val_manifest"]),
        work_size=tuple(config["data"]["work_size"]),
        limit=1,
    )[0]
    layers = tuple(int(value) for value in config["audit"]["full_model_layers"])
    qwen_config = dict(config["qwen"])
    qwen_config["lora_scale"] = 1.0
    projection_parity: dict[str, dict[str, float]] = {}
    with QwenQKExtractor(
        qwen_config,
        device=device,
        layers=layers,
        steps=(0,),
        variants=("pre",),
    ) as extractor:
        handles = []
        attention = extractor.pipeline.transformer.transformer_blocks[layers[0]].attn

        def parity_hook(name: str):
            def hook(module, inputs, output):
                if name in projection_parity:
                    return
                value = inputs[0]
                with torch.autocast(device_type=value.device.type, enabled=False):
                    reference = F.linear(
                        value.to(module.weight.dtype), module.weight, module.bias
                    )
                difference = output.float() - reference.float()
                projection_parity[name] = {
                    "maximum_absolute_difference": float(difference.abs().max()),
                    "mean_absolute_difference": float(difference.abs().mean()),
                }

            return hook

        handles.append(attention.to_q.register_forward_hook(parity_hook("target_q")))
        handles.append(attention.to_k.register_forward_hook(parity_hook("source_k")))
        queries, keys, target_grid, source_grid = extractor.selected_pair(
            tensor_to_pil(sample["warped"]),
            seed=int(config["qwen"].get("seed", 0)),
            step=0,
            variant="pre",
        )
        for handle in handles:
            handle.remove()
        lora_report = dict(extractor.lora_report)
    maximum_parity_error = max(
        value["maximum_absolute_difference"] for value in projection_parity.values()
    )
    if maximum_parity_error > float(
        config["audit"]["max_qk_projection_parity_error"]
    ):
        raise RuntimeError(
            f"same-input Q/K fused projection parity failed: {maximum_parity_error}"
        )
    gc.collect()
    torch.cuda.empty_cache()

    model_config = config["model"]
    model = QwenQKWAFT(
        qk_channels=int(queries.shape[2]),
        layer_count=len(layers),
        stage_a_base_channels=int(model_config["stage_a_base_channels"]),
        stage_a_max_displacement_ratio=float(
            model_config["stage_a_max_displacement_ratio"]
        ),
        stage_a_control_stride=int(model_config["stage_a_control_stride"]),
        waft_model_name=str(model_config["waft_model_name"]),
        waft_patch_size=int(model_config["waft_patch_size"]),
        timm_checkpoint=resolve_path(config, model_config["timm_checkpoint"]),
    )
    waft_report = model.load_waft_pretrained(
        resolve_path(config, model_config["waft_checkpoint"])
    )
    stage_a_report = model.stage_a.load_geometry(
        resolve_path(config, model_config["stage_a_checkpoint"])
    )
    model.to(device).eval()
    warped = sample["warped"].unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            warped,
            queries,
            keys,
            iterations=1,
            use_local_encoder=True,
            use_gate=True,
        )
    coordinate_tensors = [
        output["coarse_map"],
        output["coarse_confidence"],
        output["final_map"],
        output["jacobian_determinant"],
        *output["maps"],
        *output["displacements"],
    ]
    if any(value.dtype != torch.float32 for value in coordinate_tensors):
        raise RuntimeError("full-model preflight found a non-FP32 coordinate tensor")
    determinant = map_jacobian_determinant(output["final_map"])
    report = {
        "device": str(device),
        "autocast_dtype": "bfloat16",
        "layers": list(layers),
        "target_token_grid": list(target_grid),
        "source_token_grid": list(source_grid),
        "qk_shape": list(queries.shape),
        "qk_dtype": str(queries.dtype).removeprefix("torch."),
        "feature_dtype": str(output["target_feature"].dtype).removeprefix("torch."),
        "coarse_map_dtype": str(output["coarse_map"].dtype).removeprefix("torch."),
        "final_map_dtype": str(output["final_map"].dtype).removeprefix("torch."),
        "jacobian_dtype": str(determinant.dtype).removeprefix("torch."),
        "rectified_dtype": str(output["rectified"].dtype).removeprefix("torch."),
        "fold_rate": float((determinant <= 0).float().mean()),
        "lora": lora_report,
        "same_input_qk_projection_parity": projection_parity,
        "maximum_qk_projection_parity_error": maximum_parity_error,
        "stage_a": stage_a_report,
        "waft": waft_report,
        "path": "Qwen target-Q/source-K -> DPT-Q/K -> Stage-A warm start -> WAFT-A2 -> grid_sample",
        "passed": True,
    }
    output_path = resolve_path(config, config["audit"]["full_model_report"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
