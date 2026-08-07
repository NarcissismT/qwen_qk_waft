from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from qwen_qk_waft.config import load_config, resolve_path
from qwen_qk_waft.geometry import pixel_grid
from qwen_qk_waft.losses import compute_losses
from qwen_qk_waft.models.model import QwenQKWAFT
from qwen_qk_waft.train import _evaluate, _phase_trainability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model_config = config["model"]
    model = QwenQKWAFT(
        qk_channels=32,
        layer_count=4,
        stage_a_base_channels=8,
        stage_a_max_displacement_ratio=float(
            model_config["stage_a_max_displacement_ratio"]
        ),
        stage_a_control_stride=int(model_config["stage_a_control_stride"]),
        waft_model_name=str(model_config["waft_model_name"]),
        waft_patch_size=int(model_config["waft_patch_size"]),
        timm_checkpoint=resolve_path(config, model_config["timm_checkpoint"]),
    )
    model.load_waft_pretrained(resolve_path(config, model_config["waft_checkpoint"]))
    for parameter in model.stage_a.parameters():
        parameter.requires_grad = False
    _phase_trainability(model, {"local_encoder": False, "gate": False})
    model.to(device).train()
    wrapped = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-5,
    )
    amp_dtype = torch.bfloat16
    for step in range(2):
        generator = torch.Generator(device=device).manual_seed(1000 + rank + step)
        warped = torch.rand(1, 3, 64, 64, generator=generator, device=device)
        target_q = torch.rand(
            1, 4, 32, 4, 4, generator=generator, device=device, dtype=amp_dtype
        )
        source_k = torch.rand(
            1, 4, 32, 4, 4, generator=generator, device=device, dtype=amp_dtype
        )
        target_map = pixel_grid(1, 64, 64, device=device, dtype=torch.float32)
        batch = {
            "warped": warped,
            "map": target_map,
            "valid": torch.ones(1, 1, 64, 64, device=device, dtype=torch.bool),
            "target": warped,
        }
        with torch.autocast("cuda", dtype=amp_dtype):
            output = wrapped(
                warped,
                target_q,
                source_k,
                iterations=1,
                use_local_encoder=False,
                use_gate=False,
            )
            losses = compute_losses(
                output,
                batch,
                config["loss"]["weights"],
                sequence_gamma=float(config["loss"]["sequence_gamma"]),
                gate_temperature_px=float(config["loss"]["gate_temperature_px"]),
                required_improvement_px=float(
                    config["loss"]["required_improvement_px"]
                ),
                min_jacobian=float(config["loss"]["min_jacobian"]),
            )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
    class Extractor:
        def selected_pair(self, *_args, **_kwargs):
            return target_q, source_k, (4, 4), (4, 4)

    metrics = _evaluate(
        wrapped,
        Extractor(),
        [batch],
        {"step": 0, "variant": "pre"},
        {"iterations": 1, "local_encoder": False, "gate": False},
        device,
        amp_enabled=True,
        amp_dtype=amp_dtype,
        gate_temperature_px=float(config["loss"]["gate_temperature_px"]),
        selection_weights=config["train"]["selection_weights"],
    )
    dist.barrier()
    if rank == 0:
        report = {
            "world_size": dist.get_world_size(),
            "steps": 2,
            "dtype": "bfloat16",
            "find_unused_parameters": False,
            "loss": float(losses["total"].detach()),
            "validation_selection_score": metrics["selection_score"],
            "passed": True,
        }
        output = resolve_path(config, config["audit"]["ddp_report"])
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
