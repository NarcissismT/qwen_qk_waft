from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from qwen_qk_waft.config import load_config, resolve_path
from qwen_qk_waft.data import DocumentMapDataset, tensor_to_pil
from qwen_qk_waft.losses import compute_losses
from qwen_qk_waft.models.qwen_qk import QwenQKExtractor
from qwen_qk_waft.train import (
    _abort_on_numerical_failure,
    _batch_ids,
    _build_model,
    _build_optimizer_and_scheduler,
    _loader,
    _loss_values,
    _nonfinite_loss_terms,
    _phase_trainability,
    _tensor_diagnostics,
    _to_device,
    _trainable_parameters,
)


def _iteration_levels(phase: dict[str, Any]) -> list[int]:
    schedule = phase.get("iteration_schedule", [phase["iterations"]])
    return list(dict.fromkeys(int(value) for value in schedule))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    phase = config["phases"]["b"]
    selection = json.loads(
        resolve_path(
            config, config["probe"]["output_dir"] + "/selection.json"
        ).read_text(encoding="utf-8")
    )
    steps_per_iteration = int(
        config["audit"]["phase_b_stability_steps_per_iteration"]
    )
    iteration_levels = _iteration_levels(phase)
    sample_limit = world * steps_per_iteration * len(iteration_levels)
    dataset = DocumentMapDataset(
        resolve_path(config, config["data"]["train_manifest"]),
        work_size=tuple(config["data"]["work_size"]),
        limit=sample_limit,
    )
    loader, _ = _loader(
        dataset,
        rank=rank,
        world=world,
        workers=int(config["data"]["num_workers"]),
        shuffle=False,
    )
    model = _build_model(config, selection)
    _phase_trainability(model, phase)
    model.to(device).train()
    train_model: torch.nn.Module = model
    if world > 1:
        train_model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer, scheduler = _build_optimizer_and_scheduler(
        model,
        phase,
        config["train"],
        steps_per_epoch=max(len(loader), 1),
    )
    parameters = _trainable_parameters(model)
    amp_dtype = (
        torch.bfloat16
        if config["train"]["amp_dtype"] == "bfloat16"
        else torch.float16
    )
    qwen_config = dict(config["qwen"])
    qwen_config["lora_scale"] = float(selection["lora_scale"])
    variants = (
        ("pre", "post")
        if selection["variant"] == "pre_post"
        else (selection["variant"],)
    )
    report_path = resolve_path(
        config, config["audit"]["phase_b_stability_report"]
    )
    records: list[dict[str, Any]] = []
    loader_iterator = iter(loader)
    global_step = 0

    with QwenQKExtractor(
        qwen_config,
        device=device,
        layers=selection["layers"],
        steps=(selection["step"],),
        variants=variants,
    ) as extractor:
        for iterations in iteration_levels:
            for _ in range(steps_per_iteration):
                raw_batch = next(loader_iterator)
                batch = _to_device(raw_batch, device)
                queries, keys, _, _ = extractor.selected_pair(
                    tensor_to_pil(batch["warped"][0]),
                    seed=int(config["qwen"].get("seed", 0)),
                    step=int(selection["step"]),
                    variant=str(selection["variant"]),
                )
                with torch.autocast(
                    "cuda",
                    dtype=amp_dtype,
                    enabled=bool(config["train"]["amp"]),
                ):
                    prediction = train_model(
                        batch["warped"],
                        queries,
                        keys,
                        iterations=iterations,
                        use_local_encoder=False,
                        use_gate=False,
                    )
                    loss_weights = dict(config["loss"]["weights"])
                    for name in ("gate", "protection", "correction"):
                        loss_weights[name] = 0.0
                    losses = compute_losses(
                        prediction,
                        batch,
                        loss_weights,
                        sequence_gamma=float(config["loss"]["sequence_gamma"]),
                        gate_temperature_px=float(
                            config["loss"]["gate_temperature_px"]
                        ),
                        required_improvement_px=float(
                            config["loss"]["required_improvement_px"]
                        ),
                        min_jacobian=float(config["loss"]["min_jacobian"]),
                    )
                optimizer.zero_grad(set_to_none=True)
                nonfinite_terms = _nonfinite_loss_terms(losses)
                failure = None
                if nonfinite_terms:
                    failure = {
                        "rank": rank,
                        "stage": "preflight_forward_loss",
                        "global_step": global_step,
                        "iterations": iterations,
                        "sample_ids": _batch_ids(batch),
                        "nonfinite_loss_terms": nonfinite_terms,
                        "losses": _loss_values(losses),
                        "final_map": _tensor_diagnostics(prediction["final_map"]),
                    }
                _abort_on_numerical_failure(
                    failure,
                    rank=rank,
                    world=world,
                    device=device,
                    output_dir=report_path.parent,
                )
                losses["total"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, float(config["train"]["grad_clip"])
                )
                failure = None
                if not bool(torch.isfinite(grad_norm)):
                    failure = {
                        "rank": rank,
                        "stage": "preflight_backward_gradient",
                        "global_step": global_step,
                        "iterations": iterations,
                        "sample_ids": _batch_ids(batch),
                        "gradient_norm": _loss_values(
                            {"gradient_norm": grad_norm}
                        )["gradient_norm"],
                        "losses": _loss_values(losses),
                    }
                _abort_on_numerical_failure(
                    failure,
                    rank=rank,
                    world=world,
                    device=device,
                    output_dir=report_path.parent,
                )
                optimizer.step()
                scheduler.step()
                global_step += 1

                total = losses["total"].detach().double()
                maximum_grad_norm = grad_norm.detach().float()
                if world > 1:
                    dist.all_reduce(total)
                    dist.all_reduce(maximum_grad_norm, op=dist.ReduceOp.MAX)
                if rank == 0:
                    records.append(
                        {
                            "step": global_step,
                            "iterations": iterations,
                            "mean_loss": float(total / world),
                            "maximum_gradient_norm": float(maximum_grad_norm),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "loss_terms": _loss_values(losses),
                            "final_map": _tensor_diagnostics(
                                prediction["final_map"]
                            ),
                        }
                    )

    if rank == 0:
        report = {
            "passed": True,
            "world_size": world,
            "iteration_levels": iteration_levels,
            "steps_per_iteration": steps_per_iteration,
            "records": records,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    if world > 1:
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
