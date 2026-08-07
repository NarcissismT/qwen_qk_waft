from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import load_config, resolve_path
from .data import DocumentMapDataset, tensor_to_pil
from .geometry import map_jacobian_determinant
from .losses import (
    compute_losses,
    confidence_loss,
    endpoint_error,
    image_edge_weight,
    masked_mean,
)
from .models.model import QwenQKWAFT
from .models.qwen_qk import QwenQKExtractor
from .models.stage_a import StageA, set_stage_a_trainable


def _setup(seed: int) -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    return rank, world, local_rank, device


def _loader(
    dataset: DocumentMapDataset,
    *,
    rank: int,
    world: int,
    workers: int,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = (
        DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=shuffle)
        if world > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return loader, sampler


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_confidence(config_path: str | Path) -> None:
    config = load_config(config_path)
    rank, world, _, device = _setup(int(config["seed"]))
    data_config = config["data"]
    phase = config["phases"]["confidence"]
    dataset = DocumentMapDataset(
        resolve_path(config, data_config["train_manifest"]),
        work_size=tuple(data_config["work_size"]),
        limit=phase.get("max_train_samples"),
    )
    loader, sampler = _loader(
        dataset,
        rank=rank,
        world=world,
        workers=int(data_config["num_workers"]),
        shuffle=True,
    )
    model_config = config["model"]
    stage_a = StageA(
        base_channels=int(model_config["stage_a_base_channels"]),
        max_displacement_ratio=float(model_config["stage_a_max_displacement_ratio"]),
        control_stride=int(model_config["stage_a_control_stride"]),
    )
    stage_a.load_geometry(resolve_path(config, model_config["stage_a_checkpoint"]))
    set_stage_a_trainable(stage_a, confidence_only=True)
    stage_a.to(device)
    model: nn.Module = stage_a
    if world > 1:
        model = DistributedDataParallel(stage_a, device_ids=[device.index])
    optimizer = torch.optim.AdamW(
        stage_a.confidence_head.parameters(),
        lr=float(phase["learning_rate"]),
        weight_decay=float(phase["weight_decay"]),
    )
    output = resolve_path(config, phase["checkpoint"])
    for epoch in range(int(phase["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        for step, raw_batch in enumerate(loader, 1):
            batch = _to_device(raw_batch, device)
            prediction = model(batch["warped"])
            loss = confidence_loss(
                prediction["confidence"],
                prediction["map"],
                batch["map"],
                batch["valid"],
                temperature_px=float(phase["temperature_px"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            if rank == 0 and step % int(config["train"]["log_every"]) == 0:
                print(
                    json.dumps(
                        {
                            "phase": "confidence",
                            "epoch": epoch,
                            "step": step,
                            "loss": running / step,
                        }
                    )
                )
        if rank == 0:
            _save(
                output,
                {
                    "phase": "confidence",
                    "epoch": epoch,
                    "confidence_head": stage_a.confidence_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
            )
    if world > 1:
        dist.destroy_process_group()


def _build_model(config: dict[str, Any], selection: dict[str, Any]) -> QwenQKWAFT:
    model_config = config["model"]
    model = QwenQKWAFT(
        qk_channels=int(selection["descriptor_channels"]),
        layer_count=len(selection["layers"]),
        stage_a_base_channels=int(model_config["stage_a_base_channels"]),
        stage_a_max_displacement_ratio=float(model_config["stage_a_max_displacement_ratio"]),
        stage_a_control_stride=int(model_config["stage_a_control_stride"]),
        waft_model_name=str(model_config["waft_model_name"]),
        waft_patch_size=int(model_config["waft_patch_size"]),
        timm_checkpoint=resolve_path(config, model_config["timm_checkpoint"]),
    )
    initialization = model.load_waft_pretrained(
        resolve_path(config, model_config["waft_checkpoint"])
    )
    print(json.dumps({"waft_initialization": initialization}))
    model.stage_a.load_geometry(resolve_path(config, model_config["stage_a_checkpoint"]))
    confidence = torch.load(
        resolve_path(config, config["phases"]["confidence"]["checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    model.stage_a.confidence_head.load_state_dict(confidence["confidence_head"])
    for parameter in model.stage_a.parameters():
        parameter.requires_grad = False
    return model


def _phase_trainability(model: QwenQKWAFT, phase: Mapping[str, Any]) -> None:
    local_enabled = bool(phase["local_encoder"])
    gate_enabled = bool(phase["gate"])
    for parameter in model.local_encoder.parameters():
        parameter.requires_grad = local_enabled
    for parameter in model.source_fusion.parameters():
        parameter.requires_grad = local_enabled
    for parameter in model.waft.gate.parameters():
        parameter.requires_grad = gate_enabled


def _phase_iterations(phase: Mapping[str, Any], epoch: int) -> int:
    schedule = phase.get("iteration_schedule")
    return int(schedule[epoch]) if schedule else int(phase["iterations"])


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    extractor: QwenQKExtractor,
    loader: DataLoader,
    selection: Mapping[str, Any],
    phase: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(7, device=device, dtype=torch.float64)
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        queries, keys, _, _ = extractor.selected_pair(
            tensor_to_pil(batch["warped"][0]),
            seed=0,
            step=int(selection["step"]),
            variant=str(selection["variant"]),
        )
        output = model(
            batch["warped"],
            queries,
            keys,
            iterations=int(phase["iterations"]),
            use_local_encoder=bool(phase["local_encoder"]),
            use_gate=bool(phase["gate"]),
        )
        final_error = endpoint_error(output["final_map"], batch["map"])
        prior_error = endpoint_error(output["coarse_map"], batch["map"])
        edge = image_edge_weight(batch["target"])
        epe = masked_mean(final_error, batch["valid"])
        prior_epe = masked_mean(prior_error, batch["valid"])
        line_epe = masked_mean(final_error * (1 + 4 * edge), batch["valid"])
        win = masked_mean((final_error < prior_error).float(), batch["valid"])
        determinant = map_jacobian_determinant(output["final_map"])
        fold = (determinant <= 0).float().mean()
        reconstruction = masked_mean(
            (output["rectified"] - batch["target"]).abs(), batch["valid"]
        )
        totals += torch.stack(
            (
                epe,
                prior_epe,
                line_epe,
                win,
                fold,
                reconstruction,
                epe.new_ones(()),
            )
        ).to(torch.float64)
    if dist.is_initialized():
        dist.all_reduce(totals)
    count = totals[-1].clamp_min(1)
    return {
        "epe": float(totals[0] / count),
        "prior_epe": float(totals[1] / count),
        "epe_gain": float((totals[1] - totals[0]) / count),
        "line_epe": float(totals[2] / count),
        "final_win_rate": float(totals[3] / count),
        "fold_rate": float(totals[4] / count),
        "reconstruction_l1": float(totals[5] / count),
    }


def train_waft(config_path: str | Path, phase_name: str) -> None:
    config = load_config(config_path)
    rank, world, _, device = _setup(int(config["seed"]))
    phase = config["phases"][phase_name]
    data_config = config["data"]
    selection = json.loads(
        resolve_path(config, config["probe"]["output_dir"] + "/selection.json").read_text(
            encoding="utf-8"
        )
    )
    train_dataset = DocumentMapDataset(
        resolve_path(config, data_config["train_manifest"]),
        work_size=tuple(data_config["work_size"]),
        limit=phase.get("max_train_samples"),
    )
    val_dataset = DocumentMapDataset(
        resolve_path(config, data_config["val_manifest"]),
        work_size=tuple(data_config["work_size"]),
        limit=int(config["train"]["val_samples"]),
    )
    train_loader, train_sampler = _loader(
        train_dataset,
        rank=rank,
        world=world,
        workers=int(data_config["num_workers"]),
        shuffle=True,
    )
    val_loader, _ = _loader(
        val_dataset,
        rank=rank,
        world=world,
        workers=0,
        shuffle=False,
    )
    model = _build_model(config, selection)
    resume_path = phase.get("resume")
    if resume_path:
        payload = torch.load(
            resolve_path(config, resume_path), map_location="cpu", weights_only=False
        )
        model.load_state_dict(payload["model"])
    _phase_trainability(model, phase)
    model.to(device)
    train_model: nn.Module = model
    if world > 1:
        train_model = DistributedDataParallel(model, device_ids=[device.index])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(phase["learning_rate"]),
        weight_decay=float(phase["weight_decay"]),
    )
    qwen_config = dict(config["qwen"])
    qwen_config["lora_scale"] = float(selection["lora_scale"])
    variants = ("pre", "post") if selection["variant"] == "pre_post" else (selection["variant"],)
    amp_dtype = torch.bfloat16 if config["train"]["amp_dtype"] == "bfloat16" else torch.float16
    output_dir = resolve_path(config, phase["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_value = float("inf")
    with QwenQKExtractor(
        qwen_config,
        device=device,
        layers=selection["layers"],
        steps=(selection["step"],),
        variants=variants,
    ) as extractor:
        for epoch in range(int(phase["epochs"])):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_model.train()
            running = 0.0
            iterations = _phase_iterations(phase, epoch)
            for step, raw_batch in enumerate(train_loader, 1):
                batch = _to_device(raw_batch, device)
                queries, keys, _, _ = extractor.selected_pair(
                    tensor_to_pil(batch["warped"][0]),
                    seed=int(config["qwen"].get("seed", 0)),
                    step=int(selection["step"]),
                    variant=str(selection["variant"]),
                )
                with torch.autocast("cuda", dtype=amp_dtype, enabled=bool(config["train"]["amp"])):
                    prediction = train_model(
                        batch["warped"],
                        queries,
                        keys,
                        iterations=iterations,
                        use_local_encoder=bool(phase["local_encoder"]),
                        use_gate=bool(phase["gate"]),
                    )
                    loss_weights = dict(config["loss"]["weights"])
                    if not phase["gate"]:
                        for name in ("gate", "protection", "correction"):
                            loss_weights[name] = 0.0
                    losses = compute_losses(
                        prediction,
                        batch,
                        loss_weights,
                        sequence_gamma=float(config["loss"]["sequence_gamma"]),
                        gate_temperature_px=float(config["loss"]["gate_temperature_px"]),
                        min_jacobian=float(config["loss"]["min_jacobian"]),
                    )
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(config["train"]["grad_clip"]),
                )
                optimizer.step()
                running += float(losses["total"].detach())
                if rank == 0 and step % int(config["train"]["log_every"]) == 0:
                    print(
                        json.dumps(
                            {
                                "phase": phase_name,
                                "epoch": epoch,
                                "step": step,
                                "loss": running / step,
                            }
                        )
                    )

            metrics = _evaluate(
                train_model, extractor, val_loader, selection, phase, device
            )
            if rank == 0:
                payload = {
                    "architecture": (
                        "Qwen target-Q/source-K + official pretrained WAFT "
                        "DPT-Q/K + Stage-A warm-start WAFT-A2"
                    ),
                    "phase": phase_name,
                    "epoch": epoch,
                    "selection": selection,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                }
                _save(output_dir / "latest.pt", payload)
                _save(output_dir / f"epoch_{epoch:04d}.pt", payload)
                if metrics["line_epe"] < best_value:
                    best_value = metrics["line_epe"]
                    _save(output_dir / "best.pt", payload)
                print(json.dumps({"phase": phase_name, "epoch": epoch, **metrics}))
    if world > 1:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("confidence", "b", "c", "d"), required=True)
    args = parser.parse_args()
    if args.phase == "confidence":
        train_confidence(args.config)
    else:
        train_waft(args.config, args.phase)


if __name__ == "__main__":
    main()
