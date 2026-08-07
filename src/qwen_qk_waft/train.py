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
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import load_config, resolve_path
from .data import DocumentMapDataset, tensor_to_pil
from .geometry import map_jacobian_determinant, sample_by_map, valid_map_mask
from .losses import (
    compute_losses,
    confidence_loss,
    endpoint_error,
    masked_mean,
)
from .metrics import (
    calibration_metrics,
    document_region_masks,
    gate_histogram,
    map_curvature_error,
    masked_minimum,
    masked_quantile,
    reconstruction_psnr,
    reconstruction_ssim,
    text_line_fit_residual,
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
    stage_a_report = stage_a.load_geometry(
        resolve_path(config, model_config["stage_a_checkpoint"])
    )
    if rank == 0:
        print(json.dumps({"stage_a_initialization": stage_a_report}))
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
    stage_a_initialization = model.stage_a.load_geometry(
        resolve_path(config, model_config["stage_a_checkpoint"])
    )
    confidence = torch.load(
        resolve_path(config, config["phases"]["confidence"]["checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    model.stage_a.confidence_head.load_state_dict(confidence["confidence_head"])
    for parameter in model.stage_a.parameters():
        parameter.requires_grad = False
    model.initialization_report = {
        "waft": initialization,
        "stage_a": stage_a_initialization,
    }
    print(json.dumps({"stage_a_initialization": stage_a_initialization}))
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
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    gate_temperature_px: float,
    selection_weights: Mapping[str, float],
    geometry_criteria: Mapping[str, float] | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, Tensor] = {}
    jacobian_minimum = torch.full((), torch.inf, device=device, dtype=torch.float32)

    def add(name: str, value: Tensor) -> None:
        totals[name] = totals.get(
            name, torch.zeros((), device=device, dtype=torch.float64)
        ) + value.detach().to(torch.float64)

    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        queries, keys, _, _ = extractor.selected_pair(
            tensor_to_pil(batch["warped"][0]),
            seed=0,
            step=int(selection["step"]),
            variant=str(selection["variant"]),
        )
        with torch.autocast(
            "cuda", dtype=amp_dtype, enabled=amp_enabled
        ):
            output = model(
                batch["warped"],
                queries,
                keys,
                iterations=int(phase["iterations"]),
                use_local_encoder=bool(phase["local_encoder"]),
                use_gate=bool(phase["gate"]),
            )
        target_map = batch["map"].float()
        final_map = output["final_map"].float()
        coarse_map = output["coarse_map"].float()
        valid = batch["valid"]
        final_error = endpoint_error(final_map, target_map)
        prior_error = endpoint_error(coarse_map, target_map)
        regions = document_region_masks(batch["target"], valid)
        line_available = batch.get("line_mask_available")
        has_annotated_lines = bool(
            line_available is not None and line_available.bool().all()
        )
        if has_annotated_lines:
            regions["line"] = valid & batch["line_mask"].bool()

        add("epe", masked_mean(final_error, valid))
        add("epe_p95", masked_quantile(final_error, valid, 0.95))
        add("prior_epe", masked_mean(prior_error, valid))
        add("line_epe", masked_mean(final_error, regions["line"]))
        add("prior_line_epe", masked_mean(prior_error, regions["line"]))
        add("edge_epe", masked_mean(final_error, regions["page_edge"]))
        add("corner_epe", masked_mean(final_error, regions["corner"]))
        add(
            "line_straightness_error",
            map_curvature_error(final_map, target_map, regions["line"]),
        )
        add(
            "prior_line_straightness_error",
            map_curvature_error(coarse_map, target_map, regions["line"]),
        )
        add("final_win_rate", masked_mean((final_error < prior_error).float(), valid))
        residual_error = endpoint_error(
            final_map - coarse_map, target_map - coarse_map
        )
        add("residual_epe", masked_mean(residual_error, valid))

        determinant = map_jacobian_determinant(final_map)
        prior_determinant = map_jacobian_determinant(coarse_map)
        fold_valid = valid[:, :, :-1, :-1]
        add("fold_rate", masked_mean((determinant <= 0).float().unsqueeze(1), fold_valid))
        add(
            "prior_fold_rate",
            masked_mean((prior_determinant <= 0).float().unsqueeze(1), fold_valid),
        )
        jacobian_minimum = torch.minimum(
            jacobian_minimum,
            masked_minimum(determinant.unsqueeze(1), fold_valid).float(),
        )
        predicted_valid = valid_map_mask(final_map, batch["warped"].shape[-2:])
        prior_predicted_valid = valid_map_mask(
            coarse_map, batch["warped"].shape[-2:]
        )
        add("invalid_rate", masked_mean((~predicted_valid).float(), valid))
        add("prior_invalid_rate", masked_mean((~prior_predicted_valid).float(), valid))

        rectified = output["rectified"].float()
        target_image = batch["target"].float()
        add("reconstruction_l1", masked_mean((rectified - target_image).abs(), valid))
        add("reconstruction_psnr", reconstruction_psnr(rectified, target_image, valid))
        add("reconstruction_ssim", reconstruction_ssim(rectified, target_image, valid))

        line_instances_available = batch.get("line_instances_available")
        has_line_instances = bool(
            line_instances_available is not None
            and line_instances_available.bool().all()
        )
        add(
            "annotated_line_samples",
            final_error.new_tensor(float(has_line_instances)),
        )
        if has_line_instances:
            prior_rectified = sample_by_map(batch["warped"], coarse_map)
            add(
                "annotated_line_fit_residual",
                text_line_fit_residual(
                    rectified, batch["line_instances"], valid
                ),
            )
            add(
                "prior_annotated_line_fit_residual",
                text_line_fit_residual(
                    prior_rectified, batch["line_instances"], valid
                ),
            )

        confidence_target = torch.exp(-prior_error / gate_temperature_px)
        confidence = output["coarse_confidence"].float()
        confidence_brier, confidence_ece = calibration_metrics(
            confidence, confidence_target, valid
        )
        add("confidence_brier", confidence_brier)
        add("confidence_ece", confidence_ece)
        add("high_confidence_epe", masked_mean(final_error, valid & (confidence >= 0.5)))
        add("low_confidence_epe", masked_mean(final_error, valid & (confidence < 0.5)))
        high_confidence = valid & (confidence >= 0.5)
        damage_margin = float(
            (geometry_criteria or {}).get("high_confidence_damage_margin_px", 0.5)
        )
        add(
            "high_confidence_damage_rate",
            masked_mean(
                (final_error > prior_error + damage_margin).float(), high_confidence
            ),
        )

        gate_target = (prior_error / gate_temperature_px).clamp(0, 1)
        for gate_index, iteration_gate in enumerate(output["gates"], 1):
            gate = F.interpolate(
                iteration_gate.float(),
                target_map.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            gate_brier, gate_ece = calibration_metrics(gate, gate_target, valid)
            add(f"iteration_{gate_index}_gate_brier", gate_brier)
            add(f"iteration_{gate_index}_gate_ece", gate_ece)
            for bin_index, value in enumerate(gate_histogram(gate, valid)):
                add(
                    f"iteration_{gate_index}_gate_histogram_{bin_index}", value
                )
        gate_brier = calibration_metrics(gate, gate_target, valid)
        add("gate_brier", gate_brier[0])
        add("gate_ece", gate_brier[1])
        for index, value in enumerate(gate_histogram(gate, valid)):
            add(f"gate_histogram_{index}", value)

        previous_map = coarse_map
        for index, iteration_map in enumerate(output["maps"], 1):
            iteration_map = iteration_map.float()
            add(
                f"iteration_{index}_epe",
                masked_mean(endpoint_error(iteration_map, target_map), valid),
            )
            add(
                f"iteration_{index}_update_px",
                masked_mean(endpoint_error(iteration_map, previous_map), valid),
            )
            previous_map = iteration_map
        add("samples", final_error.new_ones(()))
    if dist.is_initialized():
        for value in totals.values():
            dist.all_reduce(value)
        dist.all_reduce(jacobian_minimum, op=dist.ReduceOp.MIN)
    count = totals.pop("samples").clamp_min(1)
    result = {name: float(value / count) for name, value in totals.items()}
    annotated_fraction = result.pop("annotated_line_samples", 0.0)
    result["line_annotation_fraction"] = annotated_fraction
    for name in (
        "annotated_line_fit_residual",
        "prior_annotated_line_fit_residual",
    ):
        result[name] = (
            result.get(name, 0.0) / annotated_fraction
            if annotated_fraction > 0
            else 0.0
        )
    result["jacobian_min"] = float(jacobian_minimum)
    result["epe_gain"] = result["prior_epe"] - result["epe"]
    result["line_epe_gain"] = result["prior_line_epe"] - result["line_epe"]
    result["line_straightness_gain"] = (
        result["prior_line_straightness_error"]
        - result["line_straightness_error"]
    )
    result["annotated_line_fit_gain"] = (
        result["prior_annotated_line_fit_residual"]
        - result["annotated_line_fit_residual"]
    )
    result["line_geometry_gain"] = (
        result["annotated_line_fit_gain"]
        if annotated_fraction > 0
        else result["line_straightness_gain"]
    )
    result["line_geometry_uses_annotations"] = float(annotated_fraction > 0)
    result["selection_score"] = sum(
        float(weight) * result[name] for name, weight in selection_weights.items()
    )
    criteria = dict(geometry_criteria or {})
    result["fold_rate_increase"] = result["fold_rate"] - result["prior_fold_rate"]
    result["invalid_rate_increase"] = (
        result["invalid_rate"] - result["prior_invalid_rate"]
    )
    result["meets_minimum_geometry_criteria"] = float(
        result["epe_gain"] > float(criteria.get("min_epe_gain_px", 0.0))
        and result["final_win_rate"]
        > float(criteria.get("min_final_win_rate", 0.5))
        and result["line_epe_gain"] > float(criteria.get("min_line_epe_gain_px", 0.0))
        and result["line_geometry_gain"]
        > float(criteria.get("min_line_straightness_gain", 0.0))
        and result["fold_rate_increase"]
        <= float(criteria.get("max_fold_rate_increase", 0.0))
        and result["invalid_rate_increase"]
        <= float(criteria.get("max_invalid_rate_increase", 0.0))
        and result["high_confidence_damage_rate"]
        <= float(criteria.get("max_high_confidence_damage_rate", 0.05))
    )
    return result


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
    best_saved = False
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
                        required_improvement_px=float(
                            config["loss"]["required_improvement_px"]
                        ),
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
                train_model,
                extractor,
                val_loader,
                selection,
                phase,
                device,
                amp_enabled=bool(config["train"]["amp"]),
                amp_dtype=amp_dtype,
                gate_temperature_px=float(config["loss"]["gate_temperature_px"]),
                selection_weights=config["train"]["selection_weights"],
                geometry_criteria=config["train"].get("geometry_criteria", {}),
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
                    "initialization": model.initialization_report,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                }
                _save(output_dir / "latest.pt", payload)
                _save(output_dir / f"epoch_{epoch:04d}.pt", payload)
                if (
                    metrics["meets_minimum_geometry_criteria"] > 0
                    and metrics["selection_score"] < best_value
                ):
                    best_value = metrics["selection_score"]
                    best_saved = True
                    _save(output_dir / "best.pt", payload)
                print(json.dumps({"phase": phase_name, "epoch": epoch, **metrics}))
    best_flag = torch.tensor(
        int(best_saved) if rank == 0 else 0, device=device, dtype=torch.int32
    )
    if world > 1:
        dist.broadcast(best_flag, src=0)
        dist.destroy_process_group()
    if not bool(best_flag.item()):
        raise RuntimeError(
            f"Phase {phase_name} produced no checkpoint that satisfies "
            "train.geometry_criteria; inspect latest.pt metrics"
        )


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
