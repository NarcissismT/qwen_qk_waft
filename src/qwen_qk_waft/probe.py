from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .config import load_config, resolve_path
from .data import DocumentMapDataset, tensor_to_pil
from .models.qwen_qk import FeaturePacket, QwenQKExtractor


def _distributed() -> tuple[int, int, int]:
    import os

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    return rank, world, local_rank


def _probe_regions(image: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Deterministic image regions used by the Q/K correspondence audit."""

    gray = image.mean(dim=0, keepdim=True)
    dx = F.pad((gray[:, :, 1:] - gray[:, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[:, 1:, :] - gray[:, :-1, :]).abs(), (0, 0, 0, 1))
    texture = F.max_pool2d(
        (dx + dy).unsqueeze(0), 5, stride=1, padding=2
    )[0] > 0.08
    _, height, width = valid.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=valid.device),
        torch.arange(width, device=valid.device),
        indexing="ij",
    )
    border_width = max(2, round(min(height, width) * 0.08))
    corner_width = max(2, round(min(height, width) * 0.18))
    border = (
        (x < border_width)
        | (x >= width - border_width)
        | (y < border_width)
        | (y >= height - border_width)
    ).unsqueeze(0)
    corners = (
        ((x < corner_width) | (x >= width - corner_width))
        & ((y < corner_width) | (y >= height - corner_width))
    ).unsqueeze(0)
    interior = ~border
    return {
        "all": valid,
        "text_line": valid & texture,
        "page_edge": valid & border,
        "corners": valid & corners,
        "interior_texture": valid & interior & texture,
        "blank_background": valid & interior & ~texture & (gray > 0.7),
    }


def _match_metrics(
    packet: FeaturePacket,
    target_map: Tensor,
    regions: dict[str, Tensor],
    *,
    max_points: int,
) -> dict[str, float]:
    target = F.normalize(packet.target[0].float(), dim=-1)
    source = F.normalize(packet.source[0].float(), dim=-1)
    target_h, target_w = packet.target_grid
    source_h, source_w = packet.source_grid
    sampled_map = F.interpolate(
        target_map.unsqueeze(0),
        (target_h, target_w),
        mode="bilinear",
        align_corners=True,
    )[0]
    target_scale_x = (target_map.shape[-1] - 1) / max(target_w - 1, 1)
    target_scale_y = (target_map.shape[-2] - 1) / max(target_h - 1, 1)
    source_scale_x = (target_map.shape[-1] - 1) / max(source_w - 1, 1)
    source_scale_y = (target_map.shape[-2] - 1) / max(source_h - 1, 1)

    def evaluate(indices: Tensor) -> dict[str, float]:
        if indices.numel() > max_points:
            positions = torch.linspace(
                0, indices.numel() - 1, max_points, device=indices.device
            ).long()
            indices = indices[positions]
        if not indices.numel():
            return {
                "count": 0.0,
                "epe": 0.0,
                "pck1": 0.0,
                "pck3": 0.0,
                "pck5": 0.0,
                "margin": 0.0,
                "cycle_epe": 0.0,
            }
        truth = sampled_map.flatten(1).transpose(0, 1)[indices]
        similarity = target[indices] @ source.transpose(0, 1)
        predicted_index = similarity.argmax(dim=1)
        predicted = torch.stack(
            (
                (predicted_index % source_w).float() * source_scale_x,
                torch.div(predicted_index, source_w, rounding_mode="floor").float()
                * source_scale_y,
            ),
            dim=1,
        )
        error = torch.linalg.vector_norm(predicted - truth, dim=1)
        true_x = torch.round(truth[:, 0] / source_scale_x).long().clamp(0, source_w - 1)
        true_y = torch.round(truth[:, 1] / source_scale_y).long().clamp(0, source_h - 1)
        true_index = true_y * source_w + true_x
        true_similarity = similarity.gather(1, true_index[:, None]).squeeze(1)
        nonmatching = similarity.clone()
        nonmatching.scatter_(1, true_index[:, None], -torch.inf)
        strongest_nonmatch = nonmatching.max(dim=1).values

        backward = source[predicted_index] @ target.transpose(0, 1)
        returned_index = backward.argmax(dim=1)
        original_xy = torch.stack(
            (
                (indices % target_w).float() * target_scale_x,
                torch.div(indices, target_w, rounding_mode="floor").float()
                * target_scale_y,
            ),
            dim=1,
        )
        returned_xy = torch.stack(
            (
                (returned_index % target_w).float() * target_scale_x,
                torch.div(returned_index, target_w, rounding_mode="floor").float()
                * target_scale_y,
            ),
            dim=1,
        )
        cycle_error = torch.linalg.vector_norm(returned_xy - original_xy, dim=1)
        return {
            "count": float(error.numel()),
            "epe": float(error.sum().item()),
            "pck1": float((error <= 1).sum().item()),
            "pck3": float((error <= 3).sum().item()),
            "pck5": float((error <= 5).sum().item()),
            "margin": float((true_similarity - strongest_nonmatch).sum().item()),
            "cycle_epe": float(cycle_error.sum().item()),
        }

    result: dict[str, float] = {"channels": float(packet.target.shape[-1])}
    for region_name, region_mask in regions.items():
        sampled_region = F.interpolate(
            region_mask.float().unsqueeze(0),
            (target_h, target_w),
            mode="nearest",
        )[0, 0].flatten()
        values = evaluate(torch.nonzero(sampled_region > 0.5).flatten())
        prefix = "" if region_name == "all" else f"{region_name}_"
        result.update({f"{prefix}{name}": value for name, value in values.items()})
    return result


def _merge_stats(files: list[Path]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for path in files:
        value = json.loads(path.read_text(encoding="utf-8"))
        for key, metrics in value.items():
            for name, number in metrics.items():
                if name == "channels":
                    merged[key][name] = number
                else:
                    merged[key][name] += number
    return {key: dict(value) for key, value in merged.items()}


def _select(
    stats: dict[str, dict[str, float]],
    top_layers: int,
    region_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    region_weights = region_weights or {
        "epe": 0.35,
        "text_line_epe": 0.25,
        "page_edge_epe": 0.15,
        "corners_epe": 0.10,
        "interior_texture_epe": 0.05,
        "cycle_epe": 0.10,
    }
    rows = []
    for key, sums in stats.items():
        scale, variant, step, layer = key.split("|")
        count = sums["count"]
        row: dict[str, Any] = {
            "lora_scale": float(scale),
            "variant": variant,
            "step": int(step),
            "layer": int(layer),
            "epe": sums["epe"] / count,
            "pck1": sums["pck1"] / count,
            "pck3": sums["pck3"] / count,
            "pck5": sums["pck5"] / count,
            "margin": sums["margin"] / count,
            "cycle_epe": sums["cycle_epe"] / count,
            "descriptor_channels": int(sums["channels"]),
        }
        for name in (
            "text_line",
            "page_edge",
            "corners",
            "interior_texture",
            "blank_background",
        ):
            region_count = sums.get(f"{name}_count", 0.0)
            row[f"{name}_count"] = region_count
            for metric in ("epe", "pck1", "pck3", "pck5", "margin", "cycle_epe"):
                total = sums.get(f"{name}_{metric}", 0.0)
                row[f"{name}_{metric}"] = total / region_count if region_count else None
        row["selection_score"] = sum(
            weight * float(row.get(name) if row.get(name) is not None else row["epe"])
            for name, weight in region_weights.items()
        )
        rows.append(row)
    grouped: dict[tuple[float, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["lora_scale"], row["variant"], row["step"])].append(row)

    candidates = []
    for group, values in grouped.items():
        if group[1] == "hidden":
            continue
        best_layers = sorted(values, key=lambda item: item["selection_score"])[:top_layers]
        candidates.append(
            (
                sum(item["selection_score"] for item in best_layers) / len(best_layers),
                group,
                best_layers,
            )
        )
    _, best_group, best_layers = min(candidates, key=lambda item: item[0])
    hidden_rows = [row for row in rows if row["variant"] == "hidden"]
    best_hidden = min(hidden_rows, key=lambda item: item["epe"]) if hidden_rows else None
    descriptor_channels = best_layers[0]["descriptor_channels"]
    return {
        "lora_scale": best_group[0],
        "variant": best_group[1],
        "step": best_group[2],
        "layers": sorted(item["layer"] for item in best_layers),
        "descriptor_channels": descriptor_channels,
        "top_layer_metrics": best_layers,
        "best_hidden_diagnostic": best_hidden,
        "selection_weights": region_weights,
        "all_metrics": rows,
    }


def run_probe(config_path: str | Path) -> None:
    config = load_config(config_path)
    rank, world, local_rank = _distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    probe_config = config["probe"]
    dataset = DocumentMapDataset(
        resolve_path(config, config["data"]["val_manifest"]),
        work_size=tuple(config["data"]["work_size"]),
        limit=int(probe_config["samples"]),
    )
    output_dir = resolve_path(config, probe_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    layers = tuple(int(v) for v in probe_config["layers"])
    steps = tuple(int(v) for v in probe_config["steps"])
    scales = tuple(float(v) for v in probe_config["lora_scales"])
    seeds = tuple(int(v) for v in probe_config.get("seeds", [config["qwen"].get("seed", 0)]))
    max_points = int(probe_config.get("target_points", 64))
    base_qk: Tensor | None = None
    qk_deltas: dict[str, float] = {}
    lora_reports: dict[str, dict[str, object]] = {}

    for scale in scales:
        qwen_config = dict(config["qwen"])
        qwen_config["lora_scale"] = scale
        pending_pre: dict[tuple[int, int], FeaturePacket] = {}
        scale_qk: Tensor | None = None
        extractor = QwenQKExtractor(
            qwen_config,
            device=device,
            layers=layers,
            steps=steps,
            variants=("pre", "post", "hidden"),
        )
        try:
            lora_reports[str(scale)] = extractor.lora_report
            for index in range(rank, len(dataset), world):
                sample = dataset[index]
                regions = _probe_regions(
                    sample["target"].to(device), sample["valid"].to(device)
                )

                def consume(packet: FeaturePacket) -> None:
                    nonlocal scale_qk
                    if (
                        scale_qk is None
                        and packet.variant == "pre"
                        and packet.step == steps[0]
                        and packet.layer == layers[0]
                    ):
                        scale_qk = torch.cat(
                            (packet.target.flatten(), packet.source.flatten())
                        )[:8192].float().cpu()
                    variants = [packet]
                    marker = (packet.step, packet.layer)
                    if packet.variant == "pre":
                        pending_pre[marker] = packet
                    elif packet.variant == "post" and marker in pending_pre:
                        pre = pending_pre.pop(marker)
                        variants.append(
                            FeaturePacket(
                                step=packet.step,
                                layer=packet.layer,
                                variant="pre_post",
                                target=torch.cat((pre.target, packet.target), dim=-1),
                                source=torch.cat((pre.source, packet.source), dim=-1),
                                target_grid=packet.target_grid,
                                source_grid=packet.source_grid,
                            )
                        )
                    for item in variants:
                        result = _match_metrics(
                            item,
                            sample["map"].to(device),
                            regions,
                            max_points=max_points,
                        )
                        key = f"{scale}|{item.variant}|{item.step}|{item.layer}"
                        for name, value in result.items():
                            if name == "channels":
                                stats[key][name] = value
                            else:
                                stats[key][name] += value

                for seed in seeds:
                    pending_pre.clear()
                    extractor.run(
                        tensor_to_pil(sample["warped"]),
                        seed=seed,
                        consumer=consume,
                        store=False,
                    )
        finally:
            extractor.close()
            del extractor
            gc.collect()
            torch.cuda.empty_cache()
        if scale_qk is None:
            raise RuntimeError(f"Q/K delta audit captured no tensor for LoRA scale {scale}")
        if base_qk is None:
            base_qk = scale_qk
            qk_deltas[str(scale)] = 0.0
        else:
            delta = float((scale_qk - base_qk).abs().mean())
            qk_deltas[str(scale)] = delta
            if scale > 0 and delta <= float(probe_config.get("min_lora_qk_delta", 1.0e-7)):
                raise RuntimeError(
                    f"LoRA scale {scale} did not change captured Q/K; mean delta={delta}"
                )

    rank_path = output_dir / f"rank_{rank:02d}.json"
    rank_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = _merge_stats([output_dir / f"rank_{value:02d}.json" for value in range(world)])
        selection = _select(
            merged,
            int(probe_config.get("top_layers", 4)),
            dict(probe_config.get("selection_weights", {})) or None,
        )
        selection["seeds"] = list(seeds)
        selection["lora_loading"] = lora_reports
        selection["lora_qk_mean_absolute_delta_from_base"] = qk_deltas
        (output_dir / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(selection, indent=2))
    if world > 1:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_probe(args.config)


if __name__ == "__main__":
    main()
