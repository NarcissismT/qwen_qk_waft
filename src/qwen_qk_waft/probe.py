from __future__ import annotations

import argparse
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


def _match_metrics(
    packet: FeaturePacket,
    target_map: Tensor,
    valid: Tensor,
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
    sampled_valid = F.interpolate(
        valid.float().unsqueeze(0), (target_h, target_w), mode="nearest"
    )[0, 0].flatten()
    indices = torch.nonzero(sampled_valid > 0.5).flatten()
    if indices.numel() > max_points:
        positions = torch.linspace(
            0, indices.numel() - 1, max_points, device=indices.device
        ).long()
        indices = indices[positions]
    truth = sampled_map.flatten(1).transpose(0, 1)[indices]
    similarity = target[indices] @ source.transpose(0, 1)
    predicted_index = similarity.argmax(dim=1)
    predicted_x = (predicted_index % source_w).float()
    predicted_y = torch.div(predicted_index, source_w, rounding_mode="floor").float()
    predicted_x *= (target_map.shape[-1] - 1) / max(source_w - 1, 1)
    predicted_y *= (target_map.shape[-2] - 1) / max(source_h - 1, 1)
    predicted = torch.stack((predicted_x, predicted_y), dim=1)
    error = torch.linalg.vector_norm(predicted - truth, dim=1)
    true_x = torch.round(truth[:, 0] * (source_w - 1) / (target_map.shape[-1] - 1))
    true_y = torch.round(truth[:, 1] * (source_h - 1) / (target_map.shape[-2] - 1))
    true_index = (true_y * source_w + true_x).long().clamp(0, source.shape[0] - 1)
    true_similarity = similarity.gather(1, true_index[:, None]).squeeze(1)
    best_similarity = similarity.max(dim=1).values
    count = float(error.numel())
    return {
        "count": count,
        "epe": float(error.sum().item()),
        "pck1": float((error <= 1).sum().item()),
        "pck3": float((error <= 3).sum().item()),
        "pck5": float((error <= 5).sum().item()),
        "margin": float((true_similarity - best_similarity).sum().item()),
        "channels": float(packet.target.shape[-1]),
    }


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


def _select(stats: dict[str, dict[str, float]], top_layers: int) -> dict[str, Any]:
    rows = []
    for key, sums in stats.items():
        scale, variant, step, layer = key.split("|")
        count = sums["count"]
        rows.append(
            {
                "lora_scale": float(scale),
                "variant": variant,
                "step": int(step),
                "layer": int(layer),
                "epe": sums["epe"] / count,
                "pck1": sums["pck1"] / count,
                "pck3": sums["pck3"] / count,
                "pck5": sums["pck5"] / count,
                "margin": sums["margin"] / count,
                "descriptor_channels": int(sums["channels"]),
            }
        )
    grouped: dict[tuple[float, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["lora_scale"], row["variant"], row["step"])].append(row)

    candidates = []
    for group, values in grouped.items():
        if group[1] == "hidden":
            continue
        best_layers = sorted(values, key=lambda item: item["epe"])[:top_layers]
        candidates.append(
            (sum(item["epe"] for item in best_layers) / len(best_layers), group, best_layers)
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
    max_points = int(probe_config.get("target_points", 64))

    for scale in scales:
        qwen_config = dict(config["qwen"])
        qwen_config["lora_scale"] = scale
        pending_pre: dict[tuple[int, int], FeaturePacket] = {}
        with QwenQKExtractor(
            qwen_config,
            device=device,
            layers=layers,
            steps=steps,
            variants=("pre", "post", "hidden"),
        ) as extractor:
            for index in range(rank, len(dataset), world):
                sample = dataset[index]

                def consume(packet: FeaturePacket) -> None:
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
                            sample["valid"].to(device),
                            max_points=max_points,
                        )
                        key = f"{scale}|{item.variant}|{item.step}|{item.layer}"
                        for name, value in result.items():
                            if name == "channels":
                                stats[key][name] = value
                            else:
                                stats[key][name] += value

                extractor.run(
                    tensor_to_pil(sample["warped"]),
                    seed=int(config["qwen"].get("seed", 0)),
                    consumer=consume,
                    store=False,
                )
        torch.cuda.empty_cache()

    rank_path = output_dir / f"rank_{rank:02d}.json"
    rank_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = _merge_stats([output_dir / f"rank_{value:02d}.json" for value in range(world)])
        selection = _select(merged, int(probe_config.get("top_layers", 4)))
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
