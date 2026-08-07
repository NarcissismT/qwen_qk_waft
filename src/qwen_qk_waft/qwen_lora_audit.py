from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open

from .config import load_config, resolve_path
from .models.qwen_qk import _LORA_TARGETS, normalize_lora_key


def run_lora_audit(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    checkpoint = resolve_path(config, config["qwen"]["lora_checkpoint"])
    training_script = resolve_path(config, config["qwen"]["lora_training_script"])
    script_text = training_script.read_text(encoding="utf-8")
    rank_match = re.search(r"--lora_rank\s+(\d+)", script_text)
    if rank_match is None:
        raise RuntimeError(f"No --lora_rank was found in {training_script}")
    training_rank = int(rank_match.group(1))
    alpha_match = re.search(r"--lora_alpha\s+([0-9.eE+-]+)", script_text)
    training_alpha = float(alpha_match.group(1)) if alpha_match else float(training_rank)
    reference_loader_alpha = float(
        config["qwen"].get("lora_reference_loader_alpha", 1.0)
    )
    configured_scale = float(config["qwen"].get("lora_scale", 1.0))
    layers = tuple(int(value) for value in config["probe"]["layers"])
    attention_targets = set(_LORA_TARGETS[:8])
    expected = set()
    for layer in layers:
        for target in _LORA_TARGETS:
            module = f"attn.{target}" if target in attention_targets else target
            for side in ("A", "B"):
                expected.add(
                    f"transformer_blocks.{layer}.{module}.lora_{side}.default.weight"
                )
    shapes = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as file:
        for raw_key in file.keys():
            shapes[normalize_lora_key(raw_key)] = tuple(file.get_slice(raw_key).get_shape())
    actual = set(shapes)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    ranks = sorted(
        {shape[0] for key, shape in shapes.items() if ".lora_A." in key}
    )
    checkpoint_rank = ranks[0] if len(ranks) == 1 else None
    training_alpha_over_rank = training_alpha / training_rank
    reference_effective_scale = reference_loader_alpha
    configured_effective_scale = training_alpha_over_rank * configured_scale
    passed = (
        not missing
        and not unexpected
        and checkpoint_rank == training_rank
        and configured_effective_scale == reference_effective_scale
    )
    result: dict[str, object] = {
        "checkpoint": str(checkpoint.resolve()),
        "layers": len(layers),
        "target_modules_per_layer": len(_LORA_TARGETS),
        "expected_tensors": len(expected),
        "checkpoint_tensors": len(actual),
        "rank": checkpoint_rank if checkpoint_rank is not None else ranks,
        "training_script": str(training_script.resolve()),
        "training_rank": training_rank,
        "training_alpha": training_alpha,
        "training_alpha_source": (
            "explicit --lora_alpha" if alpha_match else "trainer default alpha=rank"
        ),
        "training_alpha_over_rank": training_alpha_over_rank,
        "original_diffsynth_loader_alpha": reference_loader_alpha,
        "original_diffsynth_effective_scale": reference_effective_scale,
        "configured_effective_scale": configured_effective_scale,
        "runtime_merge_rule": "base_weight + scale * (lora_B @ lora_A)",
        "scale_semantics_match": (
            configured_effective_scale == reference_effective_scale
        ),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_examples": missing[:20],
        "unexpected_examples": unexpected[:20],
        "passed": passed,
    }
    output = resolve_path(config, config["audit"]["lora_report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError(f"Qwen LoRA schema audit failed; see {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_lora_audit(args.config)


if __name__ == "__main__":
    main()
