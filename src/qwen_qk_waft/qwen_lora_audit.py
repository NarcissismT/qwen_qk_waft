from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open

from .config import load_config, resolve_path
from .models.qwen_qk import _LORA_TARGETS, normalize_lora_key


def run_lora_audit(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    checkpoint = resolve_path(config, config["qwen"]["lora_checkpoint"])
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
    passed = not missing and not unexpected and len(ranks) == 1
    result: dict[str, object] = {
        "checkpoint": str(checkpoint.resolve()),
        "layers": len(layers),
        "target_modules_per_layer": len(_LORA_TARGETS),
        "expected_tensors": len(expected),
        "checkpoint_tensors": len(actual),
        "rank": ranks[0] if len(ranks) == 1 else ranks,
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
