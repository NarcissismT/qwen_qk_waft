from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config, resolve_path
from .data import load_rgb, tensor_to_pil
from .geometry import resize_absolute_map, sample_by_map, valid_map_mask
from .models.qwen_qk import QwenQKExtractor
from .train import _build_model


@torch.no_grad()
def infer(
    config_path: str | Path,
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selection = checkpoint["selection"]
    device = torch.device("cuda")
    model = _build_model(config, selection)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    amp_dtype = (
        torch.bfloat16
        if config["train"]["amp_dtype"] == "bfloat16"
        else torch.float16
    )

    native = load_rgb(image_path).unsqueeze(0).to(device)
    native_size = native.shape[-2:]
    work_size = tuple(int(v) for v in config["data"]["work_size"])
    model_input = F.interpolate(
        native, work_size, mode="bilinear", align_corners=True
    )
    qwen_config = dict(config["qwen"])
    qwen_config["lora_scale"] = float(selection["lora_scale"])
    variants = ("pre", "post") if selection["variant"] == "pre_post" else (selection["variant"],)
    with QwenQKExtractor(
        qwen_config,
        device=device,
        layers=selection["layers"],
        steps=(selection["step"],),
        variants=variants,
    ) as extractor:
        queries, keys, _, _ = extractor.selected_pair(
            tensor_to_pil(model_input[0]),
            seed=int(config["qwen"].get("seed", 0)),
            step=int(selection["step"]),
            variant=str(selection["variant"]),
        )
        with torch.autocast(
            "cuda", dtype=amp_dtype, enabled=bool(config["train"]["amp"])
        ):
            output = model(
                model_input,
                queries,
                keys,
                iterations=int(config["phases"]["d"]["iterations"]),
                use_local_encoder=True,
                use_gate=True,
                render=False,
            )

    native_map = resize_absolute_map(
        output["final_map"].float(),
        native_size,
        source_size_from=work_size,
        source_size_to=native_size,
    )
    # This is the only native-RGB resampling operation in inference.
    rectified = sample_by_map(native, native_map)
    valid = valid_map_mask(native_map, native_size)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    image_output = output_path / f"{stem}_rectified.png"
    map_output = output_path / f"{stem}_backward_map.npy"
    valid_output = output_path / f"{stem}_valid.png"
    tensor_to_pil(rectified[0]).save(image_output)
    np.save(map_output, native_map[0].permute(1, 2, 0).float().cpu().numpy())
    tensor_to_pil(valid[0].expand(3, -1, -1).float()).save(valid_output)
    result = {
        "rectified": str(image_output),
        "backward_map": str(map_output),
        "valid_mask": str(valid_output),
    }
    (output_path / f"{stem}_metadata.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(infer(args.config, args.checkpoint, args.image, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
