from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_qk_waft.config import load_config, resolve_path
from qwen_qk_waft.models.model import QwenQKWAFT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    model = QwenQKWAFT(
        qk_channels=32,
        layer_count=4,
        stage_a_base_channels=int(model_config["stage_a_base_channels"]),
        stage_a_max_displacement_ratio=float(
            model_config["stage_a_max_displacement_ratio"]
        ),
        stage_a_control_stride=int(model_config["stage_a_control_stride"]),
        waft_model_name=str(model_config["waft_model_name"]),
        waft_patch_size=int(model_config["waft_patch_size"]),
        timm_checkpoint=resolve_path(config, model_config["timm_checkpoint"]),
    )
    report = model.load_waft_pretrained(
        resolve_path(config, model_config["waft_checkpoint"])
    )
    report["vit_blocks"] = len(model.waft.refine_net.blks)
    report["vit_feature_layers"] = model.waft.refine_net.idx
    report["vit_embed_dim"] = model.waft.refine_net.embed_dim
    report["waft_feature_dim"] = model.waft.iter_dim
    report["vit_imagenet_checkpoint"] = model.waft.refine_net.timm_checkpoint
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
