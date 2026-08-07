from __future__ import annotations

import argparse

import torch

from qwen_qk_waft.config import load_config, resolve_path
from qwen_qk_waft.data import DocumentMapDataset, tensor_to_pil
from qwen_qk_waft.models.qwen_qk import QwenQKExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen_qk_waft.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    sample = DocumentMapDataset(
        resolve_path(config, config["data"]["val_manifest"]),
        work_size=tuple(config["data"]["work_size"]),
        limit=1,
    )[0]
    qwen = dict(config["qwen"])
    qwen["lora_scale"] = 0.0
    qwen["num_inference_steps"] = 2
    with QwenQKExtractor(
        qwen,
        device=torch.device("cuda:0"),
        layers=(0,),
        steps=(0,),
        variants=("pre", "post", "hidden"),
    ) as extractor:
        captured = extractor.run(tensor_to_pil(sample["warped"]), seed=0)
    for key, packet in sorted(captured.items()):
        print(key, tuple(packet.target.shape), tuple(packet.source.shape))


if __name__ == "__main__":
    main()
