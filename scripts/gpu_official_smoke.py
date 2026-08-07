from __future__ import annotations

import torch

from qwen_qk_waft.models.model import QwenQKWAFT


def main() -> None:
    checkpoint = (
        "/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/"
        "models/WAFT/waft_dav2_a2_zero_shot.ckpt"
    )
    timm_checkpoint = (
        "/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/"
        "models/WAFT/vit_small_patch16_224_imagenet.safetensors"
    )
    model = QwenQKWAFT(
        qk_channels=32,
        layer_count=4,
        stage_a_base_channels=8,
        waft_model_name="vits",
        waft_patch_size=8,
        timm_checkpoint=timm_checkpoint,
    )
    model.load_waft_pretrained(checkpoint)
    model.cuda().eval()
    warped = torch.rand(1, 3, 64, 64, device="cuda")
    target_q = torch.rand(1, 4, 32, 4, 4, device="cuda")
    source_k = torch.rand(1, 4, 32, 4, 4, device="cuda")
    with torch.inference_mode():
        output = model(
            warped,
            target_q,
            source_k,
            iterations=1,
            use_local_encoder=True,
            use_gate=True,
        )
    assert torch.isfinite(output["final_map"]).all()
    print(
        {
            "device": torch.cuda.get_device_name(),
            "final_map": list(output["final_map"].shape),
            "target_feature": list(output["target_feature"].shape),
            "source_feature": list(output["source_feature"].shape),
        }
    )


if __name__ == "__main__":
    main()
