"""Official WAFT recurrent VisionTransformer with offline weight control."""

import math
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .dpt import DPTHead
from .patch_embed import PatchEmbed


MODEL_CONFIGS = {
    "vitl": {
        "encoder": "vit_large_patch16_224",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitb": {
        "encoder": "vit_base_patch16_224",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vits": {
        "encoder": "vit_small_patch16_224",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitt": {
        "encoder": "vit_tiny_patch16_224",
        "features": 32,
        "out_channels": [24, 48, 96, 192],
    },
}


class VisionTransformer(nn.Module):
    """WAFT's timm ViT blocks, PatchEmbed and DepthAnythingV2 DPTHead."""

    def __init__(
        self,
        model_name: str,
        input_dim: int,
        patch_size: int = 16,
        *,
        timm_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        model = timm.create_model(
            MODEL_CONFIGS[model_name]["encoder"],
            pretrained=timm_checkpoint is None,
            num_classes=0,
        )
        if timm_checkpoint is not None:
            timm_state = load_file(str(timm_checkpoint))
            timm_state.pop("head.weight")
            timm_state.pop("head.bias")
            model.load_state_dict(timm_state, strict=True)
        self.intermediate_layer_idx = {
            "vitt": [2, 5, 8, 11],
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }
        self.idx = self.intermediate_layer_idx[model_name]
        self.blks = model.blocks
        self.embed_dim = model.embed_dim
        self.input_dim = input_dim
        self.img_size = (224, 224)
        self.patch_size = patch_size
        self.output_dim = MODEL_CONFIGS[model_name]["features"]
        self.timm_checkpoint = (
            str(timm_checkpoint) if timm_checkpoint is not None else "timm"
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, self.embed_dim))
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=input_dim,
            embed_dim=self.embed_dim,
        )
        self.dpt_head = DPTHead(
            self.embed_dim,
            MODEL_CONFIGS[model_name]["features"],
            out_channels=MODEL_CONFIGS[model_name]["out_channels"],
        )

    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        count = self.pos_embed.shape[1]
        if npatch == count and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_n = math.sqrt(count)
        sx, sy = float(w0) / sqrt_n, float(h0) / sqrt_n
        pos_embed = F.interpolate(
            pos_embed.reshape(1, int(sqrt_n), int(sqrt_n), dim).permute(0, 3, 1, 2),
            scale_factor=(sy, sx),
            mode="bicubic",
            antialias=False,
        )
        assert int(w0) == pos_embed.shape[-1]
        assert int(h0) == pos_embed.shape[-2]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x, h, w)
        outputs = []
        for index, block in enumerate(self.blks):
            x = block(x)
            if index in self.idx:
                outputs.append([x])
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        out, path_1, path_2, path_3, path_4 = self.dpt_head(
            outputs,
            patch_h,
            patch_w,
            return_intermediate=True,
        )
        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
        return {
            "out": out,
            "path_1": path_1,
            "path_2": path_2,
            "path_3": path_3,
            "path_4": path_4,
        }
