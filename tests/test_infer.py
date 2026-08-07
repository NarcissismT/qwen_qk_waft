from importlib import import_module
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from qwen_qk_waft.geometry import pixel_grid


def test_native_inference_resamples_rgb_exactly_once(tmp_path) -> None:
    infer_module = import_module("qwen_qk_waft.infer")
    image_path = tmp_path / "warped.png"
    Image.fromarray(np.full((32, 48, 3), 127, dtype=np.uint8)).save(image_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  work_size: [32, 32]
qwen:
  lora_scale: 1.0
  seed: 0
train:
  amp: false
  amp_dtype: bfloat16
phases:
  d:
    iterations: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    class Model:
        def load_state_dict(self, _state):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, image, *_args, **_kwargs):
            return {
                "final_map": pixel_grid(
                    1,
                    image.shape[-2],
                    image.shape[-1],
                    device=image.device,
                    dtype=image.dtype,
                )
            }

    class Extractor:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def selected_pair(self, *_args, **_kwargs):
            value = torch.zeros(1, 1, 1, 1, 1)
            return value, value, (1, 1), (1, 1)

    original_sample = infer_module.sample_by_map
    sample_calls = 0

    def counted_sample(*args, **kwargs):
        nonlocal sample_calls
        sample_calls += 1
        return original_sample(*args, **kwargs)

    checkpoint = {
        "selection": {
            "layers": [0],
            "step": 0,
            "variant": "pre",
            "lora_scale": 0.0,
        },
        "model": {},
    }
    with (
        patch.object(infer_module, "_build_model", return_value=Model()),
        patch.object(infer_module, "QwenQKExtractor", Extractor),
        patch.object(infer_module.torch, "load", return_value=checkpoint),
        patch.object(infer_module.torch, "device", return_value=torch.device("cpu")),
        patch.object(infer_module, "sample_by_map", side_effect=counted_sample),
    ):
        result = infer_module.infer(
            config_path,
            tmp_path / "checkpoint.pt",
            image_path,
            tmp_path / "output",
        )
    assert sample_calls == 1
    assert (tmp_path / "output" / "warped_rectified.png").exists()
    assert result["rectified"].endswith("warped_rectified.png")
