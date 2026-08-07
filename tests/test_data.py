import json

import numpy as np
from PIL import Image

from qwen_qk_waft.data import DocumentMapDataset


def test_dataset_converts_displacement_to_work_canvas(tmp_path) -> None:
    height, width = 8, 10
    pixels = np.arange(height * width, dtype=np.uint8).reshape(height, width)
    image = np.stack((pixels, pixels, pixels), axis=-1)
    Image.fromarray(image).save(tmp_path / "warped.png")
    Image.fromarray(image).save(tmp_path / "target.png")
    flow = np.zeros((height, width, 2), dtype=np.float32)
    np.save(tmp_path / "flow.npy", flow)
    record = {
        "id": "identity",
        "warped": "warped.png",
        "target": "target.png",
        "flow": "flow.npy",
        "flow_format": "displacement",
        "flow_source_size": [height, width],
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    sample = DocumentMapDataset(tmp_path / "manifest.jsonl", work_size=(height, width))[0]
    assert sample["map"].shape == (2, height, width)
    assert sample["valid"].all()

