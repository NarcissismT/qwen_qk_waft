import torch

from qwen_qk_waft.models.qwen_qk import TokenLayout, apply_rotary


def test_runtime_img_shapes_split_target_then_source() -> None:
    layout = TokenLayout.from_img_shapes([[(1, 2, 3), (1, 3, 2)]])
    value = torch.arange(12).reshape(1, 12, 1)
    assert layout.target_grid == (2, 3)
    assert layout.source_grid == (3, 2)
    assert layout.target(value).flatten().tolist() == list(range(6))
    assert layout.source(value).flatten().tolist() == list(range(6, 12))


def test_unit_rotary_frequency_preserves_qk() -> None:
    value = torch.randn(1, 5, 2, 8)
    frequencies = torch.ones(5, 4, dtype=torch.complex64)
    torch.testing.assert_close(apply_rotary(value, frequencies), value)

