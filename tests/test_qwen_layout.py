import torch
from safetensors.torch import save_file
from torch import nn

from qwen_qk_waft.models.qwen_qk import TokenLayout, _load_lora, apply_rotary


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


def test_lora_loader_matches_original_diffsynth_fused_ba_rule(tmp_path) -> None:
    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = nn.Linear(3, 2, bias=False)
            self.to_k = nn.Linear(3, 2, bias=False)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = Attention()

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = nn.ModuleList([Block()])

    transformer = Transformer()
    original = {
        name: module.weight.detach().clone()
        for name, module in transformer.named_modules()
        if isinstance(module, nn.Linear)
    }
    state = {}
    for name in original:
        state[f"{name}.lora_A.weight"] = torch.tensor([[1.0, 2.0, 3.0]])
        state[f"{name}.lora_B.weight"] = torch.tensor([[4.0], [5.0]])
    checkpoint = tmp_path / "lora.safetensors"
    save_file(state, checkpoint)
    _, report = _load_lora(transformer, checkpoint, 0.5)
    for name, module in transformer.named_modules():
        if name in original:
            expected = original[name] + 0.5 * (
                state[f"{name}.lora_B.weight"] @ state[f"{name}.lora_A.weight"]
            )
            torch.testing.assert_close(module.weight, expected)
    assert report["matches_original_diffsynth_loader"]
    assert report["updated_linear_modules"] == 2
