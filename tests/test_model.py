import torch

from qwen_qk_waft.geometry import pixel_grid
from qwen_qk_waft.losses import compute_losses
from qwen_qk_waft.models.model import QwenQKWAFT
from qwen_qk_waft.official_waft import DPTHead, PatchEmbed


TIMM_CHECKPOINT = (
    "/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/"
    "models/WAFT/vit_small_patch16_224_imagenet.safetensors"
)


def _model() -> QwenQKWAFT:
    return QwenQKWAFT(
        qk_channels=32,
        layer_count=4,
        stage_a_base_channels=8,
        waft_model_name="vits",
        waft_patch_size=8,
        timm_checkpoint=TIMM_CHECKPOINT,
    )


def test_complete_forward_uses_absolute_backward_map() -> None:
    torch.manual_seed(3)
    model = _model()
    warped = torch.rand(1, 3, 64, 64)
    target_q = torch.rand(1, 4, 32, 4, 4)
    source_k = torch.rand(1, 4, 32, 4, 4)
    output = model(
        warped,
        target_q,
        source_k,
        iterations=2,
        use_local_encoder=True,
        use_gate=True,
    )
    assert len(output["maps"]) == 2
    assert output["coarse_map"].shape == (1, 2, 64, 64)
    assert output["final_map"].shape == (1, 2, 64, 64)
    assert output["rectified"].shape == warped.shape
    assert output["target_feature"].shape[-2:] == (32, 32)
    assert output["source_feature"].shape[-2:] == (32, 32)
    assert output["gates"][-1].min() >= 0
    assert output["gates"][-1].max() <= 1


def test_training_losses_are_finite_and_backpropagate() -> None:
    torch.manual_seed(4)
    model = _model()
    warped = torch.rand(1, 3, 64, 64)
    target_q = torch.rand(1, 4, 32, 4, 4)
    source_k = torch.rand(1, 4, 32, 4, 4)
    output = model(warped, target_q, source_k, iterations=1)
    target_map = pixel_grid(1, 64, 64, device="cpu", dtype=torch.float32)
    batch = {
        "map": target_map,
        "valid": torch.ones(1, 1, 64, 64, dtype=torch.bool),
        "target": warped.clone(),
    }
    weights = {
        "sequence": 1.0,
        "uncertainty": 0.05,
        "edge": 0.1,
        "reconstruction": 0.1,
        "line": 0.1,
        "bending": 0.01,
        "anti_fold": 0.1,
        "gate": 0.05,
        "protection": 0.1,
        "correction": 0.1,
    }
    losses = compute_losses(
        output,
        batch,
        weights,
        sequence_gamma=0.85,
        gate_temperature_px=3.0,
        min_jacobian=0.05,
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.dpt_q.adapters[0].projection.weight.grad is not None


def test_official_checkpoint_strictly_loads_vit_dpt_and_update_heads() -> None:
    checkpoint = (
        "/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/"
        "models/WAFT/waft_dav2_a2_zero_shot.ckpt"
    )
    model = _model()
    report = model.load_waft_pretrained(checkpoint)
    assert report["refine_net"] > 100
    assert report["dpt_q"] == report["dpt_k"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    assert torch.equal(
        model.waft.refine_net.pos_embed,
        state["refine_net.pos_embed"],
    )
    assert torch.equal(
        model.dpt_q.dpt_head.projects[0].weight,
        state["refine_net.dpt_head.projects.0.weight"],
    )
    assert torch.equal(
        model.dpt_k.dpt_head.projects[0].weight,
        state["refine_net.dpt_head.projects.0.weight"],
    )


def test_official_waft_architecture_contract() -> None:
    model = _model()
    assert model.waft.iter_dim == 64
    assert model.waft.refine_net.embed_dim == 384
    assert model.waft.refine_net.timm_checkpoint == TIMM_CHECKPOINT
    assert model.waft.refine_net.idx == [2, 5, 8, 11]
    assert len(model.waft.refine_net.blks) == 12
    assert isinstance(model.waft.refine_net.patch_embed, PatchEmbed)
    assert isinstance(model.waft.refine_net.dpt_head, DPTHead)
    assert isinstance(model.dpt_q.dpt_head, DPTHead)
    assert isinstance(model.dpt_k.dpt_head, DPTHead)
    assert model.waft.warp_linear.in_channels == 3 * 64 + 2
    assert model.waft.refine_transform.in_channels == 96
    assert model.waft.flow_head[-1].out_channels == 6
    assert model.waft.upsample_weight[-1].out_channels == 36
