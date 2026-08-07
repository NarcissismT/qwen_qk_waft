import torch

from qwen_qk_waft.models.qwen_qk import FeaturePacket
from qwen_qk_waft.probe import _match_metrics


def test_similarity_margin_excludes_the_true_match_and_cycle_is_consistent() -> None:
    features = torch.eye(4).unsqueeze(0)
    packet = FeaturePacket(
        step=0,
        layer=0,
        variant="pre",
        target=features,
        source=features,
        target_grid=(2, 2),
        source_grid=(2, 2),
    )
    target_map = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]
    )
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    metrics = _match_metrics(
        packet,
        target_map,
        {"all": valid},
        max_points=4,
    )
    assert metrics["epe"] == 0.0
    assert metrics["cycle_epe"] == 0.0
    assert metrics["margin"] == 4.0
