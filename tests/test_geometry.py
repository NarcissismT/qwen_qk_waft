import torch

from qwen_qk_waft.geometry import (
    displacement_to_map,
    map_jacobian_determinant,
    map_to_displacement,
    pixel_grid,
    resize_absolute_map,
    sample_by_map,
    sample_feature_at_displacement,
)


def test_identity_map_samples_native_pixels() -> None:
    image = torch.arange(3 * 8 * 10, dtype=torch.float32).reshape(1, 3, 8, 10)
    identity = pixel_grid(1, 8, 10, device="cpu", dtype=torch.float32)
    sampled = sample_by_map(image, identity)
    torch.testing.assert_close(sampled, image)


def test_translation_and_map_displacement_round_trip() -> None:
    displacement = torch.zeros(1, 2, 6, 7)
    displacement[:, 0] = 2.0
    displacement[:, 1] = -1.0
    pixel_map = displacement_to_map(displacement)
    torch.testing.assert_close(map_to_displacement(pixel_map), displacement)


def test_absolute_map_resize_scales_source_coordinates() -> None:
    source = pixel_grid(1, 5, 7, device="cpu", dtype=torch.float32)
    resized = resize_absolute_map(
        source,
        (9, 13),
        source_size_from=(5, 7),
        source_size_to=(9, 13),
    )
    expected = pixel_grid(1, 9, 13, device="cpu", dtype=torch.float32)
    torch.testing.assert_close(resized, expected, atol=1.0e-5, rtol=1.0e-5)


def test_waft_feature_warp_uses_official_zero_padding() -> None:
    feature = torch.ones(1, 2, 4, 4)
    displacement = torch.full((1, 2, 4, 4), 100.0)
    sampled = sample_feature_at_displacement(feature, displacement)
    assert torch.count_nonzero(sampled) == 0


def test_bfloat16_autocast_keeps_512_identity_coordinates_in_float32() -> None:
    with torch.autocast("cpu", dtype=torch.bfloat16):
        identity = pixel_grid(1, 512, 512, device="cpu", dtype=torch.bfloat16)
        determinant = map_jacobian_determinant(identity)
    assert identity.dtype == torch.float32
    assert determinant.dtype == torch.float32
    assert torch.unique(identity[0, 0, 0]).numel() == 512
    assert torch.unique(identity[0, 1, :, 0]).numel() == 512
    assert torch.count_nonzero(determinant <= 0) == 0
    torch.testing.assert_close(determinant, torch.ones_like(determinant))
