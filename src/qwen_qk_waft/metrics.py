from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .losses import image_edge_weight, masked_mean


def document_region_masks(image: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Create reproducible text-structure, page-edge and corner masks."""

    line = F.max_pool2d(image_edge_weight(image), 5, stride=1, padding=2) > 0.08
    _, _, height, width = valid.shape
    band = max(2, round(min(height, width) * 0.08))
    padded_invalid = F.pad((~valid).float(), (band, band, band, band), value=1.0)
    eroded = 1.0 - F.max_pool2d(padded_invalid, 2 * band + 1, stride=1)
    page_edge = valid & (eroded < 0.5)
    corner_width = max(2, round(min(height, width) * 0.18))
    y, x = torch.meshgrid(
        torch.arange(height, device=valid.device),
        torch.arange(width, device=valid.device),
        indexing="ij",
    )
    corners = (
        ((x < corner_width) | (x >= width - corner_width))
        & ((y < corner_width) | (y >= height - corner_width))
    ).view(1, 1, height, width)
    return {
        "line": valid & line,
        "page_edge": page_edge,
        "corner": valid & corners,
    }


def masked_quantile(value: Tensor, valid: Tensor, quantile: float) -> Tensor:
    mask = valid.expand_as(value)
    selected = value[mask]
    if not selected.numel():
        return value.new_zeros(())
    return torch.quantile(selected.float(), quantile).to(value.dtype)


def map_curvature_error(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
) -> Tensor:
    prediction_x = prediction[:, :, :, 2:] - 2 * prediction[:, :, :, 1:-1]
    prediction_x = prediction_x + prediction[:, :, :, :-2]
    target_x = target[:, :, :, 2:] - 2 * target[:, :, :, 1:-1]
    target_x = target_x + target[:, :, :, :-2]
    prediction_y = prediction[:, :, 2:, :] - 2 * prediction[:, :, 1:-1, :]
    prediction_y = prediction_y + prediction[:, :, :-2, :]
    target_y = target[:, :, 2:, :] - 2 * target[:, :, 1:-1, :]
    target_y = target_y + target[:, :, :-2, :]
    error_x = torch.linalg.vector_norm(prediction_x - target_x, dim=1, keepdim=True)
    error_y = torch.linalg.vector_norm(prediction_y - target_y, dim=1, keepdim=True)
    return masked_mean(error_x, valid[:, :, :, 1:-1]) + masked_mean(
        error_y, valid[:, :, 1:-1, :]
    )


def calibration_metrics(
    probability: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    bins: int = 10,
) -> tuple[Tensor, Tensor]:
    mask = valid.expand_as(probability)
    probability = probability[mask].float()
    target = target.expand_as(mask)[mask].float()
    if not probability.numel():
        zero = probability.new_zeros(())
        return zero, zero
    brier = (probability - target).square().mean()
    ece = probability.new_zeros(())
    boundaries = torch.linspace(0, 1, bins + 1, device=probability.device)
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= boundaries[index]) & (
                probability <= boundaries[index + 1]
            )
        else:
            selected = (probability >= boundaries[index]) & (
                probability < boundaries[index + 1]
            )
        if selected.any():
            weight = selected.float().mean()
            ece = ece + weight * (
                probability[selected].mean() - target[selected].mean()
            ).abs()
    return brier, ece


def gate_histogram(gate: Tensor, valid: Tensor, *, bins: int = 10) -> Tensor:
    values = gate[valid.expand_as(gate)].float()
    if not values.numel():
        return gate.new_zeros(bins)
    boundaries = torch.linspace(0, 1, bins + 1, device=values.device)
    counts = []
    for index in range(bins):
        upper = (
            values <= boundaries[index + 1]
            if index == bins - 1
            else values < boundaries[index + 1]
        )
        counts.append(((values >= boundaries[index]) & upper).float().mean())
    return torch.stack(counts)


def reconstruction_psnr(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    mse = masked_mean((prediction - target).square(), valid)
    return -10.0 * torch.log10(mse.clamp_min(1.0e-12))


def reconstruction_ssim(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    prediction = prediction.mean(dim=1, keepdim=True)
    target = target.mean(dim=1, keepdim=True)
    mean_prediction = F.avg_pool2d(prediction, 11, stride=1, padding=5)
    mean_target = F.avg_pool2d(target, 11, stride=1, padding=5)
    variance_prediction = F.avg_pool2d(prediction.square(), 11, 1, 5)
    variance_prediction = (
        variance_prediction - mean_prediction.square()
    ).clamp_min(0)
    variance_target = F.avg_pool2d(target.square(), 11, 1, 5)
    variance_target = (variance_target - mean_target.square()).clamp_min(0)
    covariance = F.avg_pool2d(prediction * target, 11, 1, 5)
    covariance = covariance - mean_prediction * mean_target
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)
    ) / (
        (mean_prediction.square() + mean_target.square() + c1)
        * (variance_prediction + variance_target + c2)
    ).clamp_min(1.0e-12)
    return masked_mean(score, valid)


def masked_minimum(value: Tensor, valid: Tensor) -> Tensor:
    selected = value[valid.expand_as(value)]
    return selected.min() if selected.numel() else value.new_tensor(math.nan)
