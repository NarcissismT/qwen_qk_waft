from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .geometry import map_jacobian_determinant


def masked_mean(value: Tensor, valid: Tensor) -> Tensor:
    mask = valid.bool()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(value)
    selected = torch.where(mask, value, torch.zeros_like(value))
    return selected.sum() / mask.sum().clamp_min(1)


def weighted_masked_mean(value: Tensor, valid: Tensor, weight: Tensor) -> Tensor:
    mask = valid.bool()
    weight = weight.to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(1)
    combined = torch.where(mask, weight, torch.zeros_like(weight))
    combined = combined.expand_as(value)
    selected = torch.where(
        mask.expand_as(value), value * combined, torch.zeros_like(value)
    )
    return selected.sum() / combined.sum().clamp_min(1)


def endpoint_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.linalg.vector_norm(prediction - target, dim=1, keepdim=True)


def image_edge_weight(image: Tensor) -> Tensor:
    gray = image.mean(dim=1, keepdim=True)
    dx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return (dx + dy).clamp(0, 1)


def line_reconstruction_loss(
    prediction: Tensor, target: Tensor, valid: Tensor, kernel: int = 15
) -> Tensor:
    prediction_gray = prediction.mean(dim=1, keepdim=True)
    target_gray = target.mean(dim=1, keepdim=True)
    horizontal_prediction = F.avg_pool2d(
        prediction_gray, (1, kernel), stride=1, padding=(0, kernel // 2)
    )
    horizontal_target = F.avg_pool2d(
        target_gray, (1, kernel), stride=1, padding=(0, kernel // 2)
    )
    vertical_prediction = F.avg_pool2d(
        prediction_gray, (kernel, 1), stride=1, padding=(kernel // 2, 0)
    )
    vertical_target = F.avg_pool2d(
        target_gray, (kernel, 1), stride=1, padding=(kernel // 2, 0)
    )
    return masked_mean(
        (horizontal_prediction - horizontal_target).abs()
        + (vertical_prediction - vertical_target).abs(),
        valid,
    )


def bending_loss(pixel_map: Tensor, valid: Tensor, flat_weight: Tensor) -> Tensor:
    pixel_map = pixel_map.float()
    flat_weight = flat_weight.float()
    dxx = pixel_map[:, :, :, 2:] - 2 * pixel_map[:, :, :, 1:-1] + pixel_map[:, :, :, :-2]
    dyy = pixel_map[:, :, 2:, :] - 2 * pixel_map[:, :, 1:-1, :] + pixel_map[:, :, :-2, :]
    loss_x = weighted_masked_mean(
        dxx.abs(), valid[:, :, :, 1:-1], flat_weight[:, :, :, 1:-1]
    )
    loss_y = weighted_masked_mean(
        dyy.abs(), valid[:, :, 1:-1, :], flat_weight[:, :, 1:-1, :]
    )
    return loss_x + loss_y


def mixture_laplace_nll(
    prediction: Tensor, target: Tensor, info: Tensor, valid: Tensor
) -> Tensor:
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction = prediction.float()
        target = target.float()
        info = info.float()
        weights = info[:, :2]
        raw_scale = info[:, 2:]
        log_scale = torch.stack(
            (raw_scale[:, 0].clamp(0, 10), raw_scale[:, 1].clamp(-10, 0)),
            dim=1,
        )
        error = (target - prediction).abs().unsqueeze(2)
        component = weights - math.log(2.0) - log_scale
        component = component.unsqueeze(1) - error * torch.exp(
            -log_scale
        ).unsqueeze(1)
        nll = torch.logsumexp(weights, dim=1, keepdim=True) - torch.logsumexp(
            component, dim=2
        )
    return masked_mean(nll, valid)


def compute_losses(
    output: Mapping[str, Any],
    batch: Mapping[str, Tensor],
    weights: Mapping[str, float],
    *,
    sequence_gamma: float,
    gate_temperature_px: float,
    required_improvement_px: float,
    min_jacobian: float,
) -> dict[str, Tensor]:
    target_map = batch["map"].float()
    valid = batch["valid"]
    target_image = batch["target"].float()
    maps: list[Tensor] = output["maps"]
    infos: list[Tensor] = output["infos"]
    sequence = target_map.new_zeros(())
    uncertainty = target_map.new_zeros(())
    sequence_weight_sum = 0.0
    for index, (prediction, info) in enumerate(zip(maps, infos)):
        factor = sequence_gamma ** (len(maps) - index - 1)
        prediction = prediction.float()
        robust = torch.sqrt((prediction - target_map).square() + 1.0e-4)
        sequence = sequence + factor * masked_mean(robust, valid)
        uncertainty = uncertainty + factor * mixture_laplace_nll(
            prediction, target_map, info, valid
        )
        sequence_weight_sum += factor
    sequence = sequence / sequence_weight_sum
    uncertainty = uncertainty / sequence_weight_sum

    final_map = output["final_map"].float()
    edge = image_edge_weight(target_image)
    epe = endpoint_error(final_map, target_map)
    edge_loss = masked_mean(epe * (1 + 4 * edge), valid)
    rectified = output["rectified"].float()
    reconstruction = masked_mean((rectified - target_image).abs(), valid)
    line = line_reconstruction_loss(rectified, target_image, valid)
    flat = 1 - edge
    bending = bending_loss(final_map, valid, flat)
    determinant = map_jacobian_determinant(final_map)
    fold_valid = valid[:, :, :-1, :-1]
    anti_fold = masked_mean(F.relu(min_jacobian - determinant).unsqueeze(1), fold_valid)

    coarse_error = endpoint_error(output["coarse_map"], target_map)
    gate_target = (coarse_error / gate_temperature_px).clamp(0, 1)
    gate_loss = target_map.new_zeros(())
    protection = target_map.new_zeros(())
    correction = target_map.new_zeros(())
    gates: list[Tensor] = output["gates"]
    if gates:
        gate_weight_sum = 0.0
        for index, iteration_gate in enumerate(gates):
            factor = sequence_gamma ** (len(gates) - index - 1)
            gate = F.interpolate(
                iteration_gate.float(),
                target_map.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            with torch.autocast(device_type=gate.device.type, enabled=False):
                gate_loss = gate_loss + factor * masked_mean(
                    F.binary_cross_entropy(
                        gate.float(), gate_target.float(), reduction="none"
                    ),
                    valid,
                )
            gate_weight_sum += factor
        gate_loss = gate_loss / gate_weight_sum
        correct_prior = valid & (coarse_error < gate_temperature_px)
        wrong_prior = valid & (coarse_error >= gate_temperature_px)
        final_error = endpoint_error(final_map, target_map)
        protection = masked_mean(F.relu(final_error - coarse_error), correct_prior)
        correction = masked_mean(
            F.relu(final_error - coarse_error + required_improvement_px),
            wrong_prior,
        )

    terms = {
        "sequence": sequence,
        "uncertainty": uncertainty,
        "edge": edge_loss,
        "reconstruction": reconstruction,
        "line": line,
        "bending": bending,
        "anti_fold": anti_fold,
        "gate": gate_loss,
        "protection": protection,
        "correction": correction,
    }
    total = sum(float(weights.get(name, 0.0)) * value for name, value in terms.items())
    return {"total": total, **terms}


def confidence_loss(
    confidence: Tensor,
    coarse_map: Tensor,
    target_map: Tensor,
    valid: Tensor,
    *,
    temperature_px: float,
) -> Tensor:
    error = endpoint_error(coarse_map, target_map)
    target = torch.exp(-error / float(temperature_px))
    return masked_mean(
        F.binary_cross_entropy(confidence, target, reduction="none"), valid
    )
