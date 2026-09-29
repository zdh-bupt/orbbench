"""Training losses for probabilistic voxel-count forecasts."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F


Reduction = Literal["none", "mean", "sum"]
ProbabilitySmoothing = Literal["clamp", "additive"]
PoolSize3D = tuple[int, int, int]


def _reduce(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"unsupported reduction: {reduction}")


def _validate_fields(probability: Tensor, target_count: Tensor) -> None:
    if probability.shape != target_count.shape:
        raise ValueError("probability and target_count must have identical shapes")
    if probability.ndim < 3:
        raise ValueError("forecast fields must include at least one spatial dimension")
    if not torch.isfinite(probability).all() or not torch.isfinite(target_count).all():
        raise ValueError("probability and target_count must be finite")
    if torch.any(probability < 0):
        raise ValueError("probability cannot contain negative values")
    if torch.any(target_count < 0):
        raise ValueError("target_count cannot contain negative values")


def multinomial_nll(
    probability: Tensor,
    target_count: Tensor,
    *,
    reduction: Reduction = "mean",
    include_constant: bool = True,
    epsilon: float = 1.0e-8,
    smoothing: ProbabilitySmoothing = "clamp",
) -> Tensor:
    """Return multinomial negative log likelihood per batch and lead.

    The final three dimensions are normally altitude, latitude, and longitude;
    all dimensions after the first two are treated as categories. Fractional
    cloud-in-cell targets are supported through the gamma-function extension.
    """

    _validate_fields(probability, target_count)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    probability_flat = probability.flatten(start_dim=2)
    target_flat = target_count.to(dtype=probability.dtype).flatten(start_dim=2)
    probability_total = probability_flat.sum(dim=-1, keepdim=True)
    if torch.any(probability_total <= 0):
        raise ValueError("each probability field must contain positive mass")
    normalized = probability_flat / probability_total
    if smoothing == "clamp":
        log_probability = torch.log(normalized.clamp_min(epsilon))
    elif smoothing == "additive":
        category_count = normalized.shape[-1]
        normalized = (normalized + epsilon) / (1.0 + category_count * epsilon)
        log_probability = torch.log(normalized)
    else:
        raise ValueError(f"unsupported probability smoothing: {smoothing}")
    log_likelihood = (target_flat * log_probability).sum(dim=-1)
    if include_constant:
        target_total = target_flat.sum(dim=-1)
        log_likelihood = (
            torch.lgamma(target_total + 1.0)
            - torch.lgamma(target_flat + 1.0).sum(dim=-1)
            + log_likelihood
        )
    return _reduce(-log_likelihood, reduction)


def multinomial_nll_from_logits(
    logits: Tensor,
    target_count: Tensor,
    *,
    reduction: Reduction = "mean",
    include_constant: bool = False,
) -> Tensor:
    """Return multinomial NLL from unconstrained category logits.

    This path keeps gradients defined for every category through ``log_softmax``
    and is intended for sparse transition supervision.
    """

    if logits.shape != target_count.shape:
        raise ValueError("logits and target_count must have identical shapes")
    if logits.ndim < 3:
        raise ValueError("logits must include batch, lead, and category dimensions")
    if not torch.isfinite(logits).all() or not torch.isfinite(target_count).all():
        raise ValueError("logits and target_count must be finite")
    if torch.any(target_count < 0):
        raise ValueError("target_count cannot contain negative values")

    logits_flat = logits.flatten(start_dim=2)
    target_flat = target_count.to(dtype=logits.dtype).flatten(start_dim=2)
    log_likelihood = (target_flat * torch.log_softmax(logits_flat, dim=-1)).sum(dim=-1)
    if include_constant:
        target_total = target_flat.sum(dim=-1)
        log_likelihood = (
            torch.lgamma(target_total + 1.0)
            - torch.lgamma(target_flat + 1.0).sum(dim=-1)
            + log_likelihood
        )
    return _reduce(-log_likelihood, reduction)


def poisson_deviance(
    expected_count: Tensor,
    target_count: Tensor,
    *,
    reduction: Reduction = "mean",
    epsilon: float = 1.0e-8,
) -> Tensor:
    """Compute mean Poisson deviance over spatial cells per sample and lead."""

    if expected_count.shape != target_count.shape:
        raise ValueError("expected_count and target_count must have identical shapes")
    if not torch.isfinite(expected_count).all() or not torch.isfinite(target_count).all():
        raise ValueError("count fields must be finite")
    if torch.any(expected_count < 0) or torch.any(target_count < 0):
        raise ValueError("count fields cannot contain negative values")
    expected = expected_count.clamp_min(epsilon)
    target = target_count.to(dtype=expected.dtype)
    ratio_term = torch.where(
        target > 0,
        target * torch.log(target.clamp_min(epsilon) / expected),
        torch.zeros_like(target),
    )
    deviance = 2.0 * (ratio_term - (target - expected))
    spatial_dims = tuple(range(2, deviance.ndim))
    return _reduce(deviance.mean(dim=spatial_dims), reduction)


def conservation_loss(
    expected_count: Tensor,
    future_total: Tensor,
    *,
    reduction: Reduction = "mean",
    relative: bool = True,
    epsilon: float = 1.0e-8,
) -> Tensor:
    """Measure absolute total-count mismatch for diagnostic use."""

    if expected_count.ndim < 3:
        raise ValueError("expected_count must have batch, lead, and spatial dimensions")
    if future_total.shape != expected_count.shape[:2]:
        raise ValueError("future_total must match batch and lead dimensions")
    if not torch.isfinite(expected_count).all() or not torch.isfinite(future_total).all():
        raise ValueError("expected_count and future_total must be finite")
    spatial_dims = tuple(range(2, expected_count.ndim))
    error = (expected_count.sum(dim=spatial_dims) - future_total).abs()
    if relative:
        denominator = future_total.abs().clamp_min(epsilon)
        error = torch.where(future_total.abs() > epsilon, error / denominator, error)
    return _reduce(error, reduction)


def joint_multinomial_nll(
    voxel_probability: Tensor,
    outside_probability: Tensor,
    target_voxel_count: Tensor,
    target_outside_count: Tensor,
    *,
    reduction: Reduction = "mean",
    include_constant: bool = False,
    epsilon: float = 1.0e-8,
    smoothing: ProbabilitySmoothing = "clamp",
) -> Tensor:
    if voxel_probability.shape != target_voxel_count.shape:
        raise ValueError("voxel probability and target shapes differ")
    if outside_probability.shape != target_outside_count.shape:
        raise ValueError("outside probability and target shapes differ")
    if outside_probability.shape[:2] != voxel_probability.shape[:2] or outside_probability.shape[-1] != 2:
        raise ValueError("outside categories must have shape [B,K,2]")
    probability = torch.cat(
        (voxel_probability.flatten(start_dim=2), outside_probability), dim=2
    )
    target = torch.cat((target_voxel_count.flatten(start_dim=2), target_outside_count), dim=2)
    return multinomial_nll(
        probability,
        target,
        reduction=reduction,
        include_constant=include_constant,
        epsilon=epsilon,
        smoothing=smoothing,
    )


def pooled_distribution_l1(
    voxel_probability: Tensor,
    outside_probability: Tensor,
    target_voxel_count: Tensor,
    target_outside_count: Tensor,
    *,
    pool_size: PoolSize3D,
) -> Tensor:
    """Return joint distribution L1 after non-overlapping 3D pooling.

    The voxel term is pooled as probability mass, not as an average density.
    The lower/upper outside categories remain part of every pooled comparison.
    """
    if voxel_probability.shape != target_voxel_count.shape:
        raise ValueError("voxel probability and target shapes differ")
    if outside_probability.shape != target_outside_count.shape:
        raise ValueError("outside probability and target shapes differ")
    if voxel_probability.ndim != 5:
        raise ValueError("voxel probability must have shape [batch, lead, altitude, latitude, longitude]")
    if outside_probability.ndim != 3:
        raise ValueError("outside probability must have shape [batch, lead, outside_category]")
    if voxel_probability.shape[:2] != outside_probability.shape[:2]:
        raise ValueError("voxel and outside probabilities must share batch and lead dimensions")
    if len(pool_size) != 3 or any(int(width) != width or int(width) <= 0 for width in pool_size):
        raise ValueError("pool_size must contain three positive integer widths")
    pool_size = tuple(int(width) for width in pool_size)
    spatial_shape = voxel_probability.shape[-3:]
    if any(dimension % width for dimension, width in zip(spatial_shape, pool_size, strict=True)):
        raise ValueError(f"pool_size {pool_size} must evenly divide voxel spatial shape {spatial_shape}")
    target_total = target_voxel_count.flatten(start_dim=2).sum(dim=2) + target_outside_count.sum(dim=2)
    target_probability = target_voxel_count / target_total[:, :, None, None, None].clamp_min(1.0e-8)
    target_outside_probability = target_outside_count / target_total[:, :, None].clamp_min(1.0e-8)
    batch, leads = voxel_probability.shape[:2]
    flat_prediction = voxel_probability.reshape(batch * leads, 1, *voxel_probability.shape[-3:])
    flat_target = target_probability.reshape(batch * leads, 1, *target_probability.shape[-3:])
    factor = float(pool_size[0] * pool_size[1] * pool_size[2])
    pooled_prediction = F.avg_pool3d(flat_prediction, pool_size, pool_size) * factor
    pooled_target = F.avg_pool3d(flat_target, pool_size, pool_size) * factor
    spatial_l1 = torch.abs(pooled_prediction - pooled_target).flatten(start_dim=1).sum(dim=1)
    outside_l1 = torch.abs(outside_probability - target_outside_probability).sum(dim=2).reshape(-1)
    return 0.5 * (spatial_l1 + outside_l1).mean()


def multiscale_distribution_l1(
    voxel_probability: Tensor,
    outside_probability: Tensor,
    target_voxel_count: Tensor,
    target_outside_count: Tensor,
    *,
    pool_size: PoolSize3D = (2, 2, 2),
) -> Tensor:
    """Return the legacy single-level pooled distribution L1 auxiliary loss."""
    return pooled_distribution_l1(
        voxel_probability,
        outside_probability,
        target_voxel_count,
        target_outside_count,
        pool_size=pool_size,
    )


def joint_conservation_loss(
    expected_count: Tensor,
    outside_expected_count: Tensor,
    future_total: Tensor,
    *,
    reduction: Reduction = "mean",
    epsilon: float = 1.0e-8,
) -> Tensor:
    total = expected_count.flatten(start_dim=2).sum(dim=2) + outside_expected_count.sum(dim=2)
    error = torch.abs(total - future_total)
    error = torch.where(future_total > epsilon, error / future_total.clamp_min(epsilon), error)
    return _reduce(error, reduction)
