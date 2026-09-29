"""Count-conserving physics-residual models for voxel density forecasts."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ForecastOutput(NamedTuple):
    """Forecast probabilities and expected counts.

    Both tensors have shape ``[batch, lead, altitude, latitude, longitude]``.
    """

    probability: Tensor
    expected_count: Tensor


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        if len(value) != 3:
            raise ValueError("kernel_size must contain three values")
        result = tuple(int(item) for item in value)
    if any(item <= 0 or item % 2 == 0 for item in result):
        raise ValueError("kernel_size values must be positive odd integers")
    return result


class PeriodicConv3d(nn.Module):
    """A same-size 3D convolution with a periodic longitude boundary.

    Altitude and latitude use replicated boundary values. Only the final
    spatial axis (longitude) wraps around.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel = _triple(kernel_size)
        self.padding = tuple(item // 2 for item in kernel)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            padding=0,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 5:
            raise ValueError("PeriodicConv3d expects [B,C,D,H,W]")
        pad_depth, pad_latitude, pad_longitude = self.padding
        if pad_longitude:
            inputs = F.pad(
                inputs,
                (pad_longitude, pad_longitude, 0, 0, 0, 0),
                mode="circular",
            )
        if pad_depth or pad_latitude:
            inputs = F.pad(
                inputs,
                (0, 0, pad_latitude, pad_latitude, pad_depth, pad_depth),
                mode="replicate",
            )
        return self.conv(inputs)


class ConvGRU3DCell(nn.Module):
    """A compact ConvGRU cell operating on 3D voxel fields."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int | Sequence[int] = 3,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        combined_channels = int(input_channels) + self.hidden_channels
        self.gates = PeriodicConv3d(
            combined_channels,
            2 * self.hidden_channels,
            kernel_size,
        )
        self.candidate = PeriodicConv3d(
            combined_channels,
            self.hidden_channels,
            kernel_size,
        )

    def forward(self, inputs: Tensor, hidden: Tensor | None = None) -> Tensor:
        if inputs.ndim != 5:
            raise ValueError("ConvGRU3DCell expects [B,C,D,H,W]")
        if hidden is None:
            hidden = inputs.new_zeros(
                inputs.shape[0],
                self.hidden_channels,
                *inputs.shape[-3:],
            )
        if hidden.shape[0] != inputs.shape[0] or hidden.shape[-3:] != inputs.shape[-3:]:
            raise ValueError("hidden state and inputs must share batch and spatial shapes")

        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, hidden), dim=1))
        ).chunk(2, dim=1)
        proposal = torch.tanh(
            self.candidate(torch.cat((inputs, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * proposal


class PhysicsResidualConvGRU3D(nn.Module):
    """Forecast a conserved voxel count field around a physics prior.

    Parameters
    ----------
    input_channels:
        Number of channels in each historical voxel frame.
    hidden_channels:
        Width of the lightweight encoder and recurrent state.
    residual_scale:
        Symmetric bound applied to learned logit corrections through tanh.
    epsilon:
        Positive floor applied before taking the physics-prior logarithm.

    Notes
    -----
    ``physics_prior`` must contain non-negative voxel mass or counts, not a
    volume-normalized density. It is normalized independently at every lead.
    The output probability sums to one over all voxels. Multiplication by
    ``future_total`` therefore conserves the requested count by construction.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 16,
        kernel_size: int | Sequence[int] = 3,
        *,
        residual_scale: float = 1.0,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("channel counts must be positive")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.residual_scale = float(residual_scale)
        self.epsilon = float(epsilon)

        self.history_encoder = nn.Sequential(
            PeriodicConv3d(self.input_channels, self.hidden_channels, kernel_size),
            nn.GroupNorm(1, self.hidden_channels),
            nn.GELU(),
            PeriodicConv3d(self.hidden_channels, self.hidden_channels, kernel_size),
            nn.GELU(),
        )
        self.history_cell = ConvGRU3DCell(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size,
        )
        self.mass_encoder = nn.Sequential(
            PeriodicConv3d(1, self.hidden_channels, kernel_size),
            nn.GELU(),
        )
        self.forecast_cell = ConvGRU3DCell(
            2 * self.hidden_channels,
            self.hidden_channels,
            kernel_size,
        )
        self.residual_decoder = nn.Sequential(
            PeriodicConv3d(2 * self.hidden_channels, self.hidden_channels, kernel_size),
            nn.GELU(),
        )
        self.residual_out = PeriodicConv3d(
            self.hidden_channels,
            1,
            kernel_size,
        )

        # The untrained model reproduces the normalized physics prior.
        nn.init.zeros_(self.residual_out.conv.weight)
        if self.residual_out.conv.bias is not None:
            nn.init.zeros_(self.residual_out.conv.bias)

    def _normalize_prior(self, prior: Tensor) -> Tensor:
        flat = prior.flatten(start_dim=1)
        total = flat.sum(dim=1, keepdim=True)
        normalized = flat / total.clamp_min(self.epsilon)
        uniform = torch.full_like(flat, 1.0 / flat.shape[1])
        normalized = torch.where(total > self.epsilon, normalized, uniform)
        return normalized.reshape_as(prior)

    def _validate_inputs(
        self,
        history: Tensor,
        physics_prior: Tensor,
        future_total: Tensor,
    ) -> tuple[int, int, tuple[int, int, int]]:
        if history.ndim != 6:
            raise ValueError("history must have shape [B,T,C,D,H,W]")
        if physics_prior.ndim != 5:
            raise ValueError("physics_prior must have shape [B,K,D,H,W]")
        if future_total.ndim != 2:
            raise ValueError("future_total must have shape [B,K]")
        batch, time_steps, channels = history.shape[:3]
        prior_batch, leads = physics_prior.shape[:2]
        if time_steps < 1:
            raise ValueError("history must contain at least one time step")
        if leads < 1:
            raise ValueError("physics_prior must contain at least one lead")
        if channels != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} history channels, got {channels}"
            )
        if prior_batch != batch or future_total.shape != (batch, leads):
            raise ValueError("batch and lead dimensions must agree")
        spatial_shape = tuple(history.shape[-3:])
        if tuple(physics_prior.shape[-3:]) != spatial_shape:
            raise ValueError("history and physics_prior spatial shapes must agree")
        if not torch.is_floating_point(history) or not torch.is_floating_point(physics_prior):
            raise TypeError("history and physics_prior must be floating-point tensors")
        if (
            not torch.isfinite(history).all()
            or not torch.isfinite(physics_prior).all()
            or not torch.isfinite(future_total).all()
        ):
            raise ValueError("history, physics_prior, and future_total must be finite")
        if torch.any(physics_prior < 0):
            raise ValueError("physics_prior cannot contain negative values")
        if torch.any(future_total < 0):
            raise ValueError("future_total cannot contain negative values")
        return batch, leads, spatial_shape

    def forward(
        self,
        history: Tensor,
        physics_prior: Tensor,
        future_total: Tensor,
    ) -> ForecastOutput:
        """Run a multi-lead forecast.

        Parameters use shapes ``[B,T,C,D,H,W]``, ``[B,K,D,H,W]``, and
        ``[B,K]`` respectively.
        """

        _, leads, _ = self._validate_inputs(history, physics_prior, future_total)
        physics_prior = physics_prior.to(dtype=history.dtype, device=history.device)
        future_total = future_total.to(dtype=history.dtype, device=history.device)

        hidden: Tensor | None = None
        for time_index in range(history.shape[1]):
            encoded = self.history_encoder(history[:, time_index])
            hidden = self.history_cell(encoded, hidden)
        if hidden is None:
            raise RuntimeError("history encoding produced no hidden state")

        probabilities: list[Tensor] = []
        expected_counts: list[Tensor] = []
        previous_probability = self._normalize_prior(physics_prior[:, 0])

        for lead_index in range(leads):
            prior_probability = self._normalize_prior(physics_prior[:, lead_index])
            prior_features = self.mass_encoder(prior_probability.unsqueeze(1))
            previous_features = self.mass_encoder(previous_probability.unsqueeze(1))
            hidden = self.forecast_cell(
                torch.cat((prior_features, previous_features), dim=1),
                hidden,
            )
            residual = self.residual_out(
                self.residual_decoder(torch.cat((hidden, prior_features), dim=1))
            ).squeeze(1)
            logits = (
                torch.log(prior_probability.clamp_min(self.epsilon))
                + self.residual_scale * torch.tanh(residual)
            )
            probability = torch.softmax(logits.flatten(start_dim=1), dim=1).reshape_as(
                prior_probability
            )
            expected_count = probability * future_total[:, lead_index].view(
                -1, 1, 1, 1
            )
            probabilities.append(probability)
            expected_counts.append(expected_count)
            previous_probability = probability

        return ForecastOutput(
            probability=torch.stack(probabilities, dim=1),
            expected_count=torch.stack(expected_counts, dim=1),
        )


class FormalForecastOutput(NamedTuple):
    voxel_probability: Tensor
    outside_probability: Tensor
    expected_count: Tensor
    outside_expected_count: Tensor
    residual_logits: Tensor
    outside_residual_logits: Tensor


class SphericalConv3d(nn.Module):
    """Same-size convolution with longitude wrapping and pole-aware latitude padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel = _triple(kernel_size)
        self.padding = tuple(value // 2 for value in kernel)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel, padding=0, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 5:
            raise ValueError("SphericalConv3d expects [B,C,D,H,W]")
        depth_pad, latitude_pad, longitude_pad = self.padding
        if latitude_pad:
            if inputs.shape[-1] % 2:
                raise ValueError("pole-aware padding requires an even longitude dimension")
            south = torch.roll(
                inputs[..., :latitude_pad, :].flip(-2),
                shifts=inputs.shape[-1] // 2,
                dims=-1,
            )
            north = torch.roll(
                inputs[..., -latitude_pad:, :].flip(-2),
                shifts=inputs.shape[-1] // 2,
                dims=-1,
            )
            inputs = torch.cat((south, inputs, north), dim=-2)
        if longitude_pad:
            inputs = F.pad(inputs, (longitude_pad, longitude_pad, 0, 0, 0, 0), mode="circular")
        if depth_pad:
            inputs = F.pad(inputs, (0, 0, 0, 0, depth_pad, depth_pad), mode="replicate")
        return self.conv(inputs)


class SphericalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            SphericalConv3d(in_channels, out_channels, 3),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            SphericalConv3d(out_channels, out_channels, 3),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class SphericalConvGRU3DCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        combined = int(input_channels) + self.hidden_channels
        self.gates = SphericalConv3d(combined, 2 * self.hidden_channels, 3)
        self.candidate = SphericalConv3d(combined, self.hidden_channels, 3)

    def forward(self, inputs: Tensor, hidden: Tensor | None = None) -> Tensor:
        if hidden is None:
            hidden = inputs.new_zeros(inputs.shape[0], self.hidden_channels, *inputs.shape[-3:])
        reset, update = torch.sigmoid(self.gates(torch.cat((inputs, hidden), dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((inputs, reset * hidden), dim=1)))
        return (1.0 - update) * hidden + update * candidate


class _LeadDecoder(nn.Module):
    def __init__(self, base_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.prior_encoder = nn.Sequential(
            SphericalConv3d(1, base_channels, 3),
            nn.GELU(),
        )
        self.bottleneck = SphericalBlock(hidden_channels + base_channels, 4 * base_channels)
        self.decode_one = SphericalBlock(4 * base_channels + 2 * base_channels, 2 * base_channels)
        self.decode_zero = SphericalBlock(2 * base_channels + base_channels, base_channels)
        self.voxel_out = SphericalConv3d(base_channels, 1, 3)
        self.outside_out = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(4 * base_channels, 2),
        )
        nn.init.zeros_(self.voxel_out.conv.weight)
        nn.init.zeros_(self.voxel_out.conv.bias)
        nn.init.zeros_(self.outside_out[-1].weight)
        nn.init.zeros_(self.outside_out[-1].bias)

    def forward(
        self,
        hidden: Tensor,
        skip_zero: Tensor,
        skip_one: Tensor,
        prior_probability: Tensor,
    ) -> tuple[Tensor, Tensor]:
        prior = self.prior_encoder(prior_probability.unsqueeze(1))
        prior_bottleneck = F.adaptive_avg_pool3d(prior, hidden.shape[-3:])
        decoded = self.bottleneck(torch.cat((hidden, prior_bottleneck), dim=1))
        outside_residual = self.outside_out(decoded)
        decoded = F.interpolate(decoded, size=skip_one.shape[-3:], mode="trilinear", align_corners=False)
        decoded = self.decode_one(torch.cat((decoded, skip_one), dim=1))
        decoded = F.interpolate(decoded, size=skip_zero.shape[-3:], mode="trilinear", align_corners=False)
        decoded = self.decode_zero(torch.cat((decoded, skip_zero), dim=1))
        return self.voxel_out(decoded).squeeze(1), outside_residual


class PhysicsResidualUNetConvGRU3D(nn.Module):
    """Direct multi-lead, closed-cohort forecast around a causal physics prior."""

    def __init__(
        self,
        input_channels: int,
        lead_count: int,
        *,
        base_channels: int = 8,
        hidden_channels: int = 16,
        residual_scale: float = 4.0,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if min(input_channels, lead_count, base_channels, hidden_channels) <= 0:
            raise ValueError("channel and lead counts must be positive")
        if residual_scale <= 0 or epsilon <= 0:
            raise ValueError("residual_scale and epsilon must be positive")
        self.input_channels = int(input_channels)
        self.lead_count = int(lead_count)
        self.residual_scale = float(residual_scale)
        self.epsilon = float(epsilon)
        self.encoder_zero = SphericalBlock(input_channels, base_channels)
        self.encoder_one = SphericalBlock(base_channels, 2 * base_channels)
        self.encoder_two = SphericalBlock(2 * base_channels, 4 * base_channels)
        self.bottleneck_projection = SphericalConv3d(4 * base_channels, hidden_channels, 1)
        self.temporal_cell = SphericalConvGRU3DCell(hidden_channels, hidden_channels)
        self.decoders = nn.ModuleList(
            [_LeadDecoder(base_channels, hidden_channels) for _ in range(lead_count)]
        )

    def _encode(self, frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        zero = self.encoder_zero(frame)
        one = self.encoder_one(F.avg_pool3d(zero, kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        two = self.encoder_two(F.avg_pool3d(one, kernel_size=2, stride=2))
        return zero, one, self.bottleneck_projection(two)

    def _validate(
        self,
        history: Tensor,
        prior_voxel: Tensor,
        prior_outside: Tensor,
        future_total: Tensor,
    ) -> None:
        if history.ndim != 6 or history.shape[2] != self.input_channels:
            raise ValueError("history must have shape [B,T,input_channels,D,H,W]")
        if prior_voxel.ndim != 5 or prior_voxel.shape[1] != self.lead_count:
            raise ValueError("physics_prior must have shape [B,K,D,H,W]")
        if prior_outside.shape != (*prior_voxel.shape[:2], 2):
            raise ValueError("physics_prior_outside must have shape [B,K,2]")
        if future_total.shape != prior_voxel.shape[:2]:
            raise ValueError("future_total must have shape [B,K]")
        if history.shape[0] != prior_voxel.shape[0] or history.shape[-3:] != prior_voxel.shape[-3:]:
            raise ValueError("history and prior batch/spatial dimensions must agree")
        for value in (history, prior_voxel, prior_outside, future_total):
            if not torch.isfinite(value).all():
                raise ValueError("model inputs must be finite")
        if torch.any(prior_voxel < 0) or torch.any(prior_outside < 0) or torch.any(future_total < 0):
            raise ValueError("prior masses and totals cannot be negative")
        if min(history.shape[-3:]) < 4:
            raise ValueError("each spatial dimension must be at least four cells")
        if history.shape[-2] % 4 or history.shape[-1] % 4 or history.shape[-1] % 2:
            raise ValueError("latitude/longitude dimensions must be divisible by four")

    def forward(
        self,
        history: Tensor,
        physics_prior: Tensor,
        physics_prior_outside: Tensor,
        future_total: Tensor,
    ) -> FormalForecastOutput:
        self._validate(history, physics_prior, physics_prior_outside, future_total)
        hidden: Tensor | None = None
        skip_zero: Tensor | None = None
        skip_one: Tensor | None = None
        for time_index in range(history.shape[1]):
            skip_zero, skip_one, encoded = self._encode(history[:, time_index])
            hidden = self.temporal_cell(encoded, hidden)
        if hidden is None or skip_zero is None or skip_one is None:
            raise RuntimeError("history contains no time step")

        voxel_probabilities: list[Tensor] = []
        outside_probabilities: list[Tensor] = []
        voxel_counts: list[Tensor] = []
        outside_counts: list[Tensor] = []
        voxel_residuals: list[Tensor] = []
        outside_residuals: list[Tensor] = []
        for lead_index, decoder in enumerate(self.decoders):
            prior_voxel_mass = physics_prior[:, lead_index].to(dtype=history.dtype)
            prior_outside_mass = physics_prior_outside[:, lead_index].to(dtype=history.dtype)
            prior_categories = torch.cat(
                (prior_voxel_mass.flatten(start_dim=1), prior_outside_mass), dim=1
            )
            prior_categories = prior_categories / prior_categories.sum(dim=1, keepdim=True).clamp_min(self.epsilon)
            prior_voxel_probability = prior_categories[:, :-2].reshape_as(prior_voxel_mass)
            residual_voxel, residual_outside = decoder(
                hidden, skip_zero, skip_one, prior_voxel_probability
            )
            residual_categories = torch.cat(
                (residual_voxel.flatten(start_dim=1), residual_outside), dim=1
            )
            logits = torch.log(prior_categories.clamp_min(self.epsilon)) + self.residual_scale * torch.tanh(
                residual_categories
            )
            probability = torch.softmax(logits, dim=1)
            voxel_probability = probability[:, :-2].reshape_as(prior_voxel_mass)
            outside_probability = probability[:, -2:]
            total = future_total[:, lead_index].to(dtype=history.dtype)
            voxel_probabilities.append(voxel_probability)
            outside_probabilities.append(outside_probability)
            voxel_counts.append(voxel_probability * total[:, None, None, None])
            outside_counts.append(outside_probability * total[:, None])
            voxel_residuals.append(residual_voxel)
            outside_residuals.append(residual_outside)
        return FormalForecastOutput(
            voxel_probability=torch.stack(voxel_probabilities, dim=1),
            outside_probability=torch.stack(outside_probabilities, dim=1),
            expected_count=torch.stack(voxel_counts, dim=1),
            outside_expected_count=torch.stack(outside_counts, dim=1),
            residual_logits=torch.stack(voxel_residuals, dim=1),
            outside_residual_logits=torch.stack(outside_residuals, dim=1),
        )


class PhysicsResidualUNetLastFrame3D(PhysicsResidualUNetConvGRU3D):
    """A non-recurrent U-Net baseline around the same causal physics prior.

    This model deliberately shares the V1 encoder, lead-specific decoders,
    prior-residual parameterization, and count-conservation construction.  Its
    only architectural difference is that it encodes ``history[:, -1]`` once
    instead of passing the six historical frames through ConvGRU.  It is used
    by the Phase C ablation as V0, not by the formal trainer.
    """

    def __init__(
        self,
        input_channels: int,
        lead_count: int,
        *,
        base_channels: int = 8,
        hidden_channels: int = 16,
        residual_scale: float = 4.0,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__(
            input_channels,
            lead_count,
            base_channels=base_channels,
            hidden_channels=hidden_channels,
            residual_scale=residual_scale,
            epsilon=epsilon,
        )
        # Registering no recurrent module is the defining V0 ablation.
        del self.temporal_cell

    def forward(
        self,
        history: Tensor,
        physics_prior: Tensor,
        physics_prior_outside: Tensor,
        future_total: Tensor,
    ) -> FormalForecastOutput:
        self._validate(history, physics_prior, physics_prior_outside, future_total)
        skip_zero, skip_one, hidden = self._encode(history[:, -1])

        voxel_probabilities: list[Tensor] = []
        outside_probabilities: list[Tensor] = []
        voxel_counts: list[Tensor] = []
        outside_counts: list[Tensor] = []
        voxel_residuals: list[Tensor] = []
        outside_residuals: list[Tensor] = []
        for lead_index, decoder in enumerate(self.decoders):
            prior_voxel_mass = physics_prior[:, lead_index].to(dtype=history.dtype)
            prior_outside_mass = physics_prior_outside[:, lead_index].to(dtype=history.dtype)
            prior_categories = torch.cat(
                (prior_voxel_mass.flatten(start_dim=1), prior_outside_mass), dim=1
            )
            prior_categories = prior_categories / prior_categories.sum(dim=1, keepdim=True).clamp_min(self.epsilon)
            prior_voxel_probability = prior_categories[:, :-2].reshape_as(prior_voxel_mass)
            residual_voxel, residual_outside = decoder(
                hidden, skip_zero, skip_one, prior_voxel_probability
            )
            residual_categories = torch.cat(
                (residual_voxel.flatten(start_dim=1), residual_outside), dim=1
            )
            logits = torch.log(prior_categories.clamp_min(self.epsilon)) + self.residual_scale * torch.tanh(
                residual_categories
            )
            probability = torch.softmax(logits, dim=1)
            voxel_probability = probability[:, :-2].reshape_as(prior_voxel_mass)
            outside_probability = probability[:, -2:]
            total = future_total[:, lead_index].to(dtype=history.dtype)
            voxel_probabilities.append(voxel_probability)
            outside_probabilities.append(outside_probability)
            voxel_counts.append(voxel_probability * total[:, None, None, None])
            outside_counts.append(outside_probability * total[:, None])
            voxel_residuals.append(residual_voxel)
            outside_residuals.append(residual_outside)
        return FormalForecastOutput(
            voxel_probability=torch.stack(voxel_probabilities, dim=1),
            outside_probability=torch.stack(outside_probabilities, dim=1),
            expected_count=torch.stack(voxel_counts, dim=1),
            outside_expected_count=torch.stack(outside_counts, dim=1),
            residual_logits=torch.stack(voxel_residuals, dim=1),
            outside_residual_logits=torch.stack(outside_residuals, dim=1),
        )
