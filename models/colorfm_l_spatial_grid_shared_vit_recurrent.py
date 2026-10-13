"""Shared ViT with a recurrent feature state for consecutive video frames."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.colorfm_l_spatial_grid_shared_vit import ColorFM_L as SharedViT


class SimpleGateFeatureFusion(nn.Module):
    """NAFNet-style gated residual fusion of current and previous features."""

    def __init__(self, channels):
        super().__init__()
        self.project_in = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.depthwise = nn.Conv2d(
            2 * channels, 2 * channels, 3, padding=1, groups=2 * channels,
        )
        self.project_out = nn.Conv2d(channels, channels, 1)
        # NAFNet-style residual scaling: start as an identity on current features
        # and let training learn how much temporal correction to inject.
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, current, previous):
        interaction = self.depthwise(self.project_in(torch.cat((current, previous), dim=1)))
        gate_a, gate_b = interaction.chunk(2, dim=1)
        temporal_update = self.project_out(gate_a * gate_b)
        return current + self.beta * temporal_update


class ColorFM_L(SharedViT):
    """Encode all clip frames, recurrently fuse features, then predict per-frame parameters.

    Clip inputs and outputs use [B,N,C,H,W]. Encoder execution is batched over
    B*N frames. A SimpleGate block then updates a low-resolution feature state in time
    order; the first frame uses its own encoded feature as the previous state.
    """

    supports_temporal_recurrence = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.clip_length = int(cfg.data.get("train_clip_length", 10))
        if self.clip_length < 2:
            raise ValueError("Recurrent shared ViT requires train_clip_length >= 2.")
        self.temporal_fusion = SimpleGateFeatureFusion(self.encoder.embed_dim)
        self.reset_temporal_state()

    def reset_temporal_state(self):
        """Clear streaming inference state at a new video or clip boundary."""
        self._temporal_state = None
        self._temporal_frame_index = None

    def _encode_feature_map(self, context):
        context = F.interpolate(context, size=(self.context_size, self.context_size),
                                mode="bilinear", align_corners=False)
        tokens = self.encoder.forward_features(context)["x_norm_patchtokens"]
        side = self.context_size // self.patch_size
        return tokens, tokens.transpose(1, 2).reshape(tokens.shape[0], -1, side, side)

    def _parameters_from_feature_maps(self, feature_maps):
        tokens = feature_maps.flatten(2).transpose(1, 2).contiguous()
        kernels = self.param_gen(tokens)
        grid = self.spatial_grid_head(feature_maps)
        self.params = {f"kernel{i}_single": kernel for i, kernel in enumerate(kernels, 1)}
        self.params["spatial_grid"] = grid
        return kernels, grid

    def predict_clip_parameters(self, context):
        if context.ndim != 5:
            raise ValueError(f"Expected clip context [B,N,C,H,W], got {tuple(context.shape)}.")
        batch, frames, channels, height, width = context.shape
        if frames != self.clip_length:
            raise ValueError(f"Expected {self.clip_length} consecutive frames, got {frames}.")
        _, encoded = self._encode_feature_map(context.reshape(batch * frames, channels, height, width))
        encoded = encoded.reshape(batch, frames, *encoded.shape[1:])

        state = encoded[:, 0]
        fused = []
        for index in range(frames):
            current = encoded[:, index]
            previous = current if index == 0 else state
            state = self.temporal_fusion(current, previous)
            fused.append(state)
        fused = torch.stack(fused, dim=1).reshape(batch * frames, *encoded.shape[2:])
        return self._parameters_from_feature_maps(fused)

    def forward_clip_stages(self, source, context=None, parameters=None):
        if source.ndim != 5:
            raise ValueError(f"Expected source [B,N,C,H,W], got {tuple(source.shape)}.")
        batch, frames, channels, height, width = source.shape
        if frames != self.clip_length:
            raise ValueError(f"Expected {self.clip_length} consecutive frames, got {frames}.")
        if parameters is None:
            if context is None:
                context = source
            parameters = self.predict_clip_parameters(context)
        source_flat = source.reshape(batch * frames, channels, height, width)
        global_image, residual, final_image = self.forward_single_stages(source_flat, parameters)
        return tuple(value.reshape(batch, frames, *value.shape[1:])
                     for value in (global_image, residual, final_image))

    def predict_single_parameters(self, context, frame_index=None):
        """Stateful one-frame inference; call reset_temporal_state per clip/video."""
        if context.ndim != 4 or context.shape[0] != 1:
            raise ValueError("Streaming recurrent inference expects one [1,C,H,W] frame at a time.")
        if (frame_index is not None and self._temporal_frame_index is not None
                and int(frame_index) != self._temporal_frame_index + 1):
            self.reset_temporal_state()
        _, current = self._encode_feature_map(context)
        previous = current if self._temporal_state is None else self._temporal_state
        if previous.shape != current.shape or previous.device != current.device:
            previous = current
        fused = self.temporal_fusion(current, previous)
        self._temporal_state = fused.detach()
        self._temporal_frame_index = None if frame_index is None else int(frame_index)
        return self._parameters_from_feature_maps(fused)
